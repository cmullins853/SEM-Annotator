from __future__ import annotations
import urllib.parse
from nicegui import ui
from nicegui.events import MouseEventArguments
from state.app_state import AppState
from backend.scale import pixel_distance as euclidean_dist, pixel_distance_horizontal
from ui.theme import SCALE_COLOR, ROI_COLOR


# ── LayerManager ───────────────────────────────────────────────────────────────

class LayerManager:
    """Manages named SVG <g> layers for ui.interactive_image.

    NiceGUI's interactive_image exposes a single .content SVG string.
    LayerManager maintains named layers and recomposes the full SVG on each
    update so overlays (scale, ROI, annotations) coexist without overwriting.
    See spec §7.1.
    """

    def __init__(self, image: ui.interactive_image):
        self._image = image
        self._layers: dict[str, str] = {}
        self._order: list[str] = []

    def set_layer(self, name: str, svg_content: str) -> None:
        if name not in self._layers:
            self._order.append(name)
        self._layers[name] = svg_content
        self._recompose()

    def clear_layer(self, name: str) -> None:
        self._layers.pop(name, None)
        if name in self._order:
            self._order.remove(name)
        self._recompose()

    def clear_all(self) -> None:
        self._layers.clear()
        self._order.clear()
        self._recompose()

    def _recompose(self) -> None:
        parts = [
            f'<g id="layer-{n}">{self._layers[n]}</g>'
            for n in self._order
            if n in self._layers and self._layers[n]
        ]
        self._image.content = '\n'.join(parts)


# ── SVG helpers ────────────────────────────────────────────────────────────────

def _scale_line_svg(x1: float, y1: float, x2: float, y2: float, px_dist: float) -> str:
    mx = (x1 + x2) / 2
    my = min(y1, y2) - 14
    c = SCALE_COLOR
    return (
        f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="5" fill="{c}" opacity="0.9"/>'
        f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="5" fill="{c}" opacity="0.9"/>'
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{c}" stroke-width="2" stroke-dasharray="8,4" opacity="0.9"/>'
        f'<text x="{mx:.1f}" y="{my:.1f}" fill="{c}" text-anchor="middle" '
        f'font-size="16" font-weight="bold" paint-order="stroke" '
        f'stroke="#000000" stroke-width="3">{px_dist:.1f} px</text>'
    )


def _roi_overlay_svg(w: int, h: int, crop_pct: float) -> str:
    if crop_pct <= 0 or w == 0 or h == 0:
        return ''
    crop_y = int(h * (1.0 - crop_pct / 100.0))
    rect_h = h - crop_y
    c = ROI_COLOR
    return (
        f'<rect x="0" y="{crop_y}" width="{w}" height="{rect_h}" '
        f'fill="{c}" opacity="0.35"/>'
        f'<line x1="0" y1="{crop_y}" x2="{w}" y2="{crop_y}" '
        f'stroke="{c}" stroke-width="2" stroke-dasharray="10,5"/>'
    )


# ── Main builder ───────────────────────────────────────────────────────────────

def build_image_viewer(state: AppState) -> None:
    """Build IMAGE tab content and register UI callbacks on state.

    Coordinate note (Task 0 §7.2 — VERIFIED):
        NiceGUI calculates image_x = offsetX / clientWidth * naturalWidth.
        CSS-width zoom changes both offsetX and clientWidth proportionally,
        so image-pixel coordinates remain correct at any zoom level.

    Sets on state:
        state.activate_scale_draw()    — start 2-click draw mode
        state.deactivate_scale_draw()  — cancel draw, clear overlay
        state.update_roi_overlay()     — redraw ROI preview
    """

    _draw: dict = {'active': False, 'pt1': None}
    _zoom: dict = {'pct': 100}

    with ui.column().classes('w-full gap-2'):

        # ── Filename + zoom controls ───────────────────────────────────────────
        with ui.row().classes('w-full items-center justify-between px-1'):
            filename_label = ui.label('No image loaded').classes(
                'text-xs fs-text-subtle truncate max-w-[60%]')
            with ui.row().classes('items-center gap-1'):
                ui.button(icon='zoom_out', on_click=lambda: _do_zoom(-25)).props(
                    'flat dense color=gray size=sm').tooltip('Zoom out  (−25%)')
                zoom_label = ui.label('100%').classes('text-xs fs-text-muted w-10 text-center')
                ui.button(icon='zoom_in', on_click=lambda: _do_zoom(+25)).props(
                    'flat dense color=gray size=sm').tooltip('Zoom in  (+25%)')
                ui.button(icon='fit_screen', on_click=lambda: _do_zoom(0)).props(
                    'flat dense color=gray size=sm').tooltip('Reset zoom')

        # ── Drawing tips banner (hidden when not drawing) ──────────────────────
        tips_row = ui.row().classes(
            'w-full items-start gap-2 px-3 py-2 rounded '
            'fs-bg-tip border fs-border-tip fs-text-tip text-xs'
        )
        with tips_row:
            ui.icon('info', size='1rem').classes('fs-text-primary mt-0.5 shrink-0')
            with ui.column().classes('gap-0.5'):
                ui.label('Scale bar drawing mode').classes('font-semibold')
                tips_step = ui.label('Click the START point of the scale bar.').classes(
                    'fs-text-primary-dim')
                ui.label(
                    'Tip: Zoom in for accuracy. '
                    'Hold Shift to constrain line to horizontal.'
                ).classes('fs-text-primary italic')
        tips_row.set_visibility(False)

        # ── Scrollable image container ─────────────────────────────────────────
        # No max-height at 100% — let the image's natural aspect ratio fill the
        # panel. max-height is applied only when zoomed so the page stays tidy.
        with ui.element('div').classes(
                'w-full rounded border fs-border-muted overflow-auto bg-black'
        ).style('cursor: default;') as scroll_div:

            img = ui.interactive_image(
                source='',
                events=['click'],
                cross=False,
            ).style('width: 100%; display: block;')

    layer_mgr = LayerManager(img)

    # ── Container style helper ─────────────────────────────────────────────────
    def _container_style(cursor: str = 'default') -> str:
        pct = _zoom['pct']
        max_h = 'max-height: 70vh; ' if pct > 100 else ''
        return f'{max_h}cursor: {cursor};'

    # ── Zoom ───────────────────────────────────────────────────────────────────
    def _do_zoom(delta: int) -> None:
        if delta == 0:
            _zoom['pct'] = 100
        else:
            _zoom['pct'] = max(50, min(500, _zoom['pct'] + delta))
        pct = _zoom['pct']
        img.style(f'width: {pct}%; display: block;')
        zoom_label.set_text(f'{pct}%')
        scroll_div.style(_container_style('crosshair' if _draw['active'] else 'default'))

    # ── Mouse handler ──────────────────────────────────────────────────────────
    def on_mouse(e: MouseEventArguments) -> None:
        if not _draw['active']:
            return

        x, y, shift = e.image_x, e.image_y, e.shift

        if _draw['pt1'] is None:
            # First point placed
            _draw['pt1'] = (x, y)
            layer_mgr.set_layer(
                'scale',
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{SCALE_COLOR}" opacity="0.9"/>'
            )
            tips_step.set_text('Click the END point of the scale bar.')
        else:
            # Second point — draw full line
            x1, y1 = _draw['pt1']
            x2 = x
            y2 = y1 if shift else y
            dist = pixel_distance_horizontal(x1, x2) if shift else euclidean_dist(x1, y1, x2, y2)
            layer_mgr.set_layer('scale', _scale_line_svg(x1, y1, x2, y2, dist))
            _draw['active'] = False
            _draw['pt1'] = None
            scroll_div.style('max-height: calc(100vh - 260px); cursor: default;')
            tips_step.set_text('Line drawn. Confirm or retry in the sidebar.')
            state.notify_scale_draw_complete(x1, y1, x2, y2, dist)

    img.on_mouse(on_mouse)

    # ── Image refresh ──────────────────────────────────────────────────────────
    def refresh_image() -> None:
        path = state.selected_image_path
        if path:
            img.set_source(f'/image?path={urllib.parse.quote(path)}')
            filename_label.set_text(state.selected_image)
        else:
            img.set_source('')
            filename_label.set_text('No image loaded')
        layer_mgr.clear_layer('annotations')

    # ── Draw controls (called by scale_calibration sidebar) ───────────────────
    def do_activate_draw() -> None:
        _draw['active'] = True
        _draw['pt1'] = None
        layer_mgr.clear_layer('scale')
        tips_step.set_text('Click the START point of the scale bar.')
        tips_row.set_visibility(True)
        scroll_div.style('max-height: calc(100vh - 260px); cursor: crosshair;')

    def do_deactivate_draw() -> None:
        _draw['active'] = False
        _draw['pt1'] = None
        layer_mgr.clear_layer('scale')
        tips_row.set_visibility(False)
        scroll_div.style('max-height: calc(100vh - 260px); cursor: default;')

    # ── ROI overlay ────────────────────────────────────────────────────────────
    def do_update_roi() -> None:
        if state.roi_preview_active and state.image_width > 0:
            layer_mgr.set_layer(
                'roi',
                _roi_overlay_svg(state.image_width, state.image_height,
                                 state.roi_crop_bottom_pct)
            )
        else:
            layer_mgr.clear_layer('roi')

    # ── Register on state ──────────────────────────────────────────────────────
    state.activate_scale_draw = do_activate_draw
    state.deactivate_scale_draw = do_deactivate_draw
    state.update_roi_overlay = do_update_roi

    state.on_image_dir_changed(refresh_image)
    state.on_image_selected(refresh_image)
    state.on_roi_changed(do_update_roi)
