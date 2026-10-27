from __future__ import annotations
import math
import urllib.parse
from typing import Callable
from nicegui import ui
from nicegui.events import MouseEventArguments
from fibersight.state import STATE
from fibersight.ui.theme import SCALE_COLOR, ROI_COLOR


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

def build_image_viewer(page: dict) -> Callable[[], None]:
    """Build IMAGE tab content. Returns its reconcile step.

    Coordinate note (VERIFIED): NiceGUI calculates image_x = offsetX / clientWidth
    * naturalWidth, so image-pixel coordinates stay correct at any CSS zoom. The image
    is served at full resolution, so they are also full-resolution pixels.

    Sets on `page` (per-page, for the scale sidebar):
        page['start_draw']()   — start 2-click draw mode
        page['cancel_draw']()  — cancel draw, clear overlay
    Calls page['line_drawn'](px_dist) when the second point is placed.
    """

    draw: dict = {'active': False, 'pt1': None}
    zoom: dict = {'pct': 100}
    shown: dict = {'path': None, 'roi': None}
    # Named SVG overlay layers, composed in insertion order into img.content.
    layers: dict[str, str] = {'roi': '', 'scale': ''}

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
        with ui.element('div').classes(
                'w-full rounded border fs-border-muted overflow-auto bg-black'
        ).style('cursor: default;') as scroll_div:

            img = ui.interactive_image(
                source='',
                events=['click'],
                cross=False,
            ).style('width: 100%; display: block;')

    def paint() -> None:
        img.content = '\n'.join(f'<g id="layer-{n}">{s}</g>' for n, s in layers.items() if s)

    # ── Zoom ───────────────────────────────────────────────────────────────────
    def _do_zoom(delta: int) -> None:
        zoom['pct'] = 100 if delta == 0 else max(50, min(500, zoom['pct'] + delta))
        pct = zoom['pct']
        img.style(f'width: {pct}%; display: block;')
        zoom_label.set_text(f'{pct}%')
        max_h = 'max-height: 70vh; ' if pct > 100 else ''
        scroll_div.style(f'{max_h}cursor: {"crosshair" if draw["active"] else "default"};')

    # ── Mouse handler ──────────────────────────────────────────────────────────
    def on_mouse(e: MouseEventArguments) -> None:
        if not draw['active']:
            return
        x, y = e.image_x, e.image_y
        if draw['pt1'] is None:
            draw['pt1'] = (x, y)
            layers['scale'] = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{SCALE_COLOR}" opacity="0.9"/>'
            paint()
            tips_step.set_text('Click the END point of the scale bar.')
            return
        x1, y1 = draw['pt1']
        x2, y2 = x, (y1 if e.shift else y)          # Shift: horizontal line
        dist = math.dist((x1, y1), (x2, y2))
        layers['scale'] = _scale_line_svg(x1, y1, x2, y2, dist)
        paint()
        draw['active'] = False
        draw['pt1'] = None
        scroll_div.style('max-height: calc(100vh - 260px); cursor: default;')
        tips_step.set_text('Line drawn. Confirm or retry in the sidebar.')
        page['line_drawn'](dist)

    img.on_mouse(on_mouse)

    # ── Draw controls (called by the scale sidebar) ────────────────────────────
    def start_draw() -> None:
        draw['active'] = True
        draw['pt1'] = None
        layers['scale'] = ''
        paint()
        tips_step.set_text('Click the START point of the scale bar.')
        tips_row.set_visibility(True)
        scroll_div.style('max-height: calc(100vh - 260px); cursor: crosshair;')

    def cancel_draw() -> None:
        draw['active'] = False
        draw['pt1'] = None
        layers['scale'] = ''
        paint()
        tips_row.set_visibility(False)
        scroll_div.style('max-height: calc(100vh - 260px); cursor: default;')

    page['start_draw'] = start_draw
    page['cancel_draw'] = cancel_draw

    # ── Reconcile step: repaint only what changed (REQ-13) ─────────────────────
    def sync() -> None:
        path = STATE.selected_image_path
        if path != shown['path']:
            shown['path'] = path
            img.set_source(f'/image?path={urllib.parse.quote(path)}' if path else '')
            filename_label.set_text(STATE.selected_image or 'No image loaded')
        roi = (STATE.roi_preview_active, STATE.roi_crop_bottom_pct,
               STATE.image_width, STATE.image_height)
        if roi != shown['roi']:
            shown['roi'] = roi
            active, pct, w, h = roi
            layers['roi'] = _roi_overlay_svg(w, h, pct) if active else ''
            paint()

    return sync
