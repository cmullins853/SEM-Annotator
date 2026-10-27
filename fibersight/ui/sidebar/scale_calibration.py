from typing import Callable
from nicegui import ui
from fibersight.state import STATE


def build_scale_calibration(page: dict) -> Callable[[], None]:
    """Sidebar card: scale calibration — draw line, enter µm, confirm/retry.
    Returns its reconcile step."""

    pending: dict = {'px_dist': None}       # drawn but not yet confirmed (this page only)

    with ui.expansion('Scale Calibration', icon='straighten', value=True).classes(
            'w-full fs-sidebar-card').props('dark'):
        with ui.column().classes('w-full gap-2 pt-1'):

            # ── Lock mode ──────────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2'):
                ui.label('Per-image').classes('text-xs fs-text-muted')
                ui.switch('Apply to all').bind_value(
                    STATE, 'scale_mode',
                    forward=lambda v: 'locked' if v else 'per-image',
                    backward=lambda m: m == 'locked',
                ).props('dense color=primary').classes('text-xs fs-text-muted')

            # ── Pixel distance (read-only, filled after draw) ──────────────────
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Pixel dist:').classes('text-xs fs-text-muted w-16 shrink-0')
                px_dist_label = ui.label('—').classes(
                    'text-sm font-mono fs-text-primary-dim flex-1')

            # ── Known real-world length ────────────────────────────────────────
            # No format= on physical values: ui.number rounds the stored value (REQ-48).
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Known (µm):').classes('text-xs fs-text-muted w-16 shrink-0')
                known_length_input = ui.number(
                    value=None, min=0, step=0.1, placeholder='e.g. 10.0',
                ).props('dense dark outlined suffix=µm').classes('flex-1')

            # ── Computed µm/px ─────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Scale:').classes('text-xs fs-text-muted w-16 shrink-0')
                scale_value_input = ui.number(
                    value=None, min=0, step=0.00001, placeholder='auto-calc',
                ).props('dense dark outlined suffix=µm/px').classes('flex-1')

            ui.separator().classes('my-1')

            # ── Button group — three states ────────────────────────────────────
            # idle:     [Draw Scale Line]
            # drawing:  [Cancel]
            # drawn:    [Retry]  [Confirm]
            draw_btn = ui.button('Draw Scale Line', icon='edit').props(
                'color=primary outline').classes('w-full')
            cancel_btn = ui.button('Cancel', icon='close').props(
                'color=negative outline').classes('w-full')
            with ui.row().classes('w-full gap-2'):
                retry_btn = ui.button('Retry', icon='refresh').props(
                    'color=warning outline').classes('flex-1')
                confirm_btn = ui.button('Confirm', icon='check').props(
                    'color=positive').classes('flex-1')

            # ── Status ─────────────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2 mt-1'):
                ui.label('Status:').classes('text-xs fs-text-muted')
                status_label = ui.label('Not Set').classes(
                    'text-xs font-semibold px-2 py-0.5 rounded '
                    'fs-bg-warn-status fs-text-warn-status')

    def show(mode: str) -> None:
        draw_btn.set_visibility(mode == 'idle')
        cancel_btn.set_visibility(mode == 'drawing')
        retry_btn.set_visibility(mode == 'drawn')
        confirm_btn.set_visibility(mode == 'drawn')

    def recompute() -> None:
        known = known_length_input.value
        if pending['px_dist'] and known and known > 0:
            scale_value_input.set_value(known / pending['px_dist'])

    # ── Button actions ─────────────────────────────────────────────────────────
    def on_draw() -> None:
        px_dist_label.set_text('—')
        page['start_draw']()
        show('drawing')

    def on_cancel() -> None:
        page['cancel_draw']()
        show('idle')

    def on_confirm() -> None:
        um_per_px = scale_value_input.value
        if not um_per_px or um_per_px <= 0:
            ui.notify('Enter or draw a valid scale value.', type='warning')
            return
        STATE.confirm_scale(um_per_px, known_length_input.value or 0.0, pending['px_dist'] or 0.0)
        page['cancel_draw']()
        show('idle')

    def line_drawn(px_dist: float) -> None:
        pending['px_dist'] = px_dist
        px_dist_label.set_text(f'{px_dist:.1f} px')
        recompute()
        show('drawn')

    draw_btn.on_click(on_draw)
    cancel_btn.on_click(on_cancel)
    retry_btn.on_click(on_draw)
    confirm_btn.on_click(on_confirm)
    known_length_input.on_value_change(recompute)
    page['line_drawn'] = line_drawn
    show('idle')

    # ── Reconcile step: follow STATE (restore, image switch, other tabs) ───────
    shown: dict = {'key': ()}

    def sync() -> None:
        key = (STATE.selected_image, STATE.scale_um_per_px,
               STATE.scale_known_length, STATE.scale_pixel_distance)
        if key == shown['key']:
            return
        shown['key'] = key
        if STATE.scale_confirmed:
            status_label.set_text(f'{STATE.scale_um_per_px:.5f} µm/px  ✓')
            status_label.classes(remove='fs-bg-warn-status fs-text-warn-status',
                                 add='fs-bg-ok-status fs-text-ok')
            scale_value_input.set_value(STATE.scale_um_per_px)
            if STATE.scale_pixel_distance:
                px_dist_label.set_text(f'{STATE.scale_pixel_distance:.1f} px')
            if STATE.scale_known_length:
                known_length_input.set_value(STATE.scale_known_length)
        else:
            # Clear numbers left over from the previous image; keep the known length,
            # since a folder usually shares one scale-bar length.
            pending['px_dist'] = None
            px_dist_label.set_text('—')
            scale_value_input.set_value(None)
            status_label.set_text('Not Set')
            status_label.classes(remove='fs-bg-ok-status fs-text-ok',
                                 add='fs-bg-warn-status fs-text-warn-status')

    return sync
