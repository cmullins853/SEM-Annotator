from nicegui import ui
from state.app_state import AppState
from backend.scale import compute_scale


def build_scale_calibration(state: AppState) -> None:
    """Sidebar card: scale calibration — draw line, enter µm, confirm/retry."""

    with ui.expansion('Scale Calibration', icon='straighten', value=True).classes(
            'w-full fs-sidebar-card').props('dark'):
        with ui.column().classes('w-full gap-2 pt-1'):

            # ── Lock mode ──────────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2'):
                ui.label('Per-image').classes('text-xs fs-text-muted')
                ui.switch(
                    'Apply to all',
                    value=False,
                    on_change=lambda e: setattr(
                        state, 'scale_mode',
                        'locked' if e.value else 'per-image'
                    ),
                ).props('dense color=primary').classes('text-xs fs-text-muted')

            # ── Pixel distance (read-only, filled after draw) ──────────────────
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Pixel dist:').classes('text-xs fs-text-muted w-16 shrink-0')
                px_dist_label = ui.label('—').classes(
                    'text-sm font-mono fs-text-primary-dim flex-1')

            # ── Known real-world length ────────────────────────────────────────
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Known (µm):').classes('text-xs fs-text-muted w-16 shrink-0')
                known_length_input = ui.number(
                    value=None, min=0, step=0.1, format='%.3f',
                    placeholder='e.g. 10.0',
                ).props('dense dark outlined suffix=µm').classes('flex-1')

            # ── Computed µm/px ─────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Scale:').classes('text-xs fs-text-muted w-16 shrink-0')
                scale_value_input = ui.number(
                    value=None, min=0, step=0.00001, format='%.5f',
                    placeholder='auto-calc',
                ).props('dense dark outlined suffix=µm/px').classes('flex-1')

            ui.separator().classes('my-1')

            # ── Button group — three states ────────────────────────────────────
            # State A (idle):      [Draw Scale Line]
            # State B (drawing):   [Cancel]
            # State C (drawn):     [Retry]  [Confirm Scale]

            draw_btn = ui.button('Draw Scale Line', icon='edit').props(
                'color=primary outline').classes('w-full')

            cancel_btn = ui.button('Cancel', icon='close').props(
                'color=negative outline').classes('w-full')
            cancel_btn.set_visibility(False)

            with ui.row().classes('w-full gap-2'):
                retry_btn = ui.button('Retry', icon='refresh').props(
                    'color=warning outline').classes('flex-1')
                confirm_btn = ui.button('Confirm', icon='check').props(
                    'color=positive').classes('flex-1')
            retry_btn.set_visibility(False)
            confirm_btn.set_visibility(False)

            # ── Status ─────────────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2 mt-1'):
                ui.label('Status:').classes('text-xs fs-text-muted')
                status_label = ui.label('Not Set').classes(
                    'text-xs font-semibold px-2 py-0.5 rounded '
                    'fs-bg-warn-status fs-text-warn-status')

            # ── Button state helpers ────────────────────────────────────────────
            def _show_idle():
                draw_btn.set_visibility(True)
                cancel_btn.set_visibility(False)
                retry_btn.set_visibility(False)
                confirm_btn.set_visibility(False)

            def _show_drawing():
                draw_btn.set_visibility(False)
                cancel_btn.set_visibility(True)
                retry_btn.set_visibility(False)
                confirm_btn.set_visibility(False)

            def _show_drawn():
                draw_btn.set_visibility(False)
                cancel_btn.set_visibility(False)
                retry_btn.set_visibility(True)
                confirm_btn.set_visibility(True)

            # ── Button actions ─────────────────────────────────────────────────
            def _on_draw():
                if state.activate_scale_draw:
                    state.activate_scale_draw()
                _show_drawing()

            def _on_cancel():
                if state.deactivate_scale_draw:
                    state.deactivate_scale_draw()
                _show_idle()

            def _on_retry():
                px_dist_label.set_text('—')
                if state.activate_scale_draw:
                    state.activate_scale_draw()
                _show_drawing()

            def _on_confirm():
                um_per_px = scale_value_input.value
                known = known_length_input.value or 0.0
                px_dist_val = (state.pending_scale_draw['px_dist']
                               if state.pending_scale_draw else 0.0)
                if not um_per_px or um_per_px <= 0:
                    ui.notify('Enter or draw a valid scale value.', type='warning')
                    return
                state.set_scale_confirmed(um_per_px, known, px_dist_val)
                if state.deactivate_scale_draw:
                    state.deactivate_scale_draw()
                _show_idle()

            draw_btn.on('click', _on_draw)
            cancel_btn.on('click', _on_cancel)
            retry_btn.on('click', _on_retry)
            confirm_btn.on('click', _on_confirm)

            # ── Draw complete callback ──────────────────────────────────────────
            def _on_scale_draw_complete():
                data = state.pending_scale_draw
                if not data:
                    return
                px_dist = data['px_dist']
                px_dist_label.set_text(f'{px_dist:.1f} px')
                if known_length_input.value and known_length_input.value > 0:
                    try:
                        val = compute_scale(px_dist, known_length_input.value)
                        scale_value_input.set_value(round(val, 6))
                    except ValueError:
                        pass
                _show_drawn()

            state.on_scale_draw_complete(_on_scale_draw_complete)

            # ── Auto-compute on known length change ────────────────────────────
            def _on_known_length_change():
                if (state.pending_scale_draw
                        and known_length_input.value
                        and known_length_input.value > 0):
                    try:
                        val = compute_scale(
                            state.pending_scale_draw['px_dist'],
                            known_length_input.value,
                        )
                        scale_value_input.set_value(round(val, 6))
                    except ValueError:
                        pass

            known_length_input.on('update:model-value',
                                  lambda _: _on_known_length_change())

            # ── Scale state change (e.g. restored from session) ────────────────
            def _on_scale_changed():
                if state.scale_confirmed and state.scale_um_per_px:
                    status_label.set_text(f'{state.scale_um_per_px:.5f} µm/px  ✓')
                    status_label.classes(
                        remove='fs-bg-warn-status fs-text-warn-status',
                        add='fs-bg-ok-status fs-text-ok')
                    scale_value_input.set_value(round(state.scale_um_per_px, 6))
                    if state.scale_pixel_distance:
                        px_dist_label.set_text(f'{state.scale_pixel_distance:.1f} px')
                    if state.scale_known_length:
                        known_length_input.set_value(state.scale_known_length)
                    _show_idle()
                else:
                    status_label.set_text('Not Set')
                    status_label.classes(
                        remove='fs-bg-ok-status fs-text-ok',
                        add='fs-bg-warn-status fs-text-warn-status')

            state.on_scale_changed(_on_scale_changed)
