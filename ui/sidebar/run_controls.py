from nicegui import ui
from state.app_state import AppState


def build_run_controls(state: AppState) -> None:
    """Sidebar card: run scope, step-through toggle."""

    with ui.expansion('Run Controls', icon='play_circle', value=True).classes(
            'w-full fs-sidebar-card').props('dark'):
        with ui.column().classes('w-full gap-3 pt-1'):

            # ── Scope toggle ──────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2'):
                ui.label('Scope:').classes('text-xs text-gray-400')
                ui.toggle(
                    ['Single Image', 'Batch Directory'],
                    value='Single Image',
                    on_change=lambda e: setattr(
                        state, 'batch_mode', e.value == 'Batch Directory'
                    ),
                ).props('dense color=teal').classes('text-xs')

            # ── Step-through ──────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2'):
                ui.switch(
                    'Step-through mode',
                    on_change=lambda e: setattr(state, 'step_through', e.value),
                ).props('color=teal dense')

            # ── Run status hint ───────────────────────────────────────────────
            run_hint = ui.label('').classes('text-xs text-gray-500 italic')

            def _update_hint():
                if not state.selected_image:
                    run_hint.set_text('Load an image first.')
                elif not state.scale_confirmed:
                    run_hint.set_text('Calibrate scale to enable run.')
                else:
                    run_hint.set_text('')

            state.on_image_selected(_update_hint)
            state.on_scale_changed(_update_hint)
            state.on_image_dir_changed(_update_hint)
            _update_hint()
