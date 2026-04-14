from nicegui import ui
from state.app_state import AppState


def build_roi_crop(state: AppState) -> None:
    """Sidebar card: ROI / metadata bar crop (Advanced mode only)."""

    with ui.expansion('ROI / Crop', icon='crop', value=False).classes(
            'w-full fs-sidebar-card').props('dark') as card:

        with ui.column().classes('w-full gap-3 pt-1'):

            ui.label(
                'Exclude bottom N% of image from analysis.\n'
                'Useful for SEM metadata bars.'
            ).classes('text-xs text-gray-400')

            # ── Bottom crop slider ────────────────────────────────────────────
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Crop:').classes('text-xs text-gray-400 w-8')
                crop_slider = ui.slider(min=0, max=50, step=1, value=0).props(
                    'color=teal label').classes('flex-1')
                crop_pct_label = ui.label('0%').classes('text-xs text-teal-300 w-8 text-right')

            # ── Preview toggle ────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2'):
                preview_switch = ui.switch('Show excluded region').props('color=teal dense')
                preview_switch.set_value(False)

            def _update_roi():
                pct = crop_slider.value
                crop_pct_label.set_text(f'{pct}%')
                state.set_roi(pct, preview_switch.value)

            crop_slider.on('update:model-value', lambda _: _update_roi())
            preview_switch.on('update:model-value', lambda _: _update_roi())

    # Hide/show based on advanced mode
    def _on_advanced():
        card.set_visibility(state.advanced_mode)

    state.on_advanced_mode_changed(_on_advanced)
    _on_advanced()
