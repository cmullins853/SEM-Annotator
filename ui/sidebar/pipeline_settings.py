from nicegui import ui
from state.app_state import AppState


def build_pipeline_settings(state: AppState) -> None:
    """Sidebar card: pipeline settings (Advanced mode only)."""

    with ui.expansion('Pipeline Settings', icon='tune', value=False).classes(
            'w-full fs-sidebar-card').props('dark') as card:

        with ui.column().classes('w-full gap-3 pt-1'):

            # ── Depth threshold ───────────────────────────────────────────────
            ui.label('Depth Threshold').classes('text-xs fs-text-muted')

            with ui.row().classes('items-center gap-2'):
                auto_check = ui.checkbox('Auto (Otsu)', value=True).props('color=primary dense')

            with ui.row().classes('items-center gap-2 w-full'):
                threshold_slider = ui.slider(min=0, max=255, step=1, value=128).props(
                    'color=primary label').classes('flex-1')
                threshold_label = ui.label('128').classes('text-xs fs-text-primary-dim w-8 text-right')

            threshold_slider.set_enabled(False)

            def _on_auto_change():
                is_auto = auto_check.value
                state.depth_threshold_auto = is_auto
                threshold_slider.set_enabled(not is_auto)
                if is_auto:
                    threshold_label.set_text('Auto')

            def _on_threshold_change():
                val = int(threshold_slider.value)
                state.depth_threshold_manual = val
                threshold_label.set_text(str(val))

            auto_check.on('update:model-value', lambda _: _on_auto_change())
            threshold_slider.on('update:model-value', lambda _: _on_threshold_change())

    # Hide/show based on advanced mode
    def _on_advanced():
        card.set_visibility(state.advanced_mode)

    state.on_advanced_mode_changed(_on_advanced)
    _on_advanced()
