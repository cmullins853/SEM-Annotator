from nicegui import ui
from state.app_state import AppState


def build_pipeline_log(state: AppState) -> None:
    """Collapsible pipeline output log panel."""

    with ui.expansion('Pipeline Output', icon='terminal', value=False).classes(
            'w-full fs-sidebar-card').props('dense dark'):

        with ui.column().classes('w-full gap-1 pt-1'):
            # Progress bar — shown while pipeline runs
            progress = ui.linear_progress(value=0).props('color=primary indeterminate').classes(
                'w-full').style('height: 4px')
            progress.set_visibility(False)

            # Log view
            log = ui.log(max_lines=500).classes('w-full h-48 text-xs')

    state.log_element = log

    def _on_pipeline_state_change():
        progress.set_visibility(state.pipeline_running)

    state._on_pipeline_state = _on_pipeline_state_change
