from nicegui import ui
from state.app_state import AppState


def build_depth_tab(state: AppState) -> None:
    """DEPTH MAP tab — placeholder; Phase 2 wires real pipeline output."""
    with ui.column().classes('w-full items-center gap-4 p-8'):
        ui.icon('layers', size='3rem').classes('text-gray-600')
        ui.label('Depth Map').classes('text-lg font-semibold text-gray-400')
        ui.label('Run pipeline to generate depth map.').classes('text-gray-600 italic text-sm')
        # Phase 2:
        #   side-by-side: original | colorized depth map
        #   "Show mask overlay" toggle → 50% opacity composite
        #   depth slider → live mask update (debounced ~200ms)
