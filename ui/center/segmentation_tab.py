from nicegui import ui
from state.app_state import AppState


def build_segmentation_tab(state: AppState) -> None:
    """SEGMENTATION tab — placeholder; Phase 2 wires real pipeline output."""
    with ui.column().classes('w-full items-center gap-4 p-8'):
        ui.icon('grain', size='3rem').classes('text-gray-600')
        ui.label('Segmentation').classes('text-lg font-semibold text-gray-400')
        ui.label('Run pipeline to generate segmentation masks.').classes(
            'text-gray-600 italic text-sm')
        # Phase 2:
        #   color-coded instance mask overlay on original image
        #   fiber count badge (top-right)
        #   "Show mask only" toggle
