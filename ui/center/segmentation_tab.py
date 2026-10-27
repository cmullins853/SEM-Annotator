from nicegui import ui
from state.app_state import AppState


def build_segmentation_tab(state: AppState) -> None:
    """SEGMENTATION tab — placeholder; Phase 2 wires real pipeline output."""
    with ui.column().classes('w-full items-center gap-4 p-8'):
        ui.icon('grain', size='3rem').classes('fs-text-faint')
        ui.label('Segmentation').classes('text-lg font-semibold fs-text-muted')
        ui.label('Run pipeline to generate segmentation masks.').classes(
            'fs-text-faint italic text-sm')
        # Phase 2:
        #   color-coded instance mask overlay on original image
        #   fiber count badge (top-right)
        #   "Show mask only" toggle
