from nicegui import ui
from state.app_state import AppState


def build_results_tab(state: AppState) -> None:
    """RESULTS tab — summary stats + histogram placeholder + export buttons."""

    with ui.column().classes('w-full gap-4 p-4'):

        # ── Summary stat cards ─────────────────────────────────────────────────
        with ui.row().classes('w-full gap-3 flex-wrap'):
            stat_labels = {}
            for key, title in [
                ('mean', 'Mean Diameter'),
                ('std', 'Std Dev'),
                ('median', 'Median'),
                ('count', 'Fiber Count'),
            ]:
                with ui.card().classes('bg-[#16213e] p-4 min-w-[130px] items-center gap-1'):
                    lbl = ui.label('—').classes('text-2xl font-bold text-teal-400')
                    ui.label(title).classes('text-xs text-gray-400')
                    stat_labels[key] = lbl

        # ── Histogram placeholder ──────────────────────────────────────────────
        with ui.card().classes('w-full bg-[#16213e] p-4 gap-2'):
            with ui.row().classes('items-center justify-between w-full'):
                ui.label('Diameter Histogram').classes('text-sm text-gray-300 font-medium')
            with ui.column().classes('w-full items-center py-8'):
                ui.icon('bar_chart', size='3rem').classes('text-gray-600')
                ui.label('Run pipeline to see results.').classes(
                    'text-gray-600 italic text-sm')
            # Phase 2: ui.plotly histogram with adjustable bin count slider

        # ── Export buttons ─────────────────────────────────────────────────────
        with ui.row().classes('gap-2 flex-wrap'):
            ui.button('Download CSV', icon='download').props(
                'outline color=teal').classes('text-sm').tooltip('Export per-fiber measurements')
            ui.button('Download Annotated Image', icon='image').props(
                'outline color=teal').classes('text-sm').tooltip('PNG with fiber overlays')
            ui.button('Download All (ZIP)', icon='folder_zip').props(
                'outline color=teal').classes('text-sm').tooltip(
                'CSV + annotated image + stats JSON')
