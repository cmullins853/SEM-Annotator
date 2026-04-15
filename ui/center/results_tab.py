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
                with ui.card().classes('fs-bg-surface p-4 min-w-[130px] items-center gap-1'):
                    lbl = ui.label('—').classes('text-2xl font-bold fs-text-primary')
                    ui.label(title).classes('text-xs fs-text-muted')
                    stat_labels[key] = lbl

        # ── Histogram placeholder ──────────────────────────────────────────────
        with ui.card().classes('w-full fs-bg-surface p-4 gap-2'):
            with ui.row().classes('items-center justify-between w-full'):
                ui.label('Diameter Histogram').classes('text-sm fs-text-normal font-medium')
            with ui.column().classes('w-full items-center py-8'):
                ui.icon('bar_chart', size='3rem').classes('fs-text-faint')
                ui.label('Run pipeline to see results.').classes(
                    'fs-text-faint italic text-sm')
            # Phase 2: ui.plotly histogram with adjustable bin count slider

        # ── Export buttons ─────────────────────────────────────────────────────
        with ui.row().classes('gap-2 flex-wrap'):
            ui.button('Download CSV', icon='download').props(
                'outline color=primary').classes('text-sm').tooltip('Export per-fiber measurements')
            ui.button('Download Annotated Image', icon='image').props(
                'outline color=primary').classes('text-sm').tooltip('PNG with fiber overlays')
            ui.button('Download All (ZIP)', icon='folder_zip').props(
                'outline color=primary').classes('text-sm').tooltip(
                'CSV + annotated image + stats JSON')
