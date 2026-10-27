from typing import Callable
from nicegui import ui
from fibersight.state import STATE


def build_pipeline_log() -> Callable[[], None]:
    """Collapsible pipeline output log panel. Returns its reconcile step."""

    with ui.expansion('Pipeline Output', icon='terminal', value=False).classes(
            'w-full fs-sidebar-card').props('dense dark'):

        with ui.column().classes('w-full gap-1 pt-1'):
            ui.linear_progress(value=0, show_value=False).props('color=primary').classes(
                'w-full').style('height: 4px') \
                .bind_value_from(STATE, 'progress') \
                .bind_visibility_from(STATE, 'status', backward=lambda s: s == 'running')
            log = ui.log(max_lines=500).classes('w-full h-48 text-xs')

    shown = {'log_total': 0}

    def sync() -> None:
        # REQ-27: append only what this page hasn't shown; replay if the buffer was reset.
        pending = STATE.log_total - shown['log_total']
        if pending < 0:
            log.clear()
            pending = STATE.log_total
        if pending:
            for line in STATE.log_lines[-pending:]:
                log.push(line)
            shown['log_total'] = STATE.log_total

    return sync
