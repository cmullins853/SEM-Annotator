from nicegui import ui, app
from state.app_state import AppState
from utils.config import VERSION


def build_header(state: AppState) -> None:
    """Header bar: branding, version chip, Basic/Advanced mode toggle."""

    with ui.row().classes('w-full items-center gap-3 px-2'):

        # ── Branding ───────────────────────────────────────────────────────────
        ui.icon('biotech', size='1.8rem').classes('fs-text-primary')
        with ui.column().classes('gap-0'):
            ui.label('FiberSight').classes('text-lg font-bold text-white leading-tight')
            ui.label('SEM Fiber Analysis').classes('text-xs fs-text-muted leading-tight')

        ui.space()

        # ── Version chip ───────────────────────────────────────────────────────
        ui.chip(f'v{VERSION}').props('outline color=primary dense').classes('text-xs')

        # ── Basic / Advanced switch ────────────────────────────────────────────
        ui.label('Advanced Mode').classes('text-xs fs-text-muted')
        ui.switch(
            '',
            value=False,
            on_change=lambda e: state.set_advanced_mode(e.value),
        ).props('dense color=primary')

        ui.button("Shutdown", on_click=app.shutdown).props('dense color=accent')