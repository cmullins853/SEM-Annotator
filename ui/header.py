from nicegui import ui
from state.app_state import AppState
from utils.config import VERSION


def build_header(state: AppState) -> None:
    """Header bar: branding, version chip, Basic/Advanced mode toggle."""

    with ui.row().classes('w-full items-center gap-3 px-2'):

        # ── Branding ───────────────────────────────────────────────────────────
        ui.icon('biotech', size='1.8rem').classes('text-teal-400')
        with ui.column().classes('gap-0'):
            ui.label('FiberSight').classes('text-lg font-bold text-white leading-tight')
            ui.label('SEM Fiber Analysis').classes('text-xs text-gray-400 leading-tight')

        ui.space()

        # ── Version chip ───────────────────────────────────────────────────────
        ui.chip(f'v{VERSION}').props('outline color=teal dense').classes('text-xs')

        # ── Basic / Advanced toggle ────────────────────────────────────────────
        ui.toggle(
            ['Basic', 'Advanced'],
            value='Basic',
            on_change=lambda e: state.set_advanced_mode(e.value == 'Advanced'),
        ).props('dense color=teal').classes('text-xs')
