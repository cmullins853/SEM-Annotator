from nicegui import ui
from fibersight.state import STATE


def build_run_controls() -> None:
    """Sidebar card: run scope, step-through toggle, run hint."""

    with ui.expansion('Run Controls', icon='play_circle', value=True).classes(
            'w-full fs-sidebar-card').props('dark'):
        with ui.column().classes('w-full gap-3 pt-1'):

            # ── Scope toggle ──────────────────────────────────────────────────
            with ui.row().classes('items-center gap-2'):
                ui.label('Scope:').classes('text-xs fs-text-muted')
                ui.toggle(['Single Image', 'Batch Directory']).bind_value(
                    STATE, 'batch_mode',
                    forward=lambda v: v == 'Batch Directory',
                    backward=lambda b: 'Batch Directory' if b else 'Single Image',
                ).props('dense color=grey-9 text-color=grey-5 toggle-color=primary').classes('text-xs')

            # ── Step-through ──────────────────────────────────────────────────
            ui.switch('Step-through mode').props('color=primary dense') \
                .bind_value(STATE, 'step_through')

            # ── Run status hint ───────────────────────────────────────────────
            ui.label().classes('text-xs fs-text-subtle italic').bind_text_from(STATE, 'run_hint')
