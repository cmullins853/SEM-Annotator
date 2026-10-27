from nicegui import ui
from fibersight.state import STATE


def build_pipeline_settings() -> None:
    """Sidebar card: pipeline settings (Advanced mode only)."""

    with ui.expansion('Pipeline Settings', icon='tune', value=False).classes(
            'w-full fs-sidebar-card').props('dark') \
            .bind_visibility_from(STATE, 'advanced_mode'):

        with ui.column().classes('w-full gap-3 pt-1'):

            # ── Depth threshold ───────────────────────────────────────────────
            ui.label('Depth Threshold').classes('text-xs fs-text-muted')
            ui.checkbox('Auto (Otsu)').props('color=primary dense') \
                .bind_value(STATE, 'depth_threshold_auto')

            with ui.row().classes('items-center gap-2 w-full'):
                ui.slider(min=0, max=255, step=1).props('color=primary label').classes('flex-1') \
                    .bind_value(STATE, 'depth_threshold_manual') \
                    .bind_enabled_from(STATE, 'depth_threshold_auto', backward=lambda a: not a)
                ui.label().classes('text-xs fs-text-primary-dim w-8 text-right') \
                    .bind_text_from(STATE, 'depth_threshold_manual', backward=lambda v: str(int(v)))
