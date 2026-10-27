from nicegui import ui
from fibersight.state import STATE


def build_roi_crop() -> None:
    """Sidebar card: ROI / metadata bar crop (Advanced mode only).

    Bound straight to STATE; the image viewer's reconcile step repaints the
    overlay at most every 0.5 s, so dragging the slider can't flood the socket (REQ-44).
    """

    with ui.expansion('ROI / Crop', icon='crop', value=False).classes(
            'w-full fs-sidebar-card').props('dark') \
            .bind_visibility_from(STATE, 'advanced_mode'):

        with ui.column().classes('w-full gap-3 pt-1'):

            ui.label(
                'Exclude bottom N% of image from analysis.\n'
                'Useful for SEM metadata bars.'
            ).classes('text-xs fs-text-muted')

            # ── Bottom crop slider ────────────────────────────────────────────
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Crop:').classes('text-xs fs-text-muted w-8')
                ui.slider(min=0, max=50, step=1).props('color=primary label').classes('flex-1') \
                    .bind_value(STATE, 'roi_crop_bottom_pct')
                ui.label().classes('text-xs fs-text-primary-dim w-8 text-right') \
                    .bind_text_from(STATE, 'roi_crop_bottom_pct', backward=lambda v: f'{v:.0f}%')

            # ── Preview toggle ────────────────────────────────────────────────
            ui.switch('Show excluded region').props('color=primary dense') \
                .bind_value(STATE, 'roi_preview_active')
