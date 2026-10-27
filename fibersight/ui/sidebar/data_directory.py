from typing import Callable
from nicegui import ui, run
from fibersight.files import browse_directory
from fibersight.state import STATE


def build_data_directory() -> Callable[[], None]:
    """Sidebar card: image/output directory selection + image picker.
    Returns its reconcile step."""

    async def set_dir(path: str) -> None:
        # Writes STATE only, so it is safe even if this page is gone after the await (REQ-30).
        if path and path != STATE.image_dir:
            await run.io_bound(STATE.set_image_dir, path)

    async def browse_images() -> None:
        await set_dir(await run.io_bound(browse_directory))

    async def browse_output() -> None:
        path = await run.io_bound(browse_directory)
        if path:
            STATE.output_dir = path

    with ui.expansion('Data Directory', icon='folder_open', value=True).classes(
            'w-full fs-sidebar-card').props('dark'):
        with ui.column().classes('w-full gap-3 pt-1'):

            # ── Image directory ────────────────────────────────────────────────
            ui.label('Image Directory').classes('text-xs fs-text-muted')
            with ui.row().classes('w-full gap-1 items-center'):
                # Not bind_value_from: on a plain object it re-forces the value every
                # 0.1 s and wipes typing. sync() updates it only when STATE changes.
                img_dir_input = ui.input(placeholder='Path to SEM images...') \
                    .classes('flex-1').props('dense dark outlined')
                img_dir_input.on('keydown.enter', lambda: set_dir(img_dir_input.value.strip()))
                img_dir_input.on('blur', lambda: set_dir(img_dir_input.value.strip()))
                ui.button(icon='folder_open', on_click=browse_images).props(
                    'flat dense color=accent').tooltip('Browse for images')

            # ── Output directory ───────────────────────────────────────────────
            ui.label('Output Directory').classes('text-xs fs-text-muted mt-1')
            with ui.row().classes('w-full gap-1 items-center'):
                ui.input(placeholder='Path for results...').classes('flex-1') \
                    .props('dense dark outlined').bind_value(STATE, 'output_dir')
                ui.button(icon='folder_open', on_click=browse_output).props(
                    'flat dense color=primary')

            # ── Image selector ─────────────────────────────────────────────────
            ui.label('Image').classes('text-xs fs-text-muted mt-1')
            image_select = ui.select(
                options=[],
                label='Select image',
                on_change=lambda e: STATE.select_image(e.value) if e.value else None,
            ).classes('w-full').props('dense dark outlined')

            # ── Prev / Next ────────────────────────────────────────────────────
            with ui.row().classes('w-full gap-2 justify-center'):
                ui.button(icon='chevron_left', on_click=lambda: STATE.step_image(-1)).props(
                    'flat dense color=primary').tooltip('Previous image')
                ui.button(icon='chevron_right', on_click=lambda: STATE.step_image(+1)).props(
                    'flat dense color=primary').tooltip('Next image')

            # ── Preview cache progress (hidden when done) ──────────────────────
            ui.label().classes('text-xs fs-text-subtle italic w-full text-center') \
                .bind_text_from(STATE, 'preview_done',
                                backward=lambda d: f'Preparing previews {d} / {STATE.preview_total}') \
                .bind_visibility_from(STATE, 'preview_done', backward=lambda d: d < STATE.preview_total)

    # ── Reconcile step: the select follows STATE (options need set_options) ────
    shown: dict = {'image_dir': None}

    def sync() -> None:
        if STATE.image_dir != shown['image_dir']:
            shown['image_dir'] = STATE.image_dir
            img_dir_input.set_value(STATE.image_dir)
        if image_select.options != STATE.image_files:
            image_select.set_options(list(STATE.image_files), value=STATE.selected_image or None)
        elif image_select.value != (STATE.selected_image or None):
            image_select.set_value(STATE.selected_image or None)

    return sync
