from nicegui import ui, run
from state.app_state import AppState
from utils.file_io import browse_directory


def build_data_directory(state: AppState) -> None:
    """Sidebar card: image/output directory selection + image picker."""

    with ui.expansion('Data Directory', icon='folder_open', value=True).classes(
            'w-full fs-sidebar-card').props('dark'):
        with ui.column().classes('w-full gap-3 pt-1'):

            # ── Image directory ────────────────────────────────────────────────
            ui.label('Image Directory').classes('text-xs text-gray-400')
            with ui.row().classes('w-full gap-1 items-center'):
                img_dir_input = ui.input(
                    placeholder='Path to SEM images...',
                ).classes('flex-1').props('dense dark outlined')

                async def _browse_img():
                    path = await run.io_bound(browse_directory)
                    if path:
                        img_dir_input.set_value(path)
                        state.set_image_dir(path)

                ui.button(icon='folder_open', on_click=_browse_img).props(
                    'flat dense color=teal')

            # ── Output directory ───────────────────────────────────────────────
            ui.label('Output Directory').classes('text-xs text-gray-400 mt-1')
            with ui.row().classes('w-full gap-1 items-center'):
                out_dir_input = ui.input(
                    placeholder='Path for results...',
                ).classes('flex-1').props('dense dark outlined')

                async def _browse_out():
                    path = await run.io_bound(browse_directory)
                    if path:
                        out_dir_input.set_value(path)
                        state.output_dir = path

                ui.button(icon='folder_open', on_click=_browse_out).props(
                    'flat dense color=teal')

            # ── Image selector ─────────────────────────────────────────────────
            ui.label('Image').classes('text-xs text-gray-400 mt-1')

            image_select = ui.select(
                options=[],
                label='Select image',
                on_change=lambda e: state.set_selected_image(e.value) if e.value else None,
            ).classes('w-full').props('dense dark outlined')

            # ── Prev / Next ────────────────────────────────────────────────────
            with ui.row().classes('w-full gap-2 justify-center'):

                def _prev():
                    if not state.image_files or not state.selected_image:
                        return
                    try:
                        idx = state.image_files.index(state.selected_image)
                    except ValueError:
                        return
                    if idx > 0:
                        _select(state.image_files[idx - 1])

                def _next():
                    if not state.image_files or not state.selected_image:
                        return
                    try:
                        idx = state.image_files.index(state.selected_image)
                    except ValueError:
                        return
                    if idx < len(state.image_files) - 1:
                        _select(state.image_files[idx + 1])

                def _select(filename: str):
                    image_select.set_value(filename)
                    state.set_selected_image(filename)

                ui.button(icon='chevron_left', on_click=_prev).props(
                    'flat dense color=teal').tooltip('Previous image')
                ui.button(icon='chevron_right', on_click=_next).props(
                    'flat dense color=teal').tooltip('Next image')

            # ── Wire events ────────────────────────────────────────────────────
            def _refresh_dropdown():
                opts = state.image_files
                image_select.options = opts
                image_select.update()
                if opts:
                    image_select.set_value(opts[0])
                else:
                    image_select.set_value(None)

            def _on_img_dir_input():
                path = img_dir_input.value.strip()
                if path and path != state.image_dir:
                    state.set_image_dir(path)

            img_dir_input.on('keydown.enter', _on_img_dir_input)
            img_dir_input.on('blur', _on_img_dir_input)

            state.on_image_dir_changed(_refresh_dropdown)
