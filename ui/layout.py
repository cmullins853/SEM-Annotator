from nicegui import ui
from state.app_state import AppState
from ui.theme import apply_theme
from ui.header import build_header
from ui.pipeline_log import build_pipeline_log
from ui.sidebar.data_directory import build_data_directory
from ui.sidebar.scale_calibration import build_scale_calibration
from ui.sidebar.roi_crop import build_roi_crop
from ui.sidebar.pipeline_settings import build_pipeline_settings
from ui.sidebar.run_controls import build_run_controls
from ui.center.image_viewer import build_image_viewer
from ui.center.depth_tab import build_depth_tab
from ui.center.segmentation_tab import build_segmentation_tab
from ui.center.results_tab import build_results_tab


def build_page(state: AppState) -> None:
    """Assemble the full FiberSight page layout."""

    apply_theme()
    ui.dark_mode(True)

    # ── Model warning banner (hidden until startup check fires) ────────────────
    with ui.element('div').classes('w-full') as banner:
        with ui.row().classes(
                'w-full items-center gap-2 px-4 py-2 bg-amber-950 border-b border-amber-800'):
            ui.icon('warning', color='amber-400').classes('text-amber-400')
            ui.label(
                'Model weights not found — pipeline disabled. '
                'See models/README.md for setup.'
            ).classes('text-amber-200 text-sm')
    banner.set_visibility(False)
    state.model_banner = banner

    # ── Header ─────────────────────────────────────────────────────────────────
    with ui.header(elevated=True).classes(
            'bg-[#16213e] border-b border-[#2dd4bf]/20 items-center py-2'):
        build_header(state)

    # ── Left drawer — build image viewer FIRST so callbacks are on state ───────
    # Image viewer sets state.activate_scale_draw etc. before sidebar is built.
    # NiceGUI layout slots (header/drawer/main) are position-independent of
    # definition order, so we can define the center content before the drawer.

    # ── Center content (defines callbacks on state) ────────────────────────────
    with ui.column().classes('w-full min-h-screen bg-[#1a1a2e] p-4 gap-4'):

        with ui.tabs().classes('w-full') as tabs:
            img_tab = ui.tab('IMAGE', icon='image')
            depth_tab_el = ui.tab('DEPTH MAP', icon='layers')
            seg_tab_el = ui.tab('SEG', icon='grain')
            results_tab_el = ui.tab('RESULTS', icon='bar_chart')

        with ui.tab_panels(tabs, value=img_tab).classes('w-full'):
            with ui.tab_panel(img_tab):
                # image_viewer registers activate/deactivate_scale_draw on state
                build_image_viewer(state)

            with ui.tab_panel(depth_tab_el):
                build_depth_tab(state)

            with ui.tab_panel(seg_tab_el):
                build_segmentation_tab(state)

            with ui.tab_panel(results_tab_el):
                build_results_tab(state)

        build_pipeline_log(state)

    # ── Left drawer (sidebar) — built AFTER image_viewer sets state callbacks ──
    with ui.left_drawer(value=True, fixed=True, bordered=True).classes(
            'bg-[#16213e] overflow-y-auto py-3 px-2 gap-2'):
        build_data_directory(state)
        build_scale_calibration(state)
        build_roi_crop(state)
        build_pipeline_settings(state)
        build_run_controls(state)

    # ── Advanced mode: hide depth/seg tabs in Basic mode ──────────────────────
    def _update_tab_visibility():
        depth_tab_el.set_visibility(state.advanced_mode)
        seg_tab_el.set_visibility(state.advanced_mode)

    state.on_advanced_mode_changed(_update_tab_visibility)
    _update_tab_visibility()

    # ── FAB — Run Pipeline (fixed bottom-right) ────────────────────────────────
    with ui.page_sticky(position='bottom-right', x_offset=18, y_offset=18):
        run_fab = ui.button('Run Pipeline', icon='play_arrow').props(
            'fab color=teal').classes('shadow-lg')
        run_fab.set_enabled(False)

        def _update_fab():
            run_fab.set_enabled(state.can_run)

        state.on_scale_changed(_update_fab)
        state.on_image_selected(_update_fab)
        state.on_image_dir_changed(_update_fab)
        _update_fab()
