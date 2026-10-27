"""Page assembly, the per-page reconcile loop, and the Resume prompt.

How a page stays in sync (docs/SPEC.md §3):
- Simple fields use NiceGUI bindings to STATE (removed automatically when the page dies).
- Everything else goes in a builder's `sync()`, called from one reconcile timer per page.
  A refresh just rebuilds the page and the first reconcile repaints it from STATE.
"""
import os
import time

from nicegui import background_tasks, run, ui

from fibersight import session
from fibersight.state import STATE, run_job
from fibersight.ui.theme import apply_theme
from fibersight.ui.header import build_header
from fibersight.ui.pipeline_log import build_pipeline_log
from fibersight.ui.sidebar.data_directory import build_data_directory
from fibersight.ui.sidebar.scale_calibration import build_scale_calibration
from fibersight.ui.sidebar.roi_crop import build_roi_crop
from fibersight.ui.sidebar.pipeline_settings import build_pipeline_settings
from fibersight.ui.sidebar.run_controls import build_run_controls
from fibersight.ui.center.image_viewer import build_image_viewer
from fibersight.ui.center.depth_tab import build_depth_tab
from fibersight.ui.center.segmentation_tab import build_segmentation_tab
from fibersight.ui.center.results_tab import build_results_tab

_resume: dict = {'done': False, 'holder': None}   # server scope, once per server start (REQ-59)


def build_page() -> None:
    """Assemble the full FiberSight page layout."""
    apply_theme()
    ui.dark_mode(True)

    # Per-page registry linking the image viewer and the scale sidebar (REQ-06).
    page: dict = {}

    # ── Model warning banner ───────────────────────────────────────────────────
    with ui.element('div').classes('w-full') as banner:
        with ui.row().classes(
                'w-full items-center gap-2 px-4 py-2 fs-bg-warn border-b fs-border-warn'):
            ui.icon('warning', color='warning').classes('fs-text-warn-icon')
            ui.label(
                'Model weights not found — pipeline disabled. '
                'See weights/README.md for setup, then restart.'
            ).classes('fs-text-warn text-sm')
    banner.bind_visibility_from(STATE, 'models_ok', backward=lambda ok: not ok)

    # ── Header ─────────────────────────────────────────────────────────────────
    with ui.header(elevated=True).classes(
            'fs-bg-surface border-b fs-border-primary items-center py-2'):
        build_header()

    # ── Center content ─────────────────────────────────────────────────────────
    # Built before the drawer so the viewer's draw hooks are in `page` first.
    with ui.column().classes('w-full min-h-screen fs-bg-main p-4 gap-4'):

        with ui.tabs().classes('w-full') as tabs:
            img_tab = ui.tab('IMAGE', icon='image')
            depth_tab_el = ui.tab('DEPTH MAP', icon='layers')
            seg_tab_el = ui.tab('SEG', icon='grain')
            results_tab_el = ui.tab('RESULTS', icon='bar_chart')
        depth_tab_el.bind_visibility_from(STATE, 'advanced_mode')
        seg_tab_el.bind_visibility_from(STATE, 'advanced_mode')

        with ui.tab_panels(tabs, value=img_tab).classes('w-full'):
            with ui.tab_panel(img_tab):
                viewer_sync = build_image_viewer(page)
            with ui.tab_panel(depth_tab_el):
                build_depth_tab()
            with ui.tab_panel(seg_tab_el):
                build_segmentation_tab()
            with ui.tab_panel(results_tab_el):
                build_results_tab()

        log_sync = build_pipeline_log()

    # ── Left drawer (sidebar) ──────────────────────────────────────────────────
    with ui.left_drawer(value=True, fixed=True, bordered=True).classes(
            'fs-bg-surface overflow-y-auto py-3 px-2 gap-2'):
        dir_sync = build_data_directory()
        scale_sync = build_scale_calibration(page)
        build_roi_crop()
        build_pipeline_settings()
        build_run_controls()

    # ── FAB — Run Pipeline ─────────────────────────────────────────────────────
    with ui.page_sticky(position='bottom-right', x_offset=18, y_offset=18):
        ui.button('Run Pipeline', icon='play_arrow',
                  on_click=lambda: background_tasks.create(run_job(), name='pipeline'),
                  ).props('fab color=primary').classes('shadow-lg').bind_enabled_from(STATE, 'can_run')

    # ── Reconcile loop (REQ-10..15) ────────────────────────────────────────────
    view: dict = {'status': None}           # what this page last showed (REQ-05)

    def reconcile() -> None:
        try:
            viewer_sync()
            dir_sync()
            scale_sync()
            log_sync()
            if view['status'] is not None and STATE.status != view['status']:
                if STATE.status == 'complete':
                    ui.notify('Pipeline finished', type='positive')
                elif STATE.status == 'error':
                    ui.notify(f'Pipeline error: {STATE.error or "see log"}', type='negative')
            view['status'] = STATE.status
            session.autosave(STATE)
        except Exception as exc:                      # REQ-12: the timer must never die
            print(f'[gui] reconcile failed: {exc!r}')

    reconcile()
    ui.timer(0.5, reconcile)
    ui.timer(0.3, _offer_resume, once=True)


async def _offer_resume() -> None:
    """REQ-57..61: offered, never automatic; only an explicit answer consumes it.

    A refreshed page keeps its old client alive for reconnect_timeout (30 s), so a
    plain "offered" flag let that dead page hold the prompt and the new page skip it.
    Instead track which client shows it, and take over if that one is disconnected.
    """
    holder = _resume['holder']
    if _resume['done'] or STATE.image_dir or (holder is not None and holder.has_socket_connection):
        return
    saved = session.load_session()
    if not saved:
        _resume['done'] = True
        return
    client = ui.context.client
    _resume['holder'] = client
    saved_at = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(session.SESSION_PATH)))
    with ui.dialog() as dlg, ui.card():
        ui.label('Resume previous session?').classes('text-lg font-semibold')
        ui.label(saved['image_dir']).classes('text-sm font-mono')
        ui.label(f'{saved.get("selected_image") or "no image selected"} · saved {saved_at}').classes(
            'text-xs fs-text-muted')
        with ui.row().classes('w-full justify-end'):
            ui.button('Start fresh', on_click=lambda: dlg.submit('fresh')).props('flat')
            ui.button('Resume', on_click=lambda: dlg.submit('resume')).props('color=primary')
    try:
        answer = await dlg
    finally:                                # also on cancel: release only if still ours
        if _resume['holder'] is client:
            _resume['holder'] = None
    if answer is None:                      # Esc / backdrop / page closed: not an answer (REQ-31)
        return                              # next page asks again
    _resume['done'] = True
    if answer == 'fresh':
        session.clear_session()
        return
    if STATE.image_dir or STATE.status == 'running':    # another tab got there first
        return
    await run.io_bound(STATE.restore, saved)            # pages repaint via bindings/reconcile
