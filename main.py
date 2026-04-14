import sys
import io
import urllib.parse
from fastapi import Response
from nicegui import ui, app as nicegui_app
from state.app_state import AppState
from ui.layout import build_page
from backend.models import check_models_startup


@nicegui_app.get('/image')
async def serve_image(path: str):
    """Serve any local image file as PNG — converts TIFF for browser compatibility."""
    from nicegui import run as nicegui_run

    def _load() -> bytes | None:
        from PIL import Image
        decoded = urllib.parse.unquote(path)
        try:
            with Image.open(decoded) as im:
                buf = io.BytesIO()
                im.convert('RGB').save(buf, format='PNG')
                return buf.getvalue()
        except Exception:
            return None

    data = await nicegui_run.io_bound(_load)
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type='image/png')

DEBUG = '--debug' in sys.argv


@ui.page('/')
def index():
    state = AppState()
    state.debug = DEBUG
    build_page(state)
    if DEBUG:
        # Skip weight check — mark models available, keep banner hidden
        state.depth_model_available = True
        state.seg_model_available = True
    else:
        ui.timer(0.2, lambda: check_models_startup(state), once=True)


if __name__ in {'__main__', '__mp_main__'}:
    title = 'FiberSight [DEBUG]' if DEBUG else 'FiberSight'
    ui.run(
        title=title,
        dark=True,
        port=8888,
        reload=False,
        show=True,
        favicon='🔬',
    )
