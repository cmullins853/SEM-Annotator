"""FiberSight entry point: `python main.py [--debug]` (--debug skips the weights check)."""
import sys
from pathlib import Path

from fastapi import Response
from fastapi.responses import FileResponse
from nicegui import app, run, ui

from fibersight import preview
from fibersight.pipeline import weights_present
from fibersight.state import STATE
from fibersight.ui.layout import build_page

DEBUG = '--debug' in sys.argv
STATE.models_ok = DEBUG or weights_present()        # once, at server start


@app.get('/image')
async def serve_image(path: str):
    """Serve an image from the current image folder; TIFFs via the PNG preview cache.
    Served over HTTP, not the websocket, so image size can't time out the page (REQ-40)."""
    p = Path(path).resolve()
    if not STATE.image_dir or p.parent != Path(STATE.image_dir).resolve() or not p.is_file():
        return Response(status_code=404)
    if p.suffix.lower() not in preview.CONVERT_EXTS:
        return FileResponse(p)                          # browser-native: send as is
    cached = await run.io_bound(preview.ensure_preview, p)
    if cached:
        return FileResponse(cached)
    try:                                                # cache not writable: convert in memory
        return Response(content=await run.io_bound(preview.render_png, p), media_type='image/png')
    except Exception:
        return Response(status_code=404)


@ui.page('/')
def index():
    build_page()


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        title='FiberSight [DEBUG]' if DEBUG else 'FiberSight',
        dark=True,
        host='127.0.0.1',           # local only: /image reads files from disk
        port=8888,
        reload=False,
        reconnect_timeout=30.0,     # don't reap backgrounded tabs after 3 s (REQ-34)
        show=True,
        favicon='🔬',
    )
