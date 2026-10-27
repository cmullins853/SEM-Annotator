"""Display previews for TIFFs (spec F-61, F-67). Browsers can't show TIFF, so each one is
converted to a full-resolution PNG once and cached in <image_dir>/.fibersight/preview/.

Display only: the pipeline always reads the original. PNG is lossless, dimensions are
kept, and 8-bit pixels are unchanged; 16-bit images are percentile-scaled to 8-bit.
"""
from __future__ import annotations
import contextlib
import io
import os
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

CONVERT_EXTS = ('.tif', '.tiff')
WORKERS = 4                 # measured: 75 TIFFs in 4.4 s on 8 cores (13.9 s single-threaded)
_locks: defaultdict[Path, threading.Lock] = defaultdict(threading.Lock)


def to_display(im: Image.Image) -> Image.Image:
    """8-bit L/RGB pass through; 16/32-bit gray is scaled 2–98 percentile (like the
    ML code's scale_to_u8) instead of clipping to white; anything else becomes RGB."""
    if im.mode in ('L', 'RGB'):
        return im
    if im.mode.startswith('I') or im.mode == 'F':
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, (2, 98))
        a = (a - lo) / max(float(hi - lo), 1e-6) * 255
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im.convert('RGB')


def preview_path(src: Path) -> Path:
    # Hidden folder, original name kept: a sibling a.png would collide with a real
    # a.png and show up in the folder scan as a new image.
    return src.parent / '.fibersight' / 'preview' / f'{src.name}.png'


def render_png(src: Path) -> bytes:
    with Image.open(src) as im:
        buf = io.BytesIO()
        to_display(im).save(buf, format='PNG')
        return buf.getvalue()


def ensure_preview(src: Path) -> Path | None:
    """Cached PNG for `src`, (re)built if missing or older than the original.
    None if the source can't be read or the cache can't be written (read-only folder)."""
    dst = preview_path(src)
    tmp = None
    with _locks[dst]:                                   # one writer per file
        try:
            if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
                return dst
            data = render_png(src)                      # read first: a bad file makes no folder
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix='.tmp')
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            os.replace(tmp, dst)
            return dst
        except Exception:
            if tmp:
                with contextlib.suppress(OSError):
                    os.remove(tmp)
            return dst if dst.exists() else None


def prewarm(state, folder: str, files: list[str], current: str) -> None:
    """Build previews for a whole folder (blocking: run in a thread). Current image and
    its neighbours first. Stops if the folder changes; waits while the pipeline runs."""
    tiffs = [f for f in files if f.lower().endswith(CONVERT_EXTS)]
    i = tiffs.index(current) if current in tiffs else 0
    order = sorted(range(len(tiffs)), key=lambda j: abs(j - i))
    state.preview_total, state.preview_done = len(tiffs), 0

    def one(j: int) -> None:
        while state.status == 'running' and state.image_dir == folder:
            time.sleep(0.5)
        if state.image_dir == folder:
            ensure_preview(Path(folder) / tiffs[j])

    with ThreadPoolExecutor(WORKERS) as pool:
        for _ in pool.map(one, order):
            if state.image_dir != folder:
                return                                  # a newer folder owns the counters
            state.preview_done += 1
