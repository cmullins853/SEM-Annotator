"""Preview cache checks (spec F-61, F-67). Run: python -m tests.test_preview"""
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fibersight import preview
from fibersight.files import scan_images
from fibersight.state import AppState


def _make_tiffs(folder: Path) -> dict[str, Image.Image]:
    rng = np.random.default_rng(0)
    gray = Image.fromarray(rng.integers(0, 256, (300, 400), dtype=np.uint8))
    pal = gray.convert('P')                                     # palette TIFF, like Zeiss exports
    deep = Image.fromarray(rng.integers(1000, 4000, (300, 400)).astype(np.uint16))  # 16-bit SEM-like
    images = {'gray.tif': gray, 'pal.tif': pal, 'deep.tif': deep}
    for name, im in images.items():
        im.save(folder / name)
    return images


def _pixels(path: Path, display: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    """Arrays only: a live Image keeps an uncompressed TIFF memory-mapped, and Windows
    can't delete a mapped file."""
    with Image.open(path) as im:
        return np.asarray(preview.to_display(im) if display else im).copy(), im.size


def test_cached_png_is_exact_and_full_size():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name in _make_tiffs(folder):
            dst = preview.ensure_preview(folder / name)
            assert dst == folder / '.fibersight' / 'preview' / f'{name}.png'
            expected, src_size = _pixels(folder / name, display=True)
            cached, cached_size = _pixels(dst)
            assert cached_size == src_size                      # scale clicks stay in image pixels
            assert np.array_equal(cached, expected), name       # PNG is lossless
        assert scan_images(tmp) == ['deep.tif', 'gray.tif', 'pal.tif']    # cache never listed


def test_8bit_pixels_unchanged_and_16bit_not_white():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        images = _make_tiffs(folder)
        gray, _ = _pixels(preview.ensure_preview(folder / 'gray.tif'))
        assert np.array_equal(gray, np.asarray(images['gray.tif']))
        pal, _ = _pixels(preview.ensure_preview(folder / 'pal.tif'))
        assert np.array_equal(pal, np.asarray(images['pal.tif'].convert('RGB')))
        deep, _ = _pixels(preview.ensure_preview(folder / 'deep.tif'))
        assert deep.dtype == np.uint8 and len(np.unique(deep)) > 200 and (deep == 255).mean() < 0.05


def test_stale_cache_rebuilds_and_bad_file_is_none():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        _make_tiffs(folder)
        src = folder / 'gray.tif'
        dst = preview.ensure_preview(src)
        Image.new('L', (400, 300), 7).save(src)                 # original edited...
        later = time.time() + 5
        os.utime(src, (later, later))                           # ...after the cache was built
        preview.ensure_preview(src)
        assert _pixels(dst)[0].max() == 7                       # rebuilt from the new original
        (folder / 'broken.tif').write_bytes(b'not a tiff')
        assert preview.ensure_preview(folder / 'broken.tif') is None


def test_prewarm_fills_cache_and_counts():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        _make_tiffs(folder)
        state = AppState(image_dir=tmp)
        files = scan_images(tmp)
        preview.prewarm(state, tmp, files, current='gray.tif')
        assert state.preview_done == state.preview_total == 3
        assert all(preview.preview_path(folder / f).exists() for f in files)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
