import json
import os
from typing import Optional

SESSION_DIR = '.fibersight'
SESSION_FILE = 'session.json'


def _path(image_dir: str) -> str:
    return os.path.join(image_dir, SESSION_DIR, SESSION_FILE)


def _default() -> dict:
    return {
        'version': '0.1',
        'scale_mode': 'per-image',
        'locked_scale_um_per_px': None,
        'roi_crop_bottom_pct': 0.0,
        'images': {},
    }


def load_session(image_dir: str) -> dict:
    p = _path(image_dir)
    if not os.path.isfile(p):
        return _default()
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return _default()


def save_session(image_dir: str, data: dict):
    d = os.path.join(image_dir, SESSION_DIR)
    os.makedirs(d, exist_ok=True)
    try:
        with open(_path(image_dir), 'w') as f:
            json.dump(data, f, indent=2)
    except IOError:
        pass


def load_image_scale(image_dir: str, filename: str) -> Optional[dict]:
    if not image_dir:
        return None
    return load_session(image_dir).get('images', {}).get(filename)


def save_image_scale(image_dir: str, filename: str,
                     scale_um_per_px: float, known_length: float, pixel_distance: float):
    if not image_dir:
        return
    session = load_session(image_dir)
    session.setdefault('images', {}).setdefault(filename, {}).update({
        'scale_um_per_px': scale_um_per_px,
        'scale_known_length': known_length,
        'scale_pixel_distance': pixel_distance,
    })
    save_session(image_dir, session)


def get_calibration_status(image_dir: str, filenames: list[str]) -> dict[str, bool]:
    """Returns {filename: is_calibrated} for the batch preview table."""
    if not image_dir:
        return {f: False for f in filenames}
    images = load_session(image_dir).get('images', {})
    return {f: images.get(f, {}).get('scale_um_per_px') is not None for f in filenames}
