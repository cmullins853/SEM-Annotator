"""On-disk persistence. Two small JSON stores, both written atomically (REQ-54):

- Restart session (REQ-50..57): `.fibersight_session.json` next to the app. Holds the
  working set (folders, selected image, settings) so a restarted server can offer
  Resume. Pointers and user edits only, never results or caches.
- Scale calibrations: `<image_dir>/.fibersight/session.json`, keyed by filename, so
  calibrations travel with the dataset. (File name kept for existing calibrations.)
"""
from __future__ import annotations
import contextlib
import json
import os
import tempfile
from pathlib import Path

SESSION_VERSION = 1                 # bump on any incompatible payload change (REQ-52)
SESSION_PATH = Path(__file__).resolve().parent.parent / '.fibersight_session.json'
FIELDS = (
    'image_dir', 'output_dir', 'selected_image',
    'scale_mode', 'scale_um_per_px', 'scale_known_length', 'scale_pixel_distance',
    'roi_crop_bottom_pct', 'roi_preview_active',
    'advanced_mode', 'step_through', 'batch_mode',
    'depth_threshold_auto', 'depth_threshold_manual',
)


def write_json_atomic(path: Path, data: dict) -> bool:
    """Temp file + os.replace; never raises (REQ-55)."""
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        print(f'[session] could not write {path}: {exc}')
        if tmp:
            with contextlib.suppress(OSError):
                os.remove(tmp)
        return False


def read_json(path: Path) -> dict | None:
    """None if missing, unreadable, or not a JSON object."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ── Restart session ────────────────────────────────────────────────────────────

def build_payload(state) -> dict:
    return {'version': SESSION_VERSION, **{k: getattr(state, k) for k in FIELDS}}


def save_session(payload: dict, path: Path = SESSION_PATH) -> bool:
    return write_json_atomic(path, payload)


def load_session(path: Path = SESSION_PATH) -> dict | None:
    """A payload worth offering, or None (REQ-57)."""
    data = read_json(path)
    if not data or data.get('version') != SESSION_VERSION:
        return None
    if not isinstance(data.get('image_dir'), str) or not os.path.isdir(data['image_dir']):
        return None
    return data


def clear_session(path: Path = SESSION_PATH) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


_last_saved: dict = {'snap': None}


def autosave(state) -> None:
    """Called from every reconcile tick; writes only when the payload changed (REQ-56)."""
    if not state.image_dir:
        return
    payload = build_payload(state)
    snap = json.dumps(payload, sort_keys=True, default=str)
    if snap != _last_saved['snap'] and save_session(payload):
        _last_saved['snap'] = snap


# ── Per-folder scale calibrations ──────────────────────────────────────────────

def _scale_path(image_dir: str) -> Path:
    return Path(image_dir) / '.fibersight' / 'session.json'


def load_image_scale(image_dir: str, filename: str) -> dict | None:
    if not image_dir:
        return None
    data = read_json(_scale_path(image_dir)) or {}
    entry = data.get('images', {}).get(filename)
    return entry if isinstance(entry, dict) else None


def save_image_scale(image_dir: str, filename: str,
                     scale_um_per_px: float, known_length: float, pixel_distance: float) -> None:
    if not image_dir:
        return
    path = _scale_path(image_dir)
    data = read_json(path) or {}
    data.setdefault('images', {}).setdefault(filename, {}).update({
        'scale_um_per_px': scale_um_per_px,
        'scale_known_length': known_length,
        'scale_pixel_distance': pixel_distance,
    })
    write_json_atomic(path, data)
