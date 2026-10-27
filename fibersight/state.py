"""Server-scope app state: one STATE shared by every browser tab.

Plain data only — never a UI element, bound method or closure over one
(docs/SPEC.md §3, REQ-01/02). Pages read it through bindings and
their reconcile timer, so a refresh, a second tab or a reconnect just repaints
from here.
"""
from __future__ import annotations
import asyncio
import datetime
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from fibersight import preview, session
from fibersight.files import scan_images
from fibersight.pipeline import STAGES, PipelineConfig, run_pipeline

LOG_MAX = 500


@dataclass
class AppState:
    # ── Directory ──────────────────────────────────────────────────────────────
    image_dir: str = ''
    output_dir: str = ''
    image_files: list[str] = field(default_factory=list)
    selected_image: str = ''
    image_width: int = 0
    image_height: int = 0

    # ── Scale ──────────────────────────────────────────────────────────────────
    scale_mode: str = 'per-image'           # 'locked' | 'per-image'
    scale_um_per_px: float | None = None
    scale_known_length: float | None = None
    scale_pixel_distance: float | None = None

    # ── ROI ────────────────────────────────────────────────────────────────────
    roi_crop_bottom_pct: float = 0.0
    roi_preview_active: bool = False

    # ── Mode / settings ────────────────────────────────────────────────────────
    advanced_mode: bool = False
    step_through: bool = False
    batch_mode: bool = False
    depth_threshold_auto: bool = True
    depth_threshold_manual: int = 128

    # ── Models / previews ──────────────────────────────────────────────────────
    models_ok: bool = False                 # set once at startup by main.py
    preview_done: int = 0                   # TIFF preview cache progress (preview.prewarm)
    preview_total: int = 0

    # ── Job (REQ-04) ───────────────────────────────────────────────────────────
    status: str = 'idle'                    # idle | running | complete | error
    generation: int = 0                     # +1 each time a new result lands
    progress: float = 0.0
    error: str | None = None
    result: Any = None                      # pipeline.PipelineResult
    log_lines: list[str] = field(default_factory=list)
    log_total: int = 0                      # monotonic; pages keep a cursor (REQ-27)

    # ── Mutations ──────────────────────────────────────────────────────────────
    def set_image_dir(self, path: str, select: str = '') -> None:
        """Blocking (directory scan + image header read): call via run.io_bound."""
        self.image_dir = path
        self.image_files = scan_images(path)
        self.selected_image = ''
        if self.image_files:
            self.select_image(select if select in self.image_files else self.image_files[0])
        threading.Thread(target=preview.prewarm, daemon=True,
                         args=(self, path, list(self.image_files), self.selected_image)).start()

    def select_image(self, filename: str) -> None:
        if filename == self.selected_image:
            return
        self.selected_image = filename
        self.image_width = self.image_height = 0
        try:
            from PIL import Image
            with Image.open(self.selected_image_path) as im:
                self.image_width, self.image_height = im.size
        except Exception:
            pass
        if self.scale_mode == 'per-image':
            data = session.load_image_scale(self.image_dir, filename) or {}
            self.scale_um_per_px = data.get('scale_um_per_px')
            self.scale_known_length = data.get('scale_known_length')
            self.scale_pixel_distance = data.get('scale_pixel_distance')

    def step_image(self, delta: int) -> None:
        if self.selected_image in self.image_files:
            i = self.image_files.index(self.selected_image) + delta
            if 0 <= i < len(self.image_files):
                self.select_image(self.image_files[i])

    def confirm_scale(self, um_per_px: float, known_length: float, pixel_distance: float) -> None:
        self.scale_um_per_px = um_per_px
        self.scale_known_length = known_length
        self.scale_pixel_distance = pixel_distance
        if self.image_dir and self.selected_image:
            session.save_image_scale(self.image_dir, self.selected_image,
                                     um_per_px, known_length, pixel_distance)

    def restore(self, saved: dict) -> None:
        """Apply a validated session payload. Blocking: call via run.io_bound."""
        for key in session.FIELDS:
            if key not in ('image_dir', 'selected_image') and key in saved:
                setattr(self, key, saved[key])
        locked = (saved.get('scale_um_per_px'), saved.get('scale_known_length'),
                  saved.get('scale_pixel_distance'))
        self.set_image_dir(saved['image_dir'], select=saved.get('selected_image', ''))
        if self.scale_mode == 'locked':
            self.scale_um_per_px, self.scale_known_length, self.scale_pixel_distance = locked

    def log(self, message: str) -> None:
        """Safe from the worker thread: appends to state, never to a widget (REQ-26)."""
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.log_lines.append(f'[{ts}] {message}')
        self.log_total += 1
        del self.log_lines[:max(0, len(self.log_lines) - LOG_MAX)]

    # ── Computed (bindable: bindings poll getattr) ─────────────────────────────
    @property
    def selected_image_path(self) -> str:
        if not self.image_dir or not self.selected_image:
            return ''
        return os.path.join(self.image_dir, self.selected_image)

    @property
    def scale_confirmed(self) -> bool:
        return self.scale_um_per_px is not None

    @property
    def run_hint(self) -> str:
        if not self.models_ok:
            return 'Model weights missing — see weights/README.md.'
        if not self.selected_image:
            return 'Load an image first.'
        if not self.scale_confirmed:
            return 'Calibrate scale to enable run.'
        if self.status == 'running':
            return 'Pipeline running…'
        return ''

    @property
    def can_run(self) -> bool:
        return not self.run_hint


STATE = AppState()


async def run_job() -> None:
    """Run the pipeline on the selected image. Start with background_tasks.create
    so no page owns it (REQ-21); it writes only to STATE (REQ-03)."""
    if not STATE.can_run:                                   # REQ-22: guard in state, not the button
        return
    image_path, name, scale = STATE.selected_image_path, STATE.selected_image, STATE.scale_um_per_px
    config = PipelineConfig(                                # REQ-25: pin inputs before awaiting
        depth_threshold_auto=STATE.depth_threshold_auto,
        depth_threshold_manual=STATE.depth_threshold_manual,
        roi_crop_bottom_pct=STATE.roi_crop_bottom_pct,
    )
    STATE.status, STATE.error, STATE.progress = 'running', None, 0.0
    STATE.log(f'Running pipeline on {name}')

    def on_stage(stage: str, res) -> None:                  # worker thread
        STATE.progress = (STAGES.index(stage) + 1) / len(STAGES)
        STATE.log(f'{stage}: {"ok" if res.success else res.error} ({res.duration_sec:.1f}s)')

    try:
        result = await asyncio.to_thread(run_pipeline, image_path, scale, config, on_stage)  # REQ-20
    except Exception as exc:
        STATE.log(f'ERROR: {exc}')
        STATE.error, STATE.status = str(exc), 'error'
        return
    STATE.result = result                                   # REQ-23: data first, flags last
    STATE.progress = 1.0
    STATE.status = 'complete' if result.fully_successful else 'error'
    STATE.generation += 1
    STATE.log(f'Finished {name}')
