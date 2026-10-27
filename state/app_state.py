from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Any


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
    scale_um_per_px: Optional[float] = None
    scale_known_length: Optional[float] = None
    scale_pixel_distance: Optional[float] = None
    scale_confirmed: bool = False

    # ── ROI ────────────────────────────────────────────────────────────────────
    roi_crop_bottom_pct: float = 0.0
    roi_preview_active: bool = False

    # ── Mode ───────────────────────────────────────────────────────────────────
    advanced_mode: bool = False
    step_through: bool = False
    batch_mode: bool = False

    # ── Pipeline ───────────────────────────────────────────────────────────────
    pipeline_running: bool = False
    current_stage: int = 0                  # 0=idle, 1=depth, 2=seg, 3=diameter

    # ── Advanced settings ──────────────────────────────────────────────────────
    depth_threshold_auto: bool = True
    depth_threshold_manual: int = 128

    # ── Model availability ─────────────────────────────────────────────────────
    depth_model_available: bool = False
    seg_model_available: bool = False
    debug: bool = False

    # ── Pending draw data (populated by image_viewer after 2-click draw) ───────
    pending_scale_draw: Optional[dict] = field(default=None, repr=False)

    # ── UI element refs (set by UI builders after page construction) ───────────
    activate_scale_draw: Optional[Callable] = field(default=None, repr=False)
    deactivate_scale_draw: Optional[Callable] = field(default=None, repr=False)
    update_roi_overlay: Optional[Callable] = field(default=None, repr=False)
    log_element: Optional[Any] = field(default=None, repr=False)
    model_banner: Optional[Any] = field(default=None, repr=False)

    # ── Event callbacks ────────────────────────────────────────────────────────
    _on_image_dir_changed: list[Callable] = field(default_factory=list, repr=False)
    _on_image_selected: list[Callable] = field(default_factory=list, repr=False)
    _on_scale_changed: list[Callable] = field(default_factory=list, repr=False)
    _on_scale_draw_complete: list[Callable] = field(default_factory=list, repr=False)
    _on_advanced_mode_changed: list[Callable] = field(default_factory=list, repr=False)
    _on_roi_changed: list[Callable] = field(default_factory=list, repr=False)

    # ── Callback registration ──────────────────────────────────────────────────
    def on_image_dir_changed(self, fn: Callable): self._on_image_dir_changed.append(fn)
    def on_image_selected(self, fn: Callable): self._on_image_selected.append(fn)
    def on_scale_changed(self, fn: Callable): self._on_scale_changed.append(fn)
    def on_scale_draw_complete(self, fn: Callable): self._on_scale_draw_complete.append(fn)
    def on_advanced_mode_changed(self, fn: Callable): self._on_advanced_mode_changed.append(fn)
    def on_roi_changed(self, fn: Callable): self._on_roi_changed.append(fn)

    def _fire(self, handlers: list[Callable]):
        for fn in handlers:
            try:
                fn()
            except Exception:
                import traceback
                traceback.print_exc()

    # ── State mutations ────────────────────────────────────────────────────────
    def set_image_dir(self, path: str):
        from utils.file_io import scan_images
        self.image_dir = path
        self.image_files = scan_images(path)
        self.selected_image = self.image_files[0] if self.image_files else ''
        self._fire(self._on_image_dir_changed)
        if self.selected_image:
            self._load_image_metadata(self.selected_image)
            if self.scale_mode == 'per-image':
                self._restore_scale(self.selected_image)
            self._fire(self._on_image_selected)

    def set_selected_image(self, filename: str):
        if filename == self.selected_image:
            return
        self.selected_image = filename
        self._load_image_metadata(filename)
        if self.scale_mode == 'per-image':
            self._restore_scale(filename)
        self._fire(self._on_image_selected)

    def _load_image_metadata(self, filename: str):
        path = self.selected_image_path
        if not path:
            self.image_width = self.image_height = 0
            return
        try:
            from PIL import Image
            with Image.open(path) as im:
                self.image_width, self.image_height = im.size
                return
        except Exception:
            pass
        self.image_width = self.image_height = 0

    def _restore_scale(self, filename: str):
        from state.session import load_image_scale
        data = load_image_scale(self.image_dir, filename)
        if data and data.get('scale_um_per_px') is not None:
            self.scale_um_per_px = data['scale_um_per_px']
            self.scale_known_length = data.get('scale_known_length')
            self.scale_pixel_distance = data.get('scale_pixel_distance')
            self.scale_confirmed = True
        else:
            self.scale_um_per_px = None
            self.scale_known_length = None
            self.scale_pixel_distance = None
            self.scale_confirmed = False
        self._fire(self._on_scale_changed)

    def notify_scale_draw_complete(self, x1: float, y1: float,
                                   x2: float, y2: float, px_dist: float):
        self.pending_scale_draw = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'px_dist': px_dist
        }
        self._fire(self._on_scale_draw_complete)

    def set_scale_confirmed(self, um_per_px: float, known_length: float, pixel_distance: float):
        self.scale_um_per_px = um_per_px
        self.scale_known_length = known_length
        self.scale_pixel_distance = pixel_distance
        self.scale_confirmed = True
        if self.image_dir and self.selected_image:
            from state.session import save_image_scale
            save_image_scale(self.image_dir, self.selected_image,
                             um_per_px, known_length, pixel_distance)
        self._fire(self._on_scale_changed)

    def clear_scale(self):
        self.scale_um_per_px = None
        self.scale_known_length = None
        self.scale_pixel_distance = None
        self.scale_confirmed = False
        self._fire(self._on_scale_changed)

    def set_roi(self, crop_pct: float, preview_active: bool):
        self.roi_crop_bottom_pct = crop_pct
        self.roi_preview_active = preview_active
        self._fire(self._on_roi_changed)

    def set_advanced_mode(self, enabled: bool):
        self.advanced_mode = enabled
        self._fire(self._on_advanced_mode_changed)

    def log(self, message: str):
        import datetime
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        if self.log_element is not None:
            self.log_element.push(f'[{ts}] {message}')

    # ── Computed ───────────────────────────────────────────────────────────────
    @property
    def selected_image_path(self) -> str:
        if not self.image_dir or not self.selected_image:
            return ''
        import os
        return os.path.join(self.image_dir, self.selected_image)

    @property
    def scale_status(self) -> str:
        if self.scale_mode == 'locked' and self.scale_um_per_px is not None:
            return f'Locked: {self.scale_um_per_px:.4f} µm/px'
        if self.scale_confirmed and self.scale_um_per_px is not None:
            return f'Calibrated: {self.scale_um_per_px:.4f} µm/px'
        return 'Not Set'

    @property
    def can_run(self) -> bool:
        return (bool(self.selected_image_path)
                and self.scale_confirmed
                and not self.pipeline_running)
