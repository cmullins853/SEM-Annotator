# FiberSight GUI Spec v0.2 — Compressed

**NiceGUI desktop app wrapping SEM-Annotator pipeline. Zero training. Inference + measurement only.**

---

## 1. Purpose

Researcher with no CLI experience: load SEM images → calibrate scale → run pipeline → export results. Bundled pre-trained model.

---

## 2. Design Principles

- **Dark theme only.** Quasar dark mode base + teal accent overrides. (See §11)
- **Separation of concerns.** `ui/` owns layout/interaction. `backend/` owns image processing/inference/math. Communicate via function calls returning dataclasses. Stubs enable UI dev without real backend.
- **Composable SVG `LayerManager`.** `ui.interactive_image` has single `.content` SVG string. `LayerManager` maintains named `<g>` groups, recomposes on update. Multiple overlays (scale, ROI, annotations) coexist. (See §7.1)
- **Progressive disclosure.** Basic = minimal. Advanced = depth thresholds, stage previews, per-fiber tables.
- **Async pipeline.** All heavy compute via `app.run_cpu_bound()`. (See §7.4)

---

## 3. Layout

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: [icon] FiberSight · SEM Fiber Analysis  v0.1.0 [Basic|Adv] │
├───────────┬─────────────────────────────────────────────────┤
│  SIDEBAR  │  Tab bar: IMAGE | DEPTH MAP | SEG | RESULTS     │
│  DataDir  │                                                 │
│  ScaleCal │         (active tab content)                    │
│  ROICrop  │                                                 │
│  Pipeline │  PIPELINE OUTPUT LOG (collapsible)              │
│  Run      │                                                 │
├───────────┴─────────────────────────────────────────────────┤
│                          [▶ Run Pipeline] FAB bottom-right  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Header

| | |
|---|---|
| Left | Icon + "FiberSight" bold + subtitle |
| Right | Version chip · **Basic/Advanced** toggle · Gear → settings |

Basic mode: hides pipeline settings + ROI sidebar cards, shows only IMAGE + RESULTS tabs, summary stats only.

---

## 5. Sidebar Cards

### 5.1 Data Directory

- `ui.input` + browse → image dir, output dir
- `ui.select` dropdown → auto-populates `.tif/.tiff/.png/.jpg`
- Prev/Next buttons cycle dropdown

Changing dir scans + repopulates. Image change resets scale to "Not Set" (unless locked). Previous results loaded from session file if available.

### 5.2 Scale Calibration

| | |
|---|---|
| Mode toggle | "Lock for directory" / "Per-image" |
| Scale value | `ui.number`, µm/px, editable, auto-filled by draw tool |
| Known length | `ui.number`, real-world µm |
| Draw button | Activates line-drawing mode |
| Status chip | "Calibrated ✓" (green) / "Not Set" (amber) |

**Draw flow:**
1. Click "Draw Scale Line" → drawing mode active
2. Click start → circle marker on overlay
3. Click end → line + distance label drawn. **Default: Euclidean.** **Shift: constrain horizontal** (snap Y to start Y)
4. Pixel distance auto-fills (read-only). User enters µm value.
5. µm/px calculated, populates field
6. Confirm → `LayerManager.clear_layer('scale')`, persist value, chip → green
7. Manual edit of µm/px always allowed

### 5.3 ROI / Metadata Bar Crop *(Advanced only)*

- `ui.slider` 0–50% → excludes bottom N% from analysis
- `ui.switch` preview → semi-transparent red overlay via LayerManager
- Full image still displayed. Pipeline only processes above crop line.
- Default: 0%. Mirrors `background_remover.py` "Bottom %"

> v2: Replace with drawable rectangular ROI.

### 5.4 Pipeline Settings *(Advanced only)*

- Depth threshold: `ui.slider` + `ui.checkbox` "Auto (Otsu)". Auto checked → slider disabled.

### 5.5 Run Controls

| | |
|---|---|
| Scope toggle | "Single Image" / "Batch Directory" |
| Step-through | Pauses after each stage, shows "Next Stage ▶" |
| Run button | Large teal FAB, disabled until scale calibrated |

**Batch:** Shows preview table (image + calibration status). Per-image scale from session file. Failed images log + skip. Aggregate CSV in output dir.

---

## 6. Center Tabs

Basic: IMAGE + RESULTS only.

### 6.1 IMAGE
- `ui.interactive_image` with pan/zoom
- SVG overlay via `LayerManager`
- No persistent annotations after scale confirm

### 6.2 DEPTH MAP *(Advanced/Step-through)*
- Side-by-side: original | colorized depth map
- "Show mask overlay" toggle → 50% opacity composite
- Depth slider → live mask update (debounced ~200ms)

### 6.3 SEGMENTATION *(Advanced/Step-through)*
- Color-coded instance mask overlay
- Fiber count badge top-right
- "Show mask only" toggle

### 6.4 RESULTS

**Always visible:**
- Summary stats: mean, std, median, count (large stat cards)
- Diameter histogram (Plotly, adjustable bin count)

**Advanced (toggleable panels):**
- Orientation rose/histogram (0°–180°)
- Per-fiber table: Fiber ID, Diameter, Orientation, Length — sortable + searchable
- Annotated overlay with diameter measurements
- Diameter vs. orientation scatter (Plotly)

**Export (always visible):**
- Download CSV / Annotated Image / All (ZIP) / Batch Summary CSV

---

## 7. Interactive Image, Layers, Threading

### 7.1 LayerManager

```python
class LayerManager:
    def __init__(self, image: ui.interactive_image):
        self._image = image
        self._layers: dict[str, str] = {}
        self._order: list[str] = []

    def set_layer(self, name: str, svg_content: str): ...
    def clear_layer(self, name: str): ...
    def clear_all(self): ...

    def _recompose(self):
        parts = [f'<g id="layer-{n}">{self._layers[n]}</g>'
                 for n in self._order if n in self._layers and self._layers[n]]
        self._image.content = '\n'.join(parts)
```

Named layers:

| Layer | Purpose | Lifecycle |
|---|---|---|
| `scale` | Scale line draw | Created on draw, cleared on confirm |
| `roi` | Excluded region preview | Toggled by ROI switch |
| `annotations` | Fiber IDs, diameter lines | Set post-pipeline, cleared on new image |

Lives in `ui/center/image_viewer.py`.

### 7.2 Coordinate System — **⚠ Phase 1 Blocker**

Verify `ui.interactive_image` mouse events return **image-space** coords under pan/zoom:

```python
img = ui.interactive_image('test_sem.tif')
img.on('mouse', lambda e: print(f'image_x={e.image_x}, image_y={e.image_y}'))
```

Zoom in, click same feature, confirm stable coords. If viewport-relative → implement transform.

### 7.3 Scale Line Drawing

- Free draw: `sqrt((x2-x1)² + (y2-y1)²)`
- Shift constrained: `abs(x2 - x1)`, end Y snapped to start Y

```python
def render_scale_line(x1, y1, x2, y2, px_dist):
    return f'''
        <circle cx="{x1}" cy="{y1}" r="4" fill="lime" />
        <circle cx="{x2}" cy="{y2}" r="4" fill="lime" />
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
              stroke="lime" stroke-width="2" stroke-dasharray="6,3" />
        <text x="{(x1+x2)/2}" y="{min(y1,y2)-10}" fill="lime"
              text-anchor="middle" font-size="14">
            {px_dist:.1f} px
        </text>
    '''
```

### 7.4 Threading

All pipeline stages via `app.run_cpu_bound()` — process pool, returns awaitable. Without this: UI freezes 5–30s. **Phase 2 architectural requirement.**

```python
async def on_run_pipeline():
    depth_result = await app.run_cpu_bound(
        backend.depth.estimate, state.selected_image_path, config
    )
    if not depth_result.success:
        log.append(f'[ERROR] {depth_result.error}')
        return
    # ... next stage
```

---

## 8. Pipeline Log

Collapsible card below center area.

- Monospace, dark bg, auto-scroll
- Indeterminate progress bar (batch: determinate)
- Per-stage indicators: ✓ ✗ ○

```
[12:04:01] Loading image: fiber_mesh_001.tif (2048×1536)
[12:04:01] Scale: 0.0342 µm/px (locked)
[12:04:02] Stage 1/3 — Depth estimation (DepthAnythingV2-Small)...
[12:04:05] ✓ Stage 1/3 — Complete. Threshold: 142 (Otsu)
[12:04:08] ✓ Stage 2/3 — Complete. 47 fibers detected.
[12:04:09] ✓ Stage 3/3 — Complete.
[12:04:09] Results: mean=1.24µm, std=0.31µm, n=47
```

---

## 9. Project Structure

```
fibersight/
├── main.py
├── ui/
│   ├── theme.py
│   ├── layout.py
│   ├── header.py
│   ├── sidebar/
│   │   ├── data_directory.py
│   │   ├── scale_calibration.py
│   │   ├── roi_crop.py
│   │   ├── pipeline_settings.py
│   │   └── run_controls.py
│   ├── center/
│   │   ├── image_viewer.py      # LayerManager lives here
│   │   ├── depth_tab.py
│   │   ├── segmentation_tab.py
│   │   └── results_tab.py
│   └── pipeline_log.py
├── backend/
│   ├── pipeline.py              # run_pipeline() → PipelineResult
│   ├── depth.py                 # DepthAnythingV2 wrapper
│   ├── segmentation.py
│   ├── diameter.py
│   ├── orientation.py
│   ├── scale.py                 # Real impl, not stub
│   └── models.py                # Lazy load + validation
├── models/
│   └── README.md
├── state/
│   ├── app_state.py
│   └── session.py
└── utils/
    ├── file_io.py
    └── config.py
```

**Backend → SEM-Annotator mapping:**

| Module | Source |
|---|---|
| `backend/depth.py` | `depth_anything_remover.py` → `DepthAnythingPredictor` |
| `backend/segmentation.py` | `model/` (TBD Connor) |
| `backend/diameter.py` | Skeletonization + cross-sections (TBD) |

### Error Model

```python
@dataclass
class StageResult(Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None
    duration_sec: float = 0.0

@dataclass
class DepthOutput:
    depth_map: np.ndarray       # uint8
    depth_mask: np.ndarray      # binary
    threshold_value: int

@dataclass
class SegmentationOutput:
    instance_mask: np.ndarray   # 0=bg, 1..N=fibers
    fiber_count: int

@dataclass
class FiberMeasurement:
    fiber_id: int
    diameter_um: float
    orientation_deg: float
    length_um: float

@dataclass
class DiameterOutput:
    fibers: list[FiberMeasurement]
    mean_diameter: float
    std_diameter: float
    median_diameter: float

@dataclass
class PipelineResult:
    depth: StageResult[DepthOutput]
    segmentation: StageResult[SegmentationOutput]
    diameters: StageResult[DiameterOutput]

    @property
    def fully_successful(self) -> bool:
        return all([self.depth.success, self.segmentation.success, self.diameters.success])
```

Partial failure: completed stages render normally, failed stage shows error, RESULTS disabled. User adjusts + re-runs.

### Backend Interface

```python
def run_pipeline(
    image_path: str,
    scale_um_per_px: float,
    config: PipelineConfig,
    roi_crop_bottom_pct: float = 0.0,
    on_stage_complete: Callable[[str, StageResult], None] | None = None,
) -> PipelineResult: ...
```

```python
# backend/scale.py — real implementation
def compute_scale(pixel_distance: float, known_length_um: float) -> float:
    return known_length_um / pixel_distance

def pixel_distance(x1, y1, x2, y2) -> float:
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def pixel_distance_horizontal(x1, x2) -> float:
    return abs(x2 - x1)
```

---

## 10. State

### 10.1 AppState

```python
class AppState:
    image_dir: str = ''
    output_dir: str = ''
    image_files: list[str] = []
    selected_image: str = ''
    scale_mode: str = 'per-image'       # 'locked' | 'per-image'
    scale_um_per_px: float | None = None
    scale_confirmed: bool = False
    roi_crop_bottom_pct: float = 0.0
    advanced_mode: bool = False
    step_through: bool = False
    batch_mode: bool = False
    pipeline_running: bool = False
    current_stage: int = 0              # 0=idle, 1–3=stages
    result: PipelineResult | None = None
    depth_threshold_auto: bool = True
    depth_threshold_manual: int = 128
```

### 10.2 Session Persistence

**File:** `<image_dir>/.fibersight/session.json`

```json
{
  "version": "0.1",
  "scale_mode": "per-image",
  "locked_scale_um_per_px": null,
  "roi_crop_bottom_pct": 0.0,
  "images": {
    "fiber_mesh_001.tif": {
      "scale_um_per_px": 0.0342,
      "scale_known_length": 10.0,
      "scale_pixel_distance": 292.4,
      "last_run": "2026-04-14T12:04:09",
      "result_file": "fiber_mesh_001_results.json"
    }
  }
}
```

- **Dir load:** Restore scale values. Dropdown shows green/amber dots per image.
- **Scale confirm:** Write immediately.
- **Pipeline complete:** Write result file pointer. Actual data goes to output dir.
- **Image switch:** Restore saved scale + show "already processed" summary if available.

---

## 11. Theme

Quasar dark mode base. Override only:

| Token | Value | |
|---|---|---|
| `primary` | `#2dd4bf` | Teal accent |
| `dark-page` | `#1a1a2e` | Page bg |
| `dark` | `#16213e` | Card/surface bg |

Log bg: `#0d1117`. Font: system sans-serif. Mono: system monospace. Everything else inherits Quasar defaults.

---

## 12. Models

**Layout:**
```
models/
├── README.md
├── depth_anything_v2_small/
└── fiber_segmentation/
```

**Lazy load:** Not loaded at startup. Loaded on first run, cached. Startup time <2s.

**Load flow:**
1. "Run Pipeline" clicked
2. `backend/models.py` checks memory cache → load from `models/` if miss
3. Missing weights → `StageResult(success=False, error="Model weights not found at models/depth_anything_v2_small/. See models/README.md.")` → error toast + log
4. Subsequent runs use cache

**Startup validation (non-blocking):** Filesystem check only (no load). If missing → persistent amber banner: "Model weights not found. See models/README.md." App remains functional for UI testing + scale calibration.

---

## 13. Phases

### Phase 1 — Skeleton + Scale *(UI only, stubs)*

- [ ] **Task 0 (blocker):** Verify coordinate space (§7.2)
- [ ] Scaffolding: `main.py`, `ui/`, `backend/` stubs, `state/`
- [ ] Theme
- [ ] Header + mode toggle
- [ ] Sidebar: Data Directory card
- [ ] Center: IMAGE tab + `ui.interactive_image`
- [ ] `LayerManager`
- [ ] Scale draw tool (free + Shift-constrain)
- [ ] Scale card: fields, lock toggle, status chip
- [ ] `backend/scale.py` (real impl)
- [ ] Session persistence
- [ ] Model startup validation + warning banner

### Phase 2 — Pipeline Integration

- [ ] Wire `backend/depth.py` → `DepthAnythingPredictor`
- [ ] Wire `backend/segmentation.py` → Connor's model
- [ ] Wire `backend/diameter.py`
- [ ] `StageResult` error model across all backends
- [ ] Async pipeline via `app.run_cpu_bound()`
- [ ] Lazy model loading + cache
- [ ] Single-image run + progress + log
- [ ] RESULTS: summary stats + histogram
- [ ] Write results to output dir + session file

### Phase 3 — Advanced + Polish

- [ ] ROI crop card + preview overlay
- [ ] DEPTH MAP tab + interactive threshold
- [ ] SEGMENTATION tab + instance masks
- [ ] Pipeline Settings card
- [ ] Step-through mode
- [ ] Advanced results: orientation, per-fiber table, annotated overlay
- [ ] Export buttons

### Phase 4 — Batch + Extras

- [ ] Batch Preview table
- [ ] Batch processing + aggregate CSV
- [ ] Per-image scale in batch (from session)
- [ ] Failed-image skip + summary
- [ ] Node detection, per-fiber profiles
- [ ] Performance/GPU feedback

---

## 14. v2 Considerations

- **Results Gallery:** Cross-image comparison tab. Thumbnails + stats per image. Side-by-side compare. Data from session + result files.
- **Drawable ROI:** Replace bottom-% slider with rectangular/polygonal selection via LayerManager.
- **Additional metrics:** Node detection, per-fiber diameter profiles, porosity estimation, batch-level statistical comparisons.

---

## 15. Open Questions for Connor

1. Segmentation model arch? (YOLOv8-seg / Mask R-CNN / U-Net?) → determines `backend/segmentation.py` wrapper
2. Model input/output contract? (depth-masked vs raw? resolution? returns instance masks vs boxes+masks?)
3. Roboflow export format? (COCO JSON / YOLO / mask PNGs?) → affects test data validation
4. Orientation extraction method? (skeleton direction from masks, or separate approach?)
5. Diameter method confirmed? (medial axis → perpendicular cross-section sampling? oblique fiber correction?)
6. Model file size? → determines distribution strategy (Git LFS / separate download)
7. GUI scope for paper? (appendix companion tool vs core contribution?)
