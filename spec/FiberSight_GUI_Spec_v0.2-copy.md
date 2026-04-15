# FiberSight — GUI Specification v0.2

**Companion interface for the SEM-Annotator pipeline**
Document version: 2026-04-14 · For review by Brenden & Connor

---

## 1. Purpose

FiberSight is a NiceGUI-based desktop web app that wraps the SEM-Annotator pipeline (depth estimation → segmentation → diameter/orientation extraction) in an interactive interface. The GUI ships with a **bundled, pre-trained model** — it performs **zero training**, only inference and measurement. It is designed so that a researcher with no CLI experience can load SEM images, calibrate scale, run the pipeline, and export publication-ready results.

---

## 2. Design Principles

- **Dark theme only.** SEM imagery is inherently dark/high-contrast; a dark UI reduces eye strain and keeps visual focus on the images. Uses Quasar's built-in dark mode as the base, with targeted overrides for primary (teal accent) and surface colors. See Section 11.
- **Separation of concerns.** The `ui/` package owns all NiceGUI layout and interaction. The `backend/` package owns all image processing, model inference, and math. Communication happens through simple function calls that accept paths/configs and return dataclasses. This means the entire UI can be developed and tested with stub backends that return dummy data.
- **Composable SVG layer manager for overlays.** NiceGUI's `ui.interactive_image` exposes a single `.content` SVG string. A dedicated `LayerManager` class maintains named SVG `<g>` groups and recomposes the full string on every update, so multiple overlays (scale lines, result annotations) can coexist without overwriting each other. See Section 7.
- **Progressive disclosure.** Basic mode shows only what a typical user needs. Advanced mode exposes depth thresholds, intermediate stage previews, and per-fiber detail tables behind toggles and checkboxes.
- **Async-safe pipeline execution.** All heavy compute (depth estimation, segmentation, diameter extraction) runs via NiceGUI's `app.run_cpu_bound()` to prevent UI freezing. Callbacks push updates to the UI thread safely. See Section 7.4.

---

## 3. Layout Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  HEADER BAR                                                      │
│  [icon] FiberSight  ·  SEM Fiber Analysis     v0.1.0  [Basic|Adv]│
├────────────┬─────────────────────────────────────────────────────┤
│  SIDEBAR   │  CENTER CONTENT AREA                                │
│            │                                                     │
│  Data Dir  │  ┌─────────────────────────────────────────────┐   │
│  Scale Cal │  │  Tab bar: IMAGE | DEPTH MAP | SEG | RESULTS │   │
│  ROI Crop  │  │                                             │   │
│  Pipeline  │  │         (active tab content)                │   │
│  Settings  │  │                                             │   │
│  Run       │  └─────────────────────────────────────────────┘   │
│            │                                                     │
│            │  ┌─────────────────────────────────────────────┐   │
│            │  │  PIPELINE OUTPUT LOG  (collapsible)          │   │
│            │  └─────────────────────────────────────────────┘   │
├────────────┴─────────────────────────────────────────────────────┤
│                     [▶ Run Pipeline]  (FAB, bottom-right)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Header Bar

| Element | Details |
|---|---|
| Left | App icon + "FiberSight" bold + "SEM Fiber Analysis" subtitle (lighter weight) |
| Center | (empty, or future breadcrumb) |
| Right | Version chip `v0.1.0` · Mode toggle: **Basic / Advanced** · Gear icon → global settings dialog |

The mode toggle controls visibility throughout the app. In Basic mode, sidebar cards for pipeline settings and ROI are hidden, center tabs show only IMAGE and RESULTS, and the results panel shows summary stats only.

---

## 5. Left Sidebar — Collapsible Cards

### 5.1 Data Directory

| Element | NiceGUI component | Notes |
|---|---|---|
| Image directory | `ui.input` + browse button | Path to folder of SEM images |
| Output directory | `ui.input` + browse button | Where results/exports are saved |
| Image selector | `ui.select` (dropdown) | Auto-populates with filenames when directory is set. Filters to common SEM formats: `.tif`, `.tiff`, `.png`, `.jpg` |
| Prev / Next | `ui.button` pair | Cycles through dropdown list |

**Behavior:** Changing the directory scans for image files and populates the dropdown. Selecting an image (or using prev/next) loads it into the center IMAGE tab. If "Lock scale" is off, the scale status resets to "Not Set" on image change. Previous results for the image are loaded from the session state file if available (see Section 10.2).

### 5.2 Scale Calibration

| Element | NiceGUI component | Notes |
|---|---|---|
| Mode toggle | `ui.toggle` | "Lock for directory" vs "Per-image" |
| Scale value | `ui.number` | Displays current µm/px. Editable. Auto-populated by the draw tool. |
| Known length | `ui.number` | User enters the real-world length of the scale bar in µm |
| Draw button | `ui.button` | "Draw Scale Line" — activates line-drawing mode on the image viewer |
| Status chip | `ui.chip` | "Calibrated ✓" (green) / "Not Set" (amber) |

**Scale line drawing UX (see Section 7 for full detail):**

1. User clicks "Draw Scale Line" → overlay layer activates on the interactive image.
2. User clicks start point → small circle marker appears on overlay.
3. User clicks end point → line is drawn between the two points. **Default: free draw** (Euclidean distance). **Hold Shift: constrain to horizontal** (snaps end Y to start Y). Both endpoints marked, pixel distance label shown on overlay.
4. Pixel distance auto-fills into a read-only field. User types the known µm value into the "Known length" field.
5. µm/px is calculated and populates the scale value field.
6. User clicks "Confirm" (or the draw button again toggles off) → **overlay layer clears** (via LayerManager, see Section 7.1), line disappears, scale value persists, status chip turns green.
7. User can manually edit the µm/px field at any time (overrides the drawn value).

### 5.3 ROI / Metadata Bar Crop (Advanced mode only)

| Element | NiceGUI component | Notes |
|---|---|---|
| Bottom crop % | `ui.slider` (0–50%) | Excludes the bottom N% of the image from analysis. Intended for SEM metadata bars (instrument info, embedded scale bar, magnification text). |
| Preview toggle | `ui.switch` | When on, a semi-transparent red overlay shows the excluded region on the IMAGE tab via the LayerManager. |

**Behavior:** The crop percentage defines an analysis ROI that is passed into the pipeline. The pipeline only processes pixels above the crop line. The full image is still displayed in the viewer — only the red overlay indicates the excluded zone. Default: 0% (no crop). This mirrors the "Bottom %" slider in the existing `background_remover.py`.

> **v2 consideration:** Replace the bottom-% slider with a user-drawable rectangular ROI selection for more flexible cropping.

### 5.4 Pipeline Settings (Advanced mode only)

| Element | NiceGUI component | Notes |
|---|---|---|
| Depth threshold | `ui.slider` + `ui.checkbox` "Auto (Otsu)" | When Auto is checked, slider is disabled. When unchecked, slider overrides the Otsu value (0–255 range). |
| (Future slots) | — | Reserved for additional tunable parameters as they arise |

### 5.5 Run Controls

| Element | NiceGUI component | Notes |
|---|---|---|
| Scope toggle | `ui.toggle` | "Single Image" / "Batch Directory" |
| Step-through toggle | `ui.switch` | When on, pipeline pauses after each stage. "Next Stage ▶" button appears in center area. |
| Run Pipeline | `ui.button` | Large teal FAB, bottom-right of viewport (fixed position). Disabled until scale is calibrated. |

**Batch mode behavior:** Before batch execution begins, a **Batch Preview table** is shown listing every image in the directory and its calibration status (calibrated / not set). If scale is locked, all images show "Locked: X µm/px". If per-image mode is active, only images with a saved scale value (from the session state file) are eligible. The user can deselect images from the batch. Images that fail during processing log an error and are skipped — processing continues. A summary at the end shows pass/fail counts. Results aggregate into a combined CSV in the output directory.

---

## 6. Center Content Area — Tabbed

Top tab bar with four tabs. In Basic mode, only **IMAGE** and **RESULTS** are visible.

### 6.1 IMAGE Tab

- `ui.interactive_image` with **pan and zoom** enabled (scroll-to-zoom, click-drag to pan).
- Filename + resolution displayed as a subtle label above the image.
- SVG overlay managed by the `LayerManager` (see Section 7.1) — handles scale lines, ROI previews, and result annotations as independent named layers.
- This is the primary working view — it stays clean. No persistent annotations once scale is confirmed.

### 6.2 DEPTH MAP Tab (Advanced / Step-through only)

- Side-by-side layout: original image (left) | colorized depth map (right).
- Toggle: "Show mask overlay" — composites the binary depth mask onto the original at 50% opacity.
- If pipeline settings card is visible, adjusting the depth slider updates the mask preview live (debounced to ~200ms).

### 6.3 SEGMENTATION Tab (Advanced / Step-through only)

- Segmented mask overlay on the original image, with color-coded instance masks.
- Fiber count badge (top-right corner of the image).
- Toggle: "Show mask only" — displays the raw segmentation mask without the underlying image.

### 6.4 RESULTS Tab

**Always visible (Basic + Advanced):**

- **Summary stats card:** Mean diameter, standard deviation, median, fiber count. Displayed as large-number stat cards in a row.
- **Diameter histogram:** Interactive Plotly chart. Bin count adjustable via small slider or input.

**Advanced mode — checkboxes to show/hide additional panels:**

| Panel | Description |
|---|---|
| Orientation plot | Rose diagram or histogram of fiber orientations (0°–180°) |
| Per-fiber table | `ui.table`, sortable columns: Fiber ID, Diameter (µm), Orientation (°), Length (µm). Searchable. |
| Annotated overlay | Original image with diameter measurements drawn on each fiber |
| Diameter vs. orientation scatter | Plotly scatter plot |

**Export row (always visible):**

- "Download CSV" — per-fiber measurements
- "Download Annotated Image" — PNG with overlays baked in
- "Download All" — ZIP of CSV + annotated image + raw stats JSON
- In batch mode: "Download Batch Summary CSV"

---

## 7. Interactive Image, Layer System & Threading

### 7.1 LayerManager — Composable SVG Overlay

NiceGUI's `ui.interactive_image` provides a single `.content` property that accepts an SVG string. Setting `.content` replaces the entire overlay. To support multiple independent overlays (scale line, ROI crop preview, result annotations) without them overwriting each other, we introduce a `LayerManager` utility class.

**Design:**

```python
class LayerManager:
    """Manages named SVG <g> layers for ui.interactive_image."""

    def __init__(self, image: ui.interactive_image):
        self._image = image
        self._layers: dict[str, str] = {}  # name → SVG fragment
        self._order: list[str] = []        # render order (bottom to top)

    def set_layer(self, name: str, svg_content: str):
        """Set or replace a named layer's SVG content."""
        if name not in self._layers:
            self._order.append(name)
        self._layers[name] = svg_content
        self._recompose()

    def clear_layer(self, name: str):
        """Remove a named layer."""
        self._layers.pop(name, None)
        if name in self._order:
            self._order.remove(name)
        self._recompose()

    def clear_all(self):
        """Remove all layers."""
        self._layers.clear()
        self._order.clear()
        self._recompose()

    def _recompose(self):
        """Rebuild the full SVG string from all active layers."""
        parts = []
        for name in self._order:
            if name in self._layers and self._layers[name]:
                parts.append(f'<g id="layer-{name}">{self._layers[name]}</g>')
        self._image.content = '\n'.join(parts)
```

**Named layers:**

| Layer name | Purpose | Lifecycle |
|---|---|---|
| `scale` | Scale line drawing (endpoints, line, distance label) | Created on draw, cleared on confirm |
| `roi` | Red overlay showing excluded crop region | Toggled by ROI preview switch |
| `annotations` | Result overlays (fiber IDs, diameter lines) | Set after pipeline run, cleared on new image |

This class lives in `ui/center/image_viewer.py` and is instantiated alongside the `ui.interactive_image`.

### 7.2 Coordinate System — Verification Required

**⚠ Phase 1 blocker:** The spec assumes `ui.interactive_image` mouse events return image-space coordinates (relative to the original image pixels) regardless of pan/zoom state. This must be verified empirically before building the scale tool.

**Phase 1, Task 0:** Create a minimal test script:

```python
from nicegui import ui

img = ui.interactive_image('test_sem.tif')
img.on('mouse', lambda e: print(f'image_x={e.image_x}, image_y={e.image_y}'))

ui.run()
```

Zoom in, click the same feature, and confirm coordinates are stable. If they are viewport-relative, a transform function is needed that accounts for current pan offset and zoom factor.

### 7.3 Scale Line Drawing — Implementation Notes

**Drawing behavior:**

- **Default: free draw.** Line is drawn between the two clicked points exactly. Pixel distance = Euclidean distance: `sqrt((x2-x1)² + (y2-y1)²)`.
- **Shift held: constrain to horizontal.** End point Y is snapped to start point Y. Pixel distance = `abs(x2 - x1)`.
- This accommodates SEM images where the scale bar may not be perfectly horizontal due to slight image rotation or user preference.

**Mouse event flow:**

1. "Draw Scale Line" button activates drawing mode (state flag).
2. First click → record `(x1, y1)`, render start marker via `LayerManager.set_layer('scale', ...)`.
3. Second click → record `(x2, y2)`, apply constraint if Shift held, render full line + distance label via `LayerManager.set_layer('scale', ...)`.
4. Confirm → `LayerManager.clear_layer('scale')`, persist µm/px to state, deactivate drawing mode.

**SVG rendering example:**

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

### 7.4 Threading Model

NiceGUI runs on asyncio. Pipeline stages (depth estimation, segmentation, diameter extraction) are heavy synchronous CPU/GPU workloads that will freeze the UI if run on the main thread.

**Required approach:**

- All pipeline stages run via `app.run_cpu_bound(fn, *args)`, which executes the function in a process pool and returns an awaitable.
- The pipeline orchestrator (`backend/pipeline.py`) is called from an `async` handler in the UI.
- Stage completion callbacks use NiceGUI's UI context to safely push updates (log messages, progress bar, tab content) back to the browser.

**Pattern:**

```python
async def on_run_pipeline():
    state.pipeline_running = True
    log.append('[...] Stage 1/3 — Depth estimation...')

    depth_result = await app.run_cpu_bound(
        backend.depth.estimate, state.selected_image_path, config
    )

    if not depth_result.success:
        log.append(f'[ERROR] Depth estimation failed: {depth_result.error}')
        state.pipeline_running = False
        return

    log.append('[...] Stage 1/3 — Complete.')
    # ... continue to next stage
```

**Why this matters:** Without this, clicking "Run Pipeline" locks the browser tab for 5–30 seconds depending on image size and GPU availability. No progress bar, no log, no cancel — just a frozen UI. This is a Phase 2 architectural requirement, not a polish item.

---

## 8. Pipeline Output Log

A collapsible card below the center content area (like the Synplex "Pipeline Output" panel).

- Monospace font, dark background, auto-scrolling.
- Indeterminate progress bar while pipeline is running; switches to determinate during batch mode.
- Stage status indicators: ✓ (green), ✗ (red), ○ (pending) shown inline per stage.
- Stage-by-stage log messages:
  ```
  [12:04:01] Loading image: fiber_mesh_001.tif (2048×1536)
  [12:04:01] Scale: 0.0342 µm/px (locked)
  [12:04:01] ROI: bottom 12% excluded
  [12:04:02] Stage 1/3 — Depth estimation (DepthAnythingV2-Small)...
  [12:04:05] ✓ Stage 1/3 — Complete. Threshold: 142 (Otsu)
  [12:04:05] Stage 2/3 — Segmentation...
  [12:04:08] ✓ Stage 2/3 — Complete. 47 fibers detected.
  [12:04:08] Stage 3/3 — Diameter extraction...
  [12:04:09] ✓ Stage 3/3 — Complete.
  [12:04:09] Results: mean=1.24µm, std=0.31µm, n=47
  ```

---

## 9. Project Structure

```
fibersight/
├── main.py                  # Entry point: ui.run(), page routing
│
├── ui/                      # All NiceGUI layout and interaction
│   ├── __init__.py
│   ├── theme.py             # Quasar dark mode config + targeted overrides
│   ├── layout.py            # Top-level page scaffold (header, sidebar, center)
│   ├── header.py            # Header bar: branding, mode toggle, settings
│   ├── sidebar/
│   │   ├── __init__.py
│   │   ├── data_directory.py    # Directory inputs, image dropdown, prev/next
│   │   ├── scale_calibration.py # Scale card: draw button, fields, status
│   │   ├── roi_crop.py          # Bottom-% slider, preview toggle
│   │   ├── pipeline_settings.py # Advanced: depth slider, future params
│   │   └── run_controls.py      # Scope toggle, step-through, batch preview, run button
│   ├── center/
│   │   ├── __init__.py
│   │   ├── image_viewer.py      # Interactive image + LayerManager
│   │   ├── depth_tab.py         # Depth map display (Advanced)
│   │   ├── segmentation_tab.py  # Segmentation mask display (Advanced)
│   │   └── results_tab.py       # Stats, plots, tables, export buttons
│   └── pipeline_log.py         # Collapsible log panel
│
├── backend/                 # All processing logic (stub-able)
│   ├── __init__.py
│   ├── pipeline.py          # Orchestrator: run_pipeline() → PipelineResult
│   ├── depth.py             # DepthAnythingV2 wrapper
│   ├── segmentation.py      # Model inference wrapper
│   ├── diameter.py          # Skeletonization, cross-section sampling, stats
│   ├── orientation.py       # Fiber orientation extraction
│   ├── scale.py             # Scale math (real implementation, not a stub)
│   └── models.py            # Model loading, validation, lazy init
│
├── models/                  # Bundled pre-trained model weights
│   └── README.md            # Instructions for obtaining/placing weights
│
├── state/
│   ├── __init__.py
│   ├── app_state.py         # Shared reactive state
│   └── session.py           # JSON sidecar persistence (see Section 10.2)
│
└── utils/
    ├── __init__.py
    ├── file_io.py           # Directory scanning, image loading, format detection
    └── config.py            # Default values, supported extensions, version string
```

### Dependency on SEM-Annotator repo

The `backend/` modules wrap functions from the existing SEM-Annotator codebase. Specifically:

| FiberSight module | SEM-Annotator source |
|---|---|
| `backend/depth.py` | `depth_anything_remover.py` → `DepthAnythingPredictor` class |
| `backend/segmentation.py` | `model/` directory + inference script (TBD by Connor) |
| `backend/diameter.py` | Diameter extraction logic (TBD, skeletonization + cross-sections) |

For v1 development, each backend module exposes a single function with a clear signature. Stubs return dummy data so the UI can be built and tested independently.

### Backend Error Model

Each pipeline stage returns a `StageResult` rather than raising exceptions. This allows the orchestrator to handle partial failures gracefully, which is critical for step-through mode and batch processing.

```python
from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar('T')

@dataclass
class StageResult(Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None
    duration_sec: float = 0.0

@dataclass
class DepthOutput:
    depth_map: np.ndarray       # uint8, same size as input
    depth_mask: np.ndarray      # binary mask of topmost layer
    threshold_value: int        # Otsu or manual threshold used

@dataclass
class SegmentationOutput:
    instance_mask: np.ndarray   # labeled int array (0=bg, 1..N=fibers)
    fiber_count: int

@dataclass
class FiberMeasurement:
    fiber_id: int
    diameter_um: float
    orientation_deg: float
    length_um: float
    # future: centroid, bounding box, node associations

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
        return all([
            self.depth.success,
            self.segmentation.success,
            self.diameters.success,
        ])
```

**UI behavior on partial failure:** If depth estimation succeeds but segmentation fails, the DEPTH MAP tab renders normally, the SEGMENTATION tab shows an error message with the failure reason, and RESULTS is disabled. The log shows ✓/✗ per stage. The user can adjust settings and re-run without restarting.

### Stub interface contract

```python
# backend/pipeline.py
def run_pipeline(
    image_path: str,
    scale_um_per_px: float,
    config: PipelineConfig,
    roi_crop_bottom_pct: float = 0.0,
    on_stage_complete: Callable[[str, StageResult], None] | None = None,
) -> PipelineResult:
    """
    Runs the full pipeline. Each stage is independently failable.
    on_stage_complete is called after each stage with (stage_name, result).
    Must be called via app.run_cpu_bound() from the UI — see Section 7.4.
    """
    ...
```

```python
# backend/scale.py  (real implementation, not a stub)
import math

def compute_scale(pixel_distance: float, known_length_um: float) -> float:
    """Returns µm/px from a drawn line measurement."""
    if pixel_distance <= 0:
        raise ValueError("Pixel distance must be positive")
    if known_length_um <= 0:
        raise ValueError("Known length must be positive")
    return known_length_um / pixel_distance

def pixel_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two image-space points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def pixel_distance_horizontal(x1: float, x2: float) -> float:
    """Horizontal-only distance (for Shift-constrained lines)."""
    return abs(x2 - x1)
```

---

## 10. State Management

### 10.1 In-Memory State

A single `AppState` object (reactive, using NiceGUI's binding system) holds all shared state. UI components bind to it; backend functions read from it.

```python
# state/app_state.py
class AppState:
    # Directory
    image_dir: str = ''
    output_dir: str = ''
    image_files: list[str] = []
    selected_image: str = ''

    # Scale
    scale_mode: str = 'per-image'        # 'locked' | 'per-image'
    scale_um_per_px: float | None = None
    scale_known_length: float | None = None
    scale_pixel_distance: float | None = None
    scale_confirmed: bool = False

    # ROI
    roi_crop_bottom_pct: float = 0.0

    # Mode
    advanced_mode: bool = False
    step_through: bool = False
    batch_mode: bool = False

    # Pipeline
    pipeline_running: bool = False
    current_stage: int = 0               # 0=idle, 1=depth, 2=seg, 3=diameter
    result: PipelineResult | None = None

    # Advanced settings
    depth_threshold_auto: bool = True
    depth_threshold_manual: int = 128
```

### 10.2 Session Persistence

To prevent data loss from crashes and avoid re-calibration after restarts, FiberSight maintains a lightweight JSON sidecar file in the image directory.

**File:** `<image_dir>/.fibersight/session.json`

**Contents:**

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
    },
    "fiber_mesh_002.tif": {
      "scale_um_per_px": null,
      "last_run": null
    }
  }
}
```

**Behavior:**

- **On directory load:** If `.fibersight/session.json` exists, load it. Per-image scale values are restored. The image dropdown shows calibration status indicators (green dot / amber dot) next to each filename.
- **On scale confirm:** Write the scale value for the current image to the session file immediately.
- **On pipeline completion:** Write a pointer to the result file. Actual result data (per-fiber CSV, annotated image) is saved to the output directory — the session file only stores metadata.
- **On image switch:** If the selected image has a saved scale value in the session, restore it to the UI. If it has a previous result, load summary stats (not full data) so the user can see "this image was already processed" without re-running.

This also enables the Batch Preview table (Section 5.5) to show calibration status for all images without requiring the user to click through each one.

---

## 11. Theme Specification

Use **Quasar's built-in dark mode** as the base. Override only where needed to achieve the desired aesthetic. This avoids fighting Quasar's CSS and reduces maintenance.

**Overrides:**

| Quasar token | Override value | Notes |
|---|---|---|
| `primary` | `#2dd4bf` (teal) | Accent: buttons, active tabs, progress bars |
| `dark-page` | `#1a1a2e` | Page background |
| `dark` | `#16213e` | Card/surface backgrounds |

**Additional CSS (minimal):**

- Log panel background: `#0d1117`
- Font: System sans-serif (no custom font loading for v1)
- Monospace: `monospace` system default

All other tokens (text colors, borders, input backgrounds, hover states) inherit from Quasar dark defaults. Exact values will be tuned live during Phase 1 implementation.

---

## 12. Model Management

### 12.1 Expected Layout

```
models/
├── README.md                    # Setup instructions
├── depth_anything_v2_small/     # DepthAnythingV2-Small weights (HuggingFace cache or local)
└── fiber_segmentation/          # Connor's trained segmentation model weights
```

### 12.2 Lazy Loading

Models are **not** loaded at app startup. They are loaded on first pipeline run and cached in memory for subsequent runs. This keeps startup time fast (<2 seconds for UI-only launch).

**Loading sequence:**

1. User clicks "Run Pipeline".
2. Log: `[...] Loading depth model (first run, may take 10–20s)...`
3. `backend/models.py` checks if the model is already in memory. If not, loads from `models/` directory.
4. If weights are missing, returns a `StageResult(success=False, error="Model weights not found at models/depth_anything_v2_small/. See models/README.md for setup instructions.")`.
5. UI shows the error in the log and a notification toast. Pipeline does not proceed.
6. On subsequent runs, the model is already cached — no reload delay.

### 12.3 Startup Validation (Non-Blocking)

On app startup, `backend/models.py` performs a quick filesystem check (do the expected weight files exist?) without loading them. If missing, a persistent amber warning banner appears at the top of the page: "Model weights not found. See models/README.md for setup." The app remains fully functional for UI testing and scale calibration.

---

## 13. Development Phases

### Phase 1 — Skeleton + Scale Tool (UI-only, no backend)

- [ ] **Task 0 (blocker):** Verify `ui.interactive_image` mouse event coordinate space under pan/zoom. Build minimal test script (see Section 7.2). If viewport-relative, implement coordinate transform.
- [ ] Project scaffolding: `main.py`, `ui/`, `backend/` with stubs, `state/`
- [ ] Theme: Quasar dark mode + targeted overrides
- [ ] Header bar with mode toggle (Basic/Advanced)
- [ ] Sidebar: Data Directory card (directory input, dropdown, prev/next)
- [ ] Center: IMAGE tab with `ui.interactive_image` (pan/zoom)
- [ ] `LayerManager` class — composable SVG overlay system
- [ ] Scale calibration: free-draw line tool with Shift-to-constrain, using LayerManager
- [ ] Scale card: fields, lock toggle, status chip
- [ ] `backend/scale.py` — real implementation (pure math)
- [ ] Session persistence: `.fibersight/session.json` read/write
- [ ] Model startup validation (filesystem check, warning banner if missing)

### Phase 2 — Pipeline Integration

- [ ] Wire `backend/depth.py` to `DepthAnythingPredictor` from SEM-Annotator
- [ ] Wire `backend/segmentation.py` to Connor's trained model
- [ ] Wire `backend/diameter.py` to extraction logic
- [ ] Implement `StageResult` error model across all backend modules
- [ ] Async pipeline execution via `app.run_cpu_bound()` (see Section 7.4)
- [ ] Lazy model loading with cache (`backend/models.py`)
- [ ] Run controls: single image execution with progress + log output
- [ ] RESULTS tab: summary stats + diameter histogram
- [ ] Write results to output directory + session file on completion

### Phase 3 — Advanced Mode + Polish

- [ ] ROI crop card: bottom-% slider with preview overlay via LayerManager
- [ ] DEPTH MAP tab with side-by-side + interactive threshold slider
- [ ] SEGMENTATION tab with color-coded instance masks
- [ ] Pipeline Settings card (depth threshold override)
- [ ] Step-through mode (pause between stages, per-stage ✓/✗ status)
- [ ] Advanced results panels: orientation plot, per-fiber table, annotated overlay
- [ ] Export buttons (CSV, annotated image, ZIP)

### Phase 4 — Batch + Extras

- [ ] Batch Preview table with per-image calibration status
- [ ] Batch directory processing with aggregate CSV
- [ ] Per-image scale management in batch mode (from session file)
- [ ] Failed-image handling: skip + log + summary
- [ ] Future stats: node detection, per-fiber diameter profiles
- [ ] Performance profiling / GPU utilization feedback

---

## 14. Future Considerations (v2)

### 14.1 Results Gallery / Cross-Image Comparison

v1 limitation: switching images clears the current results view. There is no way to compare results between two images side-by-side within the app. Results are written to disk per-image, so the data exists — but the UI doesn't aggregate or navigate it.

**v2 feature:** A "Results Gallery" tab that lists all processed images in the directory with thumbnail previews, summary stats per image, and the ability to select two images for side-by-side comparison (stats, histograms, overlays). Data source: the session file + per-image result files in the output directory.

### 14.2 Drawable ROI Selection

Replace the bottom-% slider with a user-drawable rectangular (or polygonal) ROI on the image, using the LayerManager for the selection overlay.

### 14.3 Additional Metrics

- Node detection (fiber crossover points)
- Per-fiber diameter profiles (diameter variation along length)
- Porosity estimation from segmentation masks
- Batch-level statistical comparisons across sample groups

---

## 15. Open Questions for Connor

1. **Segmentation model architecture** — What model is being trained? (YOLOv8-seg, Mask R-CNN, U-Net?) This determines the inference wrapper in `backend/segmentation.py`.
2. **Model input/output contract** — What does the model expect as input (depth-masked image? raw image? resolution?) and what does it return (instance masks? bounding boxes + masks?)?
3. **Roboflow export format** — COCO JSON, YOLO, mask PNGs? Affects how we validate test data.
4. **Orientation extraction** — Is this derived from the segmentation masks (skeleton direction), or is there a separate approach?
5. **Diameter extraction method** — Confirm: medial axis skeletonization → perpendicular cross-section sampling? Any corrections for oblique fibers?
6. **Model file size** — How large are the bundled weights? Affects distribution strategy (Git LFS, separate download, etc.).
7. **GUI scope for paper** — Should the paper describe the GUI as a "companion tool" appendix, or is it a core contribution? This affects how much we document the UX.
