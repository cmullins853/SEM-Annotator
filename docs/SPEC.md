# FiberSight Spec v0.3

**The single source of truth for what FiberSight is, how it's built, and what comes next.**
Updated 2026-10-27, after the restructure and resilience pass (branch `cleanup/restructure`).

Supersedes the GUI spec v0.2 (2026-04-14) and the Synplex resilience spec; both are in
`docs/archive/` for history and rationale. Requirement IDs:

- `F-xx`: features and behavior (new in v0.3)
- `REQ-xx`: resilience rules, numbered as in `archive/NICEGUI_RESILIENCE_SPEC.md` so the
  incident history (H1–H18) there still explains why each one exists
- Status: ✅ done · 🟡 partial / stub · 🔲 planned · ❓ needs a decision (§10)

---

## 1. Purpose and scope

FiberSight is a local, browser-based app for measuring fiber diameters in SEM images.
A researcher with no CLI experience loads a folder of images, calibrates scale from
the embedded scale bar, runs a **pre-trained** model, and exports measurements.

- **Plug and play.** End users never train. Training lives in `ml/` and is run by the
  model developers (Connor), who ship weights into `weights/`.
- **Single user, local.** One server process on the user's machine, bound to
  `127.0.0.1`. Multiple tabs are allowed but share one working set.
- **Dark theme only.** Blue accent (`fibersight/ui/theme.py`, all colors defined there).

---

## 2. Architecture

### 2.1 Repository

| Path | Role |
|---|---|
| `main.py` | Entry point. Sets `STATE.models_ok`, serves `/image`, `ui.run` |
| `fibersight/state.py` | `STATE`: all server-side state, plus `run_job` |
| `fibersight/session.py` | Restart session + per-folder scale store (atomic JSON) |
| `fibersight/pipeline.py` | **The GUI/ML seam.** `run_pipeline`, result types, weights paths |
| `fibersight/files.py` | Folder scan, native folder picker |
| `fibersight/preview.py` | TIFF → PNG display cache, 16-bit scaling, background pre-warm |
| `fibersight/ui/` | `layout.py` (page, reconcile loop, Resume), `sidebar/*`, `center/*` |
| `ml/` | Model, SHF, training, CLI inference/measurement, dataset prep, cluster job |
| `scripts/` | Legacy desktop tools, retired one by one (`scripts/README.md`) |
| `weights/` | Model weights, git-ignored (`weights/README.md`) |
| `tests/` | `python -m tests.test_session`, `python -m tests.test_preview` |

`fibersight/` never imports `ml/` except inside `pipeline.py`. `ml/` never imports
`fibersight/`.

### 2.2 Install (uv, one lockfile)

| Command | Gets | For |
|---|---|---|
| `uv sync` | NiceGUI, numpy, OpenCV, Pillow | GUI, scale calibration, browsing |
| `uv sync --extra inference` | + torch, timm, transformers, tqdm | running the trained model |
| `uv sync --extra train` | + torchvision, accelerate, matplotlib, rasterio, nvidia-ml-py | training, dataset prep |

torch comes from PyPI: a CUDA build on Linux, CPU on Windows/macOS. The Compute
Canada job (`ml/batch-run.sh`) uses the cluster's wheelhouse and Python 3.12, not the
lock. Install the `train` extra's package names there with `pip --no-index`.

### 2.3 State model (three tiers + outputs)

| Tier | Lives in | Survives | Holds |
|---|---|---|---|
| Page view | builder locals, per-page `page`/`view` dicts | nothing | widget handles, what this page last painted |
| Run state | `STATE` (module level) | refresh, tab switch, reconnect, second tab | working set, settings, job status, log, result |
| Session | `.fibersight_session.json` + `<image_dir>/.fibersight/session.json` | server restart | pointers and user edits, never results |
| Outputs | output folder (🔲) | everything | masks, polygons, CSVs, overlays |

---

## 3. Resilience rules (apply to every new feature)

These are the rules new UI code must follow. "Where" points at the code that does it today.

| Rule | Requirement | Where | Status |
|---|---|---|---|
| REQ-01/02 | State is server-scope plain data. No UI element, bound method or closure in `STATE` | `state.py` | ✅ |
| REQ-03 | Handlers mutate `STATE` and return. Painting happens via bindings or `sync()` | all builders | ✅ |
| REQ-03a | **Never `bind_value_from` an editable input to `STATE`.** On a plain object, NiceGUI re-forces the bound value every 0.1 s, wiping whatever the user is typing (found 2026-10-27: typed folder paths vanished before Enter). Use two-way `bind_value`, or set it from `sync()` only when `STATE` changes. One-way binds are fine for display-only properties (text, visibility, enabled, progress) | `data_directory.py` | ✅ |
| REQ-04 | `status` ∈ idle/running/complete/error (+ cancelled/interrupted for batches), `generation` +1 per new result | `AppState` | ✅ (batch states 🔲) |
| REQ-05/06 | Per-page dicts and widget registries are created inside the page | `layout.py` `page`, `view` | ✅ |
| REQ-10–13 | One reconcile timer per page (0.5 s), called once at build, never raises, repaints only on change | `layout.py::reconcile`, builder `sync()`s | ✅ |
| REQ-14 | Run button enabled/disabled from `STATE` on every page | `bind_enabled_from(STATE, 'can_run')` | ✅ |
| REQ-15 | Toasts only for transitions this page watched | `reconcile` | ✅ |
| REQ-16 | Results rebuilds keep the active tab | | 🔲 with real results |
| REQ-20/21 | Heavy work off the loop, started as a server task, not owned by a page | `run_job` via `background_tasks.create` + `asyncio.to_thread` | ✅ |
| REQ-22 | One job at a time, enforced in `run_job` | `can_run` check | ✅ |
| REQ-23 | Publish data first, then progress, status, generation | `run_job` | ✅ |
| REQ-24/63 | Reset per-result user edits only after a *different* input succeeds | | 🔲 when edits exist |
| REQ-25 | Pin inputs before awaiting | `run_job` | ✅ |
| REQ-26/27/28 | Worker reports via callback into a bounded log with a monotonic cursor | `AppState.log`, `pipeline_log.py` | ✅ |
| REQ-29 | `matplotlib.use("Agg")` before anything draws server-side | | 🔲 if the GUI imports matplotlib |
| REQ-30/31 | After any `await`, don't touch a dead page; a dismissed dialog (`None`) is "not now" | `_offer_resume`, data-directory handlers write `STATE` only | ✅ |
| REQ-32 | No file/DB handle held across a dialog | | 🔲 |
| REQ-33/34 | `nicegui>=3.18`, `reconnect_timeout=30` | `pyproject.toml`, `main.py` | ✅ |
| REQ-40 | Pixels never go through the websocket; numbers at full resolution | `/image` HTTP route | ✅ (overlays of real results 🔲) |
| REQ-41 | Build tab panels lazily | | 🔲 when DEPTH/SEG get content |
| REQ-42/43 | Update figures in place; keep zoom (`uirevision`) | | 🔲 with histogram |
| REQ-44 | No expensive work per slider tick | sliders bind to `STATE`; overlay repaints in `sync` | ✅ |
| REQ-45/46 | Derived caches in state, never persisted; mtime cache-bust for regenerated images | | 🔲 |
| REQ-47 | Re-read file lists on demand | re-entering a path rescans | 🟡 (Refresh button 🔲) |
| REQ-48 | No `format=` on physical-value number fields | scale card | ✅ |
| REQ-50–57 | Versioned, atomic, never-raising, validated session file; autosave on change | `session.py` | ✅ |
| REQ-58–61 | Resume offered (never automatic), once per server start, explicit answers only, restored through `STATE` | `layout.py::_offer_resume` | ✅ |
| REQ-60a | Track *which client* shows the Resume prompt, not just an "offered" flag. A refreshed page's old client lives on for `reconnect_timeout` (30 s), so with a flag that dead page held the prompt and the refreshed page skipped it (found 2026-10-27, a gap inherited from Synplex). A new page takes over when the holder has no socket connection | `_resume['holder']` | ✅ |
| REQ-62 | Actions with hard-to-undo side effects start off after Run/Resume | | 🔲 |

The batch rules (REQ-70–81) are in §7.

---

## 4. Features

### 4.1 Header
- F-01 ✅ Branding, version chip, Advanced Mode switch (bound to `STATE`), Shutdown.
- F-02 🔲 Settings dialog (gear). Contents TBD; nothing needs it yet.

**Basic vs Advanced:** Advanced shows the ROI and Pipeline Settings cards and the
DEPTH MAP / SEG tabs. Basic shows only what a typical user needs.

### 4.2 Data Directory
- F-10 ✅ Image folder (typed + Enter/blur, or native picker). Scans `.tif .tiff .png .jpg .jpeg`.
- F-11 ✅ Output folder (typed or picker), stored in `STATE` and the session.
- F-12 ✅ Image select + prev/next. Selecting loads the image header and (per-image mode) its saved scale.
- F-13 🔲 Calibration status per image in the dropdown (green/amber dot), from the scale store.
- F-14 🔲 Refresh button to rescan the folder (REQ-47).

### 4.3 Scale Calibration
- F-20 ✅ Draw a line on the image: two clicks, Shift = horizontal. Coordinates are
  full-resolution image pixels at any zoom (verified).
- F-21 ✅ Known length (µm) → µm/px, stored at full precision. Manual µm/px entry allowed.
- F-22 ✅ Confirm saves to `<image_dir>/.fibersight/session.json` keyed by filename.
- F-23 ✅ "Apply to all" (locked mode): one scale for the folder, not reset on image change.
- F-24 🔲 Clear/reset a confirmed scale (no control exists today).
- F-25 🔲 Key calibrations by content hash instead of filename (REQ-80), so renames keep them.

### 4.4 ROI / Crop (Advanced)
- F-30 ✅ Exclude bottom N% (0–50) for the SEM metadata bar, with a red preview overlay.
- F-31 🟡 Passed to the pipeline as `PipelineConfig.roi_crop_bottom_pct`. The ML code's
  equivalent is `--depth_label_pct` (default **12**). The GUI default is 0. See Q4.
- F-32 🔲 (v2) Drawable rectangle/polygon ROI.

### 4.5 Pipeline Settings (Advanced)
- F-40 🟡 Depth threshold: "Auto (Otsu)" checkbox + manual 0–255 slider. The ML code
  has no Otsu: it uses a fixed threshold (default 80) with "keep above" inversion. See Q3.
- F-41 🔲 Diameter method: MIC / min-Feret / both (the ML code supports both). See Q1.
- F-42 🔲 Probability threshold (ML default 0.5) and minimum polygon area (16 px²), if users need them.

### 4.6 Run Controls
- F-50 ✅ Run (FAB) enabled when weights are present, an image is loaded, scale is confirmed, and nothing is running. A hint says what's missing.
- F-51 ✅ Progress bar + timestamped log, per stage.
- F-52 🔲 Stop button: cooperative cancel between stages/images (REQ-77).
- F-53 🔲 Step-through mode: pause after each stage with "Next stage ▶". The toggle exists but does nothing.
- F-54 🔲 Batch Directory scope (§7). The toggle exists but does nothing.

### 4.7 Center tabs
- F-60 ✅ IMAGE: zoomable viewer with named SVG layers (`roi`, `scale`; results add `annotations`).
- F-61 ✅ Image display via the `/image` route: PNG/JPG are sent as-is; TIFFs go through the
  preview cache (F-67). 8-bit gray stays as is, palette → RGB, and 16/32-bit gray is scaled
  2–98 percentile to 8-bit (like the ML code's `scale_to_u8`). Pillow's plain conversion
  turned a 16-bit test image 100% white. Verified on real Zeiss SEM TIFFs (8-bit L and
  palette) and a 4096² JPG. Display only, never used for measurement.
- F-67 ✅ **Preview cache** for TIFFs, `fibersight/preview.py`, tested in
  `tests/test_preview.py`. Cached views take 13–30 ms against 140–470 ms uncached
  (2048–3072 px TIFFs). Not built yet: the optional CLI.
  - *Where:* `<image_dir>/.fibersight/preview/<original filename>.png` (e.g. `a.tif.png`),
    next to the scale store. **Not** alongside the originals: a cached `a.png` would
    collide with a real `a.png`, and the folder scan would list every cache as a new image.
  - *Key and staleness:* original filename. Rebuild if the original's mtime is newer than
    the cache's. Switch to content hash with F-25.
  - *When (50–100 image folders):* automatically, in the background, as soon as a folder is
    opened. Order: current image, its prev/next neighbours, then the rest. 4 worker threads
    (Pillow releases the GIL). Measured on 75 real TIFFs, 8 cores: **4.4 s** total
    (1 thread: 13.9 s), ~2.8 MB per image, so ~280 MB for 100 images. A view that hits a
    not-yet-cached image converts it on demand. Cancel the pre-warm when the folder changes;
    don't start one while the pipeline is running. The same function can back a CLI
    (`python -m fibersight.preview <dir>`) for very large folders.
  - *UI:* no dedicated widget or button. One status line in the Data Directory card,
    bound to `STATE` ("Preparing previews 62 / 100"), hidden when done. Staleness is
    automatic; deleting `.fibersight/preview/` is always safe.
  - *Not a throughput tool:* the pipeline reads the **original** files, and inference
    (seconds per image) dwarfs TIFF decoding, so pre-converting doesn't speed up a run.
    The cache only makes browsing instant.
  - *Serving:* `FileResponse` from the cache, so the browser caches too.
  - *Fidelity (validated 2026-10-27 on the 4 test images):* PNG is lossless at every
    compression level (0/1/6/9 round-trip bit-identical), so use Pillow's default level 6:
    level 1 saves ~30% time for +55% disk. Conversion keeps dimensions; L stays L; palette →
    RGB is an exact lookup. Pixel values never feed a measurement anyway: the preview only
    places scale-bar clicks, and those depend on dimensions. The one lossy step is 16-bit →
    8-bit (F-61), which is display-only. `tests/test_preview.py` checks that the cached PNG
    keeps the original's size and matches the display conversion pixel for pixel, that
    8-bit pixels are unchanged, and that 16-bit isn't white.
  - *Rules:* full resolution only. Downscaling would make scale-bar clicks land in preview
    pixels, so thumbnails would be a separate, smaller cache. A read-only image folder falls
    back to converting in memory.
- F-62 🔲 DEPTH MAP (Advanced): original | depth map side by side, mask overlay toggle, live threshold preview (debounced).
- F-63 🔲 SEG (Advanced): mask overlay on the image, polygon count badge, "mask only" toggle.
- F-64 🟡 RESULTS: stat cards bound to the result ✅. Histogram 🔲 (Plotly, from the measurement table, REQ-40/42/43).
- F-65 🔲 RESULTS advanced panels: per-polygon table, annotated overlay (MIC circles / Feret lines, like `ml/sam_polygon_diameters.py::draw_measurements`).
- F-66 🔲 Exports to the output folder: per-polygon CSV, annotated PNG, stats JSON, ZIP of all. Buttons exist, unwired.

### 4.8 Model weights
- F-70 ✅ Startup check of `weights/depth_anything_v2_small/` and `weights/fiber_segmentation/`. If either is missing: banner + Run disabled. `--debug` skips it.
- F-71 🔲 Load models lazily on first run, cache in the worker, show "loading model" in the log (REQ-71).
- F-72 🔲 Re-check weights without a restart (now: restart after adding them).

---

## 5. The pipeline contract (GUI ↔ ML)

`fibersight/pipeline.py::run_pipeline` is the only function the GUI calls. Today it is a
**stub** that returns dummy data.

```python
run_pipeline(image_path: str, scale_um_per_px: float, config: PipelineConfig,
             on_stage_complete: Callable[[str, StageResult], None] | None) -> PipelineResult
```

**Rules for the implementation:**
- Runs in a worker thread: no `nicegui`, no UI, no stdout scraping.
- Calls `on_stage_complete(name, StageResult)` once per entry in `STAGES`, in order
  (drives progress + log). Change `STAGES` if the stages change.
- Fails by raising, or by `StageResult(success=False, error=…)` with a message that names
  the cause. A failed stage means later stages report failure too, so RESULTS shows nothing.
- Measurements are in µm: pixel values × `scale_um_per_px`.
- Models load once (module cache), never per call. Weights paths are the constants at the top of the file.

**What the ML code does today** (`ml/sam_infer_diameters.py::process_image`):

1. *(optional)* Depth Anything V2 → depth map → fixed threshold → mask the image; the bottom `label_pct`% is excluded.
2. Resize to `target_size`² (1024), Canny edges → SHF patchify (`fixed_length` from coverage) → SAM-SHF ViT → probability map → resized back to the original size.
3. Threshold (0.5) → binary mask → contours ≥ `min_area` → polygons.
4. Per polygon: area, perimeter, centroid, **MIC diameter**, **min-Feret diameter** (+ the geometry to draw them).

**Where the v0.2 result types don't match the ML output** (to resolve when wiring, see §10):

| GUI type (v0.2) | ML reality |
|---|---|
| `SegmentationOutput.instance_mask` (one label per fiber) | Semantic binary mask; one *polygon* per connected region, so crossing fibers merge |
| `FiberMeasurement.orientation_deg`, `length_um` | Not computed |
| `FiberMeasurement.diameter_um` (one value) | Two methods: MIC and min-Feret |
| `DepthOutput.threshold_value` from Otsu | Fixed user threshold |

**Integration plan (Phase B):** in `ml/sam_infer_diameters.py`, split `process_image`
into a pure function (RGB array + params → probability map + polygon rows) and keep the
CLI as a thin wrapper that writes files. `run_pipeline` calls the pure function and maps
rows into the GUI types. Then fold the duplicated helpers into one place: `GDTModelWrapper` and
`reconstruct_batch_preds` (in inference + training), `get_scaling_params`/`scale_to_u8`
(inference + tiling), and move `DepthAnythingPredictor` from `scripts/` into `ml/`.

**Where new ML files go:** architectures `ml/model/`, patchify `ml/shf/`, training
`ml/train/`, inference/measurement and dataset prep in `ml/`, dependencies in the
`inference` or `train` extra of `pyproject.toml`, weights in `weights/`. Imports are
absolute from the repo root (`from ml.model.sam import …`); run modules with `python -m ml.<module>`.

---

## 6. Persistence

| Store | File | Holds | Written |
|---|---|---|---|
| Restart session ✅ | `.fibersight_session.json` (app folder) | image/output folders, selected image, scale mode + locked scale, ROI, mode/settings toggles. `version: 1` | on change, from reconcile |
| Scale calibrations ✅ | `<image_dir>/.fibersight/session.json` | `images[filename] = {scale_um_per_px, scale_known_length, scale_pixel_distance}` | on Confirm |
| Results 🔲 | `<output_dir>/…` | per-image polygons JSON, CSV, overlay PNG; batch manifest | by the pipeline |

- F-80 🔲 On image switch, if results exist in the output folder, show "already processed" + summary without re-running (v0.2 §10.2).
- Results are never stored in the session (REQ-51/81). Derived numbers are recomputed from outputs.
- Bump `SESSION_VERSION` on any incompatible payload change. Old files are then ignored, not migrated.

---

## 7. Batch processing (Phase E, all 🔲)

Scope toggle "Batch Directory" runs every eligible image in the folder.

- F-90 Batch preview table before starting: every image, its calibration (locked scale or saved per-image scale), and a checkbox to exclude it. Uncalibrated images in per-image mode are ineligible.
- REQ-70 The GUI talks to the backend through `submit / status / cancel / results`. The reconcile loop polls status.
- REQ-72 The batch spec (image list + content hashes, model id + weights hash, all params) is the key. Any change = a new batch.
- REQ-73/74 `<output_dir>/<batch>/manifest.json` tracks each image as pending/done/failed with its output path and error. Write each output atomically, then mark it done.
- REQ-75/76 Resume skips verified work. The Resume prompt shows "143 / 500 done (2 failed)". A manifest left in `running` with no live worker shows as `interrupted`.
- REQ-77 Cancel is cooperative between images and leaves a resumable `cancelled` manifest (this is the Stop button, F-52).
- REQ-78 One bad image (decode error, OOM) is marked failed and the batch continues. Failed images can be retried.
- REQ-79 Partial results are viewable while the batch runs, refreshed on a cheap counter.
- F-91 A combined CSV for all images + a pass/fail summary.

---

## 8. Roadmap

| Phase | Content | Status |
|---|---|---|
| A. Foundation | GUI skeleton, scale tool, sessions, resilience core, repo restructure, uv extras | ✅ |
| B. Wire the model | Answer §10 Q1–Q4; ML pure function; real `run_pipeline`; lazy model load (F-71); result types updated | 🔲 Connor + Brenden |
| C. Results | Histogram, per-polygon table, annotated overlay, exports to output folder (F-64–66), "already processed" (F-80). (Preview cache + 16-bit display F-67/F-61 done early) | 🔲 |
| D. Advanced views | DEPTH MAP + SEG tabs (lazy, F-62/63), step-through (F-53), Stop (F-52) | 🔲 |
| E. Batch | §7 | 🔲 |
| F. Retire legacy | Delete `scripts/` per `scripts/README.md` | 🔲 after D |
| v2 | Drawable ROI; results gallery / side-by-side comparison; node detection, diameter profiles along fibers, porosity; batch group statistics | ideas |

---

## 9. Verification checklist

Run in a browser before trusting a release. Passed on 2026-10-27 with the stub pipeline
and the four images in `test-images/`: 1, 3, 4, 5, 6, 7, and 2 except "Run disabled while
running" (the stub finishes instantly). Item 6 on a real Zeiss scale bar: 270.4 px at both
100% and 400% zoom vs 270.0 px measured from the pixels (0.15%).

1. Refresh mid-session: same folder, image, scale and results return.
2. Second tab: shows the same state, no duplicate "finished" toast. Run is disabled on both while running.
3. Restart the server → the Resume prompt shows the folder and date. Resume restores folder, image and scale.
4. Esc on the Resume prompt, then refresh → the prompt appears again and the file still exists.
5. Delete the image folder or corrupt the session file → no prompt, no crash (`tests/test_session.py`).
6. Scale line on a zoomed image gives the same pixel distance as unzoomed.
7. `/image` refuses paths outside the open folder.
8. 🔲 Refresh *during* a real (multi-second) run: progress continues live, results land.
9. 🔲 Background the tab > 60 s mid-run, return: same page.
10. 🔲 Websocket payload of a results refresh (dev tools → WS) does not grow with image size.
11. 🔲 (batch) Kill the server mid-batch, Resume → done images skipped, final results equal an uninterrupted run.
12. 🔲 (batch) Change one parameter, then Resume → treated as a new batch.

---

## 10. Open questions

Answer these before Phase B. The answers become F-requirements above.

1. **Which diameter goes on the summary cards?** MIC (inscribed circle, default in the ML CLI), min-Feret, or both? *Proposal: MIC on the cards, both in the table/CSV.*
2. **"Fiber count" vs polygon count.** Segmentation is semantic, so crossing fibers merge into one polygon. Is a polygon count acceptable as "fiber count", or should it be labelled "regions", or does the model need instance output?
3. **Depth threshold "Auto".** Implement Otsu on the depth map in the pipeline, or drop "Auto" and default the slider to the ML default (80)? And should depth preprocessing be optional (the ML CLI treats it as optional)?
4. **Bottom crop default.** GUI 0% vs ML 12%. Default to 12%, or keep 0% and make users set it?
   Data point: on both Zeiss test images the metadata bar is the bottom 10.2%. Bar heights
   vary by instrument (the JPG sample has no bar), so auto-detecting the bar may beat any default.
5. **Orientation and length.** v0.2 promised per-fiber orientation and length, which the ML code doesn't compute. Keep as a future metric (needs skeletonization), or drop from the result type?
6. **Large images.** The single pass resizes to 1024² (non-square images are stretched before inference). The tiled path measures per tile, so fibers are cut at tile edges. Which path should the GUI use, and above what size?
7. **Inference in the base install?** Plug-and-play users need torch to run the model. Once Phase B lands, should `inference` move into the base dependencies (so `uv sync` is enough), leaving only `train` as an extra?
8. **GPU on Windows.** Locked torch is CPU-only on Windows. Is CPU inference fast enough on target machines, or should we add a CUDA (or Intel XPU) torch index?
9. **Weights distribution.** How do users get `weights/`: a release download, Hugging Face, Git LFS? How big is the checkpoint?
10. **Paper scope.** Is the GUI a companion tool (appendix) or a core contribution? (v0.2 Q7, still open.)
