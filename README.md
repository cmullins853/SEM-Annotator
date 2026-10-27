# SEM-Annotator

Measure fiber diameters in SEM images. Two halves share this repo:

- **FiberSight** (`fibersight/`, `main.py`): a local NiceGUI app for browsing SEM
  images, calibrating scale, and running the measurement pipeline.
- **ML research code** (`ml/`): the SAM-SHF segmentation model, training, and the
  command-line inference/diameter tools.

They meet at one function, `fibersight/pipeline.py::run_pipeline`, which is currently
a **stub** returning dummy data.

**[docs/SPEC.md](docs/SPEC.md) is the spec**: architecture, the rules new UI code must
follow, feature status, the pipeline contract, roadmap and open questions.

## Layout

```
main.py                 FiberSight entry point
fibersight/             the GUI app
  state.py              server-scope state (STATE) + the run job
  session.py            restart session + per-folder scale calibrations (atomic JSON)
  pipeline.py           ← the GUI/ML seam: run_pipeline, result types, weights paths
  files.py              image folder scan, native folder picker
  preview.py            TIFF → PNG display cache (<image_dir>/.fibersight/preview/)
  ui/                   layout.py (page + reconcile loop), header, log, theme
    sidebar/            data directory, scale calibration, ROI crop, settings, run controls
    center/             image viewer (scale drawing, overlays), depth/seg/results tabs
ml/                     research code (Connor), run from the repo root
  model/                SAM / ViT architectures
  shf/                  Symmetrical Hierarchical Forest patchify
  train/                train_comparison.py, energy_monitor.py
  sam_infer_diameters.py    image → mask → polygons → diameters (CLI)
  sam_polygon_diameters.py  mask/polygon → diameters (CLI, library for the above)
  yolo_to_mask.py, tile_generator.py   dataset prep
  batch-run.sh          Compute Canada SLURM job
scripts/                legacy desktop tools, to be retired (see scripts/README.md)
weights/                model weights, git-ignored (see weights/README.md)
docs/                   SPEC.md (current); archive/ (v0.2 GUI spec, Synplex resilience spec)
tests/                  python -m tests.test_session ; python -m tests.test_preview
```

## FiberSight (GUI)

```bash
uv sync                          # GUI only (Python 3.13, NiceGUI 3.18)
uv sync --extra inference        # + torch/timm/transformers to run the model
uv sync --extra train            # + training and dataset-prep packages
uv run python main.py            # http://127.0.0.1:8888
uv run python main.py --debug    # skip the model-weights check
```

Everything is pinned in `uv.lock`. Run Pipeline is enabled once weights are present
(or `--debug`), an image is loaded, and its scale is calibrated. The app survives
refreshes, extra tabs and restarts: state lives on the server, and the last working set
is offered back on startup. Read [docs/SPEC.md §3](docs/SPEC.md) before adding UI.

## ML tools (command line)

```bash
uv sync --extra train            # or --extra inference for the inference CLIs only
```

All commands run from the repo root.

**End-to-end inference**: a trained checkpoint on raw images; writes masks, polygon
JSON, overlays and diameter CSVs:

```bash
python -m ml.sam_infer_diameters --input path/to/images --weights weights/fiber_segmentation/canny_cov0.3_s42/best_model_sam.pth --output_dir sam_results --diameter_method mic
python -m ml.sam_infer_diameters --input path/to/masks --input_mode mask --output_dir sam_results --methods all
```

Add `--depth_preprocess` for Depth Anything background removal and `--tile_size 1024`
for large images. Outputs include `diameter_instances.csv`,
`diameter_stats_per_polygon.csv` and `diameter_stats_per_image.csv`.

**Diameters from existing masks / polygon JSON**: maximum inscribed circle (`mic`)
and minimum Feret (`min_feret`):

```bash
python -m ml.sam_polygon_diameters --input path/to/masks --output_csv polygon_diameters.csv --method mic --overlay_dir overlays
python -m ml.sam_polygon_diameters --input path/to/masks --methods all --instances_csv polygon_instances.csv --per_polygon_stats_csv polygon_stats.csv --per_image_stats_csv image_stats.csv
```

**Dataset prep and training:**

```bash
python ml/yolo_to_mask.py --image-dir data/images --label-dir data/labels --output-dir data/masks
python ml/tile_generator.py --img_dir data/images --mask_dir data/masks --tile_size 1024 --no-gui
python ml/train/train_comparison.py --method canny --image-dir data/images/tiles_output/images --mask-dir data/images/tiles_output/masks --coverage 0.5 --mixed-precision bf16
```

`train_comparison.py` supports `--val-split`, `--test-split`, `--patience`, `--resume`,
and multi-GPU via `accelerate launch`. See `ml/batch-run.sh` for the cluster job.
