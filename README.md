# SEM-Annotator

SEM-Annotator is a suite of interactive Python tools tailored for the preprocessing, segmentation, and annotation of Scanning Electron Microscope (SEM) imagery, as well as broader remote sensing (RS) tasks. The repository integrates direct CV-based thresholding with advanced AI topographical modeling (Depth Anything V2), and incorporates Vision Transformer (ViT) and Segment Anything Model (SAM) pipelines via the Symmetrical Hierarchical Forest (SHF) protocol.

## Core Features

### 1. Interactive Background Removal (Intensity-Based)
`background_remover.py` is a GUI-based CLI tool designed to remove background from high-resolution, often 16-bit grayscale SEM images using direct intensity thresholding.
- **Dynamic Resizing**: Automatically scales high-resolution 16-bit TIFFs/PNGs down to fit the screen while preserving the original bit depth for the output.
- **Thresholding Modes**: 
  - `In-Range (Band)`: Keeps pixel intensities within a chosen range.
  - `Binary (Mask)`: Returns a binary mask.
  - `To-Zero (Lower Cut)`: Zeros out pixels below the threshold.
  - `Otsu (Automatic)`: Automatically discovers the optimal foreground/background split.
- **Controls**: Use `n`/`p` to quickly cycle images, move trackbars for real-time visualization, `a` to batch process the entire directory with the current parameters, and `s` to save.
- **Label Cropping**: Features a `Bottom %` slider to ignore SEM metadata bars during processing.

```bash
python background_remover.py --dir path/to/images
```

### 2. Topographical Background Removal (Depth-Based)
`depth_anything_remover.py` uses [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) to generate a relative 3D topography map, allowing you to slice away the background based on physical planes rather than just brightness.
- **Depth Cutoff**: An interactive slider that removes background based on the underlying depth inference, bypassing common issues where the background and foreground share the same brightness values.
- **3-Pane View**: Visualizes the original image, the Depth Map in a 'Magma' colormap, and the thresholded result side-by-side.
- **Model Caching**: Reuses depth maps as you tweak cutoff and inversion trackbars to maintain real-time 30+ FPS responsiveness.
- **Label Cropping**: Pre-crops the image before model inference, preventing the data bar from skewing the depth plane estimator.

```bash
python depth_anything_remover.py --dir path/to/images --model depth-anything/Depth-Anything-V2-Small-hf
```

### 3. Model Training Pipeline (SHF, SAM, ViTs)
The `/train`, `/model`, and `/SHF` directories contain the core training implementation and architecture definitions for the deep learning component of the project.
- **SAM & ViT Architectures**: Re-implemented Segment Anything Model and Vision Transformers explicitly configured for image patchification and representation learning.
- **Symmetrical Hierarchical Forest**: Adapts existing models to support Canny and Base features.
- **Training Scripts**: Check out `train/train_sam_shf.py` and `train/train_comparison.py`, featuring built-in distributed training support, metrics generation (Chamfer Similarity), and seamless Hugging Face Hub integration for checkpoint loading.
  - `train_sam_shf.py` features robust automated dataset splitting (`--val-split`, `--test-split`), early stopping via semantic loss tracking (`--target-dice`), mask density filtering (`--coverage`), and VRAM-optimized iteration (`--mixed-precision fp16`).

### 4. Remote Sensing & Tiling Utilities
`Tile_generator.py` is a robust batch-processing script that automatically slices paired directories of high-resolution images and masks into ML-ready datasets (e.g., 1024x1024 tiles).
- **Batch Processing**: Pass `--img_dir` and `--mask_dir` to automatically pair, align, and slice complex raster sets.
- **Robust Mask Normalization**: Includes a sophisticated `prepare_mask` utility that automatically handles:
  - 16-bit, float, and boolean mask formats.
  - Multi-channel masks and Alpha channel extraction.
  - Intelligent binarization and value scaling to ensure consistent `uint8` [0, 255] outputs.
- **Alignment & Resizing**: Automatically handles resolution mismatches between image/mask pairs using nearest-neighbor alignment to preserve label integrity.

### 5. Polygon Diameter Extraction
`sam_polygon_diameters.py` is a standalone deployment utility for extracting object diameters from SAM-style mask outputs or polygon JSON/GeoJSON files.
- **Multiple Diameter Metrics**: Reports maximum inscribed circle (`mic`) and minimum Feret (`min_feret`) measurements for each polygon.
- **Batch Friendly**: Works on a single file or a directory and exports a flat CSV that is easy to inspect downstream.
- **Optional Overlays**: Can save diagnostic PNGs that draw the chosen diameter measurement on top of each mask.

```bash
python sam_polygon_diameters.py --input path/to/masks --output_csv polygon_diameters.csv --method mic --overlay_dir overlays
```

To draw or export multiple measurements per polygon:

```bash
python sam_polygon_diameters.py --input path/to/masks --output_csv polygon_diameters.csv --methods all --measurements_csv polygon_diameters_long.csv --overlay_dir overlays
```

To export repeated measurement instances plus mean/std summaries per polygon and per image:

```bash
python sam_polygon_diameters.py --input path/to/masks --methods all --instances_csv polygon_instances.csv --per_polygon_stats_csv polygon_stats.csv --per_image_stats_csv image_stats.csv
```

For example, to draw three MIC circles inside each polygon and compute stats from those three circles:

```bash
python sam_polygon_diameters.py --input path/to/masks --methods mic --max_instances_per_method 3 --overlay_dir overlays --instances_csv polygon_instances.csv --per_polygon_stats_csv polygon_stats.csv --per_image_stats_csv image_stats.csv
```

### 6. End-to-End SAM Deployment
`sam_infer_diameters.py` runs a trained SAM-SHF checkpoint on raw images, reconstructs the predicted mask, converts it to polygons, computes diameters, and saves preview overlays with the measurements drawn on top of the image.
- **One-Step Inference**: Starts from raw images and a `best_model_sam.pth` checkpoint.
- **Polygon Export**: Writes per-image polygon JSON files in pixel coordinates.
- **Visual QA**: Saves overlay previews showing polygon outlines and one or more selected diameter geometries.
- **Mask Mode**: Can also consume preprocessed masks directly without running SAM inference.

```bash
python sam_infer_diameters.py --input path/to/images --weights SEM_comp/canny_cov0.3_s42/best_model_sam.pth --output_dir sam_results --diameter_method mic
```

For preprocessed masks:

```bash
python sam_infer_diameters.py --input path/to/preprocessed_masks --input_mode mask --output_dir sam_results --methods all
```

This will also write:
- `diameter_instances.csv`: repeated instances per polygon and method
- `diameter_stats_per_polygon.csv`: mean/std per polygon and method
- `diameter_stats_per_image.csv`: mean/std per image and method

## Setup and Requirements

Ensure your environment includes `opencv-python`, `numpy`, and standard data science tools:
```bash
pip install opencv-python numpy tqdm
```

For the deep learning modules (SAM, ViTs, Depth Anything V2), you will need Hugging Face distributions and PyTorch:
```bash
pip install torch torchvision transformers accelerate
```

## Batch Processing Options
Across the interactive tools, use the built-in batch processing capabilities:
- Tweak the sliders until visual perfection is met on a challenging image.
- Press `a` on your keyboard.
- The script automatically converts the entire directory into pristine, `.png`-enforced masks mirroring your exact configuration.
