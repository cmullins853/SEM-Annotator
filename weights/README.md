# Model Weights

FiberSight checks two folders at startup. If either is missing or empty, an amber
banner appears and **Run Pipeline** stays disabled; image browsing and scale
calibration still work. Restart the app after adding weights.
`python main.py --debug` skips the check.

Everything in this folder except this README is git-ignored.

The paths are defined in one place: `fibersight/pipeline.py` (`DEPTH_MODEL_PATH`,
`SEG_MODEL_PATH`). If the layout below changes, change it there.

## Depth Estimation: `weights/depth_anything_v2_small/`

```
huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf \
  --local-dir weights/depth_anything_v2_small
```

## Fiber Segmentation: `weights/fiber_segmentation/`

The trained SAM-SHF checkpoint (`best_model_sam.pth` from `ml/train/train_comparison.py`).
`ml/sam_infer_diameters.py::infer_params_from_weights` reads the patchify method and
coverage from the path when the checkpoint doesn't store them, so keep the run folder
name intact, e.g. `weights/fiber_segmentation/canny_cov0.3_s42/best_model_sam.pth`.
