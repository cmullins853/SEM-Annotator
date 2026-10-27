# scripts/: legacy standalone tools (to be retired)

These are the original OpenCV/tkinter desktop tools from before the FiberSight GUI.
They still work, but each is meant to be **deleted once the GUI covers its job**.
Don't build new features here; build them in `fibersight/`.

Run from the repo root after `uv sync --extra inference`:

| Script | What it does | Replaced by (in the GUI) | Delete when |
|---|---|---|---|
| `scale_pixel_calc.py` | tkinter tool: click two points on a scale bar → µm/px | Scale Calibration card + draw on IMAGE tab (**done**) | Now. Nothing imports it. Confirm nobody still uses it. |
| `background_remover.py` | Intensity-threshold background removal (in-range / binary / to-zero / Otsu) with sliders, `a` to batch the folder | Not planned for the GUI yet. The pipeline uses depth-based removal instead. | Once it's confirmed intensity thresholding isn't needed, or a threshold mode is added to Pipeline Settings |
| `depth_anything_remover.py` | Depth Anything V2 background removal with a depth-cutoff slider, 3-pane view | DEPTH MAP tab + depth threshold in Pipeline Settings (**placeholder**: wired when the pipeline is) | The depth stage runs in the GUI **and** `DepthAnythingPredictor` has moved into `ml/` (`ml/sam_infer_diameters.py` imports it from here today) |

## Retirement checklist (per script)

1. The GUI feature in the "Replaced by" column works on real SEM images.
2. `grep -rn "<script_name>" --include=*.py --include=*.sh --include=*.md .` returns nothing
   outside this folder (update docs that mention it).
3. `git rm scripts/<script>.py` and remove its row here. Delete this folder when it's empty.

Example commands, kept here so the main README stays about the app:

```bash
python scripts/background_remover.py --dir path/to/images --ext "*.tif"
python scripts/depth_anything_remover.py --dir path/to/images --model depth-anything/Depth-Anything-V2-Small-hf
python scripts/scale_pixel_calc.py
```
