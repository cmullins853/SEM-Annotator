# Model Weights

FiberSight uses two pre-trained models. Place weights in the directories below.

## Depth Estimation — DepthAnythingV2-Small

**Directory:** `models/depth_anything_v2_small/`

Download from HuggingFace:
```
huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf \
  --local-dir models/depth_anything_v2_small
```
Or via Python:
```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf",
                                   cache_dir="models/depth_anything_v2_small")
AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf",
                                            cache_dir="models/depth_anything_v2_small")
```

## Fiber Segmentation

**Directory:** `models/fiber_segmentation/`

Place Connor's trained model weights here. Expected format: TBD (see open question §15.1 in spec).

## Startup Validation

On launch, FiberSight checks whether these directories exist and are non-empty.
If either is missing, an amber warning banner appears at the top of the app.
The UI remains fully functional for scale calibration and image browsing.
The Run Pipeline button is disabled until weights are present and a scale is calibrated.
