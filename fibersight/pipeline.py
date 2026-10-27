"""THE seam between the GUI and the ML code. Everything the GUI knows about models is here.

Currently a STUB that returns dummy data. To wire real models, replace the body of
`run_pipeline` (calling into `ml/`, e.g. the steps in `ml/sam_infer_diameters.py::process_image`)
and keep its signature and return types. See docs/SPEC.md §5.

Rules for whatever goes in here:
- It runs in a worker thread (asyncio.to_thread): no `nicegui` imports, no UI.
- Report progress only through `on_stage_complete`; the GUI turns it into log + progress.
- Fail by raising, or by returning a StageResult(success=False, error='message naming the problem').
- Load models once (module-level cache), not per call.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar
import numpy as np

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / 'weights'
DEPTH_MODEL_PATH = WEIGHTS_DIR / 'depth_anything_v2_small'
SEG_MODEL_PATH = WEIGHTS_DIR / 'fiber_segmentation'

STAGES = ('depth', 'segmentation', 'diameters')   # order of on_stage_complete calls

T = TypeVar('T')


def weights_present() -> bool:
    """Checked once at startup; the GUI disables Run and shows a banner if False."""
    return all(p.is_dir() and any(p.iterdir()) for p in (DEPTH_MODEL_PATH, SEG_MODEL_PATH))


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
    threshold_value: int


@dataclass
class SegmentationOutput:
    instance_mask: np.ndarray   # 0=bg, 1..N=fibers (labeled int array)
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
        return self.depth.success and self.segmentation.success and self.diameters.success


@dataclass
class PipelineConfig:
    depth_threshold_auto: bool = True
    depth_threshold_manual: int = 128
    roi_crop_bottom_pct: float = 0.0


# STUB — returns dummy data
def run_pipeline(
    image_path: str,
    scale_um_per_px: float,
    config: PipelineConfig,
    on_stage_complete: Callable[[str, StageResult], None] | None = None,
) -> PipelineResult:
    dummy_depth = StageResult(success=True, data=DepthOutput(
        depth_map=np.zeros((100, 100), dtype=np.uint8),
        depth_mask=np.zeros((100, 100), dtype=np.uint8),
        threshold_value=142,
    ), duration_sec=0.1)
    if on_stage_complete:
        on_stage_complete('depth', dummy_depth)

    dummy_seg = StageResult(success=True, data=SegmentationOutput(
        instance_mask=np.zeros((100, 100), dtype=np.int32),
        fiber_count=47,
    ), duration_sec=0.1)
    if on_stage_complete:
        on_stage_complete('segmentation', dummy_seg)

    dummy_fibers = [
        FiberMeasurement(i + 1, 1.2 + i * 0.02, 45.0 + i * 1.5, 5.0)
        for i in range(47)
    ]
    dummy_diameters = StageResult(success=True, data=DiameterOutput(
        fibers=dummy_fibers,
        mean_diameter=1.24,
        std_diameter=0.31,
        median_diameter=1.20,
    ), duration_sec=0.1)
    if on_stage_complete:
        on_stage_complete('diameters', dummy_diameters)

    return PipelineResult(
        depth=dummy_depth,
        segmentation=dummy_seg,
        diameters=dummy_diameters,
    )
