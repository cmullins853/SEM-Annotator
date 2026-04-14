import os

DEPTH_MODEL_PATH = os.path.join('models', 'depth_anything_v2_small')
SEG_MODEL_PATH = os.path.join('models', 'fiber_segmentation')


def check_models_startup(state) -> None:
    """Non-blocking filesystem check. Updates state and toggles warning banner."""
    state.depth_model_available = (
        os.path.isdir(DEPTH_MODEL_PATH) and bool(os.listdir(DEPTH_MODEL_PATH))
    )
    state.seg_model_available = (
        os.path.isdir(SEG_MODEL_PATH) and bool(os.listdir(SEG_MODEL_PATH))
    )
    all_ok = state.depth_model_available and state.seg_model_available
    if state.model_banner is not None:
        state.model_banner.set_visibility(not all_ok)
