import torch
import numpy as np


def init_plugin(config: dict):
    """
    Optional one-time initializer.
    Return a state object used by process_frame(..., state).
    """
    params = config.get("plugin_params", {})
    backend = (config.get("backend") or "").lower()
    strength = float(params.get("strength", 0.5))

    use_cuda = backend.startswith("pytorch") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load your model/checkpoint here and move to device.
    # model = MyModel(...)
    # model.to(device).eval()

    return {
        "device": device,
        "strength": strength,
        # "model": model
    }


def process_frame(frame_hwc: np.ndarray, n: int, config: dict, state: dict) -> np.ndarray:
    """
    Pure frame-processor contract (no VapourSynth API required):
    - frame_hwc: float32 HxWxC image
    - n: frame index
    - config: full vivid config json
    - state: object returned by init_plugin()
    Return: float32 HxWxC image (same H/W/C)
    """
    _ = n
    _ = config
    _ = state
    return frame_hwc
