import numpy as np
import torch
from vivid_inference_core import resolve_torch_device


def init_plugin(config: dict):
    params = config.get("plugin_params", {})
    return {
        "device": resolve_torch_device(config.get("backend")),
        "strength": float(params.get("strength", 0.5)),
    }


def process_frame(frame_hwc: np.ndarray, n: int, config: dict, state: dict) -> np.ndarray:
    _ = n
    _ = config
    strength = max(0.0, min(1.0, float(state.get("strength", 0.5))))
    return (frame_hwc * (1.0 - 0.1 * strength)).astype(np.float32, copy=False)
