import numpy as np
import torch


def init_plugin(config: dict):
    params = config.get("plugin_params", {})
    backend = (config.get("backend") or "").lower()
    use_cuda = backend.startswith("pytorch") and torch.cuda.is_available()
    return {
        "device": torch.device("cuda" if use_cuda else "cpu"),
        "strength": float(params.get("strength", 0.5)),
    }


def process_frame(frame_hwc: np.ndarray, n: int, config: dict, state: dict) -> np.ndarray:
    _ = n
    _ = config
    strength = max(0.0, min(1.0, float(state.get("strength", 0.5))))
    return (frame_hwc * (1.0 - 0.1 * strength)).astype(np.float32, copy=False)
