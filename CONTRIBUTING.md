# Contributing

## Scope

This repo accepts contributions for:

- model logic integrations,
- backend adapters,
- plugin manifest/runtime contracts,
- compatibility and validation tooling.

This repo does not accept cloud, licensing, auth, billing, or private product policy logic.

## Pull request checklist

- Keep changes modular and focused.
- Add or update tests for any contract changes.
- Avoid breaking public contract names without deprecation.
- Update docs for behavioral changes.
- Prefer the scaffold + helper APIs (`resolve_torch_device`, `resolve_model_repo`) to keep extensions aligned with runtime patterns.

## Backend and dependency declaration

When contributing a new model pack or engine:

1. **Declare engine capabilities** using `EngineCapabilityContract`:
   - List all supported backends (e.g. `["pytorch", "pytorch-cuda", "trt", "dml"]`)
   - Declare required pip dependencies and upstream repos
   - Specify whether the engine supports ONNX, PyTorch, or both (`hybrid`)
   - Include fallback backends for when the primary backend is unavailable

2. **Set `ENGINE_CAPABILITIES`** on your model logic class so the host runtime can derive install plans and run preflight checks.

3. **Test with multiple backends** — the CI will validate that declared backends actually work with your model.

Example:
```python
from vivid_inference_core import EngineCapabilityContract, CommunityModelLogicBase

class MyModel(CommunityModelLogicBase):
    ENGINE_CAPABILITIES = EngineCapabilityContract(
        supported_backends=["pytorch", "pytorch-cuda"],
        required_pip_deps=["torch"],
        supports_pytorch=True,
        execution_mode="native-torch",
    )
```

## Contract versioning

The package declares `CONTRACT_VERSION` (currently `1`). The host supports a version range for backward compatibility. When authoring new extensions, target the current `CONTRACT_VERSION`. Avoid depending on internal host APIs — use only symbols exported from `vivid_inference_core.__init__`.

## Model contribution baseline

- Document expected input/output tensor shape.
- Include backend compatibility notes using `EngineCapabilityContract`.
- Add deterministic smoke fixture for one frame path.
- Prefer explicit CPU/CUDA/MPS fallback behavior instead of hardcoding CUDA-only paths.
- Declare all pip dependencies in `ENGINE_CAPABILITIES.required_pip_deps` so the host can install them automatically.
