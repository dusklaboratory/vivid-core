<img width="200" height="100" alt="image" src="https://github.com/user-attachments/assets/6d4c4f2e-1b10-4433-9382-5101e0a059c1" />

##
**Vivid** is an elegant GUI for **Video Frame Interpolation**, **Video Upscaling**, and **Video Restoration** powered by AI. Built with **React/TypeScript** and **Tauri (Rust)** for a lightweight, fast, and secure desktop experience. It ships with a pre-packaged runtime — no Docker, WSL, or manual Python setup required. Download vivid [here](https://vividenhance.com/)
##
<img width="1200" height="671" alt="image" src="https://github.com/user-attachments/assets/35e41db3-b1ef-4e14-be06-877f6bec9bbe" />

## `vivid-core` is Vivid's *open extension SDK* for:

- custom plugin scripts (`effect` / `inference`),
- community model architectures ("model packs"),
- backend extensions,
- host-agnostic runtime orchestration hooks.

The goal is simple: contributors can ship model logic without patching host internals.
##
<img width="1707" height="947" alt="image" src="https://github.com/user-attachments/assets/5bdf9698-8579-45e6-bec6-3c080fbb824e" />

## 2-minute quickstart

Scaffold either extension type:

```bash
python3 scripts/scaffold_extension.py --kind plugin --id my-effect --name "My Effect" --output-dir /tmp/my-effect
python3 scripts/scaffold_extension.py --kind model-pack --id myarch --output-dir /tmp/myarch-pack
```

What you get:

- `plugin`: `manifest.json` + `main.py` with a frame-processor starter.
- `model-pack`: a registerable `CommunityModelLogicBase` implementation.

## Runtime alignment

This SDK supports the same practical runtime mix most video AI authors need (ONNX and PyTorch with CPU/CUDA/MPS fallback) through stable helper APIs:

- `resolve_torch_device(backend_name)` for backend-aware CPU/CUDA/MPS device selection,
- `resolve_model_repo(...)` / `ensure_model_repo_on_path(...)` for host-managed model-repo imports,
- `ModelArtifactSpec` + `CommunityModelLogicBase.resolve_from_specs(...)` for flexible artifact discovery.

These helpers are intentionally small and composable so external packs follow the same execution style as built-in engines/effects.

## Plugin authoring (easy path)

Use frame-processor mode unless you need direct VapourSynth graph control:

```python
import numpy as np
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
```

Runtime contract:

- Input/output is HWC float array.
- Keep frame count/timeline stable unless intentionally changing it.
- Log to stderr (stdout is reserved for video stream output).

## Community model-pack authoring

```python
from vivid_inference_core import (
    CommunityModelLogicBase,
    ModelArtifactSpec,
    register_model_pack,
    resolve_model_repo,
)


class MyArch(CommunityModelLogicBase):
    PYTORCH_NATIVE = True

    def process(self, clip, config, backend, model_path):
        _ = backend
        _ = config
        _ = model_path
        return clip


def resolve_myarch_artifact(model_name: str | None, config_data: dict, current_dir: str) -> str | None:
    _ = config_data
    stem = model_name or "myarch-v1"
    model_repo_root = resolve_model_repo("myarch", current_dir=current_dir)
    model_repo_paths = [model_repo_root] if model_repo_root else []
    specs = [
        ModelArtifactSpec(
            search_roots=[
                f"{current_dir}/../models/community-myarch",
                *model_repo_paths,
            ],
            file_stems=[stem],
        )
    ]
    return CommunityModelLogicBase.resolve_from_specs(specs)


register_model_pack(
    aliases=["myarch"],
    model_logic=MyArch,
    artifact_resolver=resolve_myarch_artifact,
)
```

## Make it visible in Vivid's picker

Add entries to `catalog/community_catalog.json`:

```json
{
  "version": 1,
  "engines": [
    {
      "operation": "interpolation",
      "engine": "community:myarch",
      "label": "Community - MyArch",
      "backends": ["cpu", "pytorch-cuda"],
      "models": [{ "value": "myarch-v1", "label": "MyArch v1", "recommended": true }]
    }
  ]
}
```

## Public API highlights

- Contracts: `contracts.py`
- Plugin manifest schema: `plugin_manifest.schema.json`
- Runner: `plugin_runner.py`
- Registry: `community_registry.py`
- Runtime orchestrator: `run_pipeline(...)`
- Authoring helpers: `authoring.py`

## Compatibility + scope

- Vivid resolves `community:<slug>` engines and dispatches through this package.
- Custom plugins are local execution only.
- Private product concerns (auth, licensing, cloud policy, billing) stay out of scope.

## Versioning

- `0.x`: API stabilization phase.
- `1.x`: semver + breaking-change discipline on public contracts.

## License

- Community: AGPL-3.0-only (`LICENSE`)
- Commercial exception: `LICENSE-COMMERCIAL.md`
