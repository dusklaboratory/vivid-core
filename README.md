# vivid-inference-core

`vivid-inference-core` is the open-source inference extension surface extracted from Vivid.

It is designed for community contributions around:

- new model logic implementations,
- backend adapters and compatibility layers,
- manifest-based plugin contracts and validation.
- host-agnostic runtime orchestration hooks used by desktop hosts.

## What this repo includes

- Stable Python contracts for model/backends/plugins.
- Plugin manifest schema and parameter type definitions.
- Runtime helpers for safe custom repo path handling.
- A plugin runner contract that can execute frame-based or graph-based plugins.
- Community extension registry (`community_registry`) used by Vivid runtime.
- A public `run_pipeline(...)` runtime orchestrator with host hook injection.

## How contributions affect Vivid

Vivid now loads `vivid_inference_core` by default (`VIVID_INFERENCE_CORE_MODE=public`) and uses:

- `community_registry.resolve_model_logic(model_type)` for model/effect dispatch extensions
- `community_registry.create_backend(backend_name, ...)` for backend extensions
- `community_registry.resolve_model_artifact(...)` for model weight/checkpoint resolution

To expose a new architecture, contributors register aliases in `community_registry` (for example `["community_my_arch"]`) and provide a compatible `ModelLogic` implementation.

For Rust-side model-type parsing, Vivid supports engine tokens like `community:<slug>` and forwards `<slug>` into Python model logic resolution.

### Public runtime contract

The public `run_pipeline(...)` supports host-level hook injection for:

- source/plugin loading setup,
- model artifact lookup fallback,
- fallback policy application,
- decision tracing and logging integration,
- structured progress callbacks.

This keeps orchestration open for researchers while still allowing app hosts to wire product-specific infrastructure.

### One-command scaffolding

Generate a starter extension without boilerplate:

```bash
python3 scripts/scaffold_extension.py --kind plugin --id my-effect --name "My Effect" --output-dir /tmp/my-effect
python3 scripts/scaffold_extension.py --kind model-pack --id myarch --output-dir /tmp/myarch-pack
```

### Community model-pack contract

Use:

- `CommunityModelLogicBase` to implement the runtime model class
- `register_model_pack(...)` to register aliases + artifact resolver
- `ModelArtifactSpec` to declare artifact discovery for `.onnx`, `.pth`, `.pt`, `.pkl`, `.param`, `.safetensors`

Example:

```python
from vivid_inference_core import CommunityModelLogicBase, ModelArtifactSpec, register_model_pack

class MyArch(CommunityModelLogicBase):
    PYTORCH_NATIVE = True

    def process(self, clip, config, backend, model_path):
        # run pytorch model from model_path
        return clip

def resolve_myarch_artifact(model_name, config_data, current_dir):
    specs = [
        ModelArtifactSpec(
            search_roots=[f"{current_dir}/../external/models/community-myarch"],
            file_stems=[model_name or "myarch-v1"],
        )
    ]
    return CommunityModelLogicBase.resolve_from_specs(specs)

register_model_pack(
    aliases=["myarch"],
    model_logic=MyArch,
    artifact_resolver=resolve_myarch_artifact,
)
```

### Picker catalog bridge

To make contributions visible in Vivid's model picker, add entries to:

- `catalog/community_catalog.json`

Shape:

```json
{
  "version": 1,
  "engines": [
    {
      "operation": "interpolation",
      "engine": "community:myarch",
      "label": "Community - MyArch",
      "backends": ["cpu", "pytorch-cuda"],
      "models": [
        { "value": "myarch-v1", "label": "MyArch v1", "recommended": true }
      ]
    }
  ]
}
```

## What this repo does not include

- Vivid cloud/auth/licensing implementation.
- private scheduling heuristics or business policy.
- product-specific catalog ranking/telemetry logic.

## Versioning policy

- `0.x`: rapid iteration, API stabilization in progress.
- `1.0+`: semver with breaking-change discipline on public contracts.

## License

- Community: AGPL-3.0-only (`LICENSE`)
- Commercial exception: `LICENSE-COMMERCIAL.md`
