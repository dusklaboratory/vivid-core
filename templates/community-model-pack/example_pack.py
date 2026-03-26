from __future__ import annotations

from vivid_inference_core import (
    CommunityModelLogicBase,
    EngineCapabilityContract,
    ModelArtifactSpec,
    register_model_pack,
    resolve_model_repo,
)

EXAMPLE_CAPABILITIES = EngineCapabilityContract(
    supported_backends=["pytorch", "pytorch-cuda", "cpu", "dml", "trt", "trt-rtx", "ort-cuda"],
    required_pip_deps=["torch"],
    required_upstream_repos=[],
    supports_onnx=True,
    supports_pytorch=True,
    execution_mode="hybrid",
    fallback_backends=["pytorch", "cpu"],
)


class ExampleCommunityUpscaler(CommunityModelLogicBase):
    PYTORCH_NATIVE = True
    ENGINE_CAPABILITIES = EXAMPLE_CAPABILITIES

    def process(self, clip, config, backend, model_path):
        _ = backend
        _ = config
        _ = model_path
        return clip


def resolve_example_artifact(model_name: str | None, config_data: dict, current_dir: str) -> str | None:
    _ = config_data
    stem = model_name or "example-v1"
    model_repo_root = resolve_model_repo("community-example", current_dir=current_dir)
    model_repo_paths = [model_repo_root] if model_repo_root else []
    specs = [
        ModelArtifactSpec(
            search_roots=[
                f"{current_dir}/../models/community-example",
                *model_repo_paths,
            ],
            file_stems=[stem],
        )
    ]
    return CommunityModelLogicBase.resolve_from_specs(specs)


register_model_pack(
    aliases=["example"],
    model_logic=ExampleCommunityUpscaler,
    artifact_resolver=resolve_example_artifact,
)
