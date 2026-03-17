from __future__ import annotations

from vivid_inference_core import (
    CommunityModelLogicBase,
    ModelArtifactSpec,
    register_model_pack,
    resolve_model_repo,
)


class ExampleCommunityUpscaler(CommunityModelLogicBase):
    PYTORCH_NATIVE = True

    def process(self, clip, config, backend, model_path):
        # Replace with real inference path.
        # model_path may resolve to .onnx/.pth/.pt/.pkl/.param/.safetensors
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
