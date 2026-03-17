#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PLUGIN_MAIN = """import numpy as np
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
"""


def plugin_manifest(plugin_id: str, name: str) -> dict:
    return {
        "schema_version": 2,
        "id": plugin_id,
        "name": name,
        "kind": "effect",
        "entry_script": "main.py",
        "parameters": [
            {
                "id": "strength",
                "type": "float",
                "label": "Strength",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "required": True,
            }
        ],
        "backends": ["cpu", "pytorch-cuda"],
        "repo": {
            "allow_user_override": False,
            "required_checkpoints": [],
            "allowed_override_roots": [],
        },
    }


def model_pack_python(slug: str) -> str:
    class_name = "".join(part.capitalize() for part in slug.replace("_", "-").split("-")) or "CommunityModel"
    return f"""from __future__ import annotations

from vivid_inference_core import (
    CommunityModelLogicBase,
    ModelArtifactSpec,
    register_model_pack,
    resolve_model_repo,
)


class {class_name}(CommunityModelLogicBase):
    PYTORCH_NATIVE = True

    def process(self, clip, config, backend, model_path):
        _ = backend
        _ = config
        _ = model_path
        return clip


def resolve_{slug.replace("-", "_")}_artifact(model_name: str | None, config_data: dict, current_dir: str) -> str | None:
    _ = config_data
    stem = model_name or "{slug}-v1"
    model_repo_root = resolve_model_repo("{slug}", current_dir=current_dir)
    model_repo_paths = [model_repo_root] if model_repo_root else []
    specs = [
        ModelArtifactSpec(
            search_roots=[
                f"{{current_dir}}/../models/community-{slug}",
                *model_repo_paths,
            ],
            file_stems=[stem],
        )
    ]
    return CommunityModelLogicBase.resolve_from_specs(specs)


register_model_pack(
    aliases=["{slug}"],
    model_logic={class_name},
    artifact_resolver=resolve_{slug.replace("-", "_")}_artifact,
)
"""


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scaffold_plugin(output_dir: Path, extension_id: str, name: str) -> None:
    write_file(output_dir / "main.py", PLUGIN_MAIN)
    manifest = plugin_manifest(extension_id, name)
    write_file(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"[scaffold] Created plugin template at {output_dir}")


def scaffold_model_pack(output_dir: Path, extension_id: str) -> None:
    write_file(output_dir / f"{extension_id.replace('-', '_')}.py", model_pack_python(extension_id))
    write_file(
        output_dir / "README.md",
        (
            f"# Community Model Pack: {extension_id}\n\n"
            "Register this pack by importing the Python module above from your host entrypoint.\n"
        ),
    )
    print(f"[scaffold] Created model pack template at {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold Vivid inference extensions.")
    parser.add_argument("--kind", choices=["plugin", "model-pack"], required=True)
    parser.add_argument("--id", required=True, help="Extension id/slug, e.g. anime-research-v1")
    parser.add_argument("--name", default="Community Extension", help="Human-readable name")
    parser.add_argument("--output-dir", required=True, help="Target directory for generated files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.kind == "plugin":
        scaffold_plugin(output_dir, args.id, args.name)
        return
    scaffold_model_pack(output_dir, args.id)


if __name__ == "__main__":
    main()
