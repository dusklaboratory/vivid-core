#!/usr/bin/env python3
"""Scaffold a new Vivid extension (plugin or model-pack).

Usage:
    python3 scaffold_extension.py --kind plugin    --id my-effect --name "My Effect"    --output-dir /tmp/my-effect
    python3 scaffold_extension.py --kind model-pack --id myarch    --output-dir /tmp/myarch-pack
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  created {path}")


def scaffold_plugin(plugin_id: str, name: str, output_dir: str) -> None:
    manifest = {
        "schema_version": 1,
        "id": plugin_id,
        "name": name,
        "version": "0.1.0",
        "author": "Your Name",
        "description": f"{name} -- a custom Vivid plugin.",
        "kind": "inference",
        "entry_script": "main.py",
        "backends": ["cpu", "pytorch", "pytorch-cuda"],
        "parameters": [
            {
                "id": "strength",
                "type": "float",
                "label": "Effect Strength",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
            }
        ],
    }
    _write(
        os.path.join(output_dir, "manifest.json"),
        json.dumps(manifest, indent=2) + "\n",
    )

    main_py = textwrap.dedent("""\
        import numpy as np

        try:
            from vivid_inference_core import resolve_torch_device
        except ImportError:
            resolve_torch_device = None


        def init_plugin(config: dict) -> dict:
            params = config.get("plugin_params", {})
            device = None
            if resolve_torch_device is not None:
                device = resolve_torch_device(config.get("backend"))
            return {
                "device": device,
                "strength": float(params.get("strength", 0.5)),
            }


        def process_frame(
            frame_hwc: np.ndarray, n: int, config: dict, state: dict
        ) -> np.ndarray:
            strength = max(0.0, min(1.0, float(state.get("strength", 0.5))))
            return (frame_hwc * (1.0 - 0.1 * strength)).astype(np.float32, copy=False)
    """)
    _write(os.path.join(output_dir, "main.py"), main_py)
    print(f"\nPlugin '{name}' scaffolded at {output_dir}")


def scaffold_model_pack(pack_id: str, output_dir: str) -> None:
    model_pack_py = textwrap.dedent(f"""\
        from vivid_inference_core import (
            CommunityModelLogicBase,
            EngineCapabilityContract,
            ModelArtifactSpec,
            register_model_pack,
            resolve_model_repo,
        )


        class {pack_id.capitalize()}Model(CommunityModelLogicBase):
            PYTORCH_NATIVE = True

            ENGINE_CAPABILITIES = EngineCapabilityContract(
                supported_backends=["pytorch", "pytorch-cuda"],
                required_pip_deps=["torch"],
                supports_pytorch=True,
                execution_mode="native-torch",
            )

            def process(self, clip, config, backend, model_path):
                # TODO: implement inference logic
                return clip


        def _resolve_artifact(model_name, config_data, current_dir):
            stem = model_name or "{pack_id}-v1"
            repo_root = resolve_model_repo("{pack_id}", current_dir=current_dir)
            model_repo_paths = [repo_root] if repo_root else []
            specs = [
                ModelArtifactSpec(
                    search_roots=[
                        f"{{current_dir}}/../models/community-{pack_id}",
                        *model_repo_paths,
                    ],
                    file_stems=[stem],
                )
            ]
            return CommunityModelLogicBase.resolve_from_specs(specs)


        register_model_pack(
            aliases=["{pack_id}"],
            model_logic={pack_id.capitalize()}Model,
            artifact_resolver=_resolve_artifact,
        )
    """)
    _write(os.path.join(output_dir, f"{pack_id}_pack.py"), model_pack_py)
    print(f"\nModel pack '{pack_id}' scaffolded at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a Vivid extension.")
    parser.add_argument("--kind", required=True, choices=["plugin", "model-pack"])
    parser.add_argument("--id", required=True, help="Extension identifier slug")
    parser.add_argument("--name", default="", help="Human-readable name (plugins)")
    parser.add_argument("--output-dir", required=True, help="Where to write files")
    args = parser.parse_args()

    if args.kind == "plugin":
        name = args.name or args.id.replace("-", " ").title()
        scaffold_plugin(args.id, name, args.output_dir)
    else:
        scaffold_model_pack(args.id, args.output_dir)


if __name__ == "__main__":
    main()
