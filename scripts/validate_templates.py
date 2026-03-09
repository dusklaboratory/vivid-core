from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "templates" / "custom-python-plugin-pytorch" / "manifest.json"
    schema_path = root / "src" / "vivid_inference_core" / "plugin_manifest.schema.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    required = set(schema.get("required", []))
    missing = sorted(key for key in required if key not in manifest)
    if missing:
        raise SystemExit(f"Template manifest missing required keys: {', '.join(missing)}")

    print("template validation passed")


if __name__ == "__main__":
    main()
