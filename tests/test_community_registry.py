from __future__ import annotations

import sys
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The package imports config/runtime modules that expect vapoursynth at import time.
if "vapoursynth" not in sys.modules:
    fake_vs = types.SimpleNamespace(get_core=lambda: None, core=types.SimpleNamespace())
    sys.modules["vapoursynth"] = fake_vs

from vivid_inference_core.community_registry import (  # noqa: E402
    create_backend,
    register_backend,
    register_model_pack,
    resolve_model_artifact,
    resolve_model_logic,
)


class CommunityRegistryTests(unittest.TestCase):
    def test_register_model_pack_resolves_logic_and_artifact(self):
        def resolver(model_name, config_data, current_dir):
            _ = config_data
            _ = current_dir
            return f"/tmp/{model_name}.onnx"

        class SampleModel:
            pass

        register_model_pack(["unit-test-pack"], SampleModel, resolver)
        resolved_logic = resolve_model_logic("unit-test-pack")
        self.assertIs(resolved_logic, SampleModel)
        resolved_artifact = resolve_model_artifact("unit-test-pack", "demo", {}, "/")
        self.assertEqual(resolved_artifact, "/tmp/demo.onnx")

    def test_register_backend_resolves_factory(self):
        register_backend("unit-backend", lambda config, device_id=None: ("ok", config, device_id))
        result = create_backend("unit-backend", {"streams": 1}, device_id=3)
        self.assertEqual(result[0], "ok")
        self.assertEqual(result[2], 3)


if __name__ == "__main__":
    unittest.main()
