from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vivid_inference_core.authoring import (  # noqa: E402
    ensure_model_repo_on_path,
    get_model_repo_root,
    normalize_backend_name,
    resolve_model_repo,
)


TMP = ROOT / ".tmp-test-authoring"


class AuthoringHelpersTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("VIVID_MODEL_REPO_ROOT", None)
        if TMP.exists():
            shutil.rmtree(TMP)
        (TMP / "model-repos" / "demo-repo").mkdir(parents=True)

    def tearDown(self):
        os.environ.pop("VIVID_MODEL_REPO_ROOT", None)
        if TMP.exists():
            shutil.rmtree(TMP)

    def test_normalize_backend_name(self):
        self.assertEqual(normalize_backend_name("PyTorch_CUDA"), "pytorch-cuda")
        self.assertEqual(normalize_backend_name(""), "")
        self.assertEqual(normalize_backend_name(None), "")

    def test_model_repo_discovery_from_current_dir(self):
        current_dir = str(TMP / "src" / "inference" / "inference_impl")
        (TMP / "src" / "inference" / "inference_impl").mkdir(parents=True, exist_ok=True)
        root = get_model_repo_root(current_dir=current_dir)
        self.assertTrue(root.endswith("model-repos"))

        repo = resolve_model_repo("demo-repo", current_dir=current_dir)
        self.assertTrue(repo.endswith("demo-repo"))

    def test_ensure_model_repo_on_path(self):
        current_dir = str(TMP / "src" / "inference" / "inference_impl")
        (TMP / "src" / "inference" / "inference_impl").mkdir(parents=True, exist_ok=True)
        repo = ensure_model_repo_on_path("demo-repo", current_dir=current_dir)
        self.assertIsNotNone(repo)
        self.assertIn(repo, sys.path)


if __name__ == "__main__":
    unittest.main()
