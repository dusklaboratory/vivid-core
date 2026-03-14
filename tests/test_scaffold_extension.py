from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scaffold_extension.py"
TMP = ROOT / ".tmp-test-scaffold"


class ScaffoldExtensionTests(unittest.TestCase):
    def tearDown(self):
        if TMP.exists():
            shutil.rmtree(TMP)

    def test_scaffold_plugin(self):
        out_dir = TMP / "plugin"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--kind",
                "plugin",
                "--id",
                "unit-plugin",
                "--name",
                "Unit Plugin",
                "--output-dir",
                str(out_dir),
            ],
            check=True,
        )
        self.assertTrue((out_dir / "main.py").exists())
        self.assertTrue((out_dir / "manifest.json").exists())

    def test_scaffold_model_pack(self):
        out_dir = TMP / "pack"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--kind",
                "model-pack",
                "--id",
                "unit-pack",
                "--output-dir",
                str(out_dir),
            ],
            check=True,
        )
        self.assertTrue((out_dir / "unit_pack.py").exists())
        self.assertTrue((out_dir / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
