"""Plugin runner entry point.

The host's thin shim at ``src/inference/inference_impl/plugin_runner.py``
delegates to ``_entrypoint()`` here.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import vapoursynth as vs


def _entrypoint() -> None:
    """Main entry point invoked by the host plugin runner shim."""
    core = vs.core

    tmp_path = os.environ.get("tmp", "")
    if not tmp_path or not os.path.isfile(tmp_path):
        raise RuntimeError("[PluginRunner] No config file at $tmp")

    with open(tmp_path, "r") as f:
        config_data = json.load(f)

    manifest_path = config_data.get("plugin_manifest")
    if not manifest_path or not os.path.isfile(manifest_path):
        raise RuntimeError(f"[PluginRunner] Plugin manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    entry_script = manifest.get("entry_script", "main.py")
    plugin_dir = os.path.dirname(os.path.abspath(manifest_path))
    script_path = os.path.join(plugin_dir, entry_script)

    if not os.path.isfile(script_path):
        raise RuntimeError(f"[PluginRunner] Entry script not found: {script_path}")

    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    import importlib.util
    spec = importlib.util.spec_from_file_location("_vivid_plugin_entry", script_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"[PluginRunner] Cannot load entry script: {script_path}")

    plugin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_module)

    video_path = config_data.get("file", "")
    if not video_path or not os.path.isfile(video_path):
        raise RuntimeError(f"[PluginRunner] Video file not found: {video_path}")

    clip = core.bs.VideoSource(video_path)
    clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

    # Frame-processor mode: init_plugin + process_frame
    init_fn = getattr(plugin_module, "init_plugin", None)
    frame_fn = getattr(plugin_module, "process_frame", None)

    if callable(frame_fn):
        state = init_fn(config_data) if callable(init_fn) else {}
        blank = core.std.BlankClip(clip)

        def execute(n, f):
            planes = [np.asarray(f[0][p]) for p in range(f[0].format.num_planes)]
            frame_hwc = np.stack(planes, axis=2)
            result_hwc = frame_fn(frame_hwc, n, config_data, state)
            out = f[0].copy()
            for p in range(out.format.num_planes):
                np.copyto(np.asarray(out[p]), result_hwc[:, :, p])
            return out

        clip = core.std.ModifyFrame(blank, [clip], execute)
    else:
        # Graph mode: process(clip, config)
        process_fn = getattr(plugin_module, "process", None)
        if not callable(process_fn):
            raise RuntimeError(
                "[PluginRunner] Plugin must define either process_frame() or process()."
            )
        clip = process_fn(clip, config_data)

    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    clip.set_output()
