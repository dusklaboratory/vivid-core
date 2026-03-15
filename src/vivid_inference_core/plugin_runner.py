"""
Public plugin runner contract.

This module is the single source of truth for custom plugin execution.
Vivid desktop delegates here; no duplicate implementation exists.
"""
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import vapoursynth as vs

from .custom_repo import install_repo_to_sys_path, resolve_repo_root

core = vs.core


def _try_load_bestsource_from_env():
    if hasattr(core, "bs"):
        return True
    plugin_paths = os.environ.get("VAPOURSYNTH_PLUGIN_PATH", "")
    if not plugin_paths:
        return False

    candidate_names = ("libbestsource.dylib", "libbestsource.so", "bestsource.dll")
    for root in plugin_paths.split(os.pathsep):
        root = root.strip()
        if not root:
            continue
        base = Path(root)
        for name in candidate_names:
            candidate = base / name
            if not candidate.exists():
                continue
            try:
                core.std.LoadPlugin(path=str(candidate))
                print(f"[PluginRunner] Loaded source plugin: {candidate}", file=sys.stderr)
                if hasattr(core, "bs"):
                    return True
            except Exception as e:
                print(f"[PluginRunner] Failed loading {candidate}: {e}", file=sys.stderr)
    return hasattr(core, "bs")


def _call_process_frame(process_frame, frame_hwc, frame_index, config, state):
    signature = inspect.signature(process_frame)
    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    argc = len(positional)
    if argc <= 1:
        return process_frame(frame_hwc)
    if argc == 2:
        return process_frame(frame_hwc, frame_index)
    if argc == 3:
        return process_frame(frame_hwc, frame_index, config)
    return process_frame(frame_hwc, frame_index, config, state)


def _build_frame_processor_clip(plugin_module, clip, config):
    clip_rgb = clip
    if clip.format and clip.format.color_family != vs.RGB:
        resize_errors = []
        for kwargs in (
            {"matrix_in_s": "709"},
            {"matrix_in_s": "470bg"},
            {"matrix_in_s": "170m"},
            {},
        ):
            try:
                clip_rgb = core.resize.Bicubic(clip, format=vs.RGBS, **kwargs)
                break
            except Exception as e:
                resize_errors.append(f"{kwargs or {'auto': True}} -> {e}")
        else:
            raise RuntimeError(
                "Failed to convert source clip to RGB for frame processing. "
                + " | ".join(resize_errors)
            )
    out_blank = core.std.BlankClip(clip_rgb)
    state = plugin_module.init_plugin(config) if hasattr(plugin_module, "init_plugin") else None

    def run_frame(n, f):
        src = f[0]
        dst = f[1].copy()
        chw = np.stack([np.asarray(src[p]) for p in range(src.format.num_planes)], axis=0)
        input_hwc = np.transpose(chw, (1, 2, 0)).astype(np.float32, copy=False)

        process_frame = getattr(plugin_module, "process_frame")
        result = _call_process_frame(process_frame, input_hwc, n, config, state)
        if result is None:
            result = input_hwc
        if not isinstance(result, np.ndarray):
            raise RuntimeError("process_frame must return a numpy.ndarray or None")
        if result.ndim != 3:
            raise RuntimeError("process_frame output must be HWC ndarray")
        if result.shape[0] != input_hwc.shape[0] or result.shape[1] != input_hwc.shape[1]:
            raise RuntimeError("process_frame output height/width must match input")
        if result.shape[2] != src.format.num_planes:
            raise RuntimeError("process_frame output channel count does not match clip format")

        out_chw = np.transpose(result.astype(np.float32, copy=False), (2, 0, 1))
        for p in range(src.format.num_planes):
            np.copyto(np.asarray(dst[p]), out_chw[p])
        return dst

    return core.std.ModifyFrame(out_blank, [clip_rgb, out_blank], run_frame)


def run_plugin(plugin_script_path, tmp_json_path=None):
    _run_plugin_impl(plugin_script_path, tmp_json_path)


def _run_plugin_impl(plugin_script_path, tmp_json_path=None):
    if tmp_json_path is None:
        tmp_json_path = os.environ.get("tmp", "")
    if not tmp_json_path or not os.path.exists(tmp_json_path):
        raise RuntimeError("No tmp JSON config found")

    with open(tmp_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    runtime_plugin_path = config.get("plugin_script")
    if runtime_plugin_path:
        plugin_script_path = runtime_plugin_path
    if not plugin_script_path or not os.path.exists(plugin_script_path):
        raise RuntimeError(f"Plugin script not found: {plugin_script_path}")

    video_path = config.get("file", config.get("inputVideo", ""))
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError(f"Input video not found: {video_path}")

    print(f"[PluginRunner] Loading plugin: {plugin_script_path}", file=sys.stderr)
    print(f"[PluginRunner] Input: {video_path}", file=sys.stderr)

    plugin_root = config.get("plugin_root")
    custom_repo_root = config.get("custom_repo_root")
    allow_user_override = bool(config.get("allow_user_repo_override", False))
    allowed_roots = list(config.get("allowed_repo_roots", []))
    if plugin_root:
        allowed_roots.append(plugin_root)

    required_checkpoints = list(config.get("required_checkpoints", []))
    for rel_path in required_checkpoints:
        checkpoint_path = Path(plugin_root or "").joinpath(rel_path).resolve()
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise RuntimeError(f"Required checkpoint missing: {rel_path}")
    if custom_repo_root and not allow_user_override:
        raise RuntimeError("custom_repo_root is not allowed by plugin repo policy")
    resolved_repo = resolve_repo_root(custom_repo_root, plugin_root, allowed_roots)
    install_repo_to_sys_path(resolved_repo)

    if sys.platform == "win32" and hasattr(core, "lsmas"):
        clip = core.lsmas.LWLibavSource(source=video_path, cache=0)
    else:
        if not hasattr(core, "bs"):
            _try_load_bestsource_from_env()
        if hasattr(core, "bs"):
            clip = core.bs.VideoSource(source=video_path)
        elif hasattr(core, "ffms2"):
            clip = core.ffms2.Source(source=video_path, cache=False)
        elif hasattr(core, "lsmas"):
            clip = core.lsmas.LWLibavSource(source=video_path, cache=0)
        else:
            raise RuntimeError(
                "No video source plugin found (bs/ffms2/lsmas). "
                f"VAPOURSYNTH_PLUGIN_PATH={os.environ.get('VAPOURSYNTH_PLUGIN_PATH', '')}"
            )

    print(f"[PluginRunner] Source: {clip.width}x{clip.height}, {clip.num_frames} frames", file=sys.stderr)
    os.environ["VIVID_PLUGIN_TMP"] = tmp_json_path

    spec = importlib.util.spec_from_file_location("plugin_module", plugin_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec: {plugin_script_path}")
    plugin_module = importlib.util.module_from_spec(spec)
    plugin_module.__dict__["clip"] = clip
    plugin_module.__dict__["config"] = config
    plugin_module.__dict__["plugin_params"] = config.get("plugin_params", {})
    plugin_module.__dict__["core"] = core
    plugin_module.__dict__["vs"] = vs
    spec.loader.exec_module(plugin_module)

    if hasattr(plugin_module, "process"):
        output_clip = plugin_module.process(clip, config)
    elif hasattr(plugin_module, "process_frame"):
        output_clip = _build_frame_processor_clip(plugin_module, clip, config)
    else:
        output_clip = vs.get_output(0) if vs.get_output(0) else clip

    if output_clip.height % 2 != 0:
        output_clip = core.std.AddBorders(output_clip, bottom=1)
    if output_clip.width % 2 != 0:
        output_clip = core.std.AddBorders(output_clip, right=1)

    if output_clip.format and output_clip.format.color_family == vs.RGB:
        final = core.resize.Bicubic(output_clip, format=vs.YUV420P8, matrix_s="709")
    else:
        final = core.resize.Bicubic(output_clip, format=vs.YUV420P8)

    total_frames = final.num_frames
    start_time = time.time()
    processed = [0]

    def log_progress(n, f):
        processed[0] = n + 1
        elapsed = time.time() - start_time
        fps_rate = processed[0] / elapsed if elapsed > 0 else 0
        remaining = (total_frames - processed[0]) / fps_rate if fps_rate > 0 else 0
        payload = {
            "fps": round(fps_rate, 2),
            "frame": processed[0],
            "total": total_frames,
            "progress": round(100 * processed[0] / total_frames if total_frames > 0 else 0, 1),
            "eta": int(round(remaining)),
        }
        print(json.dumps(payload), file=sys.stderr)
        return f

    final = core.std.ModifyFrame(final, final, log_progress)
    final.set_output()
    print(f"[PluginRunner] Plugin output ready: {final.width}x{final.height}", file=sys.stderr)


def _entrypoint():
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".py"):
        script = sys.argv[1]
        tmp = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        script = ""
        tmp = os.environ.get("tmp", "")
    try:
        run_plugin(script, tmp)
    except Exception as e:
        print(f"[PluginRunner] ERROR: Plugin execution failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def _should_autorun_entrypoint() -> bool:
    if __name__ == "__main__":
        return True
    # VSPipe can evaluate scripts with a non-__main__ module name while still
    # setting the tmp env var. Only auto-run in that mode when this file is
    # executed as the entry script (no package import context).
    return bool(os.environ.get("tmp")) and (__package__ is None or __package__ == "")


if _should_autorun_entrypoint():
    _entrypoint()


__all__ = ["run_plugin"]
