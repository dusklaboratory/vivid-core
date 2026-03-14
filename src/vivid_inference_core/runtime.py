from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import cpu_count
from typing import Any, Callable

import vapoursynth as vs

from .community_registry import resolve_model_artifact
from .config import InferenceConfig
from .contracts import ProgressEventContract

core = vs.core


def _get_hook(hooks: dict[str, Any] | None, name: str, default: Any = None) -> Any:
    if not hooks:
        return default
    return hooks.get(name, default)


def _get_script_dir() -> str:
    argv0 = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    return os.path.dirname(argv0) if argv0 else os.getcwd()


def _find_model_file_default(model_name: str, _model_type: str, _script_dir: str) -> str:
    return model_name


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _apply_resource_limits() -> None:
    vram_frac = os.environ.get("VIVID_VRAM_FRACTION")
    if vram_frac:
        try:
            import torch

            frac = float(vram_frac)
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(frac)
                print(f"[ResourceLimits] CUDA memory fraction set to {frac}", file=sys.stderr)
        except Exception as exc:
            print(f"[ResourceLimits] Could not apply VRAM limit: {exc}", file=sys.stderr)

    ram_mb = os.environ.get("VIVID_RAM_LIMIT_MB")
    if ram_mb:
        try:
            import resource

            limit_bytes = max(int(ram_mb) * 1024 * 1024, 256 * 1024 * 1024)
            if hasattr(resource, "RLIMIT_AS"):
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                print(f"[ResourceLimits] RLIMIT_AS set to {limit_bytes // (1024*1024)} MB", file=sys.stderr)
        except Exception as exc:
            print(f"[ResourceLimits] Could not apply RAM limit: {exc}", file=sys.stderr)


class InferencePipeline:
    def __init__(
        self,
        model_logic_class_or_instance: Any,
        backend_factory_class: Any,
        model_type: str = "esrgan",
        runtime_hooks: dict[str, Any] | None = None,
    ) -> None:
        self.config = InferenceConfig()
        self.model_logic = (
            model_logic_class_or_instance
            if not isinstance(model_logic_class_or_instance, type)
            else model_logic_class_or_instance()
        )
        self.backend_factory = backend_factory_class
        self.model_type = model_type
        self.runtime_hooks = runtime_hooks or {}
        self.tracer = None

        self._get_script_dir: Callable[[], str] = _get_hook(self.runtime_hooks, "get_script_dir", _get_script_dir)
        self._find_model_file: Callable[[str, str, str], str] = _get_hook(
            self.runtime_hooks, "find_model_file", _find_model_file_default
        )
        self._load_vs_plugins: Callable[[str], None] = _get_hook(self.runtime_hooks, "load_vs_plugins", _noop)
        self._apply_fallback_policy: Callable[[Any, Any, Any], None] = _get_hook(
            self.runtime_hooks, "apply_fallback_policy", _noop
        )
        self._create_tracer: Callable[[str, bool], Any] = _get_hook(self.runtime_hooks, "create_tracer", lambda *_: None)
        self._log_header: Callable[[str, str], None] = _get_hook(self.runtime_hooks, "log_header", _noop)
        self._log_system_info: Callable[[str], None] = _get_hook(self.runtime_hooks, "log_system_info", _noop)
        self._log_config: Callable[[str, dict, str], None] = _get_hook(self.runtime_hooks, "log_config", _noop)
        self._prepare_environment: Callable[[], None] = _get_hook(
            self.runtime_hooks, "prepare_environment", _noop
        )
        self._emit_progress: Callable[[ProgressEventContract], None] = _get_hook(
            self.runtime_hooks, "emit_progress", _noop
        )

    def setup(self) -> None:
        self._prepare_environment()
        trace_enabled = bool(self.config.data.get("decisionTraceEnabled", True))
        trace_path = self.config.data.get("decisionTracePath") or f"{self.config.tmp_file}.decision_trace.jsonl"
        self.tracer = self._create_tracer(trace_path, trace_enabled)
        if self.tracer and hasattr(self.tracer, "emit"):
            self.tracer.emit(
                "pipeline_setup",
                {
                    "modelType": self.model_type,
                    "backend": self.config.backend,
                    "tmpFile": self.config.tmp_file,
                },
            )

        _apply_resource_limits()

        script_dir = self._get_script_dir()
        self._load_vs_plugins(script_dir)
        self._apply_fallback_policy(self.config, core, self.tracer)

        cloud_threads = int(self.config.data.get("cloudThreads", 0))
        if self.model_type == "rife" and self.config.backend == "coreml":
            core.max_cache_size = 1
            core.num_threads = 1
        elif self.model_type == "rife":
            core.max_cache_size = 4000
            core.num_threads = cloud_threads or max(1, cpu_count() // 2)
        else:
            core.max_cache_size = 8000
            core.num_threads = cloud_threads or max(2, cpu_count() // 2)

        label = f"{self.model_type.upper()}-PublicCore"
        self._log_header(label, "1.0")
        self._log_system_info(label)
        self._log_config(label, self.config.data, self.config.tmp_file)

    def load_source(self):
        video_path = self.config.video_path
        if self.config.ossystem == "Windows":
            return core.lsmas.LWLibavSource(source=video_path, cache=0)
        try:
            return core.bs.VideoSource(source=video_path)
        except Exception:
            return core.ffms2.Source(source=video_path, cache=False)

    def _resolve_model_path(self) -> str | None:
        script_dir = self._get_script_dir()
        try:
            resolved = resolve_model_artifact(
                model_type=self.model_type,
                model_name=self.config.model_input,
                config_data=self.config.data,
                current_dir=script_dir,
            )
            if resolved:
                return resolved
        except Exception as exc:
            print(f"[Init] Public artifact resolution failed: {exc}", file=sys.stderr)
        return self._find_model_file(self.config.model_input, self.model_type, script_dir)

    def attach_progress(self, clip):
        total_frames = len(clip)
        start_time = time.time()
        last_emit: list[float] = [0.0]

        def log_progress(n, f):
            current = n + 1
            elapsed = time.time() - start_time
            fps_rate = current / elapsed if elapsed > 0 else 0.0
            eta = int(round((total_frames - current) / fps_rate)) if fps_rate > 0 else 0
            progress = round(100.0 * current / total_frames if total_frames > 0 else 0.0, 1)
            now = time.time()
            if now - last_emit[0] >= 1.0 or n == total_frames - 1:
                payload = {
                    "fps": round(fps_rate, 2),
                    "frame": current,
                    "total": total_frames,
                    "progress": progress,
                    "eta": eta,
                }
                print(json.dumps(payload), file=sys.stderr)
                self._emit_progress(
                    ProgressEventContract(
                        fps=payload["fps"],
                        frame=payload["frame"],
                        total=payload["total"],
                        progress=payload["progress"],
                        eta=payload["eta"],
                    )
                )
                last_emit[0] = now
            return f

        return core.std.ModifyFrame(clip, clip, log_progress)

    def run(self, backend_name: str = "coreml") -> None:
        try:
            self.setup()
            if hasattr(self.model_logic, "prepare"):
                self.model_logic.prepare(self.config)
            if hasattr(self.model_logic, "validate"):
                self.model_logic.validate(self.config)

            clip = self.load_source()
            model_path = self._resolve_model_path()

            is_pytorch_backend = str(backend_name).startswith("pytorch")
            if getattr(self.model_logic, "PYTORCH_NATIVE", False) or is_pytorch_backend:
                backend = None
            else:
                backend = self.backend_factory.create_backend(backend_name, self.config)

            output_clip = self.model_logic.process(clip, self.config, backend, model_path)

            if output_clip.height % 2 != 0:
                output_clip = core.std.AddBorders(output_clip, bottom=1)
            if output_clip.width % 2 != 0:
                output_clip = core.std.AddBorders(output_clip, right=1)
            if output_clip.format and output_clip.format.color_family == vs.RGB:
                final_clip = core.resize.Bicubic(output_clip, format=vs.YUV420P8, matrix_s="709")
            else:
                final_clip = core.resize.Bicubic(output_clip, format=vs.YUV420P8)

            final_clip = self.attach_progress(final_clip)
            final_clip.set_output()
        except Exception as exc:
            if self.tracer and hasattr(self.tracer, "emit"):
                self.tracer.emit("pipeline_error", {"message": str(exc)})
            raise
        finally:
            if hasattr(self.model_logic, "finalize"):
                try:
                    self.model_logic.finalize(self.config)
                except Exception:
                    pass
            if self.tracer and hasattr(self.tracer, "close"):
                self.tracer.close()


def run_pipeline(
    model_logic_class: Any,
    backend_factory_class: Any,
    model_type: str = "esrgan",
    backend_name: str | None = None,
    runtime_hooks: dict[str, Any] | None = None,
) -> None:
    pipeline = InferencePipeline(
        model_logic_class_or_instance=model_logic_class,
        backend_factory_class=backend_factory_class,
        model_type=model_type,
        runtime_hooks=runtime_hooks,
    )
    pipeline.run(backend_name or pipeline.config.backend)
