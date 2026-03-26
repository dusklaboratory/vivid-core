"""Vivid Inference Core — public runtime shim.

When invoked from the Vivid desktop host (``runtime_hooks`` is provided),
this module delegates *entirely* to ``inference_impl.core.InferencePipeline``
so that community packs benefit from the host's full pipeline: BestSource
fallback chain, VFR correction, dedup, resource limits, decision tracing, and
backend validation.

When invoked standalone (e.g. tests or scripts without the host), a minimal
self-contained pipeline runs instead.  This fallback path is intentionally
limited — it is not a replacement for the full host runtime.
"""

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


# ---------------------------------------------------------------------------
# Internal helpers (standalone fallback only)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public-facing InferencePipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Host-delegating inference pipeline for community packs.

    When the Vivid host is present (``runtime_hooks`` supplied), construction
    immediately wraps the host ``inference_impl.core.InferencePipeline`` and
    all public methods delegate to it.  Standalone usage falls back to a
    minimal implementation suitable for testing.
    """

    def __init__(
        self,
        model_logic_class_or_instance: Any,
        backend_factory_class: Any,
        model_type: str = "esrgan",
        runtime_hooks: dict[str, Any] | None = None,
    ) -> None:
        self._runtime_hooks = runtime_hooks or {}
        self._host_pipeline: Any = None

        if runtime_hooks:
            # Delegate to the host pipeline.  The host exports its class via
            # the same hook dict that was passed here by _build_public_runtime_hooks().
            try:
                from inference_impl.core import InferencePipeline as HostPipeline
                self._host_pipeline = HostPipeline(
                    model_logic_class_or_instance,
                    backend_factory_class,
                    model_type,
                )
                return
            except ImportError:
                pass

        # Standalone fallback path.
        self.config = InferenceConfig()
        self.model_logic = (
            model_logic_class_or_instance
            if not isinstance(model_logic_class_or_instance, type)
            else model_logic_class_or_instance()
        )
        self.backend_factory = backend_factory_class
        self.model_type = model_type
        self.tracer = None

        self._get_script_dir: Callable[[], str] = _get_hook(self._runtime_hooks, "get_script_dir", _get_script_dir)
        self._find_model_file: Callable[[str, str, str], str] = _get_hook(
            self._runtime_hooks, "find_model_file", _find_model_file_default
        )
        self._load_vs_plugins: Callable[[str], None] = _get_hook(self._runtime_hooks, "load_vs_plugins", _noop)
        self._apply_fallback_policy: Callable[[Any, Any, Any], None] = _get_hook(
            self._runtime_hooks, "apply_fallback_policy", _noop
        )
        self._create_tracer: Callable[[str, bool], Any] = _get_hook(self._runtime_hooks, "create_tracer", lambda *_: None)
        self._log_header: Callable[[str, str], None] = _get_hook(self._runtime_hooks, "log_header", _noop)
        self._log_system_info: Callable[[str], None] = _get_hook(self._runtime_hooks, "log_system_info", _noop)
        self._log_config: Callable[[str, dict, str], None] = _get_hook(self._runtime_hooks, "log_config", _noop)
        self._prepare_environment: Callable[[], None] = _get_hook(self._runtime_hooks, "prepare_environment", _noop)
        self._emit_progress: Callable[[ProgressEventContract], None] = _get_hook(
            self._runtime_hooks, "emit_progress", _noop
        )

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def _delegate(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call the same method on the host pipeline if available."""
        if self._host_pipeline is not None:
            return getattr(self._host_pipeline, method)(*args, **kwargs)
        return getattr(self, f"_standalone_{method}")(*args, **kwargs)

    # ------------------------------------------------------------------
    # Public interface — always routes through the host when present
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self._host_pipeline is not None:
            self._host_pipeline.setup()
            return
        self._standalone_setup()

    def load_source(self):
        if self._host_pipeline is not None:
            return self._host_pipeline.load_source()
        return self._standalone_load_source()

    def run(self, backend_name: str = "coreml") -> None:
        if self._host_pipeline is not None:
            self._host_pipeline.run(backend_name)
            return
        self._standalone_run(backend_name)

    # ------------------------------------------------------------------
    # Standalone fallback implementations
    # ------------------------------------------------------------------

    def _standalone_setup(self) -> None:
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

        script_dir = self._get_script_dir()
        self._load_vs_plugins(script_dir)

        # After VS plugin dirs are registered (and optional PyTorch-first warmup on Windows),
        # resource limits may import torch — matches src/inference/inference_impl/core.py order.
        _apply_resource_limits()
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

    def _standalone_load_source(self):
        video_path = self.config.video_path
        if self.config.ossystem == "Windows":
            return core.lsmas.LWLibavSource(source=video_path, cache=0)
        try:
            return core.bs.VideoSource(source=video_path)
        except Exception:
            return core.ffms2.Source(source=video_path, cache=False)

    def _standalone_resolve_model_path(self) -> str | None:
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

    def _standalone_attach_progress(self, clip):
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

    def _standalone_run(self, backend_name: str = "coreml") -> None:
        try:
            self._standalone_setup()
            if hasattr(self.model_logic, "prepare"):
                self.model_logic.prepare(self.config)
            if hasattr(self.model_logic, "validate"):
                self.model_logic.validate(self.config)

            clip = self._standalone_load_source()
            model_path = self._standalone_resolve_model_path()

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

            final_clip = self._standalone_attach_progress(final_clip)
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    model_logic_class: Any,
    backend_factory_class: Any,
    model_type: str = "esrgan",
    backend_name: str | None = None,
    runtime_hooks: dict[str, Any] | None = None,
) -> None:
    """Run the inference pipeline.

    When called by the host with *runtime_hooks*, the host's full
    ``InferencePipeline`` (including validation, dedup, VFR, etc.) is used.
    When called standalone (no hooks), a minimal fallback executes.
    """
    pipeline = InferencePipeline(
        model_logic_class_or_instance=model_logic_class,
        backend_factory_class=backend_factory_class,
        model_type=model_type,
        runtime_hooks=runtime_hooks,
    )
    effective_backend = backend_name or (
        pipeline._host_pipeline.config.backend
        if pipeline._host_pipeline is not None
        else pipeline.config.backend
    )
    pipeline.run(effective_backend)
