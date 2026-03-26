"""Vivid Inference Core -- open extension SDK for the Vivid processing pipeline.

This package provides the stable public API for community model packs,
custom plugin scripts, and backend extensions.
"""

from __future__ import annotations

CONTRACT_VERSION: int = 1

from .authoring import (
    CommunityModelLogicBase,
    EngineCapabilityContract,
    ModelArtifactSpec,
)
from .community_registry import (
    create_backend,
    register_model_pack,
    resolve_model_artifact,
    resolve_model_logic,
)
from .helpers import (
    ensure_model_repo_on_path,
    resolve_model_repo,
    resolve_torch_device,
)


def run_pipeline(
    model_logic_class,
    backend_factory_class,
    model_type: str,
    backend_name: str | None = None,
    runtime_hooks: dict | None = None,
):
    """Run the inference pipeline, optionally delegating to host hooks.

    This is the public entry point that the host's ``public_core_adapter``
    calls when ``VIVID_INFERENCE_CORE_MODE`` is ``public`` or ``public-strict``.

    When invoked from the host, *runtime_hooks* provides host-internal
    functions (environment setup, VS plugin loading, model search, etc.)
    that the pipeline should call at the appropriate stage.  When invoked
    standalone (e.g. tests), *runtime_hooks* may be ``None`` and the
    pipeline falls back to minimal defaults.
    """
    try:
        from inference_impl.core import InferencePipeline, validate_backend, run_preflight
    except ImportError:
        raise RuntimeError(
            "vivid_inference_core.run_pipeline requires the host inference_impl "
            "package on sys.path. This function is designed to be called from "
            "the Vivid desktop runtime, not standalone."
        )

    pipeline = InferencePipeline(model_logic_class, backend_factory_class, model_type)

    if backend_name is None:
        backend_name = pipeline.config.backend

    backend_name = validate_backend(model_type, backend_name)
    run_preflight(model_type, backend_name)
    pipeline.run(backend_name)


__all__ = [
    "CONTRACT_VERSION",
    "CommunityModelLogicBase",
    "EngineCapabilityContract",
    "ModelArtifactSpec",
    "create_backend",
    "ensure_model_repo_on_path",
    "register_model_pack",
    "resolve_model_artifact",
    "resolve_model_logic",
    "resolve_model_repo",
    "resolve_torch_device",
    "run_pipeline",
]
