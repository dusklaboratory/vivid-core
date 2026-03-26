"""Authoring helpers for community model packs and extensions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class EngineCapabilityContract:
    """Declares what an engine supports so the host runtime can derive install
    plans, run preflight checks, and filter UI options."""

    supported_backends: list[str] = field(default_factory=list)
    required_pip_deps: list[str] = field(default_factory=list)
    required_upstream_repos: list[str] = field(default_factory=list)
    supports_onnx: bool = False
    supports_pytorch: bool = False
    execution_mode: str = "native-torch"
    fallback_backends: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelArtifactSpec:
    """Flexible artifact discovery spec for community model packs."""

    search_roots: list[str] = field(default_factory=list)
    file_stems: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=lambda: [".pth", ".onnx", ".safetensors", ".pkl"])


class CommunityModelLogicBase:
    """Base class for community-contributed model logic.

    Subclass this and implement ``process()`` to add a new model architecture
    to Vivid without modifying host internals.
    """

    PYTORCH_NATIVE: bool = False
    ENGINE_CAPABILITIES: EngineCapabilityContract | None = None

    def process(self, clip: Any, config: Any, backend: Any, model_path: str) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__}.process() must be implemented by the model pack author."
        )

    @staticmethod
    def resolve_from_specs(specs: list[ModelArtifactSpec]) -> str | None:
        """Walk artifact specs and return the first matching file path."""
        for spec in specs:
            for root in spec.search_roots:
                if not root or not os.path.isdir(root):
                    continue
                for stem in spec.file_stems:
                    for ext in spec.extensions:
                        candidate = os.path.join(root, stem + ext)
                        if os.path.isfile(candidate):
                            return candidate
                        candidate_lower = os.path.join(root, stem.lower() + ext)
                        if os.path.isfile(candidate_lower):
                            return candidate_lower
        return None
