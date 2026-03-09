from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class BackendFactoryProtocol(Protocol):
    @staticmethod
    def create_backend(
        backend_name: str,
        config: Any,
        device_id: int | None = None,
    ) -> Any:
        ...


class ModelLogicProtocol(Protocol):
    PYTORCH_NATIVE: bool

    def process(self, clip: Any, config: Any, backend: Any, model_path: str) -> Any:
        ...


@dataclass(frozen=True)
class PluginParameterContract:
    id: str
    param_type: str
    label: str
    min: float | None = None
    max: float | None = None
    default: Any | None = None
    options: list[str] = field(default_factory=list)
    required: bool = False


@dataclass(frozen=True)
class PluginRepoPolicyContract:
    repo_root: str | None = None
    entry_module: str | None = None
    required_checkpoints: list[str] = field(default_factory=list)
    allow_user_override: bool = False
    allowed_override_roots: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PluginManifestContract:
    schema_version: int
    id: str
    name: str
    kind: str
    entry_script: str
    backends: list[str] = field(default_factory=list)
    parameters: list[PluginParameterContract] = field(default_factory=list)
    repo: PluginRepoPolicyContract | None = None
