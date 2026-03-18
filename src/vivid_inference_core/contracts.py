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

    def prepare(self, config: Any) -> None:
        ...

    def validate(self, config: Any) -> None:
        ...

    def process(self, clip: Any, config: Any, backend: Any, model_path: str) -> Any:
        ...

    def finalize(self, config: Any) -> None:
        ...


class EffectLogicProtocol(Protocol):
    def prepare(self, config: Any) -> None:
        ...

    def validate(self, config: Any) -> None:
        ...

    def process(self, clip: Any, config: Any, backend: Any, model_path: str) -> Any:
        ...

    def finalize(self, config: Any) -> None:
        ...


@dataclass(frozen=True)
class ProgressEventContract:
    fps: float
    frame: int
    total: int
    progress: float
    eta: int


class RuntimeCallbacksProtocol(Protocol):
    def emit_log(self, message: str) -> None:
        ...

    def emit_progress(self, event: ProgressEventContract) -> None:
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
class EngineCapabilityContract:
    """
    Declarative backend/dependency metadata for community engine packs.
    Community contributors can declare this alongside their model logic
    so the host catalog can ingest the metadata without private-code patching.
    """
    supported_backends: list[str] = field(default_factory=list)
    required_pip_deps: list[str] = field(default_factory=list)
    required_upstream_repos: list[str] = field(default_factory=list)
    supports_onnx: bool = False
    supports_pytorch: bool = False
    execution_mode: str = "vsmlrt"
    fallback_backends: list[str] = field(default_factory=list)


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
    engine_capabilities: EngineCapabilityContract | None = None
