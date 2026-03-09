from __future__ import annotations

from typing import Any, Callable

from .model_extensions import ModelArtifactResolverProtocol

_MODEL_ALIASES: dict[str, Any] = {}
_BACKEND_FACTORIES: dict[str, Callable[..., Any]] = {}
_ARTIFACT_RESOLVERS: dict[str, ModelArtifactResolverProtocol] = {}


def register_model_logic(aliases: list[str], model_logic: Any) -> None:
    for alias in aliases:
        key = alias.strip().lower()
        if key:
            _MODEL_ALIASES[key] = model_logic


def register_model_pack(
    aliases: list[str],
    model_logic: Any,
    artifact_resolver: ModelArtifactResolverProtocol | None = None,
) -> None:
    register_model_logic(aliases, model_logic)
    if artifact_resolver is not None:
        for alias in aliases:
            key = alias.strip().lower()
            if key:
                _ARTIFACT_RESOLVERS[key] = artifact_resolver


def resolve_model_logic(model_type: str) -> Any | None:
    return _MODEL_ALIASES.get((model_type or "").strip().lower())


def resolve_model_artifact(
    model_type: str,
    model_name: str | None,
    config_data: dict,
    current_dir: str,
) -> str | None:
    resolver = _ARTIFACT_RESOLVERS.get((model_type or "").strip().lower())
    if resolver is None:
        return None
    return resolver(model_name=model_name, config_data=config_data, current_dir=current_dir)


def register_backend(name: str, factory: Callable[..., Any]) -> None:
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("Backend name cannot be empty")
    _BACKEND_FACTORIES[key] = factory


def create_backend(backend_name: str, config: Any, device_id: int | None = None) -> Any | None:
    factory = _BACKEND_FACTORIES.get((backend_name or "").strip().lower())
    if factory is None:
        return None
    return factory(config=config, device_id=device_id)


def list_registered_model_aliases() -> list[str]:
    return sorted(_MODEL_ALIASES.keys())


def list_registered_backends() -> list[str]:
    return sorted(_BACKEND_FACTORIES.keys())


def list_registered_artifact_resolvers() -> list[str]:
    return sorted(_ARTIFACT_RESOLVERS.keys())
