"""Community model pack registry.

Extensions call ``register_model_pack`` to make their model logic discoverable
by the host runtime. The host queries ``resolve_model_logic``,
``resolve_model_artifact``, and ``create_backend`` at delegation points.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Type

_model_packs: dict[str, dict] = {}
_backend_factories: dict[str, Callable] = {}


def register_model_pack(
    aliases: list[str],
    model_logic: Type,
    artifact_resolver: Callable[..., str | None] | None = None,
    backend_factory: Callable[..., Any] | None = None,
) -> None:
    """Register a community model pack under one or more alias slugs.

    Args:
        aliases: List of engine slug strings (e.g. ``["myarch"]``).
        model_logic: A class conforming to ``ModelLogicProtocol``.
        artifact_resolver: Optional callable
            ``(model_name, config_data, current_dir) -> path | None``.
        backend_factory: Optional callable
            ``(name, config, device_id) -> backend | None``.
    """
    entry = {
        "model_logic": model_logic,
        "artifact_resolver": artifact_resolver,
    }
    for alias in aliases:
        _model_packs[alias.lower()] = entry
        print(f"[CommunityRegistry] Registered model pack: {alias}", file=sys.stderr)

    if backend_factory is not None:
        for alias in aliases:
            _backend_factories[alias.lower()] = backend_factory


def resolve_model_logic(model_type: str) -> Type | None:
    """Return the model logic class for *model_type*, or ``None``."""
    slug = model_type.lower().removeprefix("community:")
    entry = _model_packs.get(slug)
    if entry is not None:
        return entry["model_logic"]
    return None


def resolve_model_artifact(
    *,
    model_type: str,
    model_name: str | None = None,
    config_data: dict | None = None,
    current_dir: str = "",
) -> str | None:
    """Resolve a model artifact path via the registered resolver."""
    slug = model_type.lower().removeprefix("community:")
    entry = _model_packs.get(slug)
    if entry is None or entry.get("artifact_resolver") is None:
        return None
    resolver = entry["artifact_resolver"]
    return resolver(model_name, config_data or {}, current_dir)


def create_backend(name: str, config: Any, device_id: int | None = None) -> Any | None:
    """Create a community backend, or ``None`` if no factory is registered."""
    slug = name.lower().removeprefix("community:")
    factory = _backend_factories.get(slug)
    if factory is not None:
        return factory(name, config, device_id=device_id)
    return None
