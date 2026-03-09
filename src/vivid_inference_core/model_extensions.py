from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .contracts import ModelLogicProtocol

SUPPORTED_ARTIFACT_EXTENSIONS = (".onnx", ".pth", ".pt", ".pkl", ".param", ".safetensors")


@dataclass(frozen=True)
class ModelArtifactSpec:
    """
    Declarative artifact spec for community model packs.

    - `search_roots`: directories to recursively search.
    - `file_stems`: preferred model name stems (without extension).
    - `extensions`: allowed artifact formats.
    """

    search_roots: list[str]
    file_stems: list[str]
    extensions: tuple[str, ...] = SUPPORTED_ARTIFACT_EXTENSIONS
    required: bool = True


class ModelArtifactResolverProtocol(Protocol):
    def __call__(
        self,
        model_name: str | None,
        config_data: dict,
        current_dir: str,
    ) -> str | None:
        ...


class CommunityModelLogicBase(ModelLogicProtocol):
    """
    Base class for community-contributed model architectures.

    Community packs can subclass this and override:
    - `process(...)` for model execution
    - `resolve_artifact(...)` if custom artifact discovery is needed
    """

    PYTORCH_NATIVE = False

    def process(self, clip, config, backend, model_path):
        raise NotImplementedError

    def resolve_artifact(self, model_name: str | None, config_data: dict, current_dir: str) -> str | None:
        _ = model_name
        _ = config_data
        _ = current_dir
        return None

    @staticmethod
    def resolve_from_specs(
        specs: Sequence[ModelArtifactSpec],
        fallback_roots: Iterable[str] = (),
    ) -> str | None:
        for spec in specs:
            for root_raw in [*spec.search_roots, *fallback_roots]:
                root = Path(root_raw).expanduser()
                if not root.exists() or not root.is_dir():
                    continue
                for stem in spec.file_stems:
                    for ext in spec.extensions:
                        candidate = root / f"{stem}{ext}"
                        if candidate.exists() and candidate.is_file():
                            return str(candidate.resolve())
                for found in root.rglob("*"):
                    if found.is_file() and found.suffix.lower() in spec.extensions:
                        if found.stem in spec.file_stems:
                            return str(found.resolve())
        return None

