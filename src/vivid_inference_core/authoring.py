from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def normalize_backend_name(backend_name: str | None) -> str:
    return (backend_name or "").strip().lower().replace("_", "-")


def resolve_torch_device(
    backend_name: str | None = None,
    *,
    allow_mps: bool = True,
):
    """
    Resolve a practical PyTorch device for plugin/model-pack authors.

    Strategy mirrors Vivid's runtime behavior:
    - Prefer CUDA when backend indicates GPU CUDA usage and CUDA is available.
    - Prefer MPS when backend indicates CoreML/Apple and MPS is available.
    - Otherwise fall back to best available accelerator, then CPU.
    """
    import torch

    backend = normalize_backend_name(backend_name)
    if backend.startswith("pytorch-cuda") or backend in {"trt", "trt-rtx", "ort-cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")

    if backend in {"coreml", "ort-coreml"} and allow_mps:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_repo_root(current_dir: str | None = None) -> str | None:
    """
    Locate a host-provided model repository root.

    Resolution order:
    1) VIVID_MODEL_REPO_ROOT env var
    2) common host locations
    3) walk upward from current_dir / cwd looking for repo layouts
    """
    env_override = os.environ.get("VIVID_MODEL_REPO_ROOT", "").strip()
    candidates = [
        env_override,
        "/usr/local/lib/vapoursynth/models/extensions",
    ]

    if current_dir:
        base = Path(current_dir).expanduser().resolve()
        candidates.extend(
            [
                str(base / "../models/extensions"),
                str(base / "../../models/extensions"),
                str(base / "../../../models/extensions"),
                str(base / "../model-repos"),
                str(base / "../../model-repos"),
            ]
        )

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)

    probe = Path(current_dir or os.getcwd()).expanduser().resolve()
    for _ in range(10):
        nested = probe / "models" / "extensions"
        flat = probe / "model-repos"
        if nested.is_dir():
            return str(nested.resolve())
        if flat.is_dir():
            return str(flat.resolve())
        if probe.parent == probe:
            break
        probe = probe.parent
    return None


def resolve_model_repo(repo_name: str, *, current_dir: str | None = None) -> str | None:
    root = get_model_repo_root(current_dir=current_dir)
    if not root:
        return None
    candidate = Path(root) / repo_name
    if candidate.is_dir():
        return str(candidate.resolve())
    return None


def ensure_model_repo_on_path(
    repo_name: str,
    *,
    extra_subdirs: Iterable[str] = (),
    current_dir: str | None = None,
) -> str | None:
    root = get_model_repo_root(current_dir=current_dir)
    repo_path = resolve_model_repo(repo_name, current_dir=current_dir)
    if not root or not repo_path:
        return None

    entries = [root, repo_path, *[str(Path(repo_path) / subdir) for subdir in extra_subdirs]]
    for entry in entries:
        if os.path.isdir(entry) and entry not in sys.path:
            sys.path.insert(0, entry)
    return repo_path
