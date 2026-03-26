"""Runtime helpers for extension authors.

These wrap host-internal utilities (``pytorch_utils``, ``upstream_repos``)
behind stable SDK symbols so extension packs do not import host internals.
"""

from __future__ import annotations

import os
import sys


def resolve_torch_device(backend_name: str | None = None, device_index: int = 0):
    """Backend-aware device selection.

    Mapping:
      ``"pytorch-cuda"``  -> CUDA (raises if unavailable)
      ``"pytorch"``       -> auto (CUDA > MPS > CPU)
      ``"cpu"``           -> CPU
      ``None`` / other    -> auto
    """
    try:
        from inference_impl.pytorch_utils import resolve_torch_device as _host_resolve
        return _host_resolve(backend_name, device_index)
    except ImportError:
        pass

    import torch  # type: ignore

    if backend_name == "pytorch-cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Backend 'pytorch-cuda' requested but torch.cuda.is_available()=False."
            )
        return torch.device("cuda", device_index)
    if backend_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", device_index)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_repo(repo_name: str, *, current_dir: str = "") -> str | None:
    """Resolve an upstream model repo root directory.

    Tries the host's ``upstream_repos`` resolution first, then falls back
    to searching relative to *current_dir*.
    """
    try:
        from inference_impl.upstream_repos import resolve_upstream_repo
        result = resolve_upstream_repo(repo_name)
        if result:
            return result
    except ImportError:
        pass

    if current_dir:
        candidate = os.path.join(current_dir, "..", "models", "upstream-repos", repo_name)
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate):
            return candidate
    return None


def ensure_model_repo_on_path(repo_name: str) -> str | None:
    """Ensure *repo_name* is on ``sys.path`` for direct imports."""
    try:
        from inference_impl.upstream_repos import ensure_upstream_repo_on_path
        return ensure_upstream_repo_on_path(repo_name)
    except ImportError:
        pass

    repo_root = resolve_model_repo(repo_name)
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root
