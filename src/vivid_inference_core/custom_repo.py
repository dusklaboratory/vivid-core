import os
import sys
from pathlib import Path
from typing import Iterable


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_repo_root(
    override_root: str | None,
    plugin_root: str | None,
    allowed_roots: Iterable[str],
) -> Path | None:
    if not override_root:
        return None

    candidate = _canonical(Path(override_root))
    if not candidate.exists() or not candidate.is_dir():
        raise RuntimeError(f"Custom repo root does not exist or is not a directory: {candidate}")

    allowed = [_canonical(Path(raw_root)) for raw_root in allowed_roots if raw_root]
    if plugin_root:
        allowed.append(_canonical(Path(plugin_root)))

    if allowed and not any(_is_within(candidate, root) for root in allowed):
        joined = ", ".join(str(root) for root in allowed)
        raise RuntimeError(
            f"Custom repo root is outside allowed paths: {candidate}. Allowed roots: {joined}"
        )

    return candidate


def install_repo_to_sys_path(repo_root: Path | None) -> None:
    if repo_root is None:
        return
    resolved = str(repo_root)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        os.environ["VIVID_CUSTOM_REPO_ROOT"] = resolved
