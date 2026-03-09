from __future__ import annotations

from pathlib import Path


FORBIDDEN_TOKENS = (
    "cloudExecutor",
    "cloudSubscriptionService",
    "licensing",
    "auth",
    "modal",
    "stripe",
    "convex",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "src"
    violations: list[str] = []
    for py_file in root.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in content:
                violations.append(f"{py_file}: contains forbidden token '{token}'")

    if violations:
        raise SystemExit("\n".join(violations))
    print("private-boundary audit passed")


if __name__ == "__main__":
    main()
