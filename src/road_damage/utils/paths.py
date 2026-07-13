from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path.cwd().resolve()


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = Path(root).resolve() if root is not None else project_root()
    return (base / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
