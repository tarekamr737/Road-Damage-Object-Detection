from __future__ import annotations

import os
from pathlib import Path


def configure_ultralytics_env(root: Path | None = None) -> Path:
    """Keep Ultralytics settings/cache writes inside the local project."""
    configured = os.getenv("YOLO_CONFIG_DIR")
    base_dir = Path(configured).expanduser() if configured else (root or Path.cwd())
    if not configured:
        base_dir = base_dir / ".ultralytics"
    base_dir = base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(base_dir)
    return base_dir
