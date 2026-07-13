from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from road_damage.utils.ultralytics_env import configure_ultralytics_env


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return config


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | int, require_gpu: bool = False) -> str | int:
    env_device = os.getenv("ROAD_DAMAGE_DEVICE")
    requested: str | int = env_device if env_device else device
    if isinstance(requested, str) and requested.lower() == "auto":
        if torch.cuda.is_available():
            return 0
        if require_gpu:
            raise RuntimeError("No CUDA GPU detected and --require-gpu was set.")
        return "cpu"
    if require_gpu and str(requested).lower() == "cpu":
        raise RuntimeError("--require-gpu was set but the resolved device is CPU.")
    return requested


def environment_summary() -> dict[str, Any]:
    configure_ultralytics_env()
    import ultralytics

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
    }


def write_environment(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = environment_summary()
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**2))
