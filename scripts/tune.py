from __future__ import annotations

import argparse
import json
from pathlib import Path

from road_damage.data.dataset import verify_dataset_structure
from road_damage.training.runtime import (
    load_config,
    resolve_device,
    seed_everything,
    write_environment,
)
from road_damage.utils.paths import ensure_dir, resolve_path
from road_damage.utils.ultralytics_env import configure_ultralytics_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run equal-budget Ultralytics tuning.")
    parser.add_argument("--config", default="configs/training/baseline.yaml")
    parser.add_argument("--space", default="configs/tuning/search_space.yaml")
    parser.add_argument("--model", required=True, help="YOLOv8s, YOLO11s, or YOLO26s.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    configure_ultralytics_env(root)
    from ultralytics import YOLO

    config = load_config(args.config)
    search_space = load_config(args.space)

    if args.model not in config["model_specs"]:
        raise KeyError(f"Unknown model '{args.model}'.")

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    data_yaml = resolve_path(config["data_yaml"], root)
    verify_dataset_structure(data_yaml.parent)

    reports_root = ensure_dir(resolve_path(config.get("reports_root", "reports"), root))
    write_environment(reports_root / "environment.json")

    runs_root = ensure_dir(resolve_path(config.get("runs_root", "runs"), root))
    model_runs_dir = ensure_dir(runs_root / args.model)
    train_args = dict(config["train_args"])
    device = resolve_device(
        config.get("device", "auto"),
        require_gpu=args.require_gpu or bool(config.get("require_gpu", False)),
    )
    model_spec = config["model_specs"][args.model]
    tune_name = config.get("tune_name", "tune_equal_budget")

    metadata = {
        "model": args.model,
        "weights": model_spec["weights"],
        "iterations": args.iterations,
        "epochs": args.epochs,
        "search_space": search_space,
        "seed": seed,
    }
    (model_runs_dir / "tuning_protocol.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    YOLO(model_spec["weights"]).tune(
        data=str(data_yaml),
        imgsz=int(train_args.get("imgsz", 640)),
        batch=int(train_args.get("batch", 16)),
        epochs=args.epochs,
        iterations=args.iterations,
        optimizer=str(train_args.get("optimizer", "AdamW")),
        device=device,
        workers=int(train_args.get("workers", 2)),
        seed=seed,
        deterministic=bool(train_args.get("deterministic", True)),
        amp=bool(train_args.get("amp", True)),
        space=search_space,
        project=str(model_runs_dir),
        name=tune_name,
        resume=args.resume,
        plots=True,
        save=True,
        val=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
