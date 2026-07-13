from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from road_damage.data.dataset import verify_dataset_structure
from road_damage.evaluation.detection import metrics_to_dict
from road_damage.training.runtime import (
    load_config,
    resolve_device,
    seed_everything,
    write_environment,
)
from road_damage.utils.paths import ensure_dir, resolve_path
from road_damage.utils.ultralytics_env import configure_ultralytics_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local YOLO baselines/finals.")
    parser.add_argument("--config", default="configs/training/baseline.yaml")
    parser.add_argument(
        "--model",
        default="all",
        help="YOLOv8s, YOLO11s, YOLO26s, or all.",
    )
    parser.add_argument("--stage", choices=["baseline", "final"], default="baseline")
    parser.add_argument("--force", action="store_true", help="Retrain from scratch.")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def newest_file(root: Path, filename: str) -> Path | None:
    candidates = list(root.rglob(filename))
    return (
        max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    )


def run_one(model_tag: str, config: dict, args: argparse.Namespace, root: Path) -> None:
    configure_ultralytics_env(root)
    from ultralytics import YOLO

    specs = config["model_specs"]
    if model_tag not in specs:
        raise KeyError(f"Unknown model '{model_tag}'. Options: {', '.join(specs)}")

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    data_yaml = resolve_path(config["data_yaml"], root)
    dataset_dir = data_yaml.parent
    verify_dataset_structure(dataset_dir)

    runs_root = ensure_dir(resolve_path(config.get("runs_root", "runs"), root))
    reports_root = ensure_dir(resolve_path(config.get("reports_root", "reports"), root))
    write_environment(reports_root / "environment.json")

    model_spec = specs[model_tag]
    model_runs_dir = ensure_dir(runs_root / model_tag)
    run_name = (
        config.get("baseline_name", "baseline_seed42")
        if args.stage == "baseline"
        else config.get("final_name", "final_tuned_seed42")
    )
    run_dir = model_runs_dir / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    if best.exists() and not args.force:
        print(f"{model_tag} {args.stage} exists; skipping: {best}")
    else:
        if args.force and run_dir.exists():
            shutil.rmtree(run_dir)

        device = resolve_device(
            config.get("device", "auto"),
            require_gpu=args.require_gpu or bool(config.get("require_gpu", False)),
        )
        train_args = dict(config["train_args"])
        train_args.update(
            {
                "data": str(data_yaml),
                "device": device,
                "seed": seed,
            }
        )

        if args.stage == "final":
            best_hypers_path = newest_file(
                model_runs_dir,
                "best_hyperparameters.yaml",
            )
            if best_hypers_path is None:
                raise RuntimeError(
                    f"No tuning result found for {model_tag}. "
                    "Run scripts/tune.py first."
                )
            best_hypers = (
                yaml.safe_load(best_hypers_path.read_text(encoding="utf-8")) or {}
            )
            search_space = load_config("configs/tuning/search_space.yaml")
            filtered_hypers = {
                key: value for key, value in best_hypers.items() if key in search_space
            }
            if not filtered_hypers:
                raise RuntimeError(
                    f"No supported tuned hyperparameters for {model_tag}."
                )
            train_args.update(filtered_hypers)

        config_name = f"{args.stage}_config.json"
        (model_runs_dir / config_name).write_text(
            json.dumps(train_args, indent=2),
            encoding="utf-8",
        )

        if last.exists() and not args.force:
            print(f"Resuming {model_tag}: {last}")
            YOLO(str(last)).train(resume=True)
        else:
            print(f"Training {model_tag} from {model_spec['weights']}")
            YOLO(model_spec["weights"]).train(
                **train_args,
                project=str(model_runs_dir),
                name=run_name,
                exist_ok=True,
            )

    if not best.exists():
        print(f"No checkpoint available for {model_tag}: {best}")
        return

    if not args.skip_val:
        device = resolve_device(config.get("device", "auto"))
        train_args = config["train_args"]
        model = YOLO(str(best))
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=int(train_args.get("imgsz", 640)),
            batch=int(train_args.get("batch", 16)),
            device=device,
            workers=int(train_args.get("workers", 2)),
            plots=True,
            project=str(model_runs_dir),
            name=f"{args.stage}_validation",
        )
        summary = metrics_to_dict(
            metrics=metrics,
            model=model,
            checkpoint=best,
            model_family=str(model_spec["family"]),
            model_tag=model_tag,
            stage=f"{args.stage}_validation",
        )
        (model_runs_dir / f"{args.stage}_validation_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))

    metadata = {
        "model_family": model_spec["family"],
        "model_tag": model_tag,
        "weights_source": model_spec["weights"],
        "selected_checkpoint": str(best),
        "selection_stage": args.stage,
        "dataset_yaml": str(data_yaml),
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (model_runs_dir / "selected_checkpoint.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config = load_config(args.config)
    model_tags = list(config["model_specs"]) if args.model == "all" else [args.model]
    for model_tag in model_tags:
        run_one(model_tag, config, args, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
