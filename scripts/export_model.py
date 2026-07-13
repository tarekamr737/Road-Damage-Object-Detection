from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from road_damage.training.runtime import load_config
from road_damage.utils.paths import ensure_dir, resolve_path
from road_damage.utils.ultralytics_env import configure_ultralytics_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export checkpoints and select production."
    )
    parser.add_argument("--config", default="configs/training/baseline.yaml")
    parser.add_argument(
        "--deployment-config",
        default="configs/deployment/selection.yaml",
    )
    parser.add_argument("--comparison-dir", default="reports/final_comparison")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--validate-onnx", action="store_true")
    parser.add_argument(
        "--simplify-onnx",
        action="store_true",
        help="Run ONNX simplification. Requires onnxslim to be installed.",
    )
    parser.add_argument("--skip-selection", action="store_true")
    return parser.parse_args()


def find_checkpoints(config: dict, root: Path) -> dict[str, Path]:
    runs_root = resolve_path(config.get("runs_root", "runs"), root)
    checkpoints: dict[str, Path] = {}
    for model_tag in config["model_specs"]:
        final = (
            runs_root
            / model_tag
            / config.get("final_name", "final_tuned_seed42")
            / "weights"
            / "best.pt"
        )
        baseline = (
            runs_root
            / model_tag
            / config.get("baseline_name", "baseline_seed42")
            / "weights"
            / "best.pt"
        )
        if final.exists():
            checkpoints[model_tag] = final
        elif baseline.exists():
            checkpoints[model_tag] = baseline
    if not checkpoints:
        raise FileNotFoundError("No trained checkpoints found under runs/.")
    return checkpoints


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config = load_config(args.config)
    deployment_config = load_config(args.deployment_config)
    comparison_dir = ensure_dir(resolve_path(args.comparison_dir, root))
    exports_root = ensure_dir(
        resolve_path(config.get("exports_root", "models/exports"), root)
    )
    configure_ultralytics_env(root)
    from ultralytics import YOLO

    checkpoints = find_checkpoints(config, root)
    imgsz = int(
        deployment_config.get("image_size", config["train_args"].get("imgsz", 640))
    )

    export_rows: list[dict] = []
    if args.export_onnx:
        for model_tag, checkpoint in checkpoints.items():
            print(f"Exporting {model_tag} to ONNX...")
            exported = Path(
                YOLO(str(checkpoint)).export(
                    format="onnx",
                    imgsz=imgsz,
                    batch=1,
                    dynamic=False,
                    simplify=args.simplify_onnx,
                    opset=17,
                )
            )
            destination = exports_root / f"{model_tag}_road_damage.onnx"
            shutil.copy2(exported, destination)
            row = {
                "model": model_tag,
                "format": "onnx",
                "path": str(destination),
                "size_mb": destination.stat().st_size / (1024**2),
            }
            if args.validate_onnx:
                data_yaml = resolve_path(config["data_yaml"], root)
                metrics = YOLO(str(destination), task="detect").val(
                    data=str(data_yaml),
                    split="test",
                    imgsz=imgsz,
                    batch=1,
                    plots=False,
                )
                row["onnx_map50_95"] = float(metrics.box.map)
            export_rows.append(row)

    if export_rows:
        pd.DataFrame(export_rows).to_csv(comparison_dir / "exports.csv", index=False)

    if args.skip_selection:
        return 0

    validation_path = comparison_dir / "validation_metrics.csv"
    benchmark_path = comparison_dir / "benchmark.csv"
    if not validation_path.exists() or not benchmark_path.exists():
        raise FileNotFoundError(
            "Selection needs validation_metrics.csv and benchmark.csv. "
            "Run scripts/evaluate.py and scripts/benchmark.py first."
        )

    validation_df = pd.read_csv(validation_path)
    benchmark_df = pd.read_csv(benchmark_path)
    selection_df = validation_df.merge(benchmark_df, on="model", how="left")

    eligible = selection_df[
        (
            selection_df["validation_recall"]
            >= float(deployment_config["minimum_validation_recall"])
        )
        & (
            selection_df["p95_wall_latency_ms"]
            <= float(deployment_config["maximum_p95_latency_ms"])
        )
        & (
            selection_df["model_size_mb"]
            <= float(deployment_config["maximum_model_size_mb"])
        )
    ].copy()

    if eligible.empty:
        print("No model meets the predeclared validation/deployment constraints.")
        selection_df.to_csv(comparison_dir / "selection_candidates.csv", index=False)
        return 1

    winner = eligible.sort_values(
        ["validation_map50_95", "median_wall_latency_ms"],
        ascending=[False, True],
    ).iloc[0]
    winner_name = str(winner["model"])
    winner_checkpoint = checkpoints[winner_name]
    production_pt = exports_root / "production_road_damage_model.pt"
    shutil.copy2(winner_checkpoint, production_pt)

    candidate_onnx = exports_root / f"{winner_name}_road_damage.onnx"
    production_onnx = exports_root / "production_road_damage_model.onnx"
    if candidate_onnx.exists():
        shutil.copy2(candidate_onnx, production_onnx)

    decision = {
        "selected_model": winner_name,
        "selection_basis": (
            "validation metrics plus local benchmark; test metrics excluded"
        ),
        "checkpoint": str(winner_checkpoint),
        "production_pt": str(production_pt),
        "production_onnx": str(production_onnx) if production_onnx.exists() else None,
        "constraints": {
            "minimum_validation_recall": float(
                deployment_config["minimum_validation_recall"]
            ),
            "maximum_p95_latency_ms": float(
                deployment_config["maximum_p95_latency_ms"]
            ),
            "maximum_model_size_mb": float(deployment_config["maximum_model_size_mb"]),
        },
        "selection_metrics": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in winner.to_dict().items()
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (comparison_dir / "production_model_decision.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
