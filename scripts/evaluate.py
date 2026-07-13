from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from road_damage.data.dataset import image_paths, verify_dataset_structure
from road_damage.evaluation.detection import (
    ground_truth,
    matched_ious,
    metrics_to_dict,
    names_dict,
)
from road_damage.training.runtime import load_config, resolve_device, write_environment
from road_damage.utils.paths import ensure_dir, resolve_path
from road_damage.utils.ultralytics_env import configure_ultralytics_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models on val/test.")
    parser.add_argument("--config", default="configs/training/baseline.yaml")
    parser.add_argument(
        "--deployment-config",
        default="configs/deployment/selection.yaml",
    )
    parser.add_argument("--output-dir", default="reports/final_comparison")
    parser.add_argument("--max-iou-images", type=int, default=None)
    parser.add_argument("--skip-iou", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
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
        else:
            print(f"Missing checkpoint: {model_tag}")
    if not checkpoints:
        raise FileNotFoundError("No trained checkpoints found under runs/.")
    return checkpoints


def checkpoint_stage(checkpoint: Path, config: dict) -> str:
    return (
        "final"
        if config.get("final_name", "final_tuned_seed42") in checkpoint.parts
        else "baseline"
    )


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config = load_config(args.config)
    deployment_config = load_config(args.deployment_config)
    data_yaml = resolve_path(config["data_yaml"], root)
    dataset_dir = data_yaml.parent
    verify_dataset_structure(dataset_dir)

    output_dir = ensure_dir(resolve_path(args.output_dir, root))
    test_runs_dir = ensure_dir(output_dir / "test_runs")
    predictions_path = output_dir / "test_predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")

    reports_root = ensure_dir(resolve_path(config.get("reports_root", "reports"), root))
    write_environment(reports_root / "environment.json")
    configure_ultralytics_env(root)
    from ultralytics import YOLO

    checkpoints = find_checkpoints(config, root)
    train_args = config["train_args"]
    device = resolve_device(config.get("device", "auto"))
    imgsz = int(deployment_config.get("image_size", train_args.get("imgsz", 640)))
    batch = int(train_args.get("batch", 16))
    workers = int(train_args.get("workers", 2))
    deploy_conf = float(deployment_config.get("deploy_confidence_threshold", 0.25))
    match_iou_threshold = float(deployment_config.get("match_iou_threshold", 0.5))

    validation_rows: list[dict] = []
    test_rows: list[dict] = []
    per_class_rows: list[dict] = []

    for model_tag, checkpoint in checkpoints.items():
        print(f"Evaluating validation metrics for {model_tag}...")
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            plots=False,
            project=str(output_dir / "validation_runs"),
            name=model_tag,
        )
        model_spec = config["model_specs"][model_tag]
        validation_summary = metrics_to_dict(
            metrics,
            model,
            checkpoint,
            model_spec["family"],
            model_tag,
            "validation",
        )
        validation_rows.append(
            {
                "model": model_tag,
                "checkpoint_stage": checkpoint_stage(checkpoint, config),
                "validation_precision": validation_summary["precision"],
                "validation_recall": validation_summary["recall"],
                "validation_map50": validation_summary["map50"],
                "validation_map50_95": validation_summary["map50_95"],
                "model_size_mb": validation_summary["model_size_mb"],
                "checkpoint": str(checkpoint),
            }
        )

        if args.skip_test:
            continue

        print(f"Evaluating untouched test split for {model_tag}...")
        test_metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            plots=True,
            project=str(test_runs_dir),
            name=model_tag,
        )
        precision = float(test_metrics.box.mp)
        recall = float(test_metrics.box.mr)
        test_rows.append(
            {
                "model": model_tag,
                "checkpoint_stage": checkpoint_stage(checkpoint, config),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall + 1e-12),
                "map50": float(test_metrics.box.map50),
                "map50_95": float(test_metrics.box.map),
                "parameters": int(
                    sum(parameter.numel() for parameter in model.model.parameters())
                ),
                "model_size_mb": checkpoint.stat().st_size / (1024**2),
                "val_preprocess_ms": float(
                    test_metrics.speed.get("preprocess", np.nan)
                ),
                "val_inference_ms": float(test_metrics.speed.get("inference", np.nan)),
                "val_postprocess_ms": float(
                    test_metrics.speed.get("postprocess", np.nan)
                ),
                "checkpoint": str(checkpoint),
            }
        )

        names = names_dict(model.names)
        maps = [float(value) for value in test_metrics.box.maps]
        for class_id, class_name in names.items():
            if class_id < len(maps):
                per_class_rows.append(
                    {
                        "model": model_tag,
                        "class_id": class_id,
                        "class_name": class_name,
                        "map50_95": maps[class_id],
                    }
                )

    validation_df = pd.DataFrame(validation_rows).sort_values(
        "validation_map50_95",
        ascending=False,
    )
    validation_df.to_csv(output_dir / "validation_metrics.csv", index=False)

    test_metrics_df = pd.DataFrame(test_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    if not test_metrics_df.empty:
        test_metrics_df = test_metrics_df.sort_values("map50_95", ascending=False)
        test_metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    if not per_class_df.empty:
        per_class_df.to_csv(output_dir / "per_class_map50_95.csv", index=False)

    iou_summary_rows: list[dict] = []
    iou_detail_rows: list[dict] = []
    if not args.skip_iou and not args.skip_test:
        test_images = image_paths(dataset_dir / "test" / "images")
        if args.max_iou_images is not None:
            test_images = test_images[: args.max_iou_images]

        with predictions_path.open("a", encoding="utf-8") as predictions_file:
            for model_tag, checkpoint in checkpoints.items():
                print(f"IoU analysis and prediction export for {model_tag}...")
                model = YOLO(str(checkpoint))
                names = names_dict(model.names)
                all_ious: list[float] = []

                for number, image_path in enumerate(test_images, start=1):
                    gt_boxes, gt_classes = ground_truth(image_path)
                    result = model.predict(
                        source=str(image_path),
                        imgsz=imgsz,
                        conf=deploy_conf,
                        device=device,
                        verbose=False,
                    )[0]

                    if result.boxes is None or len(result.boxes) == 0:
                        continue

                    pred_boxes = result.boxes.xyxy.detach().cpu().numpy()
                    pred_classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                    pred_conf = result.boxes.conf.detach().cpu().numpy()

                    values, classes = matched_ious(
                        pred_boxes,
                        pred_classes,
                        pred_conf,
                        gt_boxes,
                        gt_classes,
                        match_iou_threshold,
                    )
                    all_ious.extend(values)
                    for iou_value, class_id in zip(values, classes, strict=True):
                        iou_detail_rows.append(
                            {
                                "model": model_tag,
                                "image": image_path.name,
                                "class_id": class_id,
                                "class_name": names.get(class_id, str(class_id)),
                                "iou": iou_value,
                            }
                        )

                    for box, class_id, confidence in zip(
                        pred_boxes,
                        pred_classes,
                        pred_conf,
                        strict=True,
                    ):
                        predictions_file.write(
                            json.dumps(
                                {
                                    "model": model_tag,
                                    "image": image_path.name,
                                    "class_id": int(class_id),
                                    "class_name": names.get(
                                        int(class_id),
                                        str(class_id),
                                    ),
                                    "confidence": float(confidence),
                                    "x_min": float(box[0]),
                                    "y_min": float(box[1]),
                                    "x_max": float(box[2]),
                                    "y_max": float(box[3]),
                                }
                            )
                            + "\n"
                        )

                    if number % 100 == 0:
                        print(f"{model_tag}: {number}/{len(test_images)}")

                iou_summary_rows.append(
                    {
                        "model": model_tag,
                        "matched_true_positives": len(all_ious),
                        "mean_matched_iou": (
                            float(np.mean(all_ious)) if all_ious else np.nan
                        ),
                        "median_matched_iou": (
                            float(np.median(all_ious)) if all_ious else np.nan
                        ),
                        "p10_matched_iou": (
                            float(np.percentile(all_ious, 10)) if all_ious else np.nan
                        ),
                        "matching_threshold": match_iou_threshold,
                        "confidence_threshold": deploy_conf,
                    }
                )

    iou_summary_df = pd.DataFrame(iou_summary_rows)
    iou_details_df = pd.DataFrame(iou_detail_rows)
    if not iou_summary_df.empty:
        iou_summary_df.to_csv(output_dir / "matched_iou_summary.csv", index=False)
    if not iou_details_df.empty:
        iou_details_df.to_csv(output_dir / "matched_iou_details.csv", index=False)

    comparison_df = test_metrics_df.copy()
    if not comparison_df.empty and not iou_summary_df.empty:
        comparison_df = comparison_df.merge(iou_summary_df, on="model", how="left")
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_yaml": str(data_yaml),
        "models": {model: str(path) for model, path in checkpoints.items()},
        "deploy_confidence_threshold": deploy_conf,
        "match_iou_threshold": match_iou_threshold,
        "outputs": str(output_dir),
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
