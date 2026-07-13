from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from road_damage.data.dataset import find_validation_folder, image_paths
from road_damage.evaluation.detection import box_iou_xyxy
from road_damage.training.runtime import load_config
from road_damage.utils.paths import ensure_dir, resolve_path
from road_damage.utils.ultralytics_env import configure_ultralytics_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare production PyTorch and ONNX detections on a sample."
    )
    parser.add_argument("--training-config", default="configs/training/baseline.yaml")
    parser.add_argument("--deployment-config", default="configs/deployment/app.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--onnx",
        default="models/exports/production_road_damage_model.onnx",
    )
    parser.add_argument("--output-dir", default="reports/final_comparison")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-box-iou", type=float, default=0.95)
    parser.add_argument("--max-conf-delta", type=float, default=0.05)
    return parser.parse_args()


def detections_from_result(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    return [
        {
            "class_id": int(class_id),
            "confidence": float(confidence),
            "box": np.asarray(box, dtype=float),
        }
        for box, confidence, class_id in zip(boxes, confidences, classes, strict=True)
    ]


def predict(model: Any, image_path: Path, imgsz: int, conf: float, device: str) -> list:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    result = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )[0]
    return detections_from_result(result)


def compare_detections(
    pytorch_detections: list[dict[str, Any]],
    onnx_detections: list[dict[str, Any]],
) -> dict[str, Any]:
    unmatched_onnx = set(range(len(onnx_detections)))
    matched_ious: list[float] = []
    matched_conf_deltas: list[float] = []
    matched_box_deltas: list[float] = []
    unmatched_pytorch = 0

    for pytorch_detection in sorted(
        pytorch_detections,
        key=lambda detection: detection["confidence"],
        reverse=True,
    ):
        candidates = [
            index
            for index in unmatched_onnx
            if onnx_detections[index]["class_id"] == pytorch_detection["class_id"]
        ]
        if not candidates:
            unmatched_pytorch += 1
            continue

        best_index = max(
            candidates,
            key=lambda index: box_iou_xyxy(
                pytorch_detection["box"],
                onnx_detections[index]["box"],
            ),
        )
        onnx_detection = onnx_detections[best_index]
        unmatched_onnx.remove(best_index)
        matched_ious.append(
            box_iou_xyxy(pytorch_detection["box"], onnx_detection["box"])
        )
        matched_conf_deltas.append(
            abs(pytorch_detection["confidence"] - onnx_detection["confidence"])
        )
        matched_box_deltas.append(
            float(np.max(np.abs(pytorch_detection["box"] - onnx_detection["box"])))
        )

    return {
        "pytorch_detections": len(pytorch_detections),
        "onnx_detections": len(onnx_detections),
        "matched_detections": len(matched_ious),
        "unmatched_pytorch": unmatched_pytorch,
        "unmatched_onnx": len(unmatched_onnx),
        "min_matched_iou": min(matched_ious) if matched_ious else "",
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else "",
        "max_conf_delta": max(matched_conf_deltas) if matched_conf_deltas else "",
        "max_box_abs_delta_px": max(matched_box_deltas) if matched_box_deltas else "",
    }


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    training_config = load_config(args.training_config)
    deployment_config = load_config(args.deployment_config)
    data_yaml = resolve_path(training_config["data_yaml"], root)
    dataset_dir = data_yaml.parent
    checkpoint = resolve_path(
        args.checkpoint or deployment_config["model_path"],
        root,
    )
    onnx_path = resolve_path(args.onnx, root)
    output_dir = ensure_dir(resolve_path(args.output_dir, root))

    configure_ultralytics_env(root)
    from ultralytics import YOLO

    validation_folder = find_validation_folder(dataset_dir)
    sample_images = image_paths(dataset_dir / validation_folder / "images")[
        : args.max_images
    ]
    if not sample_images:
        raise RuntimeError("No validation images found for export equivalence check.")

    pytorch_model = YOLO(str(checkpoint), task="detect")
    onnx_model = YOLO(str(onnx_path), task="detect")
    imgsz = int(deployment_config.get("default_image_size", 640))
    confidence = float(deployment_config.get("default_confidence_threshold", 0.25))

    details: list[dict[str, Any]] = []
    for image_path in sample_images:
        pytorch_detections = predict(
            pytorch_model,
            image_path,
            imgsz,
            confidence,
            args.device,
        )
        onnx_detections = predict(
            onnx_model,
            image_path,
            imgsz,
            confidence,
            args.device,
        )
        row = compare_detections(pytorch_detections, onnx_detections)
        row["image"] = image_path.relative_to(dataset_dir).as_posix()
        row["passed"] = (
            row["pytorch_detections"] == row["onnx_detections"]
            and row["unmatched_pytorch"] == 0
            and row["unmatched_onnx"] == 0
            and (
                row["matched_detections"] == 0
                or (
                    float(row["min_matched_iou"]) >= args.min_box_iou
                    and float(row["max_conf_delta"]) <= args.max_conf_delta
                )
            )
        )
        details.append(row)

    details_df = pd.DataFrame(details)
    details_path = output_dir / "onnx_equivalence_details.csv"
    details_df.to_csv(details_path, index=False)

    matched = details_df[details_df["matched_detections"] > 0]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "images_checked": len(details),
        "confidence_threshold": confidence,
        "image_size": imgsz,
        "min_box_iou_tolerance": args.min_box_iou,
        "max_conf_delta_tolerance": args.max_conf_delta,
        "pytorch_detections": int(details_df["pytorch_detections"].sum()),
        "onnx_detections": int(details_df["onnx_detections"].sum()),
        "matched_detections": int(details_df["matched_detections"].sum()),
        "failed_images": int((~details_df["passed"]).sum()),
        "min_matched_iou": (
            float(matched["min_matched_iou"].min()) if not matched.empty else None
        ),
        "mean_matched_iou": (
            float(matched["mean_matched_iou"].mean()) if not matched.empty else None
        ),
        "max_conf_delta": (
            float(matched["max_conf_delta"].max()) if not matched.empty else None
        ),
        "max_box_abs_delta_px": (
            float(matched["max_box_abs_delta_px"].max()) if not matched.empty else None
        ),
        "passed": bool(details_df["passed"].all()),
        "details": str(details_path),
    }
    summary_path = output_dir / "onnx_equivalence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
