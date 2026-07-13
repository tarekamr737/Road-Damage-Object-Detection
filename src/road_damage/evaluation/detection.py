from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def names_dict(names: dict | list) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def metrics_to_dict(
    metrics: Any,
    model: Any,
    checkpoint: Path,
    model_family: str,
    model_tag: str,
    stage: str,
) -> dict:
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    names = names_dict(model.names)
    maps = [float(value) for value in metrics.box.maps]
    return {
        "model_family": model_family,
        "model_tag": model_tag,
        "stage": stage,
        "checkpoint": str(checkpoint),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-12),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": {
            str(names.get(index, index)): maps[index]
            for index in range(min(len(maps), len(names)))
        },
        "model_size_mb": checkpoint.stat().st_size / (1024**2),
        "parameters": int(
            sum(parameter.numel() for parameter in model.model.parameters())
        ),
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def ground_truth(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    with Image.open(image_path) as image:
        width, height = image.size

    boxes: list[list[float]] = []
    classes: list[int] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, box_width, box_height = map(float, parts[1:])
            boxes.append(
                [
                    (x_center - box_width / 2) * width,
                    (y_center - box_height / 2) * height,
                    (x_center + box_width / 2) * width,
                    (y_center + box_height / 2) * height,
                ]
            )
            classes.append(class_id)

    return (
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(classes, dtype=int),
    )


def matched_ious(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    threshold: float,
) -> tuple[list[float], list[int]]:
    used_gt: set[int] = set()
    ious_out: list[float] = []
    classes_out: list[int] = []

    for pred_index in np.argsort(-pred_conf):
        candidates = [
            index
            for index, gt_class in enumerate(gt_classes)
            if index not in used_gt and int(gt_class) == int(pred_classes[pred_index])
        ]
        if not candidates:
            continue

        values = [
            box_iou_xyxy(pred_boxes[pred_index], gt_boxes[index])
            for index in candidates
        ]
        position = int(np.argmax(values))
        gt_index = candidates[position]
        best_iou = float(values[position])

        if best_iou >= threshold:
            used_gt.add(gt_index)
            ious_out.append(best_iou)
            classes_out.append(int(pred_classes[pred_index]))

    return ious_out, classes_out
