from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from road_damage.data.dataset import image_paths
from road_damage.evaluation.detection import box_iou_xyxy, ground_truth
from road_damage.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build qualitative error-analysis tables from saved predictions."
    )
    parser.add_argument(
        "--data-yaml",
        default=("data/processed/road-damage-detection-bbox-v1/data_detection.yaml"),
    )
    parser.add_argument(
        "--predictions",
        default="reports/final_comparison/test_predictions.jsonl",
    )
    parser.add_argument("--output-dir", default="reports/final_comparison")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--examples-per-type", type=int, default=5)
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> dict[int, str]:
    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    names = config.get("names", [])
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def load_predictions(path: Path, confidence_threshold: float) -> dict[str, list[dict]]:
    predictions: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row["confidence"]) < confidence_threshold:
                continue
            predictions[str(row["image"])].append(row)
    return predictions


def prediction_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = [[row["x_min"], row["y_min"], row["x_max"], row["y_max"]] for row in rows]
    classes = [row["class_id"] for row in rows]
    confidence = [row["confidence"] for row in rows]
    return (
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(classes, dtype=int),
        np.asarray(confidence, dtype=float),
    )


def class_count_label(classes: np.ndarray, names: dict[int, str]) -> str:
    if len(classes) == 0:
        return ""
    values, counts = np.unique(classes, return_counts=True)
    parts = [
        f"{names.get(int(class_id), str(int(class_id)))}:{int(count)}"
        for class_id, count in zip(values, counts, strict=True)
    ]
    return "; ".join(parts)


def match_predictions(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for pred_index in np.argsort(-pred_conf):
        candidates = [
            index
            for index, gt_class in enumerate(gt_classes)
            if index not in used_gt and int(gt_class) == int(pred_classes[pred_index])
        ]
        if not candidates:
            continue

        ious = [
            box_iou_xyxy(pred_boxes[pred_index], gt_boxes[index])
            for index in candidates
        ]
        best_position = int(np.argmax(ious))
        best_gt = candidates[best_position]
        best_iou = float(ious[best_position])
        if best_iou < threshold:
            continue

        used_gt.add(best_gt)
        used_pred.add(int(pred_index))
        matches.append((int(pred_index), int(best_gt), best_iou))

    return matches, used_pred, used_gt


def build_rows(
    dataset_dir: Path,
    predictions: dict[str, list[dict]],
    names: dict[int, str],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_path in image_paths(dataset_dir / "test" / "images"):
        gt_boxes, gt_classes = ground_truth(image_path)
        pred_rows = predictions.get(image_path.name, [])
        pred_boxes, pred_classes, pred_conf = prediction_arrays(pred_rows)
        matches, used_pred, used_gt = match_predictions(
            pred_boxes,
            pred_classes,
            pred_conf,
            gt_boxes,
            gt_classes,
            threshold,
        )
        matched_ious = [match[2] for match in matches]
        fp_count = len(pred_rows) - len(used_pred)
        fn_count = len(gt_classes) - len(used_gt)
        rows.append(
            {
                "image": image_path.name,
                "ground_truth_objects": len(gt_classes),
                "predicted_objects": len(pred_rows),
                "matched_true_positives": len(matches),
                "false_positives": fp_count,
                "false_negatives": fn_count,
                "mean_matched_iou": (
                    float(np.mean(matched_ious)) if matched_ious else np.nan
                ),
                "min_matched_iou": (
                    float(np.min(matched_ious)) if matched_ious else np.nan
                ),
                "ground_truth_classes": class_count_label(gt_classes, names),
                "predicted_classes": class_count_label(pred_classes, names),
            }
        )
    return pd.DataFrame(rows)


def take_examples(
    frame: pd.DataFrame,
    example_type: str,
    sort_columns: list[str],
    ascending: list[bool],
    count: int,
    predicate: pd.Series,
) -> pd.DataFrame:
    selected = (
        frame[predicate]
        .sort_values(sort_columns, ascending=ascending)
        .head(count)
        .copy()
    )
    selected.insert(0, "example_type", example_type)
    return selected


def build_examples(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    examples = [
        take_examples(
            frame,
            "false_positive_heavy",
            ["false_positives", "predicted_objects", "image"],
            [False, False, True],
            count,
            frame["false_positives"] > 0,
        ),
        take_examples(
            frame,
            "false_negative_heavy",
            ["false_negatives", "ground_truth_objects", "image"],
            [False, False, True],
            count,
            frame["false_negatives"] > 0,
        ),
        take_examples(
            frame,
            "low_localization_iou",
            ["mean_matched_iou", "matched_true_positives", "image"],
            [True, False, True],
            count,
            frame["matched_true_positives"] > 0,
        ),
        take_examples(
            frame,
            "representative_strong_match",
            [
                "false_positives",
                "false_negatives",
                "matched_true_positives",
                "mean_matched_iou",
                "image",
            ],
            [True, True, False, False, True],
            count,
            frame["matched_true_positives"] > 0,
        ),
    ]
    return pd.concat(examples, ignore_index=True)


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    data_yaml = resolve_path(args.data_yaml, root)
    dataset_dir = data_yaml.parent
    output_dir = ensure_dir(resolve_path(args.output_dir, root))
    predictions = load_predictions(
        resolve_path(args.predictions, root),
        args.confidence_threshold,
    )
    names = load_class_names(data_yaml)
    by_image = build_rows(
        dataset_dir,
        predictions,
        names,
        args.match_iou_threshold,
    )
    examples = build_examples(by_image, args.examples_per_type)

    by_image.to_csv(output_dir / "error_analysis_by_image.csv", index=False)
    examples.to_csv(output_dir / "error_analysis_examples.csv", index=False)

    summary = {
        "images": int(len(by_image)),
        "confidence_threshold": args.confidence_threshold,
        "match_iou_threshold": args.match_iou_threshold,
        "images_with_false_positives": int((by_image["false_positives"] > 0).sum()),
        "images_with_false_negatives": int((by_image["false_negatives"] > 0).sum()),
        "images_with_matches": int((by_image["matched_true_positives"] > 0).sum()),
        "total_false_positives": int(by_image["false_positives"].sum()),
        "total_false_negatives": int(by_image["false_negatives"].sum()),
    }
    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
