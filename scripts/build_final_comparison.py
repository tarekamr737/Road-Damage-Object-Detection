from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from road_damage.utils.paths import ensure_dir, resolve_path

COLUMNS = [
    "model",
    "variant",
    "source",
    "stage",
    "dataset",
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
    "map75",
    "mean_matched_iou",
    "median_matched_iou",
    "median_latency_ms",
    "p95_latency_ms",
    "fps",
    "model_size_mb",
    "export",
    "endpoint_or_checkpoint",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the current final comparison table from saved artifacts."
    )
    parser.add_argument("--input-dir", default="reports/final_comparison")
    parser.add_argument(
        "--output",
        default="reports/final_comparison/model_comparison_current.csv",
    )
    parser.add_argument(
        "--legacy-output",
        default="reports/final_comparison/model_comparison.csv",
        help="Optional compatibility copy for older references. Use '' to skip.",
    )
    parser.add_argument(
        "--pareto-output",
        default="reports/final_comparison/deployment_pareto_frontier.csv",
        help="Optional deployment Pareto summary. Use '' to skip.",
    )
    parser.add_argument(
        "--selection-config",
        default="configs/deployment/selection.yaml",
        help="Deployment constraint config used to annotate the Pareto summary.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def index_by_model(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}
    return {
        str(row["model"]): row.dropna().to_dict()
        for _, row in frame.iterrows()
        if "model" in row
    }


def variant_from_model(model_name: str) -> str:
    return model_name[-1].lower() if model_name[-1].isalpha() else ""


def f1_score(precision: Any, recall: Any) -> float | str:
    if pd.isna(precision) or pd.isna(recall):
        return ""
    precision_float = float(precision)
    recall_float = float(recall)
    denominator = precision_float + recall_float
    if denominator == 0:
        return 0.0
    return 2 * precision_float * recall_float / denominator


def numeric(value: Any) -> float | None:
    if value == "" or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def local_common(
    model_name: str,
    benchmark_by_model: dict[str, dict[str, Any]],
    export_by_model: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    benchmark = benchmark_by_model.get(model_name, {})
    export = export_by_model.get(model_name, {})
    return {
        "model": model_name,
        "variant": variant_from_model(model_name),
        "source": "local_ultralytics",
        "median_latency_ms": benchmark.get("median_wall_latency_ms", ""),
        "p95_latency_ms": benchmark.get("p95_wall_latency_ms", ""),
        "fps": benchmark.get("fps_from_median_latency", ""),
        "export": export.get("path", ""),
    }


def build_local_rows(
    input_dir: Path,
    benchmark_by_model: dict[str, dict[str, Any]],
    export_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    validation = read_csv(input_dir / "validation_metrics.csv")
    test = read_csv(input_dir / "test_metrics.csv")
    iou_by_model = index_by_model(read_csv(input_dir / "matched_iou_summary.csv"))

    rows: list[dict[str, Any]] = []
    for _, metric in validation.iterrows():
        model_name = str(metric["model"])
        checkpoint = Path(str(metric.get("checkpoint", "")))
        if checkpoint and not checkpoint.exists():
            continue
        row = local_common(model_name, benchmark_by_model, export_by_model)
        row.update(
            {
                "stage": "validation",
                "dataset": "road-damage-detection-bbox-v1 valid",
                "precision": metric.get("validation_precision", ""),
                "recall": metric.get("validation_recall", ""),
                "f1": f1_score(
                    metric.get("validation_precision"),
                    metric.get("validation_recall"),
                ),
                "map50": metric.get("validation_map50", ""),
                "map50_95": metric.get("validation_map50_95", ""),
                "map75": "",
                "mean_matched_iou": "",
                "median_matched_iou": "",
                "model_size_mb": metric.get("model_size_mb", ""),
                "endpoint_or_checkpoint": str(checkpoint),
                "notes": "Local YOLOv8s checkpoint; selected production runtime.",
            }
        )
        rows.append(row)

    for _, metric in test.iterrows():
        model_name = str(metric["model"])
        checkpoint = Path(str(metric.get("checkpoint", "")))
        if checkpoint and not checkpoint.exists():
            continue
        iou = iou_by_model.get(model_name, {})
        row = local_common(model_name, benchmark_by_model, export_by_model)
        row.update(
            {
                "stage": "test",
                "dataset": "road-damage-detection-bbox-v1 test",
                "precision": metric.get("precision", ""),
                "recall": metric.get("recall", ""),
                "f1": metric.get("f1", ""),
                "map50": metric.get("map50", ""),
                "map50_95": metric.get("map50_95", ""),
                "map75": "",
                "mean_matched_iou": iou.get("mean_matched_iou", ""),
                "median_matched_iou": iou.get("median_matched_iou", ""),
                "model_size_mb": metric.get("model_size_mb", ""),
                "endpoint_or_checkpoint": str(checkpoint),
                "notes": (
                    "Untouched local test split; selected production runtime. "
                    "Threshold 0.25 for IoU/prediction export."
                ),
            }
        )
        rows.append(row)
    return rows


def build_hosted_rows(decision_path: Path) -> list[dict[str, Any]]:
    if not decision_path.exists():
        return []
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for available in decision.get("available_models", []):
        if available.get("source") != "roboflow_hosted":
            continue
        stage = "validation" if "valid" in str(available.get("stage", "")) else "test"
        rows.append(
            {
                "model": available.get("model", ""),
                "variant": available.get("variant", ""),
                "source": "roboflow_hosted",
                "stage": stage,
                "dataset": available.get("dataset", ""),
                "precision": available.get("precision", ""),
                "recall": available.get("recall", ""),
                "f1": available.get("f1", ""),
                "map50": available.get("map50", ""),
                "map50_95": available.get("map50_95", ""),
                "map75": available.get("map75", ""),
                "mean_matched_iou": "",
                "median_matched_iou": "",
                "median_latency_ms": "",
                "p95_latency_ms": "",
                "fps": "",
                "model_size_mb": "",
                "export": "",
                "endpoint_or_checkpoint": available.get("endpoint_or_checkpoint", ""),
                "notes": available.get("notes", ""),
            }
        )
    return rows


def load_selection_constraints(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded if isinstance(loaded, dict) else {}


def is_dominated(
    candidate: dict[str, Any],
    other: dict[str, Any],
    *,
    maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> bool:
    at_least_as_good = True
    strictly_better = False

    for field in maximize:
        candidate_value = numeric(candidate.get(field))
        other_value = numeric(other.get(field))
        if candidate_value is None or other_value is None:
            return False
        if other_value < candidate_value:
            at_least_as_good = False
        if other_value > candidate_value:
            strictly_better = True

    for field in minimize:
        candidate_value = numeric(candidate.get(field))
        other_value = numeric(other.get(field))
        if candidate_value is None or other_value is None:
            return False
        if other_value > candidate_value:
            at_least_as_good = False
        if other_value < candidate_value:
            strictly_better = True

    return at_least_as_good and strictly_better


def build_pareto_summary(
    comparison: pd.DataFrame,
    constraints: dict[str, Any],
) -> pd.DataFrame:
    selection_rows = comparison[comparison["stage"] == "validation"].copy()
    required = [
        "map50_95",
        "median_latency_ms",
        "p95_latency_ms",
        "fps",
        "model_size_mb",
    ]
    complete_candidates = []
    records: list[dict[str, Any]] = []

    for _, row in selection_rows.iterrows():
        row_dict = row.to_dict()
        missing = [field for field in required if numeric(row_dict.get(field)) is None]
        complete = not missing
        if complete:
            complete_candidates.append(row_dict)

        p95_latency = numeric(row_dict.get("p95_latency_ms"))
        fps = numeric(row_dict.get("fps"))
        size = numeric(row_dict.get("model_size_mb"))
        recall = numeric(row_dict.get("recall"))
        max_p95 = numeric(constraints.get("maximum_p95_latency_ms"))
        min_fps = numeric(constraints.get("minimum_fps"))
        max_size = numeric(constraints.get("maximum_model_size_mb"))
        min_recall = numeric(constraints.get("minimum_validation_recall"))

        blocking_reasons = []
        if missing:
            blocking_reasons.append("missing deployment metrics: " + ", ".join(missing))
        if max_p95 is not None and p95_latency is not None and p95_latency > max_p95:
            blocking_reasons.append("p95 latency exceeds constraint")
        if min_fps is not None and fps is not None and fps < min_fps:
            blocking_reasons.append("FPS is below constraint")
        if max_size is not None and size is not None and size > max_size:
            blocking_reasons.append("model size exceeds constraint")
        if min_recall is not None and recall is not None and recall < min_recall:
            blocking_reasons.append("validation recall is below constraint")

        records.append(
            {
                "model": row_dict.get("model", ""),
                "variant": row_dict.get("variant", ""),
                "source": row_dict.get("source", ""),
                "stage": row_dict.get("stage", ""),
                "map50_95": row_dict.get("map50_95", ""),
                "recall": row_dict.get("recall", ""),
                "median_latency_ms": row_dict.get("median_latency_ms", ""),
                "p95_latency_ms": row_dict.get("p95_latency_ms", ""),
                "fps": row_dict.get("fps", ""),
                "model_size_mb": row_dict.get("model_size_mb", ""),
                "deployment_metrics_complete": complete,
                "passes_p95_latency": (
                    ""
                    if max_p95 is None or p95_latency is None
                    else p95_latency <= max_p95
                ),
                "passes_fps": "" if min_fps is None or fps is None else fps >= min_fps,
                "passes_model_size": (
                    "" if max_size is None or size is None else size <= max_size
                ),
                "passes_validation_recall": (
                    "" if min_recall is None or recall is None else recall >= min_recall
                ),
                "pareto_frontier": False,
                "blocking_reasons": "; ".join(blocking_reasons),
                "notes": row_dict.get("notes", ""),
            }
        )

    for record in records:
        if not record["deployment_metrics_complete"]:
            continue
        record_candidate = next(
            candidate
            for candidate in complete_candidates
            if candidate.get("model") == record["model"]
            and candidate.get("source") == record["source"]
        )
        record["pareto_frontier"] = not any(
            is_dominated(
                record_candidate,
                other,
                maximize=("map50_95", "fps"),
                minimize=("median_latency_ms", "model_size_mb"),
            )
            for other in complete_candidates
            if other is not record_candidate
        )

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    input_dir = resolve_path(args.input_dir, root)
    output = resolve_path(args.output, root)

    benchmark_by_model = index_by_model(read_csv(input_dir / "benchmark.csv"))
    export_by_model = index_by_model(read_csv(input_dir / "exports.csv"))

    rows = build_local_rows(input_dir, benchmark_by_model, export_by_model)
    rows.extend(build_hosted_rows(input_dir / "production_model_decision.json"))

    comparison = pd.DataFrame(rows, columns=COLUMNS)
    ensure_dir(output.parent)
    comparison.to_csv(output, index=False)

    if args.legacy_output:
        legacy_output = resolve_path(args.legacy_output, root)
        ensure_dir(legacy_output.parent)
        comparison.to_csv(legacy_output, index=False)

    if args.pareto_output:
        pareto_output = resolve_path(args.pareto_output, root)
        constraints = load_selection_constraints(
            resolve_path(args.selection_config, root)
        )
        pareto = build_pareto_summary(comparison, constraints)
        ensure_dir(pareto_output.parent)
        pareto.to_csv(pareto_output, index=False)

    print(f"Wrote {len(comparison)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
