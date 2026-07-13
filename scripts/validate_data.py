from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from road_damage.data.dataset import build_ultralytics_yaml, verify_dataset_structure
from road_damage.data.validation import validate_dataset
from road_damage.training.runtime import load_config
from road_damage.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local YOLO dataset.")
    parser.add_argument("--config", default="configs/data/roboflow.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--data-yaml", default=None)
    parser.add_argument("--reports-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config = load_config(args.config)
    dataset_dir = resolve_path(
        args.dataset_dir or config["processed_dataset_dir"],
        root,
    )
    reports_dir = ensure_dir(
        resolve_path(args.reports_dir or config["reports_dir"], root)
    )
    data_yaml = (
        resolve_path(args.data_yaml, root)
        if args.data_yaml
        else build_ultralytics_yaml(dataset_dir)
    )

    counts = verify_dataset_structure(dataset_dir)
    report = validate_dataset(dataset_dir, data_yaml)

    pd.DataFrame(report.image_records).to_csv(reports_dir / "images.csv", index=False)
    pd.DataFrame(report.box_records).to_csv(reports_dir / "boxes.csv", index=False)
    pd.DataFrame(report.errors).to_csv(
        reports_dir / "validation_errors.csv", index=False
    )
    pd.DataFrame(report.duplicate_records).to_csv(
        reports_dir / "cross_split_exact_duplicates.csv", index=False
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "images_by_split": counts,
        "objects_by_split": report.object_counts,
        "critical_validation_errors": report.critical_error_count,
        "all_validation_errors": len(report.errors),
        "cross_split_exact_duplicate_groups": report.exact_cross_split_duplicate_count,
        "training_gate_passed": (
            report.critical_error_count == 0
            and report.exact_cross_split_duplicate_count == 0
        ),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))

    return 0 if summary["training_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
