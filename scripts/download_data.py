from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from road_damage.data.convert import convert_detection_or_polygon_annotations
from road_damage.data.dataset import build_ultralytics_yaml, verify_dataset_structure
from road_damage.data.validation import validate_dataset
from road_damage.training.runtime import load_config, write_environment
from road_damage.utils.env import load_dotenv
from road_damage.utils.paths import ensure_dir, resolve_path


def safe_rmtree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root = root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside the project root: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)


def write_records(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Roboflow dataset and prepare local YOLO detection data."
        )
    )
    parser.add_argument("--config", default="configs/data/roboflow.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild processed data.")
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Remove and download the raw Roboflow export again.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Prepare data but skip the audit/training gate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    load_dotenv(root / ".env")

    config = load_config(args.config)
    source_dir = resolve_path(config["source_dataset_dir"], root)
    processed_dir = resolve_path(config["processed_dataset_dir"], root)
    reports_dir = resolve_path(config["reports_dir"], root)
    ensure_dir(source_dir.parent)
    ensure_dir(processed_dir.parent)
    ensure_dir(reports_dir)

    if args.redownload:
        safe_rmtree(source_dir, root)
    if args.force:
        safe_rmtree(processed_dir, root)

    source_yaml = source_dir / "data.yaml"
    if not source_yaml.exists():
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not set. Put it in .env or your shell environment."
            )

        from roboflow import Roboflow

        print("Downloading Roboflow dataset version...")
        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(config["workspace_id"])
        project = workspace.project(config["project_id"])
        dataset = project.version(int(config["version"])).download(
            model_format=str(config.get("export_format", "yolov8")),
            location=str(source_dir),
        )
        print("Downloaded source dataset to:", dataset.location)
    else:
        print("Using existing source dataset:", source_dir)

    if not source_yaml.exists():
        raise FileNotFoundError(f"Missing source data.yaml at {source_yaml}")

    if not processed_dir.exists():
        print("Creating processed dataset copy:", processed_dir)
        shutil.copytree(source_dir, processed_dir)
    else:
        print("Using existing processed dataset:", processed_dir)

    print("Converting polygon annotations to detection boxes where needed...")
    conversion_records, conversion_errors = convert_detection_or_polygon_annotations(
        processed_dir
    )
    write_records(conversion_records, reports_dir / "conversion_records.csv")
    write_records(conversion_errors, reports_dir / "conversion_errors.csv")

    data_yaml = build_ultralytics_yaml(processed_dir)
    counts = verify_dataset_structure(processed_dir)
    print("Dataset counts:", counts)
    print("Dataset YAML:", data_yaml)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": config.get("source_url"),
        "workspace_id": config["workspace_id"],
        "project_id": config["project_id"],
        "version": int(config["version"]),
        "export_format": config.get("export_format", "yolov8"),
        "source_dataset_dir": str(source_dir),
        "processed_dataset_dir": str(processed_dir),
        "dataset_yaml": str(data_yaml),
        "images_by_split": counts,
        "conversion_rows": len(conversion_records),
        "conversion_errors": len(conversion_errors),
    }
    (reports_dir / "dataset_metadata.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (root / "dataset_handoff.json").write_text(
        json.dumps(
            {
                "processed_dataset_dir": str(processed_dir),
                "data_yaml": str(data_yaml),
                "training_gate_passed": None,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_environment(resolve_path("reports/environment.json", root))

    if args.skip_validation:
        return 0

    report = validate_dataset(processed_dir, data_yaml)
    pd.DataFrame(report.image_records).to_csv(reports_dir / "images.csv", index=False)
    pd.DataFrame(report.box_records).to_csv(reports_dir / "boxes.csv", index=False)
    pd.DataFrame(report.errors).to_csv(
        reports_dir / "validation_errors.csv", index=False
    )
    pd.DataFrame(report.duplicate_records).to_csv(
        reports_dir / "cross_split_exact_duplicates.csv", index=False
    )

    gate_summary = {
        "critical_validation_errors": report.critical_error_count,
        "all_validation_errors": len(report.errors),
        "cross_split_exact_duplicate_groups": report.exact_cross_split_duplicate_count,
        "images_by_split": report.split_counts,
        "objects_by_split": report.object_counts,
        "training_gate_passed": (
            len(conversion_errors) == 0
            and report.critical_error_count == 0
            and report.exact_cross_split_duplicate_count == 0
        ),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps({**summary, **gate_summary}, indent=2),
        encoding="utf-8",
    )
    (root / "dataset_handoff.json").write_text(
        json.dumps(
            {
                "processed_dataset_dir": str(processed_dir),
                "data_yaml": str(data_yaml),
                **gate_summary,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(gate_summary, indent=2))
    if not gate_summary["training_gate_passed"]:
        raise RuntimeError("Dataset training gate failed. See data/reports outputs.")

    print("Dataset setup and audit completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
