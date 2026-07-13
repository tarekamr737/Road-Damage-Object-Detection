from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from road_damage.data.dataset import (
    build_ultralytics_yaml,
    normalize_names,
    split_directories,
)
from road_damage.data.validation import file_sha256
from road_damage.training.runtime import load_config
from road_damage.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen split manifests with file hashes and label counts."
    )
    parser.add_argument("--config", default="configs/data/roboflow.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--data-yaml", default=None)
    parser.add_argument("--output-dir", default="data/reports/split_manifests")
    return parser.parse_args()


def count_label_classes(label_path: Path, class_names: dict[int, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(float(parts[0]))
        counts[class_names.get(class_id, str(class_id))] += 1
    return counts


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        return image.size


def build_manifest_rows(
    dataset_dir: Path,
    split: str,
    split_dir: Path,
    class_names: dict[int, str],
) -> tuple[list[dict], Counter[str]]:
    rows: list[dict] = []
    class_totals: Counter[str] = Counter()
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    for image_path in sorted(path for path in images_dir.rglob("*") if path.is_file()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        label_counts = count_label_classes(label_path, class_names)
        class_totals.update(label_counts)
        width, height = image_size(image_path)

        row = {
            "split": split,
            "image_id": image_path.stem,
            "image_path": image_path.relative_to(dataset_dir).as_posix(),
            "label_path": label_path.relative_to(dataset_dir).as_posix(),
            "sha256": file_sha256(image_path),
            "width": width,
            "height": height,
            "object_count": sum(label_counts.values()),
        }
        for class_name in class_names.values():
            row[f"objects_{class_name}"] = label_counts.get(class_name, 0)
        rows.append(row)

    return rows, class_totals


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config = load_config(args.config)
    dataset_dir = resolve_path(
        args.dataset_dir or config["processed_dataset_dir"],
        root,
    )
    data_yaml = (
        resolve_path(args.data_yaml, root)
        if args.data_yaml
        else build_ultralytics_yaml(dataset_dir)
    )
    dataset_config = load_config(data_yaml)
    class_names = normalize_names(dataset_config["names"])
    output_dir = ensure_dir(resolve_path(args.output_dir, root))

    split_rows: dict[str, list[dict]] = {}
    class_counts_by_split: dict[str, dict[str, int]] = {}
    for split, split_dir in split_directories(dataset_dir).items():
        rows, class_counts = build_manifest_rows(
            dataset_dir,
            split,
            split_dir,
            class_names,
        )
        pd.DataFrame(rows).to_csv(output_dir / f"{split}_manifest.csv", index=False)
        split_rows[split] = rows
        class_counts_by_split[split] = dict(class_counts)

    all_rows = [row for rows in split_rows.values() for row in rows]
    pd.DataFrame(all_rows).to_csv(output_dir / "all_splits_manifest.csv", index=False)

    image_hashes: dict[str, set[str]] = {
        split: {row["sha256"] for row in rows} for split, rows in split_rows.items()
    }
    exact_overlap = {
        f"{left}_vs_{right}": len(image_hashes[left] & image_hashes[right])
        for index, left in enumerate(image_hashes)
        for right in list(image_hashes)[index + 1 :]
    }
    classes_missing_by_split = {
        split: [
            class_name
            for class_name in class_names.values()
            if class_counts_by_split[split].get(class_name, 0) == 0
        ]
        for split in class_counts_by_split
    }

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml),
        "split_strategy": "preserved Roboflow v1 export split",
        "split_seed": None,
        "images_by_split": {split: len(rows) for split, rows in split_rows.items()},
        "objects_by_split": {
            split: int(sum(counts.values()))
            for split, counts in class_counts_by_split.items()
        },
        "class_counts_by_split": class_counts_by_split,
        "classes_missing_by_split": classes_missing_by_split,
        "exact_cross_split_hash_overlap": exact_overlap,
        "notes": [
            "The original Roboflow split is preserved for local experiments.",
            (
                "The rare edge class is absent from the test split, so class-level "
                "edge test metrics are not meaningful."
            ),
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
