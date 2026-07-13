from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from road_damage.data.dataset import (
    IMAGE_EXTENSIONS,
    image_paths,
    load_yaml,
    normalize_names,
    split_directories,
)

CRITICAL_ERROR_TYPES = {
    "missing_images_directory",
    "missing_labels_directory",
    "corrupt_image",
    "invalid_field_count",
    "non_numeric_label",
    "unknown_class_id",
    "non_finite_box",
    "invalid_box",
}


@dataclass(frozen=True)
class ValidationReport:
    class_names: dict[int, str]
    image_records: list[dict]
    box_records: list[dict]
    errors: list[dict]
    duplicate_records: list[dict]

    @property
    def critical_error_count(self) -> int:
        return sum(
            1 for error in self.errors if error.get("type") in CRITICAL_ERROR_TYPES
        )

    @property
    def exact_cross_split_duplicate_count(self) -> int:
        return len(self.duplicate_records)

    @property
    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.image_records:
            split = str(row["split"])
            counts[split] = counts.get(split, 0) + 1
        return counts

    @property
    def object_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.image_records:
            split = str(row["split"])
            counts[split] = counts.get(split, 0) + int(row.get("objects", 0))
        return counts


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_image_extensions() -> set[str]:
    return IMAGE_EXTENSIONS


def validate_dataset(dataset_dir: Path, data_yaml: Path) -> ValidationReport:
    dataset_config = load_yaml(data_yaml)
    class_names = normalize_names(dataset_config.get("names"))
    split_dirs = split_directories(dataset_dir)

    image_records: list[dict] = []
    box_records: list[dict] = []
    errors: list[dict] = []
    hash_entries: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split, split_dir in split_dirs.items():
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            errors.append(
                {
                    "split": split,
                    "type": "missing_images_directory",
                    "path": str(images_dir),
                }
            )
            continue

        if not labels_dir.exists():
            errors.append(
                {
                    "split": split,
                    "type": "missing_labels_directory",
                    "path": str(labels_dir),
                }
            )
            continue

        for image_path in image_paths(images_dir):
            label_path = labels_dir / f"{image_path.stem}.txt"
            image_valid = True
            width_px: int | None = None
            height_px: int | None = None

            try:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image)
                    image.load()
                    width_px, height_px = image.size
                if width_px <= 0 or height_px <= 0:
                    raise ValueError("Image has invalid dimensions.")
            except Exception as error:
                image_valid = False
                errors.append(
                    {
                        "split": split,
                        "type": "corrupt_image",
                        "path": str(image_path),
                        "detail": repr(error),
                    }
                )

            try:
                digest = file_sha256(image_path)
                hash_entries[digest].append((split, str(image_path)))
            except Exception as error:
                errors.append(
                    {
                        "split": split,
                        "type": "hash_error",
                        "path": str(image_path),
                        "detail": repr(error),
                    }
                )

            label_exists = label_path.exists()
            object_count = 0

            if not label_exists:
                errors.append(
                    {
                        "split": split,
                        "type": "missing_label_file",
                        "path": str(label_path),
                    }
                )
            else:
                lines = [
                    line.strip()
                    for line in label_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

                for line_number, line in enumerate(lines, start=1):
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(
                            {
                                "split": split,
                                "type": "invalid_field_count",
                                "path": str(label_path),
                                "line": line_number,
                                "value_count": len(parts),
                                "content": line,
                            }
                        )
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        x_center, y_center, width, height = map(float, parts[1:])
                    except ValueError:
                        errors.append(
                            {
                                "split": split,
                                "type": "non_numeric_label",
                                "path": str(label_path),
                                "line": line_number,
                                "content": line,
                            }
                        )
                        continue

                    if class_id not in class_names:
                        errors.append(
                            {
                                "split": split,
                                "type": "unknown_class_id",
                                "path": str(label_path),
                                "line": line_number,
                                "class_id": class_id,
                            }
                        )
                        continue

                    values = [x_center, y_center, width, height]
                    if not all(math.isfinite(value) for value in values):
                        errors.append(
                            {
                                "split": split,
                                "type": "non_finite_box",
                                "path": str(label_path),
                                "line": line_number,
                            }
                        )
                        continue

                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    normalized_valid = (
                        0.0 <= x_center <= 1.0
                        and 0.0 <= y_center <= 1.0
                        and 0.0 < width <= 1.0
                        and 0.0 < height <= 1.0
                    )
                    inside_image = (
                        x_min >= -1e-6
                        and y_min >= -1e-6
                        and x_max <= 1.0 + 1e-6
                        and y_max <= 1.0 + 1e-6
                    )

                    if not normalized_valid or not inside_image:
                        errors.append(
                            {
                                "split": split,
                                "type": "invalid_box",
                                "path": str(label_path),
                                "line": line_number,
                                "class_id": class_id,
                                "box": values,
                            }
                        )
                        continue

                    object_count += 1
                    box_records.append(
                        {
                            "split": split,
                            "image": str(image_path),
                            "class_id": class_id,
                            "class_name": class_names[class_id],
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "area": width * height,
                            "aspect_ratio": width / height,
                        }
                    )

            image_records.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "label": str(label_path),
                    "image_valid": image_valid,
                    "label_exists": label_exists,
                    "width": width_px,
                    "height": height_px,
                    "objects": object_count,
                }
            )

    duplicates: list[dict] = []
    for digest, entries in hash_entries.items():
        splits = sorted({split for split, _ in entries})
        if len(splits) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "splits": ", ".join(splits),
                    "count": len(entries),
                    "files": " | ".join(path for _, path in entries),
                }
            )

    return ValidationReport(
        class_names=class_names,
        image_records=image_records,
        box_records=box_records,
        errors=errors,
        duplicate_records=duplicates,
    )
