from __future__ import annotations

import math
from pathlib import Path


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def convert_detection_or_polygon_annotations(
    dataset_dir: Path,
    split_names: tuple[str, ...] = ("train", "valid", "val", "test"),
) -> tuple[list[dict], list[dict]]:
    """Convert YOLO polygon labels to YOLO detection boxes in place.

    Lines with five fields are normalized and preserved. Longer polygon rows are
    converted to the tight bounding rectangle expected by detection training.
    """
    records: list[dict] = []
    errors: list[dict] = []

    for split_name in split_names:
        labels_dir = dataset_dir / split_name / "labels"
        if not labels_dir.exists():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            converted_lines: list[str] = []
            lines = label_path.read_text(encoding="utf-8").splitlines()

            for line_number, raw_line in enumerate(lines, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()

                try:
                    class_id = int(float(parts[0]))
                except (ValueError, IndexError):
                    errors.append(
                        {
                            "file": str(label_path),
                            "line": line_number,
                            "reason": "invalid_class_id",
                            "content": line,
                        }
                    )
                    continue

                if len(parts) == 5:
                    try:
                        x_center, y_center, width, height = map(float, parts[1:])
                    except ValueError:
                        errors.append(
                            {
                                "file": str(label_path),
                                "line": line_number,
                                "reason": "non_numeric_detection_box",
                                "content": line,
                            }
                        )
                        continue

                    values = [x_center, y_center, width, height]
                    if not _finite(values):
                        errors.append(
                            {
                                "file": str(label_path),
                                "line": line_number,
                                "reason": "non_finite_detection_box",
                                "content": line,
                            }
                        )
                        continue

                    converted_lines.append(
                        f"{class_id} {x_center:.10f} {y_center:.10f} "
                        f"{width:.10f} {height:.10f}"
                    )
                    records.append(
                        {
                            "file": str(label_path),
                            "line": line_number,
                            "source_type": "bounding_box",
                            "class_id": class_id,
                        }
                    )
                    continue

                coordinate_values = parts[1:]
                if len(coordinate_values) < 6 or len(coordinate_values) % 2 != 0:
                    errors.append(
                        {
                            "file": str(label_path),
                            "line": line_number,
                            "reason": "invalid_polygon_coordinate_count",
                            "value_count": len(parts),
                            "content": line,
                        }
                    )
                    continue

                try:
                    coordinates = list(map(float, coordinate_values))
                except ValueError:
                    errors.append(
                        {
                            "file": str(label_path),
                            "line": line_number,
                            "reason": "non_numeric_polygon",
                            "content": line,
                        }
                    )
                    continue

                if not _finite(coordinates):
                    errors.append(
                        {
                            "file": str(label_path),
                            "line": line_number,
                            "reason": "non_finite_polygon",
                            "content": line,
                        }
                    )
                    continue

                x_values = [min(1.0, max(0.0, value)) for value in coordinates[0::2]]
                y_values = [min(1.0, max(0.0, value)) for value in coordinates[1::2]]
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                width = x_max - x_min
                height = y_max - y_min

                if width <= 0 or height <= 0:
                    errors.append(
                        {
                            "file": str(label_path),
                            "line": line_number,
                            "reason": "degenerate_polygon_box",
                            "content": line,
                        }
                    )
                    continue

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                converted_lines.append(
                    f"{class_id} {x_center:.10f} {y_center:.10f} "
                    f"{width:.10f} {height:.10f}"
                )
                records.append(
                    {
                        "file": str(label_path),
                        "line": line_number,
                        "source_type": "polygon",
                        "class_id": class_id,
                        "polygon_points": len(x_values),
                        "bbox_x_center": x_center,
                        "bbox_y_center": y_center,
                        "bbox_width": width,
                        "bbox_height": height,
                    }
                )

            output_text = "\n".join(converted_lines)
            if output_text:
                output_text += "\n"
            label_path.write_text(output_text, encoding="utf-8")

    return records, errors
