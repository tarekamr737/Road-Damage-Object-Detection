from __future__ import annotations

from pathlib import Path

from road_damage.data.convert import convert_detection_or_polygon_annotations


def test_polygon_annotation_is_converted_to_detection_box(tmp_path: Path) -> None:
    labels_dir = tmp_path / "train" / "labels"
    labels_dir.mkdir(parents=True)
    label_path = labels_dir / "sample.txt"
    label_path.write_text("2 0.10 0.20 0.50 0.20 0.50 0.60 0.10 0.60\n")

    records, errors = convert_detection_or_polygon_annotations(tmp_path)

    assert not errors
    assert records[0]["source_type"] == "polygon"
    expected = "2 0.3000000000 0.4000000000 0.4000000000 0.4000000000"
    assert label_path.read_text().strip() == expected


def test_detection_annotation_is_preserved_with_normalized_format(
    tmp_path: Path,
) -> None:
    labels_dir = tmp_path / "valid" / "labels"
    labels_dir.mkdir(parents=True)
    label_path = labels_dir / "sample.txt"
    label_path.write_text("1 0.5 0.5 0.25 0.125\n")

    records, errors = convert_detection_or_polygon_annotations(tmp_path)

    assert not errors
    assert records[0]["source_type"] == "bounding_box"
    expected = "1 0.5000000000 0.5000000000 0.2500000000 0.1250000000"
    assert label_path.read_text().strip() == expected
