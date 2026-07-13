from __future__ import annotations

from PIL import Image

from road_damage.inference.annotation import (
    annotate_image,
    detections_to_csv,
    filter_detections,
)
from road_damage.inference.service import Detection


def make_detection(class_name: str, confidence: float) -> Detection:
    return Detection(
        class_id=1,
        class_name=class_name,
        confidence=confidence,
        x_min=10,
        y_min=10,
        x_max=30,
        y_max=30,
    )


def test_filter_detections_applies_confidence_classes_and_limit() -> None:
    detections = [
        make_detection("pothole", 0.91),
        make_detection("crack", 0.88),
        make_detection("pothole", 0.42),
    ]

    filtered = filter_detections(
        detections,
        min_confidence=0.5,
        classes={"pothole", "crack"},
        max_detections=1,
    )

    assert [detection.class_name for detection in filtered] == ["pothole"]


def test_filter_detections_handles_empty_input() -> None:
    assert filter_detections([], min_confidence=0.9) == []


def test_detections_to_csv_writes_empty_header() -> None:
    assert detections_to_csv([]).startswith("class_id,class_name,confidence")


def test_annotate_image_returns_same_size_copy() -> None:
    image = Image.new("RGB", (80, 60), "white")
    annotated = annotate_image(image, [make_detection("pothole", 0.75)])

    assert annotated.size == image.size
    assert annotated is not image
