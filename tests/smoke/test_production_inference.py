from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from road_damage.inference.service import Detection, load_model, predict_image

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models" / "exports" / "production_road_damage_model.pt"
VALID_IMAGES = (
    ROOT / "data" / "processed" / "road-damage-detection-bbox-v1" / "valid" / "images"
)


def sortable_detection(detection: Detection) -> tuple:
    return (
        detection.class_id,
        round(detection.confidence, 6),
        round(detection.x_min, 3),
        round(detection.y_min, 3),
        round(detection.x_max, 3),
        round(detection.y_max, 3),
    )


def detection_vector(detection: Detection) -> np.ndarray:
    return np.asarray(
        [
            detection.class_id,
            detection.confidence,
            detection.x_min,
            detection.y_min,
            detection.x_max,
            detection.y_max,
        ],
        dtype=float,
    )


@pytest.mark.skipif(
    not CHECKPOINT.exists() or not VALID_IMAGES.exists(),
    reason="local production checkpoint or validation images are not available",
)
def test_production_checkpoint_predictions_are_deterministic() -> None:
    pytest.importorskip("ultralytics")
    image_path = sorted(VALID_IMAGES.glob("*"))[0]
    image = Image.open(image_path).convert("RGB")
    model = load_model(CHECKPOINT)

    first = predict_image(
        model,
        image,
        confidence=0.25,
        image_size=640,
        device="cpu",
    )
    second = predict_image(
        model,
        image,
        confidence=0.25,
        image_size=640,
        device="cpu",
    )

    first_detections = sorted(first.detections, key=sortable_detection)
    second_detections = sorted(second.detections, key=sortable_detection)
    assert len(first_detections) == len(second_detections)

    for first_detection, second_detection in zip(
        first_detections,
        second_detections,
        strict=True,
    ):
        np.testing.assert_allclose(
            detection_vector(first_detection),
            detection_vector(second_detection),
            rtol=1e-4,
            atol=1e-3,
        )
