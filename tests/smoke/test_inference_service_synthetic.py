from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from road_damage.inference.service import predict_image


class TensorLike:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    def detach(self) -> TensorLike:
        return self

    def cpu(self) -> TensorLike:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    def __init__(
        self,
        xyxy: list[list[float]],
        confidence: list[float],
        classes: list[int],
    ) -> None:
        self.xyxy = TensorLike(xyxy)
        self.conf = TensorLike(confidence)
        self.cls = TensorLike(classes)

    def __len__(self) -> int:
        return len(self.conf.values)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes
        self.speed = {
            "preprocess": 1.0,
            "inference": 2.0,
            "postprocess": 3.0,
        }


class FakeModel:
    names = {0: "crack", 5: "pothole"}

    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes
        self.last_kwargs: dict[str, Any] = {}

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        self.last_kwargs = kwargs
        return [FakeResult(self.boxes)]


def test_predict_image_synthetic_smoke_for_ci() -> None:
    boxes = FakeBoxes(
        xyxy=[[1, 2, 30, 40], [10, 12, 44, 50]],
        confidence=[0.91, 0.72],
        classes=[5, 0],
    )
    model = FakeModel(boxes)
    image = Image.new("RGB", (64, 48), "white")

    result = predict_image(
        model,
        image,
        confidence=0.4,
        iou=0.55,
        image_size=640,
        device="cpu",
        max_det=25,
        classes=[0, 5],
    )

    assert len(result.detections) == 2
    assert result.detections[0].class_name == "pothole"
    assert result.detections[0].confidence == pytest.approx(0.91)
    assert result.detections[1].class_name == "crack"
    assert result.speed_ms == {
        "preprocess": 1.0,
        "inference": 2.0,
        "postprocess": 3.0,
    }

    assert model.last_kwargs["source"].shape == (48, 64, 3)
    assert model.last_kwargs["conf"] == 0.4
    assert model.last_kwargs["iou"] == 0.55
    assert model.last_kwargs["imgsz"] == 640
    assert model.last_kwargs["device"] == "cpu"
    assert model.last_kwargs["max_det"] == 25
    assert model.last_kwargs["classes"] == [0, 5]


def test_predict_image_handles_no_detections() -> None:
    model = FakeModel(FakeBoxes(xyxy=[], confidence=[], classes=[]))
    image = Image.new("RGB", (16, 16), "white")

    result = predict_image(model, image)

    assert result.detections == []


def test_predict_image_handles_many_detections() -> None:
    detection_count = 150
    boxes = FakeBoxes(
        xyxy=[[1, 1, 8, 8] for _ in range(detection_count)],
        confidence=[0.8 for _ in range(detection_count)],
        classes=[0 for _ in range(detection_count)],
    )
    model = FakeModel(boxes)
    image = Image.new("RGB", (16, 16), "white")

    result = predict_image(model, image, max_det=detection_count)

    assert len(result.detections) == detection_count
    assert {detection.class_name for detection in result.detections} == {"crack"}
