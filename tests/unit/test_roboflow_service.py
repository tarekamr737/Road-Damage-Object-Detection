from __future__ import annotations

import pytest
from PIL import Image

from road_damage.inference import roboflow_service
from road_damage.inference.roboflow_service import (
    RoboflowHostedModel,
    predict_roboflow_image,
    roboflow_predictions_to_detections,
)


def test_roboflow_predictions_to_detections_converts_center_boxes() -> None:
    detections = roboflow_predictions_to_detections(
        [
            {
                "x": 50,
                "y": 40,
                "width": 20,
                "height": 10,
                "confidence": 0.75,
                "class": "pothole",
                "class_id": 5,
            }
        ]
    )

    assert len(detections) == 1
    assert detections[0].class_id == 5
    assert detections[0].class_name == "pothole"
    assert detections[0].confidence == 0.75
    assert detections[0].x_min == 40
    assert detections[0].y_min == 35
    assert detections[0].x_max == 60
    assert detections[0].y_max == 45


def test_direct_http_prediction_converts_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class Response:
        ok = True
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "predictions": [
                    {
                        "x": 20,
                        "y": 20,
                        "width": 10,
                        "height": 10,
                        "confidence": 0.8,
                        "class": "crack",
                        "class_id": 2,
                    }
                ]
            }

    def fake_post(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr(roboflow_service.requests, "post", fake_post)
    model = RoboflowHostedModel(
        workspace="workspace",
        project="project",
        version=1,
        endpoint="https://serverless.roboflow.com/project/1",
        api_key="secret",
    )

    result = predict_roboflow_image(
        model,
        Image.new("RGB", (32, 32), "white"),
        confidence=0.25,
        overlap=0.5,
    )

    assert captured["args"][0] == "https://serverless.roboflow.com/project/1"
    assert captured["kwargs"]["params"]["confidence"] == 25
    assert captured["kwargs"]["params"]["overlap"] == 50
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "crack"


def test_direct_http_prediction_redacts_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        ok = False
        status_code = 402
        text = ""

        @staticmethod
        def json() -> dict:
            return {"message": "secret credit cap exceeded"}

    monkeypatch.setattr(roboflow_service.requests, "post", lambda *_, **__: Response())
    model = RoboflowHostedModel(
        workspace="workspace",
        project="project",
        version=1,
        endpoint="https://serverless.roboflow.com/project/1",
        api_key="secret",
    )

    with pytest.raises(RuntimeError, match="\\[redacted\\] credit cap exceeded"):
        predict_roboflow_image(model, Image.new("RGB", (32, 32), "white"))
