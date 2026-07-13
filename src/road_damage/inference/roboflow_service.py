from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from road_damage.inference.service import Detection, PredictionResult, normalize_image
from road_damage.utils.env import load_dotenv


@dataclass(frozen=True)
class RoboflowHostedModel:
    workspace: str
    project: str
    version: int
    model: Any | None = None
    endpoint: str | None = None
    api_key: str | None = field(default=None, repr=False)


def load_roboflow_model(
    workspace: str,
    project: str,
    version: int,
    api_key: str | None = None,
    env_path: str | Path = ".env",
    endpoint: str | None = None,
) -> RoboflowHostedModel:
    load_dotenv(env_path)
    resolved_key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not resolved_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required for Roboflow hosted models.")

    if endpoint:
        return RoboflowHostedModel(
            workspace=workspace,
            project=project,
            version=version,
            endpoint=endpoint,
            api_key=resolved_key,
        )

    from roboflow import Roboflow

    model = (
        Roboflow(api_key=resolved_key)
        .workspace(workspace)
        .project(project)
        .version(version)
        .model
    )
    if model is None:
        endpoint = f"https://serverless.roboflow.com/{project}/{version}"
    return RoboflowHostedModel(
        workspace=workspace,
        project=project,
        version=version,
        model=model,
        endpoint=endpoint,
        api_key=resolved_key,
    )


def roboflow_predictions_to_detections(
    predictions: list[dict[str, Any]],
) -> list[Detection]:
    detections: list[Detection] = []
    for prediction in predictions:
        x_center = float(prediction["x"])
        y_center = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        class_id = prediction.get("class_id", -1)
        detections.append(
            Detection(
                class_id=int(class_id) if class_id is not None else -1,
                class_name=str(prediction.get("class", class_id)),
                confidence=float(prediction["confidence"]),
                x_min=x_center - width / 2,
                y_min=y_center - height / 2,
                x_max=x_center + width / 2,
                y_max=y_center + height / 2,
            )
        )
    return detections


def _image_to_base64(image: str | Path | Image.Image | np.ndarray) -> bytes | None:
    if isinstance(image, Image.Image):
        buffer = BytesIO()
        normalize_image(image).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue())

    if isinstance(image, np.ndarray):
        buffer = BytesIO()
        normalize_image(Image.fromarray(image)).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue())

    path_or_url = str(image)
    if path_or_url.startswith(("http://", "https://")):
        return None
    return base64.b64encode(Path(path_or_url).read_bytes())


def _clean_error_message(message: str, api_key: str | None) -> str:
    return message.replace(api_key, "[redacted]") if api_key else message


def _predict_direct_http(
    model: RoboflowHostedModel,
    image: str | Path | Image.Image | np.ndarray,
    confidence: float,
    overlap: float,
) -> dict[str, Any]:
    if not model.endpoint:
        raise RuntimeError("Roboflow endpoint is not configured.")
    if not model.api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required for Roboflow hosted models.")

    image_body = _image_to_base64(image)
    params: dict[str, Any] = {
        "api_key": model.api_key,
        "confidence": int(confidence * 100),
        "overlap": int(overlap * 100),
        "format": "json",
    }
    if image_body is None:
        params["image"] = str(image)

    response = requests.post(
        model.endpoint,
        params=params,
        data=image_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        message = _clean_error_message(response.text[:300], model.api_key)
        raise RuntimeError(
            f"Roboflow inference returned HTTP {response.status_code}: {message}"
        ) from exc

    if not response.ok:
        raw_message = payload.get("message") or payload.get("error") or str(payload)
        message = _clean_error_message(str(raw_message), model.api_key)
        raise RuntimeError(
            f"Roboflow inference returned HTTP {response.status_code}: {message}"
        )

    return payload


def predict_roboflow_image(
    model: RoboflowHostedModel,
    image: str | Path | Image.Image | np.ndarray,
    confidence: float = 0.25,
    overlap: float = 0.5,
) -> PredictionResult:
    started = time.perf_counter()
    if model.model is not None:
        if isinstance(image, Image.Image):
            source: str | np.ndarray = np.asarray(normalize_image(image))
        elif isinstance(image, Path):
            source = str(image)
        else:
            source = image
        result = model.model.predict(
            source,
            confidence=int(confidence * 100),
            overlap=int(overlap * 100),
        )
        raw_json = result.json()
    else:
        raw_json = _predict_direct_http(model, image, confidence, overlap)
    inference_ms = (time.perf_counter() - started) * 1000
    return PredictionResult(
        detections=roboflow_predictions_to_detections(raw_json.get("predictions", [])),
        inference_ms=inference_ms,
        speed_ms={"hosted_request": inference_ms},
        raw_result=raw_json,
    )
