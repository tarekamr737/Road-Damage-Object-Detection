from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from road_damage.utils.ultralytics_env import configure_ultralytics_env


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class PredictionResult:
    detections: list[Detection]
    inference_ms: float
    speed_ms: dict[str, float]
    raw_result: Any


def load_model(checkpoint: str | Path, device: str | int = "auto") -> Any:
    configure_ultralytics_env()
    from ultralytics import YOLO

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    model = YOLO(str(path))
    if device != "auto":
        model.to(device)
    return model


def normalize_image(
    image: Image.Image,
    max_dimension: int | None = None,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max_dimension is not None and max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension))
    return image


def predict_image(
    model: Any,
    image: Image.Image | np.ndarray,
    confidence: float = 0.25,
    iou: float | None = None,
    image_size: int = 640,
    device: str | int | None = None,
    max_det: int | None = None,
    classes: list[int] | None = None,
) -> PredictionResult:
    source = (
        np.asarray(normalize_image(image)) if isinstance(image, Image.Image) else image
    )
    kwargs: dict[str, Any] = {
        "source": source,
        "conf": confidence,
        "imgsz": image_size,
        "verbose": False,
    }
    if iou is not None:
        kwargs["iou"] = iou
    if device is not None:
        kwargs["device"] = device
    if max_det is not None:
        kwargs["max_det"] = max_det
    if classes is not None:
        kwargs["classes"] = classes

    started = time.perf_counter()
    result = model.predict(**kwargs)[0]
    inference_ms = (time.perf_counter() - started) * 1000

    names = (
        model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    )
    detections: list[Detection] = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidence_values = result.boxes.conf.detach().cpu().numpy()
        classes_out = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(
            boxes,
            confidence_values,
            classes_out,
            strict=True,
        ):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=str(names.get(int(class_id), int(class_id))),
                    confidence=float(score),
                    x_min=float(box[0]),
                    y_min=float(box[1]),
                    x_max=float(box[2]),
                    y_max=float(box[3]),
                )
            )

    return PredictionResult(
        detections=detections,
        inference_ms=inference_ms,
        speed_ms={key: float(value) for key, value in result.speed.items()},
        raw_result=result,
    )
