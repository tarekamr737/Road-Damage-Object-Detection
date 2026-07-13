from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable

from PIL import Image, ImageDraw, ImageFont

from road_damage.inference.service import Detection

CLASS_COLORS: dict[str, str] = {
    "alligator": "#C44536",
    "block": "#F2C14E",
    "crack": "#2F4858",
    "edge": "#4D96FF",
    "longitudinal": "#7A4CB0",
    "pothole": "#0B6E4F",
    "transverse": "#F78154",
}
DEFAULT_COLOR = "#E0E0E0"


def filter_detections(
    detections: Iterable[Detection],
    min_confidence: float = 0.0,
    classes: set[str] | None = None,
    max_detections: int | None = None,
) -> list[Detection]:
    filtered = [
        detection
        for detection in detections
        if detection.confidence >= min_confidence
        and (classes is None or detection.class_name in classes)
    ]
    filtered.sort(key=lambda detection: detection.confidence, reverse=True)
    return filtered[:max_detections] if max_detections else filtered


def detection_rows(
    detections: Iterable[Detection],
) -> list[dict[str, float | int | str]]:
    return [
        {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": round(detection.confidence, 6),
            "x_min": round(detection.x_min, 2),
            "y_min": round(detection.y_min, 2),
            "x_max": round(detection.x_max, 2),
            "y_max": round(detection.y_max, 2),
        }
        for detection in detections
    ]


def detections_to_csv(detections: Iterable[Detection]) -> str:
    rows = detection_rows(detections)
    if not rows:
        return "class_id,class_name,confidence,x_min,y_min,x_max,y_max\r\n"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def detections_to_json(detections: Iterable[Detection]) -> str:
    return json.dumps(detection_rows(detections), indent=2)


def annotate_image(
    image: Image.Image,
    detections: Iterable[Detection],
    class_colors: dict[str, str] | None = None,
) -> Image.Image:
    colors = class_colors or CLASS_COLORS
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    width, height = output.size
    line_width = max(2, int(min(width, height) / 220))
    font = ImageFont.load_default()

    for detection in detections:
        color = colors.get(detection.class_name, DEFAULT_COLOR)
        xy = [
            max(0, detection.x_min),
            max(0, detection.y_min),
            min(width, detection.x_max),
            min(height, detection.y_max),
        ]
        draw.rectangle(xy, outline=color, width=line_width)

        label = f"{detection.class_name} {detection.confidence:.0%}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = xy[0]
        label_y = max(0, xy[1] - text_height - line_width * 2)
        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + text_width + line_width * 3,
                label_y + text_height + line_width * 2,
            ],
            fill=color,
        )
        draw.text(
            (label_x + line_width, label_y + line_width),
            label,
            fill="#111111",
            font=font,
        )

    return output


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
