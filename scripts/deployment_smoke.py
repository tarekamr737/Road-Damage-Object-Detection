from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import tomllib
from PIL import Image

from road_damage.inference.service import load_model, predict_image
from road_damage.inference.startup import deployment_health_errors, file_sha256
from road_damage.training.runtime import load_config
from road_damage.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deployment smoke test for the Streamlit app model."
    )
    parser.add_argument("--config", default="configs/deployment/app.yaml")
    parser.add_argument("--streamlit-config", default=".streamlit/config.toml")
    parser.add_argument("--output", default="reports/deployment/deployment_smoke.json")
    parser.add_argument(
        "--device",
        default=None,
        help="Override inference device for the smoke test, for example `cpu`.",
    )
    return parser.parse_args()


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def load_streamlit_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as file:
        loaded = tomllib.load(file)
    return loaded if isinstance(loaded, dict) else {}


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config = load_config(resolve_path(args.config, root))
    streamlit_config = load_streamlit_config(resolve_path(args.streamlit_config, root))
    output = resolve_path(args.output, root)

    errors = deployment_health_errors(config, root)
    configured_upload_mb = int(config.get("max_upload_mb", 0))
    streamlit_upload_mb = int(
        streamlit_config.get("server", {}).get("maxUploadSize", 0)
    )
    if streamlit_upload_mb != configured_upload_mb:
        errors.append(
            "Streamlit maxUploadSize does not match deployment max_upload_mb."
        )

    provider = str(config.get("provider", ""))
    model_load_ms = None
    first_inference_ms = None
    detections = None
    model_sha256 = ""
    memory_before_mb = rss_mb()
    memory_after_model_mb = None
    memory_after_inference_mb = None

    if provider == "local_ultralytics" and not errors:
        model_path = resolve_path(str(config["model_path"]), root)
        model_sha256 = file_sha256(model_path)
        device = args.device or str(config.get("device", "auto"))
        predict_device = None if device == "auto" else device

        started = time.perf_counter()
        model = load_model(model_path, device=device)
        model_load_ms = (time.perf_counter() - started) * 1000
        memory_after_model_mb = rss_mb()

        image = Image.new("RGB", (640, 640), "#808080")
        started = time.perf_counter()
        result = predict_image(
            model,
            image,
            confidence=float(config["default_confidence_threshold"]),
            iou=0.50,
            image_size=int(config["default_image_size"]),
            device=predict_device,
            max_det=100,
        )
        first_inference_ms = (time.perf_counter() - started) * 1000
        detections = len(result.detections)
        memory_after_inference_mb = rss_mb()

    max_rss_mb = float(config.get("max_process_rss_mb", 0) or 0)
    peak_observed_rss_mb = max(
        value
        for value in [
            memory_before_mb,
            memory_after_model_mb,
            memory_after_inference_mb,
        ]
        if value is not None
    )
    if max_rss_mb and peak_observed_rss_mb > max_rss_mb:
        errors.append("Observed process RSS exceeded max_process_rss_mb.")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "deployment_platform": config.get("deployment_platform", ""),
        "deployment_url": config.get("deployment_url", ""),
        "provider": provider,
        "model_path": config.get("model_path", ""),
        "model_sha256": model_sha256,
        "configured_upload_mb": configured_upload_mb,
        "streamlit_upload_mb": streamlit_upload_mb,
        "max_process_rss_mb": max_rss_mb,
        "memory_before_mb": memory_before_mb,
        "memory_after_model_mb": memory_after_model_mb,
        "memory_after_inference_mb": memory_after_inference_mb,
        "peak_observed_rss_mb": peak_observed_rss_mb,
        "model_load_ms": model_load_ms,
        "first_inference_ms": first_inference_ms,
        "synthetic_detections": detections,
    }
    ensure_dir(output.parent)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
