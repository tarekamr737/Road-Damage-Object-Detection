from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SUPPORTED_PROVIDERS = {"local_ultralytics", "roboflow_hosted"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_health_errors(config: Mapping[str, Any], app_root: Path) -> list[str]:
    errors: list[str] = []
    provider = str(config.get("provider", "")).strip()

    if provider not in SUPPORTED_PROVIDERS:
        errors.append(
            "Unsupported deployment provider. Expected one of: "
            + ", ".join(sorted(SUPPORTED_PROVIDERS))
            + "."
        )
        return errors

    if provider == "local_ultralytics":
        model_path = str(config.get("model_path", "")).strip()
        if not model_path:
            errors.append("Local deployment is missing `model_path`.")
        else:
            resolved_model_path = app_root / model_path
            if not resolved_model_path.exists():
                errors.append(f"Local model checkpoint was not found: {model_path}")
            else:
                expected_sha256 = str(config.get("model_sha256", "")).strip()
                if (
                    expected_sha256
                    and file_sha256(resolved_model_path) != expected_sha256
                ):
                    errors.append(
                        "Local model checkpoint checksum does not match `model_sha256`."
                    )

    try:
        confidence = float(config.get("default_confidence_threshold", 0))
    except (TypeError, ValueError):
        errors.append("Default confidence threshold must be numeric.")
    else:
        if not 0 < confidence < 1:
            errors.append("Default confidence threshold must be between 0 and 1.")

    try:
        max_upload_mb = int(config.get("max_upload_mb", 0))
    except (TypeError, ValueError):
        errors.append("Maximum upload size must be an integer.")
    else:
        if max_upload_mb <= 0:
            errors.append("Maximum upload size must be greater than zero.")

    try:
        max_image_dimension = int(config.get("max_image_dimension", 0))
    except (TypeError, ValueError):
        errors.append("Maximum image dimension must be an integer.")
    else:
        if max_image_dimension <= 0:
            errors.append("Maximum image dimension must be greater than zero.")

    supported_types = config.get("supported_types", [])
    if not isinstance(supported_types, list) or not supported_types:
        errors.append("At least one supported image type must be configured.")

    return errors


def acceleration_notice(config: Mapping[str, Any]) -> tuple[str, bool]:
    provider = str(config.get("provider", "")).strip()
    if provider != "local_ultralytics":
        return "Hosted inference is configured; local GPU status does not apply.", False

    configured_device = str(config.get("device", "auto")).strip().lower()
    if configured_device == "cpu":
        return (
            "CPU inference is selected in the deployment config. Predictions may "
            "be slower than the GPU benchmark.",
            True,
        )

    try:
        import torch
    except ImportError:
        return (
            "PyTorch is not importable in this environment, so local inference "
            "cannot start.",
            True,
        )

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"GPU acceleration detected: {gpu_name}.", False

    return (
        "GPU acceleration was not detected. The app will use CPU inference and "
        "may be slower than the recorded GPU benchmark.",
        True,
    )
