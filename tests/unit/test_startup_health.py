from __future__ import annotations

from pathlib import Path

from road_damage.inference.startup import deployment_health_errors


def base_config(model_path: str) -> dict:
    return {
        "provider": "local_ultralytics",
        "model_path": model_path,
        "default_confidence_threshold": 0.25,
        "max_upload_mb": 15,
        "max_image_dimension": 4096,
        "supported_types": ["jpg", "jpeg", "png", "webp"],
    }


def test_deployment_health_accepts_existing_local_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")

    assert deployment_health_errors(base_config("model.pt"), tmp_path) == []


def test_deployment_health_reports_missing_local_checkpoint(tmp_path: Path) -> None:
    errors = deployment_health_errors(base_config("missing.pt"), tmp_path)

    assert any("checkpoint was not found" in error for error in errors)


def test_deployment_health_reports_invalid_provider(tmp_path: Path) -> None:
    config = base_config("model.pt")
    config["provider"] = "unknown"

    errors = deployment_health_errors(config, tmp_path)

    assert any("Unsupported deployment provider" in error for error in errors)


def test_deployment_health_reports_invalid_upload_limits(tmp_path: Path) -> None:
    config = base_config("model.pt")
    config["max_upload_mb"] = 0
    config["max_image_dimension"] = "large"
    config["supported_types"] = []

    errors = deployment_health_errors(config, tmp_path)

    assert "Maximum upload size must be greater than zero." in errors
    assert "Maximum image dimension must be an integer." in errors
    assert "At least one supported image type must be configured." in errors
