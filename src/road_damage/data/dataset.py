from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_validation_folder(dataset_dir: Path) -> str:
    if (dataset_dir / "valid").exists():
        return "valid"
    if (dataset_dir / "val").exists():
        return "val"
    raise FileNotFoundError(f"Neither 'valid' nor 'val' exists inside {dataset_dir}.")


def split_directories(dataset_dir: Path) -> dict[str, Path]:
    validation_folder = find_validation_folder(dataset_dir)
    return {
        "train": dataset_dir / "train",
        "val": dataset_dir / validation_folder,
        "test": dataset_dir / "test",
    }


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def verify_dataset_structure(dataset_dir: Path) -> dict[str, int]:
    split_dirs = split_directories(dataset_dir)
    counts: dict[str, int] = {}
    missing: list[str] = []

    for split, split_dir in split_dirs.items():
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists():
            missing.append(str(images_dir))
        if not labels_dir.exists():
            missing.append(str(labels_dir))
        counts[split] = len(image_paths(images_dir)) if images_dir.exists() else 0

    if missing:
        raise FileNotFoundError(
            "The processed dataset is incomplete. Missing:\n" + "\n".join(missing)
        )

    empty_splits = [split for split, count in counts.items() if count <= 0]
    if empty_splits:
        raise RuntimeError(f"One or more dataset splits contain no images: {counts}")

    return counts


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return data


def normalize_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise ValueError("Dataset YAML must contain class names as a list or mapping.")


def find_dataset_yaml(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "data_detection.yaml",
        dataset_dir / "data_detection_drive.yaml",
        dataset_dir / "data.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No dataset YAML was found inside {dataset_dir}.")


def build_ultralytics_yaml(dataset_dir: Path, output_yaml: Path | None = None) -> Path:
    source_yaml = find_dataset_yaml(dataset_dir)
    source_config = load_yaml(source_yaml)
    names = source_config.get("names")
    if not names:
        raise ValueError(f"The dataset YAML {source_yaml} does not contain names.")

    validation_folder = find_validation_folder(dataset_dir)
    output = output_yaml or dataset_dir / "data_detection.yaml"
    local_config = {
        "train": "train/images",
        "val": f"{validation_folder}/images",
        "test": "test/images",
        "names": names,
    }
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(local_config, file, sort_keys=False, allow_unicode=True)
    return output
