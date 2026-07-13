from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from PIL import Image, UnidentifiedImageError

from road_damage.inference.service import normalize_image


class UploadedImage(Protocol):
    name: str
    size: int
    type: str | None

    def seek(self, offset: int, whence: int = 0) -> int: ...


MIME_TO_EXTENSION = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def normalized_supported_types(supported_types: Iterable[str]) -> set[str]:
    return {file_type.lower().lstrip(".") for file_type in supported_types}


def uploaded_file_extension(uploaded_file: UploadedImage) -> str:
    return Path(getattr(uploaded_file, "name", "")).suffix.lower().lstrip(".")


def uploaded_mime_extension(uploaded_file: UploadedImage) -> str:
    return MIME_TO_EXTENSION.get(str(getattr(uploaded_file, "type", "")).lower(), "")


def validate_uploaded_image_type(
    uploaded_file: UploadedImage,
    supported_types: Iterable[str],
) -> None:
    supported = normalized_supported_types(supported_types)
    extension = uploaded_file_extension(uploaded_file)
    mime_extension = uploaded_mime_extension(uploaded_file)
    if extension in supported or mime_extension in supported:
        return

    supported_label = ", ".join(sorted(supported))
    raise ValueError(f"Unsupported image type. Supported types: {supported_label}.")


def read_uploaded_image(
    uploaded_file: UploadedImage | None,
    *,
    max_upload_mb: int,
    max_dimension: int,
    supported_types: Iterable[str],
) -> Image.Image:
    if uploaded_file is None:
        raise ValueError("No image was provided.")

    if max_upload_mb <= 0:
        raise ValueError("Maximum upload size must be greater than zero.")

    if uploaded_file.size <= 0:
        raise ValueError("Image file is empty.")

    if uploaded_file.size > max_upload_mb * 1024 * 1024:
        raise ValueError(f"Image is larger than {max_upload_mb} MB.")

    validate_uploaded_image_type(uploaded_file, supported_types)

    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    finally:
        uploaded_file.seek(0)

    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("The uploaded image has invalid dimensions.")

    return normalize_image(image, max_dimension=max_dimension)
