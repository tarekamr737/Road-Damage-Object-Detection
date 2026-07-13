from __future__ import annotations

import io

import pytest
from PIL import Image

from road_damage.inference.input_validation import read_uploaded_image


class FakeUpload(io.BytesIO):
    def __init__(
        self,
        data: bytes,
        *,
        name: str = "road.jpg",
        content_type: str = "image/jpeg",
    ) -> None:
        super().__init__(data)
        self.name = name
        self.type = content_type
        self.size = len(data)


def image_bytes(format_name: str = "JPEG", size: tuple[int, int] = (32, 24)) -> bytes:
    image = Image.new("RGB", size, "white")
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("name", "content_type", "format_name"),
    [
        ("road.jpg", "image/jpeg", "JPEG"),
        ("road.jpeg", "image/jpeg", "JPEG"),
        ("road.png", "image/png", "PNG"),
        ("road.webp", "image/webp", "WEBP"),
    ],
)
def test_read_uploaded_image_accepts_supported_types(
    name: str,
    content_type: str,
    format_name: str,
) -> None:
    upload = FakeUpload(
        image_bytes(format_name),
        name=name,
        content_type=content_type,
    )

    image = read_uploaded_image(
        upload,
        max_upload_mb=1,
        max_dimension=4096,
        supported_types=["jpg", "jpeg", "png", "webp"],
    )

    assert image.mode == "RGB"
    assert image.size == (32, 24)


def test_read_uploaded_image_accepts_very_small_image() -> None:
    upload = FakeUpload(image_bytes(size=(1, 1)))

    image = read_uploaded_image(
        upload,
        max_upload_mb=1,
        max_dimension=4096,
        supported_types=["jpg"],
    )

    assert image.size == (1, 1)


def test_read_uploaded_image_resizes_very_large_image() -> None:
    upload = FakeUpload(image_bytes(size=(120, 40)))

    image = read_uploaded_image(
        upload,
        max_upload_mb=1,
        max_dimension=60,
        supported_types=["jpg"],
    )

    assert image.size == (60, 20)


def test_read_uploaded_image_rejects_oversized_upload() -> None:
    upload = FakeUpload(image_bytes(), name="road.jpg")
    upload.size = 2 * 1024 * 1024

    with pytest.raises(ValueError, match="larger than 1 MB"):
        read_uploaded_image(
            upload,
            max_upload_mb=1,
            max_dimension=4096,
            supported_types=["jpg"],
        )


def test_read_uploaded_image_rejects_unsupported_type() -> None:
    upload = FakeUpload(image_bytes(), name="road.gif", content_type="image/gif")

    with pytest.raises(ValueError, match="Unsupported image type"):
        read_uploaded_image(
            upload,
            max_upload_mb=1,
            max_dimension=4096,
            supported_types=["jpg", "png"],
        )


def test_read_uploaded_image_rejects_renamed_non_image() -> None:
    upload = FakeUpload(b"not actually an image", name="road.jpg")

    with pytest.raises(ValueError, match="not a readable image"):
        read_uploaded_image(
            upload,
            max_upload_mb=1,
            max_dimension=4096,
            supported_types=["jpg"],
        )
