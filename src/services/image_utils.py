# src/services/image_utils.py
"""Работа с изображениями: загрузка в PIL и кодирование в base64 data URI."""

import base64
from io import BytesIO

from PIL import Image


def to_pil_image(image: "Image.Image | str | bytes") -> Image.Image:
    """Конвертирует вход (PIL Image, путь к файлу или байты) в RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(BytesIO(image)).convert("RGB")
    # Строка — путь к файлу
    return Image.open(image).convert("RGB")


def image_to_base64_data_uri(
    image: Image.Image,
    max_size: int = 448,
    quality: int = 85,
) -> str:
    """Конвертирует PIL Image в base64 data URI для OpenAI API.

    Ресайзит до max_size px по большей стороне чтобы не превышать
    лимит payload vLLM.
    """
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    with BytesIO() as buf:
        image.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
