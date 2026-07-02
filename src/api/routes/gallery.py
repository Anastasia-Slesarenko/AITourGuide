# src/api/routes/gallery.py
"""Эндпоинт для отдачи изображений из галереи достопримечательностей."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Gallery"])


@router.get("/gallery-image/{image_path:path}")
async def gallery_image(image_path: str) -> FileResponse:
    """
    Отдаёт изображение из галереи достопримечательностей.

    image_path — имя файла (или относительный путь) внутри images_base_dir.
    Защита от path traversal: разрешены только файлы внутри images_base_dir.
    """
    base = Path(settings.images_base_dir)
    if not base.is_absolute():
        base = settings.project_root / base

    # Защита от path traversal
    try:
        full_path = (base / image_path).resolve()
        full_path.relative_to(base.resolve())
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Недопустимый путь") from e

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Изображение не найдено: {image_path}",
        )

    suffix = full_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_types.get(suffix, "image/jpeg")

    return FileResponse(
        path=str(full_path),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )
