# src/api/routes/predict.py
"""Эндпоинт предсказания достопримечательности по изображению."""

import asyncio
import logging
from typing import Dict

from fastapi import (
    APIRouter, File, UploadFile, Form,
    HTTPException, Depends, Request,
)
from pydantic import BaseModel, Field

from src.services.ai_tour_guide import AITourGuide
from src.api.dependencies import get_guide, get_rate_limiter
from src.api.middleware import check_rate_limit, RateLimiter
from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Prediction"])


class PredictionResponse(BaseModel):
    """Ответ с результатом распознавания достопримечательности."""

    name: str = Field(..., description="Название достопримечательности")
    description: str = Field(..., description="Описание достопримечательности")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Уверенность (0–1)"
    )
    unknown: bool = Field(
        False,
        description="True если достопримечательность не распознана"
    )
    source: str = Field(
        ..., description="Источник: retrieval / internet / fallback"
    )
    timing: Dict[str, float] = Field(
        ..., description="Время выполнения этапов в секундах"
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Распознать достопримечательность",
    description=(
        "Загрузите фотографию для определения достопримечательности. "
        f"Лимит: {settings.rate_limit_calls} запросов "
        f"за {settings.rate_limit_period} сек. "
        f"Макс. размер файла: {settings.max_file_size_mb} МБ."
    ),
)
async def predict(
    request: Request,
    image: UploadFile = File(
        ..., description="Фотография достопримечательности"
    ),
    use_internet_search: bool = Form(
        True,
        description="Включить поиск в интернете при низкой уверенности",
    ),
    guide: AITourGuide = Depends(get_guide),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
):
    """
    Распознаёт достопримечательность на фотографии.

    Пайплайн:
    1. Валидация файла
    2. SigLIP + FAISS — поиск кандидатов
    3. VLM reranking через vLLM — выбор лучшего кандидата
    4. Интернет-поиск при низкой уверенности
    """
    await check_rate_limit(request, rate_limiter)

    content = await image.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Файл слишком большой. "
                f"Максимум: {settings.max_file_size_mb} МБ"
            ),
        )

    try:
        result = await asyncio.wait_for(
            guide.predict(
                image_input=content,
                use_internet_search=use_internet_search,
            ),
            timeout=settings.predict_timeout,
        )
    except asyncio.TimeoutError:
        logger.error("Таймаут предсказания")
        raise HTTPException(
            status_code=504,
            detail=f"Таймаут после {settings.predict_timeout} секунд",
        )
    except Exception:
        logger.exception("Ошибка предсказания")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера",
        )

    return PredictionResponse(
        name=result.get("name", ""),
        description=result.get("description", ""),
        confidence=result.get("confidence", 0.0),
        unknown=result.get("unknown", False),
        source=result.get("source", "unknown"),
        timing=result.get("timing", {}),
    )
