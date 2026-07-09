# src/api/routes/predict.py
"""Эндпоинт предсказания достопримечательности по изображению."""

import asyncio
import io
import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_guide,
    get_predict_semaphore,
    get_rate_limiter,
)
from src.api.middleware import RateLimiter, check_rate_limit
from src.core.config import settings
from src.core.metrics import METRICS
from src.services.ai_tour_guide import AITourGuide

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Prediction"])


class PredictionResponse(BaseModel):
    """Ответ с результатом распознавания достопримечательности."""

    name: str = Field(..., description="Название достопримечательности")
    description: str = Field(..., description="Описание достопримечательности")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Калиброванная уверенность P(верно), 0–1")
    confidence_band: str | None = Field(
        None, description="Уровень уверенности для показа: high / medium / low (None если unknown)"
    )
    unknown: bool = Field(
        False, description="True если достопримечательность не распознана"
    )
    source: str = Field(..., description="Источник: retrieval / internet / fallback")
    timing: dict[str, float] = Field(
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
    image: UploadFile = File(..., description="Фотография достопримечательности"),
    use_internet_search: bool = Form(
        True,
        description="Включить поиск в интернете при низкой уверенности",
    ),
    guide: AITourGuide = Depends(get_guide),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    predict_semaphore: asyncio.Semaphore = Depends(get_predict_semaphore),
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
            detail=(f"Файл слишком большой. Максимум: {settings.max_file_size_mb} МБ"),
        )

    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(
            status_code=400,
            detail="Невалидный файл изображения",
        ) from e

    # Backpressure: пытаемся занять слот обработки. Если за
    # predict_admission_timeout слот не освободился — сервис перегружен,
    # быстро отдаём 503 с Retry-After вместо бесконечного роста очереди.
    try:
        await asyncio.wait_for(
            predict_semaphore.acquire(),
            timeout=settings.predict_admission_timeout,
        )
    except (TimeoutError, asyncio.TimeoutError) as e:
        METRICS.predict_rejected_total.inc()
        logger.warning("Backpressure: predict отклонён (перегрузка)")
        raise HTTPException(
            status_code=503,
            detail="Сервис перегружен, повторите позже",
            headers={"Retry-After": "5"},
        ) from e

    try:
        result = await guide.predict(
            image_input=content,
            use_internet_search=use_internet_search,
        )
    except TimeoutError as e:
        logger.error("Таймаут предсказания")
        raise HTTPException(
            status_code=504,
            detail=f"Таймаут после {settings.predict_timeout} секунд",
        ) from e
    except Exception as e:
        logger.exception("Ошибка предсказания")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера",
        ) from e
    finally:
        predict_semaphore.release()

    return PredictionResponse(
        name=result.get("name", ""),
        description=result.get("description", ""),
        confidence=result.get("confidence", 0.0),
        confidence_band=result.get("confidence_band"),
        unknown=result.get("unknown", False),
        source=result.get("source", "unknown"),
        timing=result.get("timing", {}),
    )
