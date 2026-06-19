# src/api/routes/health.py
"""Эндпоинт проверки состояния сервиса."""

import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any

from src.services.ai_tour_guide import AITourGuide
from src.api.dependencies import get_guide

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Health"])


class HealthResponse(BaseModel):
    """Ответ на запрос проверки состояния."""

    status: str
    service: str
    components: Dict[str, Any]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния",
    description="Проверяет доступность сервиса и его компонентов.",
)
async def health_check(guide: AITourGuide = Depends(get_guide)):
    """Возвращает статус сервиса и всех его компонентов."""
    try:
        health = await guide.health_check()
        components = health.get("components", {})
        if not isinstance(components, dict):
            components = {}
        return HealthResponse(
            status="healthy" if health.get("ready") else "degraded",
            service="AITourGuide",
            components=components,
        )
    except Exception as e:
        logger.exception("Ошибка при проверке состояния")
        return HealthResponse(
            status="unhealthy",
            service="AITourGuide",
            components={"error": str(e)},
        )
