# src/api/routes/health.py
"""Эндпоинт проверки состояния сервиса."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends

from src.api.dependencies import get_guide
from src.services.ai_tour_guide import AITourGuide

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Health"])


@router.get(
    "/health",
    summary="Проверка состояния",
    description="Проверяет доступность сервиса и его компонентов.",
)
async def health_check(guide: AITourGuide = Depends(get_guide)) -> Dict[str, Any]:
    """Возвращает полный статус сервиса: компоненты, конфиг, метрики."""
    try:
        return await guide.health_check()
    except Exception as e:
        logger.exception("Ошибка при проверке состояния")
        return {
            "status": "unhealthy",
            "ready": False,
            "components": {"error": str(e)},
        }
