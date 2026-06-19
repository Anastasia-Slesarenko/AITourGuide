# src/api/dependencies.py
"""Зависимости FastAPI для внедрения сервисов."""

from typing import Optional
from fastapi import HTTPException, Request
from src.services.ai_tour_guide import AITourGuide


# Глобальный экземпляр сервиса
_guide: Optional[AITourGuide] = None


def set_guide(guide: AITourGuide) -> None:
    """Устанавливает глобальный экземпляр AITourGuide."""
    global _guide
    _guide = guide


async def get_guide() -> AITourGuide:
    """
    Зависимость FastAPI: возвращает экземпляр AITourGuide.

    Raises:
        HTTPException 503: если сервис не инициализирован
    """
    if _guide is None:
        raise HTTPException(
            status_code=503,
            detail="Сервис не готов. AITourGuide не инициализирован.",
        )
    return _guide


async def get_client_id(request: Request) -> str:
    """Возвращает IP-адрес клиента из запроса."""
    return request.client.host if request.client else "unknown"


async def get_rate_limiter(request: Request):
    """Возвращает экземпляр RateLimiter из состояния приложения."""
    return request.app.state.rate_limiter
