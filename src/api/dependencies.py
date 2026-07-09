# src/api/dependencies.py
"""Зависимости FastAPI для внедрения сервисов."""

from fastapi import HTTPException, Request

from src.services.ai_tour_guide import AITourGuide

# Глобальный экземпляр сервиса
_guide: AITourGuide | None = None


def set_guide(guide: AITourGuide | None) -> None:
    """Устанавливает (или сбрасывает) глобальный экземпляр AITourGuide."""
    global _guide
    _guide = guide


def get_guide_optional() -> AITourGuide | None:
    """Возвращает экземпляр AITourGuide или None если не инициализирован."""
    return _guide


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


async def get_rate_limiter(request: Request):
    """Возвращает экземпляр RateLimiter из состояния приложения."""
    return request.app.state.rate_limiter


async def get_predict_semaphore(request: Request):
    """Возвращает семафор допуска predict-запросов (backpressure)."""
    return request.app.state.predict_semaphore
