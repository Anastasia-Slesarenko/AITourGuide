# src/api/routes/info.py
"""Информационный эндпоинт API."""

from fastapi import APIRouter

router = APIRouter(tags=["Info"])


@router.get(
    "/",
    summary="Информация об API",
    description="Возвращает базовую информацию о сервисе.",
)
async def api_info():
    """Базовая информация о сервисе и доступных эндпоинтах."""
    return {
        "service": "AI Tour Guide API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "predict": "/v1/predict",
            "health": "/v1/health",
        },
    }
