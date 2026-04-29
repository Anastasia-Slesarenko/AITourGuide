# src/api/routes/info.py
"""
Information endpoints for AITourGuide API.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Info"])


@router.get(
    "/",
    summary="API Information",
    description="Get basic information about the API",
)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "AI Tour Guide API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "predict": "/v1/predict",
            "health": "/v1/health"
        }
    }
