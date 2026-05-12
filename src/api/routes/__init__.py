# src/api/routes/__init__.py
"""
API routes for AITourGuide.
"""

from .predict import router as predict_router
from .health import router as health_router
from .info import router as info_router
from .frontend import router as frontend_router

__all__ = ["predict_router", "health_router", "info_router", "frontend_router"]
