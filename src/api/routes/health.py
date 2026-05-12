# src/api/routes/health.py
"""
Health check endpoints for AITourGuide API.
"""

import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Dict

from src.services.ai_tour_guide import AITourGuide
from src.api.dependencies import get_guide

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Health"])


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(
        ...,
        description="Overall service status",
        example="healthy"
    )
    service: str = Field(
        ...,
        description="Service name",
        example="AITourGuide"
    )
    components: Dict = Field(
        default_factory=dict,
        description="Status of individual components",
        example={
            "model": True,
            "retriever": True,
            "internet_search": True
        }
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service and its components",
)
async def health_check(guide: AITourGuide = Depends(get_guide)):
    """
    Check service health.
    
    Returns the status of the service and all its components.
    """
    try:
        health = guide.health_check()
        return HealthResponse(
            status="healthy" if health.get("ready") else "degraded",
            service="AITourGuide",
            components=health.get("components", {}),
        )
    except Exception as e:
        logger.exception("Health check failed")
        return HealthResponse(
            status="unhealthy",
            service="AITourGuide",
            components={"error": str(e)},
        )
