# src/api/dependencies.py
"""
FastAPI dependency injection for AITourGuide.
"""

from typing import Optional
from fastapi import HTTPException, Request
from services.ai_tour_guide import AITourGuide


# Global service instance
_guide: Optional[AITourGuide] = None


def set_guide(guide: AITourGuide) -> None:
    """Set the global AITourGuide instance."""
    global _guide
    _guide = guide


async def get_guide() -> AITourGuide:
    """
    Dependency for getting AITourGuide instance.
    
    Raises:
        HTTPException: If service is not initialized
    """
    if _guide is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. AITourGuide is not initialized."
        )
    return _guide


async def get_client_id(request: Request) -> str:
    """
    Get client identifier from request.
    
    Args:
        request: FastAPI request object
    
    Returns:
        Client identifier (IP address or "unknown")
    """
    return request.client.host if request.client else "unknown"


async def get_rate_limiter(request: Request):
    """
    Dependency for getting RateLimiter instance from app state.
    
    Args:
        request: FastAPI request object
    
    Returns:
        RateLimiter instance
    """
    return request.app.state.rate_limiter
