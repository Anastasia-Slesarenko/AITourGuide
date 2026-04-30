# src/api/routes/predict.py
"""
Prediction endpoints for AITourGuide API.
"""

import asyncio
import logging
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Dict, Annotated

from services.ai_tour_guide import AITourGuide
from src.api.dependencies import get_guide, get_rate_limiter
from src.api.middleware import check_rate_limit, RateLimiter
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Prediction"])


class PredictionResponse(BaseModel):
    """Response model for landmark prediction."""
    name: str = Field(
        ...,
        description="Name of the identified landmark",
        example="Эйфелева башня"
    )
    description: str = Field(
        ...,
        description="Detailed description of the landmark",
        example="Металлическая башня в Париже, построенная в 1889 году..."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction (0.0 to 1.0)",
        example=0.95
    )
    source: str = Field(
        ...,
        description="Source of the prediction (retrieval/internet/fallback)",
        example="retrieval"
    )
    timing: Dict[str, float] = Field(
        ...,
        description="Performance timing breakdown in seconds",
        example={
            "retrieval": 0.15,
            "generation": 2.3,
            "total": 2.45
        }
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict landmark from image",
    description=f"""
    Upload an image to identify the landmark and get detailed information.
    
    **Rate Limit**: {settings.rate_limit_calls} requests per {settings.rate_limit_period} seconds per IP.
    
    **Max File Size**: {settings.max_file_size_mb} MB
    
    **Supported Formats**: JPEG, PNG, GIF, WEBP
    """,
)
async def predict(
    image: UploadFile = File(
        ...,
        description="Image file of the landmark"
    ),
    use_internet_search: bool = Form(
        True,
        description="Enable internet search for unknown landmarks"
    ),
    guide: AITourGuide = Depends(get_guide),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    request: Request = None,
):
    """
    Predict landmark from uploaded image.
    
    This endpoint:
    1. Validates the uploaded image
    2. Uses CLIP+FAISS for candidate retrieval
    3. Generates description using VLM
    4. Falls back to internet search if confidence is low
    """
    # Check rate limit
    await check_rate_limit(request, rate_limiter)
    
    # Validate file size
    content = await image.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb} MB"
        )
    
    # Perform prediction
    try:
        result = await asyncio.wait_for(
            guide.predict(
                image_input=content,
                use_internet_search=use_internet_search
            ),
            timeout=settings.predict_timeout
        )
    except asyncio.TimeoutError:
        logger.error("Prediction timeout")
        raise HTTPException(
            status_code=504,
            detail=f"Prediction timeout after {settings.predict_timeout} seconds"
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )
    
    return PredictionResponse(
        name=result.get("name", ""),
        description=result.get("description", ""),
        confidence=result.get("confidence", 0.0),
        source=result.get("source", "unknown"),
        timing=result.get("timing", {}),
    )
