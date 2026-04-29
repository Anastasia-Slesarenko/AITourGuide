# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional
import uvicorn
from services.ai_tour_guide import AITourGuide
import logging
from contextlib import asynccontextmanager
import asyncio
from PIL import Image
from io import BytesIO
import time
from collections import defaultdict
from config import settings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# RATE LIMITING
# ========================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        now = time.time()
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.period
        ]
        
        if len(self.requests[client_id]) >= self.calls:
            return False
        
        self.requests[client_id].append(now)
        return True
    
    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until next request is allowed"""
        if not self.requests[client_id]:
            return 0
        oldest = min(self.requests[client_id])
        return max(0, int(self.period - (time.time() - oldest)))


rate_limiter = RateLimiter(
    calls=settings.rate_limit_calls,
    period=settings.rate_limit_period
)


# ========================================
# DEPENDENCY INJECTION
# ========================================

# Глобальный объект сервиса
_guide: Optional[AITourGuide] = None


async def get_guide() -> AITourGuide:
    """Dependency для получения AITourGuide instance"""
    if _guide is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. AITourGuide is not initialized."
        )
    return _guide


async def check_rate_limit(request: Request):
    """Dependency для проверки rate limit"""
    client_id = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_id):
        retry_after = rate_limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )


# ========================================
# VALIDATION
# ========================================

def validate_image(content: bytes) -> Image.Image:
    """
    Validate image content and return PIL Image.
    
    Raises:
        HTTPException: If image is invalid
    """
    try:
        image = Image.open(BytesIO(content))
        image.verify()  # Verify it's a valid image
        
        # Reopen after verify (verify closes the file)
        image = Image.open(BytesIO(content))
        
        # Check format
        if image.format.lower() not in ['jpeg', 'jpg', 'png', 'gif', 'webp']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format: {image.format}. "
                       f"Allowed: JPEG, PNG, GIF, WEBP"
            )
        
        return image
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Image validation failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid image."
        )


# ========================================
# LIFESPAN
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global _guide
    
    logger.info("Starting AI Tour Guide API...")
    
    # Validate configuration paths
    path_status = settings.validate_paths()
    missing_paths = [path for path, exists in path_status.items() if not exists]
    
    if missing_paths:
        logger.warning(f"Missing paths: {missing_paths}")
        logger.warning("Service will start but may fail on first request")
    
    logger.info("Initializing AITourGuide...")
    try:
        _guide = AITourGuide(
            model_path=settings.model_path,
            mmproj_path=settings.mmproj_path,
            index_path=settings.index_path,
            facts_db_path=settings.facts_db_path,
            device=settings.device,
        )
        logger.info("AITourGuide initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AITourGuide: {e}")
        logger.warning("Service will start in degraded mode")
    
    yield
    
    # Shutdown
    if _guide:
        logger.info("Shutting down AITourGuide...")
        _guide.cleanup()
        logger.info("AITourGuide resources cleaned up")


# ========================================
# APP INITIALIZATION
# ========================================

app = FastAPI(
    title="AI Tour Guide API",
    description="""
    🏛️ AI-powered landmark recognition and tour guide service.
    
    ## Features
    
    * **Image Recognition**: Upload an image to identify landmarks
    * **RAG-based Retrieval**: Uses CLIP + FAISS for fast candidate retrieval
    * **VLM Generation**: Generates detailed descriptions using vision-language model
    * **Internet Search**: Falls back to Yandex + Wikipedia for unknown landmarks
    * **Rate Limiting**: Protects against abuse
    
    ## Usage
    
    1. Upload an image of a landmark
    2. Optionally enable/disable internet search
    3. Receive landmark name, description, and confidence score
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# ========================================
# MODELS
# ========================================

class PredictionResponse(BaseModel):
    """Response model for landmark prediction"""
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


class HealthResponse(BaseModel):
    """Response model for health check"""
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
    components: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of individual components",
        example={
            "model": True,
            "retriever": True,
            "internet_search": True
        }
    )


class ErrorResponse(BaseModel):
    """Response model for errors"""
    detail: str = Field(
        ...,
        description="Error message",
        example="Rate limit exceeded"
    )


# ========================================
# API ENDPOINTS - V1
# ========================================

@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        504: {"model": ErrorResponse, "description": "Prediction timeout"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Predict landmark from image",
    description="""
    Upload an image to identify the landmark and get detailed information.
    
    **Rate Limit**: {calls} requests per {period} seconds per IP.
    
    **Max File Size**: {max_size} MB
    
    **Supported Formats**: JPEG, PNG, GIF, WEBP
    """.format(
        calls=settings.rate_limit_calls,
        period=settings.rate_limit_period,
        max_size=settings.max_file_size_mb
    ),
    tags=["Prediction"]
)
async def predict_v1(
    image: UploadFile = File(
        ...,
        description="Image file of the landmark"
    ),
    use_internet_search: bool = Form(
        True,
        description="Enable internet search for unknown landmarks"
    ),
    guide: AITourGuide = Depends(get_guide),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Predict landmark from uploaded image.
    
    This endpoint:
    1. Validates the uploaded image
    2. Uses CLIP+FAISS for candidate retrieval
    3. Generates description using VLM
    4. Falls back to internet search if confidence is low
    """
    # Validate file size
    content = await image.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb} MB"
        )
    
    # Validate image content
    try:
        validate_image(content)
    except HTTPException:
        raise
    
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


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service and its components",
    tags=["Health"]
)
async def health_check_v1(guide: AITourGuide = Depends(get_guide)):
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


# ========================================
# LEGACY ENDPOINTS (for backward compatibility)
# ========================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    include_in_schema=False,  # Hide from docs, use v1 instead
    deprecated=True
)
async def predict_legacy(
    image: UploadFile = File(...),
    use_internet_search: bool = Form(True),
    guide: AITourGuide = Depends(get_guide),
    _rate_limit: None = Depends(check_rate_limit),
):
    """Legacy endpoint - use /v1/predict instead"""
    return await predict_v1(image, use_internet_search, guide, _rate_limit)


@app.get(
    "/health",
    response_model=HealthResponse,
    include_in_schema=False,  # Hide from docs, use v1 instead
    deprecated=True
)
async def health_check_legacy(guide: AITourGuide = Depends(get_guide)):
    """Legacy endpoint - use /v1/health instead"""
    return await health_check_v1(guide)


# ========================================
# ROOT ENDPOINT
# ========================================

@app.get(
    "/",
    summary="API Information",
    description="Get basic information about the API",
    tags=["Info"]
)
async def root():
    """Root endpoint with API information"""
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


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
