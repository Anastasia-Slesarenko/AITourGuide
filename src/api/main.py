# src/api/main.py
"""
Main FastAPI application for AITourGuide.
Modular version with separated routes and middleware.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.core.config import settings
from src.services.ai_tour_guide import AITourGuide
from src.api.dependencies import set_guide
from src.api.middleware import RateLimiter
from src.api.routes import predict_router, health_router, info_router, frontend_router
from src.core.logging import setup_logging

# Setup logging
setup_logging(
    level=getattr(settings, "log_level", "INFO"),
    log_format=getattr(settings, "log_format", "text"),
    log_file=getattr(settings, "log_file", None),
)

logger = logging.getLogger(__name__)


# ========================================
# LIFESPAN MANAGEMENT
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AI Tour Guide API...")
    
    # Validate configuration paths
    path_status = settings.validate_paths()
    missing_paths = [path for path, exists in path_status.items() if not exists]
    
    if missing_paths:
        logger.warning(f"Missing paths: {missing_paths}")
        logger.warning("Service will start but may fail on first request")
    
    logger.info("Initializing AITourGuide...")
    try:
        guide = AITourGuide(
            model_path=settings.model_path,
            mmproj_path=settings.mmproj_path,
            index_path=settings.index_path,
            facts_db_path=settings.facts_db_path,
            device=settings.device,
        )
        set_guide(guide)
        logger.info("AITourGuide initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AITourGuide: {e}")
        logger.warning("Service will start in degraded mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AITourGuide...")
    try:
        # Access guide through the dependency module
        from src.api import dependencies
        if dependencies._guide:
            dependencies._guide.cleanup()
            logger.info("AITourGuide resources cleaned up")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


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

# ========================================
# MIDDLEWARE
# ========================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Rate limiter (stored in app state for access in dependencies)
app.state.rate_limiter = RateLimiter(
    calls=settings.rate_limit_calls,
    period=settings.rate_limit_period
)

app.state.templates = Jinja2Templates(directory=str(settings.templates_path_abs))

# ========================================
# ROUTES
# ========================================

# Include routers
app.include_router(frontend_router) 
app.include_router(info_router)
app.include_router(predict_router)
app.include_router(health_router)

# ========================================
# STATIC FILES
# ========================================

# Монтируем статику (после роутеров, чтобы не конфликтовать)
app.mount(
    "/static", 
    StaticFiles(directory=str(settings.static_path_abs)), 
    name="static"
)


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
