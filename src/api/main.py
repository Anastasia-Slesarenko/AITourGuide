# src/api/main.py
"""Основное FastAPI-приложение AITourGuide."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.dependencies import get_guide_optional, set_guide
from src.api.middleware import RateLimiter
from src.api.routes import (
    frontend_router,
    gallery_router,
    health_router,
    info_router,
    metrics_router,
    predict_router,
)
from src.core.config import settings
from src.core.logging import setup_logging
from src.core.metrics import METRICS
from src.services.ai_tour_guide import AITourGuide

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения: запуск и остановка."""
    # Настраиваем логирование первым делом — до любых других операций
    setup_logging(
        level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file,
    )
    logger.info("Запуск AI Tour Guide API...")

    # Проверяем наличие файлов индекса
    path_status = settings.validate_paths()
    missing = [p for p, ok in path_status.items() if not ok]
    if missing:
        logger.warning(f"Отсутствуют файлы: {missing}")
        logger.warning("Сервис запустится, но может упасть на первом запросе")

    # Инициализация AITourGuide
    logger.info("Инициализация AITourGuide...")
    try:
        guide = AITourGuide(
            index_dir=str(settings.index_dir_abs),
            vllm_base_url=settings.vllm_base_url,
            vllm_model_name=settings.vllm_model_name,
            vllm_timeout=settings.vllm_timeout,
            vllm_max_retries=settings.vllm_max_retries,
            device=settings.device,
            siglip_model_path=settings.siglip_model_path,
            images_base_dir=settings.images_base_dir,
            top_k_retrieval=settings.top_k_retrieval,
            vlm_threshold=settings.confidence_threshold,
            enable_internet_search=settings.enable_internet_search,
            yc_folder_id=settings.yc_folder_id,
            yc_api_key=settings.yc_api_key,
        )
        set_guide(guide)
        logger.info("AITourGuide инициализирован успешно")

        # Устанавливаем размер FAISS-индекса в Gauge (один раз при старте)
        try:
            index_size = len(guide.retriever.gallery_metadata)
            METRICS.index_size.set(index_size)
            logger.info(f"FAISS index_size gauge = {index_size}")
        except Exception as _e:
            logger.warning(f"Не удалось установить index_size gauge: {_e}")
    except Exception as e:
        logger.error(f"Ошибка инициализации AITourGuide: {e}")
        logger.warning("Сервис запущен в деградированном режиме")

    yield

    # Завершение работы
    logger.info("Остановка AITourGuide...")
    try:
        guide_instance = get_guide_optional()
        if guide_instance:
            await guide_instance.cleanup()
            logger.info("Ресурсы AITourGuide освобождены")
    except Exception as e:
        logger.error(f"Ошибка при остановке: {e}")


app = FastAPI(
    title="AI Tour Guide API",
    description=(
        "🏛️ Сервис распознавания достопримечательностей по фотографиям.\n\n"
        "**Пайплайн:** SigLIP + FAISS → VLM reranking (vLLM) → "
        "интернет-поиск при низкой уверенности."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Rate limiter в состоянии приложения
app.state.rate_limiter = RateLimiter(
    calls=settings.rate_limit_calls,
    period=settings.rate_limit_period,
)

# Шаблоны Jinja2
app.state.templates = Jinja2Templates(directory=str(settings.templates_path_abs))

# Роуты
app.include_router(frontend_router)
app.include_router(info_router)
app.include_router(predict_router)
app.include_router(health_router)
app.include_router(gallery_router)
app.include_router(metrics_router)

# Статические файлы (монтируем после роутеров)
app.mount(
    "/static",
    StaticFiles(directory=str(settings.static_path_abs)),
    name="static",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
