# src/api/routes/metrics.py
"""Эндпоинт /metrics для Prometheus-скрейпинга."""

import logging
from fastapi import APIRouter, Response
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description=(
        "Экспортирует метрики в формате Prometheus text exposition. "
        "Используется Prometheus-сервером для скрейпинга."
    ),
    response_class=Response,
    include_in_schema=True,
)
async def prometheus_metrics() -> Response:
    """
    Возвращает все зарегистрированные Prometheus-метрики.

    Включает:
    - aitourguide_requests_total{status} — счётчик запросов
    - aitourguide_internet_searches_total — счётчик интернет-поисков
    - aitourguide_retrieval_duration_seconds — гистограмма retrieval
    - aitourguide_vlm_duration_seconds — гистограмма VLM reranking
    - aitourguide_total_duration_seconds — гистограмма полного времени
    - aitourguide_internet_search_duration_seconds — гистограмма поиска
    - aitourguide_confidence_last — confidence последнего запроса
    - aitourguide_index_size — размер FAISS-индекса
    - стандартные Python/process метрики от prometheus_client
    """
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
