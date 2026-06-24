# src/core/metrics.py
"""
Глобальные Prometheus-метрики AITourGuide.

Все метрики определены здесь как модульные синглтоны — они регистрируются
в REGISTRY ровно один раз при импорте модуля. Это предотвращает
ValueError: Duplicated timeseries при повторном создании AITourGuide
(например, в тестах или при hot-reload).

Использование:
    from src.core.metrics import METRICS
    METRICS.requests_total.labels(status="success").inc()
"""

from prometheus_client import Counter, Histogram, Gauge


class _AITourGuideMetrics:
    """
    Контейнер Prometheus-метрик сервиса.

    Инстанцируется один раз на уровне модуля (METRICS).
    Не создавать напрямую — использовать импортированный синглтон.
    """

    def __init__(self) -> None:
        # ── Счётчики запросов ──────────────────────────────────────────────
        self.requests_total = Counter(
            "aitourguide_requests_total",
            "Общее число вызовов predict()",
            ["status"],   # labels: success | error
        )
        self.internet_searches_total = Counter(
            "aitourguide_internet_searches_total",
            "Число запросов, для которых был выполнен интернет-поиск",
        )

        # ── Гистограммы времени выполнения ────────────────────────────────
        self.retrieval_duration = Histogram(
            "aitourguide_retrieval_duration_seconds",
            "Время SigLIP+FAISS retrieval",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self.vlm_duration = Histogram(
            "aitourguide_vlm_duration_seconds",
            "Время VLM reranking через SGLang",
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        self.total_duration = Histogram(
            "aitourguide_total_duration_seconds",
            "Полное время обработки одного запроса",
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        )
        self.internet_search_duration = Histogram(
            "aitourguide_internet_search_duration_seconds",
            "Время интернет-поиска (Yandex + Wikipedia)",
            buckets=(1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 90.0),
        )

        # ── Gauge: текущие значения ───────────────────────────────────────
        self.confidence_last = Gauge(
            "aitourguide_confidence_last",
            "Confidence последнего успешного запроса",
        )
        self.index_size = Gauge(
            "aitourguide_index_size",
            "Число объектов в FAISS-индексе галереи",
        )


# Единственный экземпляр — импортировать отсюда
METRICS = _AITourGuideMetrics()
