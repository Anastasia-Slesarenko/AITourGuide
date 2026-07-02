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

from prometheus_client import Counter, Gauge, Histogram


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
        self.unknown_total = Counter(
            "aitourguide_unknown_total",
            "Число запросов где достопримечательность не распознана "
            "(unknown=True, confidence < threshold)",
        )
        self.source_total = Counter(
            "aitourguide_source_total",
            "Число запросов по источнику итогового ответа",
            ["source"],   # labels: retrieval | internet | fallback
        )

        # ── Гистограммы времени выполнения ────────────────────────────────
        self.retrieval_duration = Histogram(
            "aitourguide_retrieval_duration_seconds",
            "Время SigLIP+FAISS retrieval",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self.vlm_duration = Histogram(
            "aitourguide_vlm_duration_seconds",
            "Время VLM reranking через vLLM",
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
        self.confidence_histogram = Histogram(
            "aitourguide_confidence",
            "Распределение confidence по всем успешным запросам",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.vlm_candidates_count = Histogram(
            "aitourguide_vlm_candidates_count",
            "Число кандидатов переданных в VLM reranking",
            buckets=(1, 2, 3, 5, 7, 10, 15),
        )
        self.image_size_bytes = Histogram(
            "aitourguide_image_size_bytes",
            "Размер входящего изображения в байтах",
            buckets=(
                50_000, 100_000, 250_000, 500_000,
                1_000_000, 2_000_000, 5_000_000, 10_000_000,
            ),
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
