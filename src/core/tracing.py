# src/core/tracing.py
"""
Инициализация OpenTelemetry трейсинга для AITourGuide.

Экспортирует трейсы в Grafana Tempo через OTLP/gRPC.
Настраивается через переменные окружения:
    OTEL_EXPORTER_OTLP_ENDPOINT  — адрес Tempo (default: http://tempo:4317)
    OTEL_SERVICE_NAME            — имя сервиса (default: aitourguide)
    OTEL_TRACES_SAMPLER_ARG      — вероятность сэмплирования 0.0–1.0 (default: 1.0)

Использование:
    from src.core.tracing import setup_tracing, get_tracer
    setup_tracing()                        # вызвать один раз при старте
    tracer = get_tracer()
    with tracer.start_as_current_span("my_span") as span:
        span.set_attribute("key", "value")
"""

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)

logger = logging.getLogger(__name__)

_tracer: Optional[trace.Tracer] = None


def setup_tracing(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    sample_rate: Optional[float] = None,
) -> None:
    """
    Инициализирует OTel TracerProvider с OTLP/gRPC экспортом в Tempo.

    Безопасно вызывать повторно — повторный вызов игнорируется.

    Args:
        service_name: Имя сервиса (переопределяет OTEL_SERVICE_NAME)
        otlp_endpoint: Адрес Tempo gRPC (переопределяет
            OTEL_EXPORTER_OTLP_ENDPOINT)
        sample_rate: Вероятность сэмплирования 0.0–1.0 (переопределяет
            OTEL_TRACES_SAMPLER_ARG)
    """
    global _tracer

    if _tracer is not None:
        return  # уже инициализирован

    _service_name = (
        service_name
        or os.getenv("OTEL_SERVICE_NAME", "aitourguide")
    )
    _endpoint = (
        otlp_endpoint
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
    )
    _sample_rate = float(
        sample_rate
        if sample_rate is not None
        else os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0")
    )

    resource = Resource(attributes={SERVICE_NAME: _service_name})

    # Сэмплер: TraceIdRatioBased при sample_rate < 1.0, иначе ALWAYS_ON
    if _sample_rate < 1.0:
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        sampler = TraceIdRatioBased(_sample_rate)
    else:
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON
        sampler = ALWAYS_ON

    provider = TracerProvider(resource=resource, sampler=sampler)

    try:
        exporter = OTLPSpanExporter(endpoint=_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(
            f"OTel трейсинг: сервис={_service_name}, "
            f"endpoint={_endpoint}, sample_rate={_sample_rate}"
        )
    except Exception as e:
        logger.warning(
            f"Не удалось подключить OTLP экспортер ({_endpoint}): {e}. "
            "Трейсы не будут отправляться."
        )

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(_service_name)
    logger.info("OpenTelemetry TracerProvider инициализирован")


def get_tracer() -> trace.Tracer:
    """
    Возвращает глобальный tracer.

    Если setup_tracing() не был вызван — возвращает no-op tracer
    (трейсы не записываются, но код не падает).
    """
    if _tracer is not None:
        return _tracer
    return trace.get_tracer("aitourguide")
