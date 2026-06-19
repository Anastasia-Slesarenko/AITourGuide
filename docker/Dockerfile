# ============================================================
# Stage 1: builder — устанавливаем зависимости
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Системные зависимости для сборки
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements для кэширования слоя
COPY requirements-prod.txt .

# Устанавливаем Python-зависимости (только продакшен-пакеты)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# ============================================================
# Stage 2: runtime
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Минимальные runtime-зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем установленные пакеты из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем исходный код
COPY src/ ./src/
COPY config/ ./config/

# Создаём директории для данных (монтируются через volume)
RUN mkdir -p data/index/siglip data/models logs

# Непривилегированный пользователь
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Переменные окружения по умолчанию
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    DEVICE=cpu \
    RELOAD=false

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
