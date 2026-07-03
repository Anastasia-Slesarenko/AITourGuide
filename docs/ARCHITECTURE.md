# AI Tour Guide — Architecture

## Обзор системы

AI Tour Guide — сервис распознавания достопримечательностей по фотографиям.  
Пайплайн: **SigLIP + FAISS** → **Qwen2-VL LoRA reranker (vLLM)** → **Yandex + Wikipedia** (при низкой уверенности).

## Схема архитектуры
![Architecture](architecture_diagram.png)

## Компоненты

### FastAPI Server (`src/api/`)

- Роуты: `/v1/predict`, `/v1/health`, `/v1/info`, `/metrics`, `/`
- `RateLimiter` — in-memory, 10 запросов / 60 секунд на IP
- `python-multipart` для загрузки файлов
- Jinja2 + StaticFiles для фронтенда

### AITourGuide Service (`src/services/ai_tour_guide.py`)

Основной оркестратор пайплайна.

**Ключевые методы:**
- `predict()` — полный пайплайн
- `_retrieve_candidates()` — SigLIP + FAISS
- `_generate_vlm_prediction()` — параллельный VLM reranking
- `_search_internet()` — Yandex + Wikipedia + VLM верификация
- `_parse_logprobs_p_yes()` — извлечение P(yes) из logprobs

**Confidence:**  
`confidence = P(yes)` — вероятность токена "Yes" из logprobs vLLM.  
Порог по умолчанию: `vlm_threshold = 0.5` (оптимальный по экспериментам: 0.547).

### LandmarkRetriever (`src/rag/landmark_retriever.py`)

- Модель: `google/siglip-base-patch16-224` (SigLIP, не CLIP)
- Индекс: FAISS `IndexFlatIP` (inner product = cosine similarity на L2-нормализованных векторах)
- Метаданные: `gallery_metadata.json` — список объектов с изображениями, описаниями, RU-названиями

### VLM Reranker (vLLM, внешний сервер)

- Модель: `Qwen2-VL-2B-Instruct` + LoRA (r=16, α=32)
- Квантизация: online (на лету при загрузке в vLLM) — для снижения VRAM. Оффлайн-варианты (AWQ, GGUF/llama.cpp) проверены, но в прод не пошли.
- Протокол: OpenAI Chat Completions API
- Промпт: Query Photo + Candidate Photo + вопрос "Are these the same landmark?"
- Ответ: `logprobs` для токенов Yes/No → `P(yes) = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))`

### Internet Search (`src/services/yandex_search.py`)

1. **YandexSearchService** — синхронный, `requests.Session`, кэш TTL 1 час
2. **WikipediaService** — асинхронный, `aiohttp`, opensearch + fulltext fallback
3. **YandexTranslator** — синхронный, Yandex Translate API v2

### Метрики (`src/core/metrics.py`)

Prometheus-метрики (синглтон `METRICS`):
- `aitourguide_requests_total{status}` — счётчик запросов
- `aitourguide_confidence` — гистограмма confidence
- `aitourguide_retrieval_duration_seconds` — время retrieval
- `aitourguide_vlm_duration_seconds` — время VLM reranking
- `aitourguide_internet_search_duration_seconds` — время интернет-поиска
- `aitourguide_index_size` — размер FAISS-индекса

## Потоки данных

### Путь через retrieval (высокая уверенность)

```
Фото → SigLIP (0.1–0.3 с) → FAISS (< 0.05 с) → VLM reranking (1–5 с)
     → P(yes) ≥ threshold → ответ
```

### Путь через интернет-поиск (низкая уверенность)

```
Фото → SigLIP → FAISS → VLM reranking → P(yes) < threshold
     → [параллельно] Yandex Image Search + VLM этап 1 (без подсказок)
     → VLM этап 2 (с pageTitle) → Wikipedia → VLM верификация
     → Yandex Translate → ответ
```

## Стек технологий

| Компонент | Технология | Версия |
|-----------|-----------|--------|
| Web Framework | FastAPI | ≥ 0.109 |
| ML Framework | PyTorch | 2.2.2 |
| Vision Encoder | SigLIP (`google/siglip-base-patch16-224`) | via transformers 4.40 |
| VLM | Qwen2-VL-2B-Instruct + LoRA | via vLLM |
| Vector Search | FAISS CPU | ≥ 1.8.0 |
| Async HTTP | httpx (vLLM), aiohttp (Wikipedia) | ≥ 0.27 / ≥ 3.9 |
| Server | Uvicorn | ≥ 0.27 |
| Python | CPython | 3.11 |

## Масштабирование

### Текущие ограничения
- Один инстанс API (нет горизонтального масштабирования)
- In-memory rate limiting (не распределённый)
- vLLM — отдельный сервер, один инстанс

### Возможные улучшения
1. Nginx + несколько инстансов API за балансировщиком
2. Redis для распределённого rate limiting и кэша
3. Несколько vLLM-серверов с балансировкой

> Квантизация модели для снижения VRAM уже реализована — online при загрузке в vLLM (см. «VLM Reranker»).

## Мониторинг

- **Prometheus** — метрики на `/metrics`
- **Grafana** — дашборды (`docker/grafana/dashboards/`)
- **Loki + Promtail** — агрегация логов
- **Структурированные логи** — JSON-формат в продакшене, correlation ID на каждый запрос
