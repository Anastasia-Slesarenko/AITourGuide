# AI Tour Guide — Architecture

## Обзор системы

AI Tour Guide — сервис распознавания достопримечательностей по фотографиям.  
Пайплайн: **SigLIP + FAISS**, затем **Qwen2-VL LoRA reranker (vLLM)**, при низкой уверенности — **Yandex + Wikipedia**.

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
- `_enhance_with_internet_search()` — fallback через `InternetSearchService` (Yandex + Wikipedia + VLM-верификация)

Извлечение P(yes) из logprobs вынесено в `src/services/scoring.py` (`parse_logprobs_p_yes`), калибровка уверенности — в `src/services/calibration.py`.

**Confidence:**  
Решение known/unknown — по сырому `P(yes)` из logprobs vLLM, порог `vlm_threshold = 0.473` (Youden-оптимум на валидации). Пользователю отдаётся isotonic-калиброванный `P(correct)`; калибровка монотонна и порог не меняет.

### LandmarkRetriever (`src/rag/landmark_retriever.py`)

- Модель: `google/siglip-base-patch16-224` (SigLIP, не CLIP)
- Индекс: FAISS `IndexFlatIP` (inner product = cosine similarity на L2-нормализованных векторах)
- Метаданные: `gallery_metadata.json` — список объектов с изображениями, описаниями, RU-названиями

### VLM Reranker (vLLM, внешний сервер)

- Модель: `Qwen2-VL-2B-Instruct` + LoRA (r=16, α=32)
- Квантизация: в проде reranker обслуживается в **fp16** (LoRA-адаптер слит с базовой моделью). Offline-квантизацию (GPTQ / AWQ / GGUF) исследовали и отклонили — на T4 (sm_75) корректных ядер нет, а для one-token reranker выигрыш околонулевой (подробности в README).
- Протокол: OpenAI Chat Completions API
- Промпт: Query Photo + Candidate Photo + вопрос "Are these the same landmark?"
- Ответ: из `logprobs` токенов Yes/No считается `P(yes) = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))`

### Internet Search (`src/services/yandex_search.py`)

1. **YandexSearchService** — синхронный, `requests.Session`, кэш TTL 1 час
2. **WikipediaService** — асинхронный, `aiohttp`, opensearch + fulltext fallback
3. **YandexTranslator** — синхронный, Yandex Translate API v2

### Метрики (`src/core/metrics.py`)

Prometheus-метрики (синглтон `METRICS`):
- `aitourguide_requests_total{status}` — счётчик запросов
- `aitourguide_confidence_last` — confidence последнего запроса
- `aitourguide_retrieval_duration_seconds` — время retrieval
- `aitourguide_vlm_duration_seconds` — время VLM reranking
- `aitourguide_internet_search_duration_seconds` — время интернет-поиска
- `aitourguide_index_size` — размер FAISS-индекса

## Потоки данных

### Путь через retrieval (высокая уверенность)

```
Фото -> SigLIP (0.1–0.3 с) -> FAISS (< 0.05 с) -> VLM reranking (1–5 с)
     -> P(yes) >= threshold -> ответ
```

### Путь через интернет-поиск (низкая уверенность)

```
Фото -> SigLIP -> FAISS -> VLM reranking -> P(yes) < threshold
     -> [параллельно] Yandex Image Search + VLM этап 1 (без подсказок)
     -> VLM этап 2 (с pageTitle) -> Wikipedia -> VLM верификация
     -> Yandex Translate -> ответ
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

> В проде reranker обслуживается в fp16 (см. «VLM Reranker»); offline-квантизацию исследовали и отклонили. Ключевой выигрыш латентности дал именно переход на fp16.

## Мониторинг

- **Prometheus** — метрики на `/metrics`
- **Grafana** — дашборды (`docker/grafana/dashboards/`)
- **Loki + Promtail** — агрегация логов
- **Структурированные логи** — JSON-формат в продакшене, correlation ID на каждый запрос
