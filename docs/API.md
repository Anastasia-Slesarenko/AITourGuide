# AI Tour Guide — API Reference

## Base URL

```
http://localhost:8000
```

## Аутентификация

API не требует аутентификации. Применяется rate limiting по IP.

## Rate Limiting

- **Лимит**: 10 запросов / 60 секунд на IP
- **Ответ при превышении**: HTTP 429
- **Заголовок**: `Retry-After` — секунды до следующего разрешённого запроса

---

## Эндпоинты

### POST /v1/predict

Распознать достопримечательность на фотографии.

**Content-Type**: `multipart/form-data`

**Параметры:**

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|---------|
| `file` | file | ✅ | Изображение (JPEG, PNG, GIF, WEBP), макс. 10 MB |
| `use_internet_search` | boolean | ❌ | Включить интернет-поиск при низкой уверенности (default: `true`) |

**Пример запроса:**

```bash
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@photo.jpg" \
  -F "use_internet_search=true"
```

**Ответ 200 OK:**

```json
{
  "name": "Исаакиевский собор",
  "description": "Исаакиевский собор — крупнейший православный храм Санкт-Петербурга...",
  "confidence": 0.923,
  "source": "retrieval",
  "unknown": false,
  "winner_landmark_id": "saint_isaac_cathedral",
  "winner_images": ["saint_isaac_1.jpg", "saint_isaac_2.jpg"],
  "retrieved_names": ["Исаакиевский собор", "Казанский собор", "..."],
  "retrieved_scores": [0.91, 0.74, "..."],
  "retrieved_p_yes": [0.923, 0.12, "..."],
  "search_query": null,
  "timing": {
    "image_load": 0.01,
    "retrieval": 0.18,
    "vlm_generation": 2.45,
    "total": 2.64
  }
}
```

**Поля ответа:**

| Поле | Тип | Описание |
|------|-----|---------|
| `name` | string | Название достопримечательности (RU). Пустая строка если `unknown=true` |
| `description` | string | Описание (RU). Пустая строка если `unknown=true` |
| `confidence` | float | P(yes) от VLM reranker, 0.0–1.0 |
| `source` | string | `retrieval` / `internet` / `fallback` |
| `unknown` | boolean | `true` если confidence < threshold после всех этапов |
| `winner_landmark_id` | string | ID победителя в галерее |
| `winner_images` | array | Пути к изображениям победителя из галереи |
| `retrieved_names` | array | Названия top-K кандидатов (RU) |
| `retrieved_scores` | array | FAISS-скоры кандидатов |
| `retrieved_p_yes` | array | P(yes) от VLM для каждого кандидата |
| `search_query` | array\|null | pageTitle из Yandex Image Search (если был интернет-поиск) |
| `timing` | object | Тайминги этапов в секундах |

**Ошибки:**

| Код | Описание |
|-----|---------|
| 400 | Невалидный файл, неподдерживаемый формат или превышен размер |
| 429 | Превышен rate limit |
| 503 | Сервис не инициализирован |
| 504 | Таймаут предсказания (> 90 с) |

---

### GET /v1/health

Статус сервиса и компонентов.

**Ответ 200 OK:**

```json
{
  "status": "healthy",
  "ready": true,
  "timestamp": "2026-07-01T10:00:00.000000",
  "components": {
    "retriever": {
      "status": "ok",
      "index_size": 5234
    },
    "vllm": {
      "status": "ok",
      "base_url": "http://localhost:30000/v1",
      "model": "qwen2-vl-2b-r16"
    }
  },
  "config": {
    "top_k_retrieval": 10,
    "vlm_threshold": 0.5,
    "enable_internet_search": true,
    "device": "cpu"
  },
  "metrics": {
    "total_requests": 142,
    "successful_requests": 138,
    "failed_requests": 4,
    "success_rate": 0.9718,
    "internet_searches": 31,
    "confidence_last": 0.847
  }
}
```

**Значения `status`**: `healthy` / `degraded` (vLLM недоступен) / `not_ready`

---

### GET /v1/info

Информация о сервисе и текущие метрики (аналог health без проверки vLLM).

---

### GET /metrics

Prometheus-метрики в формате text/plain.

```
# HELP aitourguide_requests_total Общее число вызовов predict()
# TYPE aitourguide_requests_total counter
aitourguide_requests_total{status="success"} 138.0
aitourguide_requests_total{status="error"} 4.0
...
```

---

### GET /

Базовая информация об API.

```json
{
  "service": "AI Tour Guide API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "redoc": "/redoc"
}
```

---

## Интерактивная документация

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Ограничения файлов

| Параметр | Значение |
|---------|---------|
| Максимальный размер | 10 MB |
| Форматы | JPEG, PNG, GIF, WEBP |
| MIME-типы | `image/jpeg`, `image/png`, `image/gif`, `image/webp` |

---

## Типичное время ответа

| Сценарий | Время |
|---------|-------|
| Только retrieval (без VLM) | 0.1–0.3 с |
| Retrieval + VLM reranking | 2–5 с |
| С интернет-поиском | 8–20 с |

---

## Примеры

### Python

```python
import requests

def predict_landmark(image_path: str, base_url: str = "http://localhost:8000") -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{base_url}/v1/predict",
            files={"file": f},
            data={"use_internet_search": "true"},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()

result = predict_landmark("photo.jpg")
print(f"{result['name']} (confidence: {result['confidence']:.2%})")
```

### cURL

```bash
# Распознать достопримечательность
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@photo.jpg" | python -m json.tool

# Проверить здоровье сервиса
curl http://localhost:8000/v1/health | python -m json.tool
```

---

## Обработка ошибок

Все ошибки возвращаются в формате:

```json
{
  "detail": "Описание ошибки"
}
```

HTTP-коды:

| Код | Значение |
|-----|---------|
| 200 | Успех |
| 400 | Невалидный запрос |
| 422 | Ошибка валидации (Pydantic) |
| 429 | Превышен rate limit |
| 500 | Внутренняя ошибка сервера |
| 503 | Сервис недоступен |
| 504 | Таймаут |
