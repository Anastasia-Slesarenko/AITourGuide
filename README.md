# 🏛️ AI Tour Guide

**Сервис распознавания достопримечательностей по фотографиям на русском языке.**

Загружаете фото — получаете название объекта, описание и уровень уверенности модели.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

---

## Схема работы сервиса

![Architecture](docs/architecture_diagram.png)

1. **SigLIP + FAISS** — векторный поиск по галерее, возвращает top-10 кандидатов
2. **Qwen2-VL-2B LoRA reranker** — попарное сравнение фото запроса и кандидата, вычисляет P(yes) через logprobs
3. **Интернет-поиск** — при низкой уверенности: Yandex Image Search → Wikipedia → VLM верификация
4. **Перевод** — Yandex Translate API для перевода EN→RU

---

## Системные требования

### Продакшен (API-сервер)

| Ресурс | Минимум | Рекомендуется |
|--------|---------|---------------|
| **CPU** | 4 ядра | 8+ ядер |
| **RAM** | 8 GB | 16 GB |
| **Диск** | 10 GB | 20 GB |
| **GPU** | ❌ Не требуется | ✅ Для vLLM-сервера |
| **Python** | 3.11 | 3.11 |

> **Важно:** API-сервер работает на CPU. GPU нужен только для отдельного vLLM-сервера с Qwen2-VL-2B.

### vLLM-сервер (VLM reranking)

| Ресурс | Требование |
|--------|-----------|
| **GPU** | NVIDIA, ≥ 8 GB VRAM (RTX 3080 / A10 / T4) |
| **CUDA** | 12.x |
| **RAM** | 16 GB |

---

## Быстрый старт

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/your-org/AITourGuide.git
cd AITourGuide

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Установите зависимости
pip install -r requirements-prod.txt

# 4. Настройте переменные окружения
cp .env.example .env
# Отредактируйте .env: YC_FOLDER_ID, YC_API_KEY

# 5. Соберите FAISS-индекс
make build-index

# 6. Запустите API-сервер
make run
```

API доступен на `http://localhost:8000`, документация — `http://localhost:8000/docs`.

### Docker

```bash
make docker-up
```

---

## Результаты обучения

Модель: **Qwen2-VL-2B-Instruct** + LoRA reranker  
Датасет: валидационная выборка, 13 911 примеров (5 746 known / 8 165 unknown)


| Метрика | Retrieval baseline (SigLIP + FAISS) | VLM reranker | LoRA VLM reranker (лучшая модель) |
|---------|---------|---------|---------|
| Hit@1 | 61.9% | | **61.9%** (+ пп к baseline) |
| MRR | 0.854 | |  **0.854** (+0.109)|
| Accuracy | 67.87% | |  67.87% |
| Unknown accuracy | 72.92% | |72.92% |
| Latency P95 | 2.84 с | | 2.84 с|

### LoRA VLM reranker (лучшая модель)

Модель: **Qwen2-VL-2B-Instruct** + LoRA reranker (r=16, α=32, lr=2e-5, 448px)  
Датасет: тестовая выборка, 13 889 примеров

| Метрика | Значение |
|---------|---------|
| Hit@1 | **61.9%** |
| MRR | **0.854** |
| Accuracy | 67.87% |
| Unknown accuracy | 72.92% |
| Latency P95 | 2.84 с |

Оптимальный порог (threshold = 0.4226):

| Метрика | Значение |
|---------|---------|
| F1 known | % |
| F1 unknown | % |
| F1 macro | 74.96% |
| Known accuracy | % |
| Unknown accuracy | % |

---

## Нагрузочное тестирование

Эндпоинт `POST /v1/predict`, 100 concurrent users:

| Метрика | Значение |
|---------|---------|
| RPS (sustained) | ~10 req/s (~40 req/s суммарно по всем эндпоинтам) |
| Latency P50 | 3 ms |
| Latency P95 | 5 ms |
| Latency P99 | 26 000 ms |
| Error rate | 0% (0 / 587 запросов) |

---

## Структура проекта

```
AITourGuide/
├── src/
│   ├── api/          # FastAPI: роуты, middleware, зависимости
│   ├── core/         # Конфигурация, логирование, метрики
│   ├── rag/          # FAISS-индекс, LandmarkRetriever
│   └── services/     # AITourGuide, YandexSearch, Wikipedia, Translator
├── scripts/
│   ├── experiments/  # Обучение, оценка, экспорт моделей
│   └── data_preparation/  # Подготовка данных, сборка индекса
├── docker/           # Dockerfile, docker-compose, Prometheus, Grafana
├── config/           # YAML-конфиги (base, development, production)
├── tests/            # Unit и integration тесты
├── requirements-prod.txt  # Зависимости для продакшена
└── requirements-dev.txt   # Зависимости для разработки и экспериментов
```

---

## API

| Метод | Путь | Описание |
|-------|------|---------|
| `POST` | `/v1/predict` | Распознать достопримечательность |
| `GET` | `/v1/health` | Статус сервиса и компонентов |
| `GET` | `/v1/info` | Метрики и конфигурация |
| `GET` | `/metrics` | Prometheus-метрики |
| `GET` | `/docs` | Swagger UI |

Пример запроса:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@photo.jpg"
```

Пример ответа:

```json
{
  "name": "Исаакиевский собор",
  "description": "Исаакиевский собор — крупнейший православный храм Санкт-Петербурга...",
  "confidence": 0.923,
  "source": "retrieval",
  "unknown": false
}
```

---

## Разработка

```bash
# Установить все зависимости (включая dev и эксперименты)
make install-dev

# Запустить тесты
make test

# Линтинг
make lint

# Форматирование
make format
```

---

## Лицензия

[MIT License](LICENSE)
