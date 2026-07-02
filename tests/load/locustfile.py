# tests/load/locustfile.py
"""
Нагрузочное тестирование AITourGuide API с помощью Locust.

Запуск:
    # Веб-интерфейс (http://localhost:8089)
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Headless-режим (100 пользователей, 10 новых/сек, 60 сек)
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 100 -r 10 -t 60s \
        --csv tests/load/results/report

Результаты сохраняются в tests/load/results/ (CSV + HTML).
"""

import io
import random

from locust import HttpUser, between, task
from PIL import Image


def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Генерируем минимальный JPEG в памяти для отправки в API."""
    # Случайный цвет чтобы каждый запрос был уникальным
    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class APIUser(HttpUser):
    """
    Симулирует обычного пользователя сервиса.
    Ждёт 1–5 секунд между запросами.
    """

    wait_time = between(1, 5)

    def on_start(self):
        """Проверяем доступность сервиса перед началом теста."""
        self.client.get("/v1/health")

    @task(5)
    def predict_без_интернет_поиска(self):
        """
        Основной сценарий: распознавание без интернет-поиска.
        Вес 5 — самый частый запрос.
        """
        jpeg = _make_jpeg_bytes()
        with self.client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
            data={"use_internet_search": "false"},
            name="/v1/predict (no internet)",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "confidence" not in data:
                    response.failure("Нет поля confidence в ответе")
                else:
                    response.success()
            elif response.status_code == 429:
                # Rate limit — не считаем ошибкой
                response.success()
            else:
                response.failure(f"Неожиданный статус: {response.status_code}")

    @task(2)
    def predict_с_интернет_поиском(self):
        """
        Сценарий с интернет-поиском.
        Вес 2 — реже, т.к. медленнее.
        """
        jpeg = _make_jpeg_bytes()
        with self.client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
            data={"use_internet_search": "true"},
            name="/v1/predict (with internet)",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 429, 504):
                response.success()
            else:
                response.failure(f"Неожиданный статус: {response.status_code}")

    @task(3)
    def health_check(self):
        """
        Проверка состояния сервиса.
        Вес 3 — имитирует мониторинг.
        """
        self.client.get("/v1/health", name="/v1/health")

    @task(1)
    def metrics(self):
        """
        Получение Prometheus-метрик.
        Вес 1 — редкий запрос от системы мониторинга.
        """
        self.client.get("/metrics", name="/metrics")
