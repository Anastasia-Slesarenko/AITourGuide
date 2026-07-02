# tests/integration/test_api.py
"""
Интеграционные тесты FastAPI-эндпоинтов.
Используют замоканный AITourGuide — реальные модели не нужны.
"""

import io

from PIL import Image


def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Создаём минимальный JPEG в памяти."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /v1/health
# ---------------------------------------------------------------------------


class TestHealth:
    """Тесты эндпоинта проверки состояния сервиса."""

    def test_health_возвращает_200(self, api_client):
        response = api_client.get("/v1/health")
        assert response.status_code == 200

    def test_health_содержит_статус(self, api_client):
        data = api_client.get("/v1/health").json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "not_ready")

    def test_health_содержит_компоненты(self, api_client):
        data = api_client.get("/v1/health").json()
        assert "components" in data


# ---------------------------------------------------------------------------
# /v1/predict
# ---------------------------------------------------------------------------


class TestPredict:
    """Тесты эндпоинта распознавания достопримечательности."""

    def test_predict_без_файла_возвращает_422(self, api_client):
        """Запрос без файла должен вернуть 422 Unprocessable Entity."""
        response = api_client.post("/v1/predict")
        assert response.status_code == 422

    def test_predict_с_валидным_изображением_возвращает_200(self, api_client):
        """Корректный JPEG должен вернуть 200 с полями ответа."""
        jpeg = _make_jpeg_bytes()
        response = api_client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_predict_структура_ответа(self, api_client):
        """Ответ должен содержать все обязательные поля."""
        jpeg = _make_jpeg_bytes()
        data = api_client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
        ).json()

        assert "name" in data
        assert "description" in data
        assert "confidence" in data
        assert "source" in data
        assert "unknown" in data
        assert "timing" in data

    def test_predict_confidence_в_диапазоне_0_1(self, api_client):
        """Confidence должен быть в диапазоне [0, 1]."""
        jpeg = _make_jpeg_bytes()
        data = api_client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
        ).json()

        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_source_допустимое_значение(self, api_client):
        """Поле source должно быть одним из допустимых значений."""
        jpeg = _make_jpeg_bytes()
        data = api_client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
        ).json()

        assert data["source"] in ("retrieval", "internet", "fallback")

    def test_predict_с_отключённым_интернет_поиском(self, api_client):
        """Параметр use_internet_search=false должен приниматься."""
        jpeg = _make_jpeg_bytes()
        response = api_client.post(
            "/v1/predict",
            files={"image": ("photo.jpg", jpeg, "image/jpeg")},
            data={"use_internet_search": "false"},
        )
        assert response.status_code == 200

    def test_predict_слишком_большой_файл_возвращает_400(self, api_client):
        """Файл больше 10 МБ должен быть отклонён с кодом 400."""
        # Создаём данные размером > 10 МБ
        big_data = b"x" * (11 * 1024 * 1024)
        response = api_client.post(
            "/v1/predict",
            files={"image": ("big.jpg", big_data, "image/jpeg")},
        )
        assert response.status_code == 400

    def test_predict_невалидные_байты_возвращают_ошибку(self, api_client):
        """Невалидные данные изображения должны вернуть ошибку."""
        response = api_client.post(
            "/v1/predict",
            files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        # Сервис должен вернуть ошибку (400 или 422), но не падать с 500
        assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Тесты Prometheus-эндпоинта."""

    def test_metrics_возвращает_200(self, api_client):
        response = api_client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_содержит_prometheus_формат(self, api_client):
        """Ответ должен содержать строки в формате Prometheus."""
        text = api_client.get("/metrics").text
        assert "# HELP" in text or "# TYPE" in text


# ---------------------------------------------------------------------------
# /docs, /redoc, /openapi.json
# ---------------------------------------------------------------------------


class TestDocumentation:
    """Тесты доступности документации."""

    def test_swagger_ui_доступен(self, api_client):
        assert api_client.get("/docs").status_code == 200

    def test_redoc_доступен(self, api_client):
        assert api_client.get("/redoc").status_code == 200

    def test_openapi_json_доступен(self, api_client):
        response = api_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
