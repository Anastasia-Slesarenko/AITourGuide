# tests/conftest.py
"""Общие фикстуры для тестов AITourGuide."""

import io
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Изображения

@pytest.fixture
def sample_image() -> Image.Image:
    """Минимальное RGB-изображение для тестов."""
    return Image.new("RGB", (224, 224), color=(100, 150, 200))


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Байты JPEG-изображения."""
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_image_file(sample_image: Image.Image) -> Generator[Path, None, None]:
    """Временный файл изображения; удаляется после теста."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        sample_image.save(f, format="JPEG")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


# Мок AITourGuide

@pytest.fixture
def mock_guide() -> MagicMock:
    """
    Мок AITourGuide с предустановленным ответом predict().
    Используется в интеграционных тестах чтобы не поднимать реальные модели.
    """
    guide = MagicMock()
    guide.predict = AsyncMock(return_value={
        "name": "Исаакиевский собор",
        "description": "Крупнейший православный храм Санкт-Петербурга.",
        "confidence": 0.92,
        "unknown": False,
        "source": "retrieval",
        "winner_landmark_id": "saint_isaac",
        "winner_images": ["saint_isaac_1.jpg"],
        "retrieved_names": ["Исаакиевский собор", "Казанский собор"],
        "retrieved_scores": [0.91, 0.74],
        "retrieved_p_yes": [0.92, 0.11],
        "search_query": None,
        "error": None,
        "timing": {"image_load": 0.01, "retrieval": 0.15, "vlm_generation": 1.8},
    })
    guide.health_check = AsyncMock(return_value={
        "status": "healthy",
        "ready": True,
        "components": {
            "retriever": {"status": "ok", "index_size": 100},
            "vllm": {"status": "ok"},
        },
        "config": {"top_k_retrieval": 10, "vlm_threshold": 0.473},
        "metrics": {"total_requests": 0},
    })
    guide.cleanup = AsyncMock()
    return guide


# FastAPI TestClient

@pytest.fixture
def api_client(mock_guide: MagicMock) -> TestClient:
    """
    TestClient с подменённым AITourGuide.
    Не требует реальных моделей и индекса.
    """
    from src.api import dependencies
    from src.api.dependencies import get_guide
    from src.api.main import app

    # Подменяем guide и на уровне глобального экземпляра, и через
    # dependency_overrides. Подмена через зависимость здесь обязательна:
    # lifespan при старте создаёт реальный AITourGuide и перезаписывает
    # глобальный _guide (main.py: set_guide), из-за чего результат теста
    # зависел бы от того, поднялся реальный сервис или упал. Так маршруты
    # всегда берут мок.
    dependencies.set_guide(mock_guide)
    app.dependency_overrides[get_guide] = lambda: mock_guide

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    # Сбрасываем после теста
    app.dependency_overrides.clear()
    dependencies.set_guide(None)
