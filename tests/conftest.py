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

# ---------------------------------------------------------------------------
# Изображения
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Мок AITourGuide
# ---------------------------------------------------------------------------

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
    guide.health_check = AsyncMock(return_value={"status": "healthy"})
    guide.cleanup = AsyncMock()
    return guide


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(mock_guide: MagicMock) -> TestClient:
    """
    TestClient с подменённым AITourGuide.
    Не требует реальных моделей и индекса.
    """
    from src.api import dependencies
    from src.api.main import app

    # Подменяем глобальный экземпляр guide
    dependencies.set_guide(mock_guide)

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    # Сбрасываем после теста
    dependencies.set_guide(None)
