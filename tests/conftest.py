# tests/conftest.py
"""
Pytest configuration and shared fixtures for AITourGuide tests.
"""

import pytest
from pathlib import Path
from PIL import Image
import io
import tempfile
from typing import Generator
from fastapi.testclient import TestClient


# ========================================
# PATH FIXTURES
# ========================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Return the tests directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def fixtures_dir(tests_dir: Path) -> Path:
    """Return the fixtures directory."""
    return tests_dir / "fixtures"


# ========================================
# IMAGE FIXTURES
# ========================================

@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Convert sample image to bytes."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_file(sample_image: Image.Image) -> Generator[Path, None, None]:
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        sample_image.save(f, format="JPEG")
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def large_image() -> Image.Image:
    """Create a large image for testing size limits."""
    return Image.new("RGB", (4000, 4000), color="blue")


# ========================================
# API FIXTURES
# ========================================

@pytest.fixture
def api_client() -> TestClient:
    """Create a FastAPI test client."""
    from main import app
    return TestClient(app)


@pytest.fixture
def mock_guide_config():
    """Mock configuration for AITourGuide."""
    return {
        "model_path": "models/test-model.gguf",
        "mmproj_path": "models/test-mmproj.gguf",
        "index_path": "rag/test_index",
        "facts_db_path": "rag/test_facts.pkl",
        "device": "cpu",
        "top_k_retrieval": 5,
        "confidence_threshold": 0.78,
    }


# ========================================
# RAG FIXTURES
# ========================================

@pytest.fixture
def sample_retrieved_results():
    """Sample retrieval results for testing."""
    return [
        {
            "landmark_id": "1",
            "name": "Эйфелева башня",
            "description": "Металлическая башня в Париже",
            "score": 0.95,
            "image_path": "images/eiffel.jpg",
        },
        {
            "landmark_id": "2",
            "name": "Колизей",
            "description": "Древний амфитеатр в Риме",
            "score": 0.85,
            "image_path": "images/colosseum.jpg",
        },
        {
            "landmark_id": "3",
            "name": "Тадж-Махал",
            "description": "Мавзолей в Индии",
            "score": 0.75,
            "image_path": "images/taj_mahal.jpg",
        },
    ]


@pytest.fixture
def sample_facts_db():
    """Sample facts database for testing."""
    return {
        "1": {
            "name_ru": "Эйфелева башня",
            "name_en": "Eiffel Tower",
            "ground_truth": "Металлическая башня в Париже, построенная в 1889 году",
            "image_path": "images/eiffel.jpg",
        },
        "2": {
            "name_ru": "Колизей",
            "name_en": "Colosseum",
            "ground_truth": "Древний амфитеатр в Риме, построенный в 80 году н.э.",
            "image_path": "images/colosseum.jpg",
        },
    }


# ========================================
# MOCK FIXTURES
# ========================================

@pytest.fixture
def mock_clip_embeddings():
    """Mock CLIP embeddings for testing."""
    import numpy as np
    return np.random.rand(1, 768).astype("float32")


@pytest.fixture
def mock_vlm_response():
    """Mock VLM response for testing."""
    return {
        "name": "Эйфелева башня",
        "description": "Металлическая башня в Париже, построенная в 1889 году для Всемирной выставки.",
    }


# ========================================
# ENVIRONMENT FIXTURES
# ========================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("YC_FOLDER_ID", "test_folder_id")
    monkeypatch.setenv("YC_API_KEY", "test_api_key")
    monkeypatch.setenv("MODEL_PATH", "models/test-model.gguf")
    monkeypatch.setenv("DEVICE", "cpu")


# ========================================
# PYTEST CONFIGURATION
# ========================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "requires_api_keys: marks tests that require API keys"
    )


# ========================================
# CLEANUP
# ========================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Add any cleanup logic here
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
