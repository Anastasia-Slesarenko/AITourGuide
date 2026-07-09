# tests/unit/test_config.py
"""Юнит-тесты конфигурации приложения."""

from src.core.config import Settings


class TestSettings:
    """Тесты класса Settings."""

    def test_default_values(self):
        """Проверяем дефолтные значения настроек."""
        s = Settings()

        assert s.port == 8000
        assert s.host == "0.0.0.0"
        assert s.device == "cpu"
        assert s.rate_limit_calls == 10
        assert s.rate_limit_period == 60
        assert s.predict_timeout == 90
        assert s.max_file_size_mb == 10
        assert s.top_k_retrieval == 10
        assert s.confidence_threshold == 0.472656

    def test_file_size_conversion_to_bytes(self):
        """max_file_size_bytes = max_file_size_mb * 1024 * 1024."""
        s = Settings()
        assert s.max_file_size_bytes == 10 * 1024 * 1024

    def test_allowed_extensions(self):
        """Проверяем набор допустимых расширений изображений."""
        s = Settings()
        assert {".jpg", ".jpeg", ".png", ".gif", ".webp"} <= s.allowed_extensions

    def test_allowed_mime_types(self):
        """Проверяем набор допустимых MIME-типов."""
        s = Settings()
        assert {
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        } <= s.allowed_mime_types

    def test_cors_defaults(self):
        """CORS по умолчанию открыт для всех."""
        s = Settings()
        assert s.cors_origins == ["*"]
        assert s.cors_allow_credentials is True

    def test_environment_variables(self, monkeypatch):
        """Настройки читаются из переменных окружения."""
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("HOST", "127.0.0.1")
        monkeypatch.setenv("DEVICE", "cuda")
        monkeypatch.setenv("RATE_LIMIT_CALLS", "20")

        s = Settings()

        assert s.port == 9000
        assert s.host == "127.0.0.1"
        assert s.device == "cuda"
        assert s.rate_limit_calls == 20

    def test_validate_paths_returns_dict(self):
        """validate_paths() возвращает словарь с булевыми значениями."""
        s = Settings()
        result = s.validate_paths()

        assert isinstance(result, dict)
        assert "index_faiss" in result
        assert "index_metadata" in result
        assert all(isinstance(v, bool) for v in result.values())
