# tests/unit/test_config.py
"""
Unit tests for configuration module.
"""

import pytest
from pathlib import Path
from core.config import Settings


class TestSettings:
    """Tests for Settings class."""
    
    def test_settings_default_values(self):
        """Test that settings have correct default values."""
        settings = Settings()
        
        assert settings.port == 8000
        assert settings.host == "0.0.0.0"
        assert settings.device == "cuda"
        assert settings.rate_limit_calls == 10
        assert settings.rate_limit_period == 60
        assert settings.predict_timeout == 90
        assert settings.max_file_size_mb == 10
    
    def test_max_file_size_bytes_conversion(self):
        """Test conversion from MB to bytes."""
        settings = Settings()
        expected_bytes = 10 * 1024 * 1024
        assert settings.max_file_size_bytes == expected_bytes
    
    def test_validate_paths(self, tmp_path):
        """Test path validation."""
        # Create temporary files
        model_path = tmp_path / "model.gguf"
        model_path.touch()
        
        settings = Settings()
        settings.model_path = str(model_path)
        
        validation = settings.validate_paths()
        
        assert "model_path" in validation
        assert validation["model_path"] is True
    
    def test_validate_paths_missing_files(self):
        """Test path validation with missing files."""
        settings = Settings()
        settings.model_path = "nonexistent/model.gguf"
        
        validation = settings.validate_paths()
        
        assert validation["model_path"] is False
    
    def test_allowed_extensions(self):
        """Test allowed file extensions."""
        settings = Settings()
        
        assert ".jpg" in settings.allowed_extensions
        assert ".jpeg" in settings.allowed_extensions
        assert ".png" in settings.allowed_extensions
        assert ".gif" in settings.allowed_extensions
        assert ".webp" in settings.allowed_extensions
    
    def test_allowed_mime_types(self):
        """Test allowed MIME types."""
        settings = Settings()
        
        assert "image/jpeg" in settings.allowed_mime_types
        assert "image/png" in settings.allowed_mime_types
        assert "image/gif" in settings.allowed_mime_types
        assert "image/webp" in settings.allowed_mime_types
    
    def test_cors_configuration(self):
        """Test CORS configuration."""
        settings = Settings()
        
        assert settings.cors_origins == ["*"]
        assert settings.cors_allow_credentials is True
        assert settings.cors_allow_methods == ["*"]
        assert settings.cors_allow_headers == ["*"]
    
    def test_settings_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("HOST", "127.0.0.1")
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("RATE_LIMIT_CALLS", "20")
        
        settings = Settings()
        
        assert settings.port == 9000
        assert settings.host == "127.0.0.1"
        assert settings.device == "cpu"
        assert settings.rate_limit_calls == 20
