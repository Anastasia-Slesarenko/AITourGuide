# src/core/config.py
"""Конфигурация приложения через переменные окружения."""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Set


class Settings(BaseSettings):
    """Настройки приложения с валидацией через pydantic."""

    project_root: Path = Path(__file__).resolve().parent.parent.parent

    # Путь к SigLIP FAISS-индексу
    index_dir: str = "data/index/siglip"

    # Базовая директория изображений галереи.
    # image_path в gallery_metadata.json — просто имя файла (photo.jpg),
    # полный путь = images_base_dir / image_path
    images_base_dir: str = "data/images"

    # Локальный путь к SigLIP модели (пустая строка = загрузка с HuggingFace)
    siglip_model_path: str = "data/models/siglip-base-patch16-224"

    # SGLang сервер (VLM reranking)
    sglang_base_url: str = "http://localhost:30000/v1"
    sglang_model_name: str = "qwen2-vl-2b-r16"
    sglang_timeout: float = 30.0
    sglang_max_retries: int = 3

    # RAG / retrieval
    top_k_retrieval: int = 10
    confidence_threshold: float = 0.5
    enable_internet_search: bool = True

    # Пути к фронтенду
    templates_dir: str = "src/api/templates"
    static_dir: str = "src/api/static"

    # Логирование
    log_level: str = "INFO"
    log_format: str = "text"
    log_file: str | None = None

    # Устройство (cpu/cuda)
    device: str = "cpu"

    # Таймауты и лимиты
    predict_timeout: int = 90
    max_file_size_mb: int = 10

    # Rate limiting
    rate_limit_calls: int = 10
    rate_limit_period: int = 60  # секунды

    # Сервер
    port: int = 8000
    host: str = "0.0.0.0"
    reload: bool = False

    # CORS
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Допустимые форматы файлов
    allowed_extensions: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    allowed_mime_types: Set[str] = {
        "image/jpeg", "image/png", "image/gif", "image/webp"
    }

    @property
    def max_file_size_bytes(self) -> int:
        """Максимальный размер файла в байтах."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def templates_path_abs(self) -> Path:
        """Абсолютный путь к директории шаблонов."""
        return self.project_root / self.templates_dir

    @property
    def static_path_abs(self) -> Path:
        """Абсолютный путь к директории статики."""
        return self.project_root / self.static_dir

    @property
    def index_dir_abs(self) -> Path:
        """Абсолютный путь к директории FAISS-индекса."""
        p = Path(self.index_dir)
        if p.is_absolute():
            return p
        return self.project_root / p

    def validate_paths(self) -> dict[str, bool]:
        """Проверяет наличие обязательных файлов индекса."""
        return {
            "index_faiss": (self.index_dir_abs / "gallery_index.faiss").exists(),
            "index_metadata": (
                self.index_dir_abs / "gallery_metadata.json"
            ).exists(),
        }

    class Config:
        env_file = ".env"
        env_prefix = ""
        extra = "ignore"
        case_sensitive = False


# Глобальный экземпляр настроек
settings = Settings()
