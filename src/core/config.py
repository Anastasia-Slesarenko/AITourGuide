# src/core/config.py
"""Конфигурация приложения через переменные окружения."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # vLLM сервер (VLM reranking)
    vllm_base_url: str = "http://localhost:30000/v1"
    vllm_model_name: str = "qwen2-vl-2b-r16"
    vllm_timeout: float = 30.0
    vllm_max_retries: int = 3

    # RAG / retrieval
    top_k_retrieval: int = 10
    # Порог по сырому p_yes: accept/reject (known/unknown) и запуск интернет-поиска.
    # Youden-оптимум LoRA; совпадает с ACCEPT_THRESHOLD в calibration.py.
    confidence_threshold: float = 0.472656
    enable_internet_search: bool = True

    # Калибровка отдаваемой уверенности (isotonic-кривая, фит на val)
    calibrate_confidence: bool = True
    calibration_curve_path: str = "data/calibration/isotonic_reranker.json"

    # Yandex Cloud API (перевод и поиск по изображению)
    yc_folder_id: str = ""
    yc_api_key: str = ""

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
    rate_limit_enabled: bool = True  # отключать на нагрузочных тестах
    rate_limit_calls: int = 10
    rate_limit_period: int = 60  # секунды

    # Backpressure: максимум одновременно обрабатываемых predict-запросов
    # на процесс. Сверх лимита запросы быстро отклоняются (503), чтобы
    # задержка успешных оставалась ограниченной вместо роста очереди.
    max_concurrent_predicts: int = 8
    # Сколько ждать освобождения слота перед тем как отдать 503, секунды.
    predict_admission_timeout: float = 0.5

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
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    allowed_mime_types: set[str] = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
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

    @property
    def calibration_curve_path_abs(self) -> Path:
        """Абсолютный путь к кривой калибровки уверенности."""
        p = Path(self.calibration_curve_path)
        return p if p.is_absolute() else self.project_root / p

    def validate_paths(self) -> dict[str, bool]:
        """Проверяет наличие обязательных файлов индекса."""
        return {
            "index_faiss": (self.index_dir_abs / "gallery_index.faiss").exists(),
            "index_metadata": (self.index_dir_abs / "gallery_metadata.json").exists(),
        }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )


# Глобальный экземпляр настроек
settings = Settings()
