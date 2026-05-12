# config.py
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Set
import os


class Settings(BaseSettings):
    """Application configuration with validation"""
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    
    # Model paths
    model_path: str = "data/models/qwen2-vl-2b-r16/model-q5_k_m.gguf"
    mmproj_path: str = "data/models/qwen2-vl-2b-r16/mmproj-model-f16.gguf"
    index_path: str = "artifacts/indexes/clip_index"
    facts_db_path: str = "artifacts/databases/facts_db.pkl"

    # Пути к фронтенду
    templates_dir: str = "src/api/templates"
    static_dir: str = "src/api/static"
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "text"
    log_file: str | None = None
    
    # Device configuration
    device: str = "cpu"
    
    # Timeouts and limits
    predict_timeout: int = 90
    max_file_size_mb: int = 10
    
    # Rate limiting
    rate_limit_calls: int = 10
    rate_limit_period: int = 60  # seconds
    
    # Server configuration
    port: int = 8000
    host: str = "0.0.0.0"
    reload: bool = False
    
    # CORS
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    # File validation
    allowed_extensions: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    allowed_mime_types: Set[str] = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def templates_path_abs(self) -> Path:
        """Absolute path to templates directory"""
        return self.project_root / self.templates_dir
    
    @property
    def static_path_abs(self) -> Path:
        """Absolute path to static directory"""
        return self.project_root / self.static_dir
    
    def validate_paths(self) -> dict[str, bool]:
        """Validate that required paths exist"""
        # For index_path, check if .faiss file exists
        index_exists = os.path.exists(f"{self.index_path}.faiss")
        
        return {
            "model_path": os.path.exists(self.model_path),
            "mmproj_path": os.path.exists(self.mmproj_path),
            "index_path": index_exists,
            "facts_db_path": os.path.exists(self.facts_db_path),
        }
    
    class Config:
        env_file = ".env"
        env_prefix = ""
        extra = "ignore"  # Игнорировать неизвестные переменные окружения
        case_sensitive = False


# Global settings instance
settings = Settings()
