# config.py
from pydantic_settings import BaseSettings
from typing import Set
import os


class Settings(BaseSettings):
    """Application configuration with validation"""
    
    # Model paths
    model_path: str = "models/qwen2-vl-2b-r16/model-q5_k_m.gguf"
    mmproj_path: str = "models/qwen2-vl-2b-r16/mmproj-model-f16.gguf"
    index_path: str = "rag/clip_index"
    facts_db_path: str = "rag/facts_db.pkl"
    
    # Device configuration
    device: str = "cuda"
    
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
    
    def validate_paths(self) -> dict[str, bool]:
        """Validate that required paths exist"""
        return {
            "model_path": os.path.exists(self.model_path),
            "mmproj_path": os.path.exists(self.mmproj_path),
            "index_path": os.path.exists(self.index_path),
            "facts_db_path": os.path.exists(self.facts_db_path),
        }
    
    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False


# Global settings instance
settings = Settings()
