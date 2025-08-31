"""
Core configuration for the UI Validation API
"""
import os
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    app_name: str = "UI Validation API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    api_v1_prefix: str = "/api/v1"
    
    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"], 
        env="ALLOWED_ORIGINS"
    )
    
    @field_validator('allowed_origins', 'allowed_file_types', mode='before')
    @classmethod
    def parse_comma_separated(cls, value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(',') if v.strip()]
        return value
    
    # AI Model Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    use_openrouter: bool = Field(default=False, env="USE_OPENROUTER")
    huggingface_cache_dir: Optional[str] = Field(default=None, env="HF_CACHE_DIR")
    detection_confidence: float = Field(default=0.1, env="DETECTION_CONFIDENCE")
    use_gpu: bool = Field(default=True, env="USE_GPU")
    
    # GPT Configuration
    gpt_model: str = Field(default="gpt-4o-mini", env="GPT_MODEL")
    gpt_vision_model: str = Field(default="gpt-4o", env="GPT_VISION_MODEL")
    openrouter_model: str = Field(default="anthropic/claude-3.5-sonnet", env="OPENROUTER_MODEL")
    max_gpt_calls: int = Field(default=20, env="MAX_GPT_CALLS")
    gpt_batch_size: int = Field(default=4, env="GPT_BATCH_SIZE")
    
    # Debug Configuration
    enable_debug_crops: bool = Field(default=False, env="ENABLE_DEBUG_CROPS")
    debug_output_dir: str = Field(default="./debug_output", env="DEBUG_OUTPUT_DIR")
    
    # CLIP Configuration
    clip_model: str = Field(default="openai/clip-vit-large-patch14", env="CLIP_MODEL")
    clip_max_elements: int = Field(default=10, env="CLIP_MAX_ELEMENTS")
    clip_similarity_threshold: float = Field(default=0.1, env="CLIP_SIMILARITY_THRESHOLD")
    
    # Redis Configuration (for caching)
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # File Upload Configuration
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"], 
        env="ALLOWED_FILE_TYPES"
    )
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    
    # Background Tasks
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # Database (optional)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()