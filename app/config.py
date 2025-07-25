import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # App Settings
    app_name: str = Field(default="Waste Detection API")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    # Model Settings
    model_path: str = Field(default="./models/model.onnx")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=640)
    
    # Upload Settings
    upload_dir: str = Field(default="./uploads")
    max_file_size: int = Field(default=10 * 1024 * 1024)  # 10MB
    allowed_extensions: List[str] = Field(default=["jpg", "jpeg", "png", "bmp", "webp"])
    
    # Firebase Settings (optional)
    firebase_credentials_path: str = Field(default="")
    firebase_storage_bucket: str = Field(default="")
    
    # CORS Settings
    allowed_origins: List[str] = Field(default=["http://localhost:3000"])
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=30)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.upload_dir, exist_ok=True)