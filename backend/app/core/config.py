"""
AcousticIQ Core Configuration Module.
Provides centralized typed configuration using Pydantic Settings.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "AcousticIQ"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api"

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Directories (fully configurable for local and external storage)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATASET_DIR: Path = BASE_DIR / "ml" / "data"
    DATASET_RAW_DIR: Path = BASE_DIR / "ml" / "data" / "raw" / "MIMII"
    DATASET_INTERIM_DIR: Path = BASE_DIR / "ml" / "data" / "interim"
    DATASET_PROCESSED_DIR: Path = BASE_DIR / "ml" / "data" / "processed"
    DATASET_METADATA_DIR: Path = BASE_DIR / "ml" / "data" / "metadata"
    MODEL_REGISTRY_DIR: Path = BASE_DIR / "ml" / "checkpoints"
    EXPERIMENTS_DIR: Path = BASE_DIR / "ml" / "experiments"
    UPLOAD_DIR: Path = BASE_DIR / "backend" / "uploads"
    
    # Persistence
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'backend' / 'acousticiq.db'}"
    AUDIO_RETENTION_ENABLED: bool = False
    MAX_UPLOAD_SIZE_MB: int = 25

    # Audio Signal Processing & Model Hyperparameters
    DEFAULT_SAMPLE_RATE: int = 16000
    N_FFT: int = 1024
    HOP_LENGTH: int = 512
    N_MELS: int = 64
    F_MIN: float = 0.0
    F_MAX: float = 8000.0
    TARGET_DURATION_SECONDS: float = 10.0
    DECISION_THRESHOLD_PERCENTILE: float = 95.0

    # Supported Industrial Machine Categories (MIMII compliant)
    SUPPORTED_MACHINES: List[str] = ["valve", "pump", "fan", "slider"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure runtime directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
settings.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASET_RAW_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASET_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASET_METADATA_DIR.mkdir(parents=True, exist_ok=True)

