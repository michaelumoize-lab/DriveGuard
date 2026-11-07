"""
Application Configuration and Environment Settings using Pydantic Settings.
"""

from pathlib import Path
from typing import List
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """DriveGuard Backend Settings."""

    model_config = SettingsConfigDict(
        env_file=(
            ".env",
            "backend/.env",
            str(Path(__file__).resolve().parents[2] / ".env"),
            str(Path(__file__).resolve().parents[2] / "backend" / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "DriveGuard"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server Binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # LiveKit Cloud WebRTC
    LIVEKIT_URL: str = Field(
        default="",
        validation_alias=AliasChoices("LIVEKIT_URL", "NEXT_PUBLIC_LIVEKIT_URL", "livekit_url"),
    )
    LIVEKIT_API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("LIVEKIT_API_KEY", "livekit_api_key"),
    )
    LIVEKIT_API_SECRET: str = Field(
        default="",
        validation_alias=AliasChoices("LIVEKIT_API_SECRET", "livekit_api_secret"),
    )

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    DATASETS_DIR: Path = BASE_DIR / "datasets"
    PROFILES_DIR: Path = BASE_DIR / "research" / "outputs" / "profiles"

    # AI Defaults
    DEFAULT_EXPERIMENT_MODE: str = "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"
    FIXED_EAR_CUTOFF: float = 0.22
    DEFAULT_ZONE_CALIBRATION_DURATION_SEC: float = 5.0


settings = Settings()
