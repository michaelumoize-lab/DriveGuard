"""
Application Configuration and Environment Settings using Pydantic Settings.
"""

from pathlib import Path
from typing import List
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DEV_AUTH_SECRET = "driveguard-dev-secret-key-32-bytes-minimum-length-strictly-enforced"


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

    # Deployment Environment
    ENV: str = Field(
        default="development",
        validation_alias=AliasChoices("ENV", "APP_ENV", "NODE_ENV", "env"),
    )
    RENDER: bool = Field(
        default=False,
        validation_alias=AliasChoices("RENDER", "render"),
    )

    # Server Binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Security & Authentication
    AUTH_SECRET: str = Field(
        default=DEFAULT_DEV_AUTH_SECRET,
        validation_alias=AliasChoices("AUTH_SECRET", "BETTER_AUTH_SECRET", "auth_secret"),
    )
    AUTH_TOKEN_ISSUER: str = "driveguard-frontend"
    AUTH_TOKEN_AUDIENCE: str = "driveguard-backend"
    MAX_WS_FRAME_SIZE_BYTES: int = 2 * 1024 * 1024  # 2 MB max per video frame
    WS_IDLE_TIMEOUT_SEC: float = 60.0

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

    # CORS Allowed Origins (Wildcards removed; regex handles Vercel & localhost securely)
    CORS_ORIGINS: List[str] = [
        "https://drive-guard-mu.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    DATASETS_DIR: Path = BASE_DIR / "datasets"
    PROFILES_DIR: Path = BASE_DIR / "data" / "profiles"
    LOCAL_DB_PATH: Path = BASE_DIR / "data" / "driveguard_local.db"

    # Standalone Local Python Camera API (Loopback-only, hard opt-in; defaults False for cloud safety)
    LOCAL_CAMERA_API_ENABLED: bool = False

    # AI Defaults
    DEFAULT_EXPERIMENT_MODE: str = "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"
    FIXED_EAR_CUTOFF: float = 0.22
    DEFAULT_ZONE_CALIBRATION_DURATION_SEC: float = 5.0

    @model_validator(mode="after")
    def _validate_production_security(self) -> "Settings":
        is_prod = str(self.ENV).strip().lower() == "production" or bool(self.RENDER)
        if is_prod:
            secret = (self.AUTH_SECRET or "").strip()
            if not secret or secret == DEFAULT_DEV_AUTH_SECRET or len(secret) < 32:
                raise ValueError(
                    "Production startup blocked: AUTH_SECRET (or BETTER_AUTH_SECRET) must be "
                    "explicitly set to a non-default secret of at least 32 bytes."
                )
        return self


settings = Settings()
