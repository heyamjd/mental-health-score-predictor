"""
backend/app/settings.py
========================
Application settings loaded from environment variables.
Uses pydantic-settings for validation and .env file support.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration is driven by environment variables.
    Defaults make the app work out of the box locally.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    app_name: str = "Mental Health Score Predictor API"
    app_version: str = "1.0.0"
    debug: bool = False

    # CORS — comma-separated list of allowed origins
    # Example: "https://my-site.onrender.com,http://localhost:5500"
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated allowed CORS origins. Use specific domains in production.",
    )

    # Model artifact path (relative to backend/ or absolute)
    model_path: Path = Field(
        default=Path(__file__).resolve().parents[2] / "models" / "pipeline.joblib",
        description="Path to the trained sklearn pipeline artifact.",
    )
    metadata_path: Path = Field(
        default=Path(__file__).resolve().parents[2] / "models" / "metadata.json",
        description="Path to model metadata JSON.",
    )

    # Rate limiting
    rate_limit: str = "30/minute"   # slowapi format

    @property
    def cors_origins(self) -> list[str]:
        """Return ALLOWED_ORIGINS as a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


# Singleton — imported everywhere
settings = Settings()
