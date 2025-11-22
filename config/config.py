"""
Configuration management using Pydantic Settings.
Loads from .env file and provides type-safe configuration access.
"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    logs_dir: Path = Field(default=Path("logs"))

    # Data paths
    raw_data_path: Path = Field(default=Path("data/raw/transactions.csv"))
    processed_data_path: Path = Field(default=Path("data/processed/processed_data.parquet"))

    # Model Configuration
    model_type: Literal["xgboost", "lightgbm", "random_forest", "logistic"] = "xgboost"
    random_seed: int = 42
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    validation_size: float = Field(default=0.1, ge=0.0, le=1.0)

    # MLflow Configuration
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "fraud-detection"

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["text", "json"] = "json"

    # Feature Engineering
    n_jobs: int = -1
    use_gpu: bool = False

    # Model Thresholds
    fraud_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    high_risk_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1024, le=65535)

    # Database (optional)
    database_url: Optional[str] = None

    @field_validator("data_dir", "models_dir", "logs_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v)

    @field_validator("raw_data_path", "processed_data_path", mode="before")
    @classmethod
    def resolve_data_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v)

    def get_absolute_path(self, path: Path) -> Path:
        """Get absolute path relative to project root."""
        if path.is_absolute():
            return path
        return self.project_root / path

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "external",
            self.models_dir,
            self.models_dir / "saved_models",
            self.models_dir / "experiments",
            self.logs_dir,
        ]:
            abs_path = self.get_absolute_path(dir_path)
            abs_path.mkdir(parents=True, exist_ok=True)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()
