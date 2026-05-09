"""FastAPI application entry point for ML Fraud Scorer.

Lifespan loads SingleTransactionScorer and ClickHouse client once at startup
and stores them in the shared _state dict from dependencies.py.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import clickhouse_connect
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from fraud_detector.scoring.scorer import SingleTransactionScorer
from scorer import dependencies as _deps


class ScorerSettings(BaseSettings):
    """Scorer-specific configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_dir: Path = Path("output/models")
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "pbp_productionDB_optimized"
    clickhouse_secure: bool = False


settings = ScorerSettings()


def _load_model_version(model_dir: Path) -> str:
    """Read model version from thresholds.json, default to 'IF-31-v1'."""
    thresholds_path = model_dir / "thresholds.json"
    try:
        with open(thresholds_path) as f:
            data = json.load(f)
        return data.get("model_version", "IF-31-v1")
    except Exception:
        return "IF-31-v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and ClickHouse client once at startup; clean up on shutdown."""
    model_dir = settings.model_dir

    # Load scorer
    scorer = SingleTransactionScorer(
        model_path=str(model_dir / "isolation_forest.joblib"),
        scaler_path=str(model_dir / "scaler.joblib"),
        feature_engineer_path=str(model_dir / "feature_engineer.joblib"),
        thresholds_path=str(model_dir / "thresholds.json"),
        ch_connector=None,  # context queries use the shared client below
    )

    # Load ClickHouse client
    ch_client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
        secure=settings.clickhouse_secure,
    )

    model_version = _load_model_version(model_dir)

    # Populate shared state
    _deps._state["scorer"] = scorer
    _deps._state["ch_client"] = ch_client
    _deps._state["model_loaded"] = True
    _deps._state["model_version"] = model_version
    _deps._state["last_batch_at"] = None

    yield

    # Shutdown: close ClickHouse connection and clear state
    try:
        ch_client.close()
    except Exception:
        pass
    _deps._state.clear()


app = FastAPI(
    title="ML Fraud Scorer",
    version="1.0.0",
    lifespan=lifespan,
)

# Routers registered after their modules are created
from scorer.routers.model import router as model_router  # noqa: E402
from scorer.routers.score import router as score_router  # noqa: E402

app.include_router(model_router, prefix="/api/v1")
app.include_router(score_router, prefix="/api/v1")
