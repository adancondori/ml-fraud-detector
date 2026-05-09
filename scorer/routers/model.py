"""Model management endpoints: /health, /model/info, /model/reload."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from fraud_detector.scoring.classifier import RISK_LEVELS
from fraud_detector.scoring.scorer import SingleTransactionScorer
from scorer import dependencies as _deps
from scorer.dependencies import get_scorer
from scorer.schemas import HealthResponse, ModelInfoResponse, ReloadResponse

router = APIRouter(tags=["model"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return application health including model and ClickHouse status."""
    model_loaded = _deps._state.get("model_loaded", False)

    # Probe ClickHouse with a lightweight command
    clickhouse_connected = False
    ch_client = _deps._state.get("ch_client")
    if ch_client is not None:
        try:
            ch_client.command("SELECT 1")
            clickhouse_connected = True
        except Exception:
            clickhouse_connected = False

    return HealthResponse(
        model_loaded=model_loaded,
        clickhouse_connected=clickhouse_connected,
        model_version=_deps._state.get("model_version", "unknown"),
        last_batch_at=_deps._state.get("last_batch_at"),
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(scorer: SingleTransactionScorer = Depends(get_scorer)) -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    return ModelInfoResponse(
        model_version=_deps._state.get("model_version", "unknown"),
        feature_count=31,
        threshold=scorer._classifier._threshold,
        risk_levels={level: list(bounds) for level, bounds in RISK_LEVELS.items()},
    )


@router.post("/model/reload", response_model=ReloadResponse)
def model_reload() -> ReloadResponse:
    """Atomically replace the scorer instance with a freshly loaded model."""
    from scorer.main import settings  # import here to avoid circular at module level

    model_dir = settings.model_dir

    def _load_version(model_dir: Path) -> str:
        thresholds_path = model_dir / "thresholds.json"
        try:
            with open(thresholds_path) as f:
                data = json.load(f)
            return data.get("model_version", "IF-31-v1")
        except Exception:
            return "IF-31-v1"

    new_scorer = SingleTransactionScorer(
        model_path=str(model_dir / "isolation_forest.joblib"),
        scaler_path=str(model_dir / "scaler.joblib"),
        feature_engineer_path=str(model_dir / "feature_engineer.joblib"),
        thresholds_path=str(model_dir / "thresholds.json"),
        ch_connector=None,
    )

    new_version = _load_version(model_dir)

    # Atomic replacement
    _deps._state["scorer"] = new_scorer
    _deps._state["model_version"] = new_version

    return ReloadResponse(status="reloaded", model_version=new_version)
