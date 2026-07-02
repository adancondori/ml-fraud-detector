"""Model management endpoints: /health, /model/info, /model/reload."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fraud_detector.scoring.classifier import RISK_LEVELS
from fraud_detector.scoring.scorer import SingleTransactionScorer
from scorer import dependencies as _deps
from scorer.artifact_loader import load_artifacts
from scorer.dependencies import get_scorer
from scorer.schemas import HealthResponse, ModelInfoResponse, ReloadResponse

router = APIRouter(tags=["model"])


def _probe_ch(client) -> bool:
    """Return True if a lightweight SELECT 1 succeeds on the client."""
    if client is None:
        return False
    try:
        client.command("SELECT 1")
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return application health including model and ClickHouse status.

    clickhouse_connected is the conjunction of READ and WRITE health: a batch
    run needs both the production read client and the local write client.
    """
    model_loaded = _deps._state.get("model_loaded", False)

    read_ok = _probe_ch(
        _deps._state.get("read_ch_client", _deps._state.get("ch_client"))
    )
    write_ok = _probe_ch(_deps._state.get("write_ch_client"))
    clickhouse_connected = read_ok and write_ok

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
        model_version=scorer._model_version,
        feature_count=len(scorer._feature_names),
        feature_version=scorer._feature_version,
        threshold_version=scorer._threshold_version,
        score_function=scorer._score_function,
        threshold=scorer._classifier._threshold,
        risk_levels={level: list(bounds) for level, bounds in RISK_LEVELS.items()},
    )


@router.post("/model/reload", response_model=ReloadResponse)
def model_reload() -> ReloadResponse:
    """Atomically replace the scorer instance with a freshly loaded model."""
    from scorer.main import settings  # import here to avoid circular at module level

    model_dir = settings.model_dir
    artifacts = load_artifacts(model_dir)

    new_scorer = SingleTransactionScorer(
        feature_engineer_path=str(model_dir / "feature_engineer.joblib"),
        ch_connector=_deps._state.get("read_ch_client", _deps._state.get("ch_client")),
        artifacts=artifacts,
    )

    # Atomic replacement
    _deps._state["scorer"] = new_scorer
    _deps._state["model_version"] = new_scorer._model_version

    return ReloadResponse(status="reloaded", model_version=new_scorer._model_version)
