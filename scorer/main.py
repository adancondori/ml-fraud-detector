"""FastAPI application entry point for ML Fraud Scorer.

Lifespan loads SingleTransactionScorer and ClickHouse client once at startup
and stores them in the shared _state dict from dependencies.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import clickhouse_connect
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from fraud_detector.scoring.scorer import SingleTransactionScorer
from scorer.artifact_loader import load_artifacts
from scorer import dependencies as _deps
from scorer.batch.scorer import DEFAULT_ANOMALY_SCORES_TABLE, ch_fingerprint


class ScorerSettings(BaseSettings):
    """Scorer-specific configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_dir: Path = Path("output/models")

    # READ target — production ClickHouse (read-only). Used for cursor
    # resolution, payment fetch, batch context and single-score context.
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "pbp_productionDB_optimized"
    clickhouse_secure: bool = True

    # WRITE target — local ClickHouse where anomaly_scores are inserted.
    # Defaults point at the local docker service so the guardrail's
    # local-host check passes out of the box; never default to production.
    anomaly_scores_ch_host: str = "clickhouse"
    anomaly_scores_ch_port: int = 8123
    anomaly_scores_ch_user: str = "default"
    anomaly_scores_ch_password: str = ""
    anomaly_scores_ch_database: str = "pbp_productionDB_optimized"
    anomaly_scores_ch_secure: bool = False

    # Destination table for the INSERT. Local reuses the prod DB *name*.
    anomaly_scores_table: str = DEFAULT_ANOMALY_SCORES_TABLE

    # Explicit, off-by-default bypass for the non-local WRITE host guardrail.
    allow_nonlocal_anomaly_score_writes: bool = False

    # Shadow dual-run configuration (Fase 4 — SHAD-01).
    # scoring_mode='active'      → single scorer, no dual-run (default, backward compat).
    # scoring_mode='shadow_dual' → loads champion (IF-40) + challenger (frame-v1),
    #                              produces 2 rows/payment in anomaly_scores.
    scoring_mode: str = "active"
    # Metadata filenames used to distinguish the two models when both live in model_dir.
    shadow_champion_metadata: str = "model_metadata.json"
    shadow_new_metadata: str = "model_metadata_frame_v1.json"


settings = ScorerSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and ClickHouse client once at startup; clean up on shutdown."""
    model_dir = settings.model_dir

    artifacts = load_artifacts(model_dir)

    # READ client — production ClickHouse (read-only).
    # autogenerate_session_id=False: avoids a server-side session lock so the
    # /health probe ("SELECT 1") can run concurrently with a long batch on the
    # same client without raising "concurrent queries within the same session".
    read_ch_client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
        secure=settings.clickhouse_secure,
        autogenerate_session_id=False,
    )

    # WRITE client — local ClickHouse for anomaly_scores INSERT.
    write_ch_client = clickhouse_connect.get_client(
        host=settings.anomaly_scores_ch_host,
        port=settings.anomaly_scores_ch_port,
        username=settings.anomaly_scores_ch_user,
        password=settings.anomaly_scores_ch_password,
        database=settings.anomaly_scores_ch_database,
        secure=settings.anomaly_scores_ch_secure,
        autogenerate_session_id=False,
    )

    scorer = SingleTransactionScorer(
        feature_engineer_path=str(model_dir / "feature_engineer.joblib"),
        ch_connector=read_ch_client,
        artifacts=artifacts,
    )

    # Populate shared state
    _deps._state["scorer"] = scorer
    _deps._state["read_ch_client"] = read_ch_client
    _deps._state["write_ch_client"] = write_ch_client
    _deps._state["ch_client"] = read_ch_client  # backward-compat alias (READ)
    _deps._state["model_loaded"] = True
    _deps._state["model_version"] = scorer._model_version
    _deps._state["last_batch_at"] = None
    _deps._state["scoring_mode"] = settings.scoring_mode

    # Shadow dual-run (SHAD-01): load champion + challenger when scoring_mode='shadow_dual'.
    # Both models live in the same model_dir; they are distinguished by metadata filename.
    # The active scorer above is unaffected — this block is purely additive.
    if settings.scoring_mode == "shadow_dual":
        import logging as _logging

        _dual_logger = _logging.getLogger(__name__)

        champion_artifacts = load_artifacts(
            model_dir, metadata_filename=settings.shadow_champion_metadata
        )
        new_artifacts = load_artifacts(
            model_dir, metadata_filename=settings.shadow_new_metadata
        )
        scorer_champion = SingleTransactionScorer(
            feature_engineer_path=str(model_dir / "feature_engineer.joblib"),
            ch_connector=read_ch_client,
            artifacts=champion_artifacts,
        )
        scorer_new = SingleTransactionScorer(
            feature_engineer_path=str(model_dir / "feature_engineer.joblib"),
            ch_connector=read_ch_client,
            artifacts=new_artifacts,
        )

        assert scorer_champion._model_version == "IF-40-v1", (
            f"shadow_dual: expected champion model_version='IF-40-v1', "
            f"got {scorer_champion._model_version!r}. "
            f"Check shadow_champion_metadata setting."
        )
        assert scorer_new._model_version == "frame-v1", (
            f"shadow_dual: expected challenger model_version='frame-v1', "
            f"got {scorer_new._model_version!r}. "
            f"Check shadow_new_metadata setting."
        )

        _dual_logger.info(
            "shadow_dual: champion=%s challenger=%s",
            scorer_champion._model_version,
            scorer_new._model_version,
        )

        _deps._state["scorer_champion"] = scorer_champion
        _deps._state["scorer_new"] = scorer_new

    # Guardrail metadata consumed by the /score/batch route.
    _deps._state["anomaly_scores_table"] = settings.anomaly_scores_table
    _deps._state["read_fingerprint"] = ch_fingerprint(
        settings.clickhouse_host,
        settings.clickhouse_port,
        settings.clickhouse_database,
        settings.clickhouse_secure,
        settings.clickhouse_user,
    )
    _deps._state["write_fingerprint"] = ch_fingerprint(
        settings.anomaly_scores_ch_host,
        settings.anomaly_scores_ch_port,
        settings.anomaly_scores_ch_database,
        settings.anomaly_scores_ch_secure,
        settings.anomaly_scores_ch_user,
    )
    _deps._state["write_host"] = settings.anomaly_scores_ch_host
    _deps._state["allow_nonlocal_write"] = settings.allow_nonlocal_anomaly_score_writes

    yield

    # Shutdown: close both ClickHouse connections and clear state
    for _client in (read_ch_client, write_ch_client):
        try:
            _client.close()
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
