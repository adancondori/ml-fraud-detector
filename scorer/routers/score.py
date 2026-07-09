"""Scoring endpoints: POST /score (single) and POST /score/batch."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from loguru import logger

from fraud_detector.scoring.scorer import SingleTransactionScorer
from scorer import dependencies as _deps
from scorer.dependencies import get_read_ch_client, get_scorer, get_write_ch_client
from scorer.batch.scorer import DEFAULT_ANOMALY_SCORES_TABLE
from scorer.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    CriticalAlert,
    FactorItem,
    FrameFlags,
    ScoreRequest,
    ScoreResponse,
)

from scorer.batch.scorer import BatchScorer

router = APIRouter(tags=["scoring"])


@router.post("/score", response_model=ScoreResponse)
def score_single(
    request: ScoreRequest,
    scorer: SingleTransactionScorer = Depends(get_scorer),
) -> ScoreResponse:
    """Score a single transaction end-to-end.

    Delegates entirely to SingleTransactionScorer.score() — no feature
    reimplementation here.  FastAPI runs sync def endpoints in a threadpool,
    which is correct for CPU-bound sklearn work.
    """
    payment = request.model_dump()
    result = scorer.score(payment)

    frame_flags_obj = None
    if result.frame_flags is not None:
        frame_flags_obj = FrameFlags(**result.frame_flags)

    return ScoreResponse(
        raw_score=result.score,
        percentile=result.percentile,
        risk_level=result.risk_level,
        is_anomaly=result.is_anomaly,
        factors=[FactorItem(**f) for f in result.factors],
        model_version=result.model_version,
        feature_version=result.feature_version,
        threshold_version=result.threshold_version,
        calibration_segment=result.calibration_segment,
        fallback_level=result.fallback_level,
        frame_flags=frame_flags_obj,
    )


@router.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(
    request: BatchScoreRequest,
    scorer: SingleTransactionScorer = Depends(get_scorer),
    read_ch_client=Depends(get_read_ch_client),
    write_ch_client=Depends(get_write_ch_client),
) -> BatchScoreResponse:
    """Trigger a batch scoring run from a cursor datetime.

    Creates a BatchScorer on every call so the scorer + clients injected by DI
    are always the current (potentially reloaded) instances. READ runs against
    production (read-only); the anomaly_scores INSERT runs against the local
    WRITE client, guarded by metadata resolved at startup.
    FastAPI runs sync def in a threadpool — correct for CPU-bound work.
    """
    logger.info(f"POST /score/batch — cursor={request.cursor.isoformat()}")

    # Dual-run wiring: when the service started in shadow_dual mode, lifespan
    # populates _state["scorer_new"] with the frame-v1 challenger. Passing it
    # here activates BatchScorer's dual path (2 rows/payment: shadow_old +
    # shadow_new). In active mode the key is absent → None → single-model path,
    # so this is backward compatible.
    batch_scorer = BatchScorer(
        scorer=scorer,
        scorer_shadow=_deps._state.get("scorer_new"),
        read_ch_client=read_ch_client,
        write_ch_client=write_ch_client,
        anomaly_scores_table=_deps._state.get(
            "anomaly_scores_table", DEFAULT_ANOMALY_SCORES_TABLE
        ),
        read_fingerprint=_deps._state.get("read_fingerprint"),
        write_fingerprint=_deps._state.get("write_fingerprint"),
        write_host=_deps._state.get("write_host"),
        allow_nonlocal_write=_deps._state.get("allow_nonlocal_write", False),
    )
    result = batch_scorer.score_batch(cursor=request.cursor)

    _deps._state["last_batch_at"] = datetime.utcnow()

    logger.info(
        f"POST /score/batch — processed={result['processed']} "
        f"scored={result['scored']} "
        f"critical_alerts={len(result['critical_alerts'])}"
    )

    return BatchScoreResponse(
        processed=result["processed"],
        scored=result["scored"],
        critical_alerts=[CriticalAlert(**a) for a in result["critical_alerts"]],
        next_cursor=result.get("next_cursor"),
    )
