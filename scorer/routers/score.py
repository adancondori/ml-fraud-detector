"""Scoring endpoints: POST /score (single) and POST /score/batch."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from loguru import logger

from fraud_detector.scoring.scorer import SingleTransactionScorer
from scorer import dependencies as _deps
from scorer.dependencies import get_ch_client, get_scorer
from scorer.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    CriticalAlert,
    FactorItem,
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

    return ScoreResponse(
        raw_score=result.score,
        percentile=result.percentile,
        risk_level=result.risk_level,
        is_anomaly=result.is_anomaly,
        factors=[FactorItem(**f) for f in result.factors],
    )


@router.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(
    request: BatchScoreRequest,
    scorer: SingleTransactionScorer = Depends(get_scorer),
    ch_client=Depends(get_ch_client),
) -> BatchScoreResponse:
    """Trigger a batch scoring run from a cursor datetime.

    Creates a BatchScorer on every call so the scorer + ch_client injected
    by DI are always the current (potentially reloaded) instances.
    FastAPI runs sync def in a threadpool — correct for CPU-bound work.
    """
    logger.info(f"POST /score/batch — cursor={request.cursor.isoformat()}")

    batch_scorer = BatchScorer(scorer=scorer, ch_client=ch_client)
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
