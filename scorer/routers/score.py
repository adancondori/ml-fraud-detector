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
