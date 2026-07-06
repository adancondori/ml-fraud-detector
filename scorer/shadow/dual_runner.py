"""ShadowDualRunner — scores each payment with champion and challenger simultaneously.

SHAD-01 invariants:
  - score_pair NEVER raises: a failure in one model produces an error ScoringResult
    for that model without affecting the other (partial-failure isolation).
  - Both models receive the same UserContext (factual rolling aggregates — not
    model-specific).  The context is built once upstream (BatchContextProvider) and
    shared here.  This avoids duplicating the 6 bulk queries per payment.
  - The champion model_version is always 'IF-40-v1'; the challenger is 'frame-v1'.
"""

from __future__ import annotations

from typing import Tuple

from loguru import logger

from fraud_detector.scoring.classifier import ScoringResult
from fraud_detector.scoring.context import UserContext
from fraud_detector.scoring.scorer import SingleTransactionScorer


def _error_result(model_version: str, exc: Exception) -> ScoringResult:
    """Construct a safe zero-score ScoringResult when a model raises."""
    return ScoringResult(
        score=0.0,
        is_anomaly=False,
        risk_level="minimal",
        percentile=0.0,
        factors=[],
        model_version=model_version,
        feature_version="error",
        threshold_version="error",
        calibration_segment=None,
        fallback_level=None,
        frame_flags=None,
    )


class ShadowDualRunner:
    """Scores one payment with both the champion (IF-40) and challenger (frame-v1).

    The caller is responsible for building the UserContext once via
    BatchContextProvider and passing it to score_pair.  The context is factual
    (rolling aggregates from the READ ClickHouse client) and is independent of
    the model — sharing it avoids duplicating the 6 bulk queries per payment.

    If the IF-40 scorer uses the per-payment enriched-context path internally
    (``len(feature_names) == 40``), that enrichment happens inside
    ``SingleTransactionScorer.score()``; the same base ``UserContext`` is a
    superset of the bulk factual context used by frame-v1.

    Args:
        scorer_champion: Loaded SingleTransactionScorer for IF-40 (model_version='IF-40-v1').
        scorer_new: Loaded SingleTransactionScorer for frame-v1 (model_version='frame-v1').
    """

    def __init__(
        self,
        scorer_champion: SingleTransactionScorer,
        scorer_new: SingleTransactionScorer,
    ) -> None:
        self._champion = scorer_champion
        self._new = scorer_new

    def score_pair(
        self,
        payment: dict,
        context_champion: UserContext,
        context_new: UserContext,
    ) -> Tuple[ScoringResult, ScoringResult]:
        """Score *payment* with both models.  Never raises.

        Args:
            payment: Payment dict (same format as BatchScorer._score_all).
            context_champion: UserContext for the champion scorer (IF-40 path may
                internally enrich it; frame-v1 path uses it directly).
            context_new: UserContext for the challenger scorer.  Typically the
                same object as context_champion since the context is factual and
                model-independent.

        Returns:
            ``(result_champion, result_new)`` — a ScoringResult for each model.
            If a model fails its result will have ``score=0.0``,
            ``is_anomaly=False``, ``risk_level='minimal'``.
        """
        payment_id = payment.get("payment_id")

        try:
            r_old = self._champion.score(payment, context=context_champion)
        except Exception as exc:
            logger.warning(
                "shadow champion failed payment_id={}: {}", payment_id, exc
            )
            r_old = _error_result("IF-40-v1", exc)

        try:
            r_new = self._new.score(payment, context=context_new)
        except Exception as exc:
            logger.warning(
                "shadow frame-v1 failed payment_id={}: {}", payment_id, exc
            )
            r_new = _error_result("frame-v1", exc)

        delta = abs(r_old.score - r_new.score)
        logger.debug(
            "shadow delta={:.4f} fid={} old_risk={} new_risk={}",
            delta,
            payment.get("facility_id"),
            r_old.risk_level,
            r_new.risk_level,
        )
        return r_old, r_new
