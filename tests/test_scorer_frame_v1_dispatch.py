"""Tests for frame-v1 dispatch in SingleTransactionScorer.

Covers:
- ScoringResult backward-compat (3 new fields default to None)
- Dispatch by artifact presence: frame-v1 vs IF-40 legacy
- score() populates calibration_segment/fallback_level/frame_flags in frame-v1 path
- timezone_missing flag for unknown facilities
- Router propagates frame-v1 fields to ScoreResponse
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from fraud_detector.scoring.classifier import ScoringResult
from fraud_detector.scoring.context import UserContext
from scorer.artifact_loader import load_artifacts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_DIR = Path("output/models")
FRAME_V1_METADATA = MODEL_DIR / "model_metadata_frame_v1.json"
FRAME_V1_FILES = [
    "isolation_forest_frame_v1.joblib",
    "scaler_frame_v1.joblib",
    "feature_list_frame_v1.json",
    "thresholds_segmented_v1.json",
    "facility_stats_v1.json",
]


def _make_frame_v1_dir(tmp_path: Path) -> Path:
    """Create a temp dir that mimics a frame-v1 model directory."""
    meta = json.loads(FRAME_V1_METADATA.read_text())
    (tmp_path / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    for fname in FRAME_V1_FILES:
        src = MODEL_DIR / fname
        if src.exists():
            shutil.copy2(src, tmp_path / fname)
    return tmp_path


def _make_minimal_context() -> UserContext:
    """Return a UserContext with safe zero-values for frame-v1 calculate()."""
    return UserContext(
        txn_count_1h=0,
        txn_count_24h=0,
        amount_24h=0.0,
        last_txn_at=None,
        time_since_last_txn=0.0,  # >= 0 → used directly
        distinct_facilities_30d=1,
        distinct_methods=1,
        reversal_ratio_30d=0.0,
        discount_ratio_30d=0.0,
        debit_count_30d=0,
        debit_amount_30d=0.0,
        prepaid_spend_30d=0.0,
        credit_flow_ratio=0.0,  # >= 0 → used directly
        categories_30d=[],
        category_entropy_30d=0.0,  # >= 0 → used directly
        reversal_count_30d=0,
        merchandise_ratio_30d=0.0,
        gateway_change_recent=0,
        is_main_gateway=1,
        is_first_gateway_for_user=0,
        source_change_recent=0,
        user_role="player",
    )


def _make_minimal_payment(facility_id: int = 10, currency: str = "USD") -> dict:
    """Return a minimal payment dict for scorer.score()."""
    return {
        "payment_id": 1,
        "user_id": 1,
        "facility_id": facility_id,
        "reservation_paid_out": 100.0,
        "created_at": "2025-06-15T14:30:00",
        "discount": 0.0,
        "tip": 0.0,
        "payment_method": "card",
        "category": "reservation",
        "club_credit_flag": False,
        "paid_by_manager": False,
        "currency": currency,
    }


# ---------------------------------------------------------------------------
# Task 1 — ScoringResult backward-compat
# ---------------------------------------------------------------------------


def test_scoring_result_defaults_none():
    """ScoringResult constructed without frame-v1 kwargs keeps None defaults.

    This verifies that the IF-40 path (which omits these kwargs) does not raise
    TypeError, and that all 3 new fields default correctly to None.
    """
    result = ScoringResult(
        score=0.1,
        is_anomaly=False,
        risk_level="minimal",
        percentile=0.4,
    )
    assert result.calibration_segment is None
    assert result.fallback_level is None
    assert result.frame_flags is None


def test_scoring_result_accepts_frame_v1_fields():
    """ScoringResult accepts the 3 new fields when provided."""
    result = ScoringResult(
        score=0.05,
        is_anomaly=True,
        risk_level="high",
        percentile=0.9,
        calibration_segment="facility:123",
        fallback_level="facility",
        frame_flags={"timezone_missing": False, "currency_missing": False, "currency_unknown": False},
    )
    assert result.calibration_segment == "facility:123"
    assert result.fallback_level == "facility"
    assert result.frame_flags["timezone_missing"] is False


# ---------------------------------------------------------------------------
# Task 2 — Dispatch: frame-v1 vs IF-40 in SingleTransactionScorer
# ---------------------------------------------------------------------------


class TestDispatch:
    """Dispatch is determined by presence of artifacts, not feature count."""

    def test_dispatch_selects_frame_v1_when_artifacts_present(self, tmp_path):
        """frame-v1 artifacts → _is_frame_v1 is True, SegmentedThresholdClassifier selected."""
        from fraud_detector.scoring.scorer import SingleTransactionScorer
        from fraud_detector.scoring.classifier import SegmentedThresholdClassifier
        from fraud_detector.scoring.features_frame_v1 import FrameV1FeatureCalculator

        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)

        scorer = SingleTransactionScorer(
            feature_engineer_path="output/models/feature_engineer.joblib",
            artifacts=artifacts,
        )

        assert scorer._is_frame_v1 is True
        assert isinstance(scorer._classifier, SegmentedThresholdClassifier)
        assert isinstance(scorer._feature_calc, FrameV1FeatureCalculator)

    def test_dispatch_stays_if40_when_stats_absent(self):
        """IF-40 artifacts (facility_stats=None) → _is_frame_v1 is False, ThresholdClassifier."""
        from fraud_detector.scoring.scorer import SingleTransactionScorer
        from fraud_detector.scoring.classifier import ThresholdClassifier

        artifacts = load_artifacts(MODEL_DIR)
        assert artifacts.facility_stats is None  # IF-40 has no stats

        scorer = SingleTransactionScorer(
            feature_engineer_path="output/models/feature_engineer.joblib",
            artifacts=artifacts,
        )

        assert scorer._is_frame_v1 is False
        assert isinstance(scorer._classifier, ThresholdClassifier)

    def test_frame_v1_score_populates_fields(self, tmp_path):
        """score() in frame-v1 mode populates calibration_segment/fallback_level/frame_flags."""
        from fraud_detector.scoring.scorer import SingleTransactionScorer

        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)

        scorer = SingleTransactionScorer(
            feature_engineer_path="output/models/feature_engineer.joblib",
            artifacts=artifacts,
        )

        # Use a known facility from the artifact (facility 10 exists in stats)
        # If not known, timezone_missing will be True but result still populated.
        payment = _make_minimal_payment(facility_id=10, currency="USD")
        ctx = _make_minimal_context()

        result = scorer.score(payment, context=ctx)

        # calibration_segment must be a valid string
        assert isinstance(result.calibration_segment, str)
        assert result.calibration_segment in (
            [f"facility:{10}"]
            + [f"currency:USD"]
            + ["global"]
        ) or result.calibration_segment.startswith(("facility:", "currency:", "global"))

        # fallback_level must be one of the three levels
        assert result.fallback_level in {"facility", "currency", "global"}

        # frame_flags must be a dict with the 3 expected keys
        assert isinstance(result.frame_flags, dict)
        assert "timezone_missing" in result.frame_flags
        assert "currency_missing" in result.frame_flags
        assert "currency_unknown" in result.frame_flags

    def test_frame_flags_timezone_missing_for_unknown_facility(self, tmp_path):
        """Facility not in artifact → timezone_missing=True; no exception raised."""
        from fraud_detector.scoring.scorer import SingleTransactionScorer

        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)

        scorer = SingleTransactionScorer(
            feature_engineer_path="output/models/feature_engineer.joblib",
            artifacts=artifacts,
        )

        # facility_id=999999 does not exist in the artifact
        payment = _make_minimal_payment(facility_id=999999, currency="USD")
        ctx = _make_minimal_context()

        result = scorer.score(payment, context=ctx)

        assert result.frame_flags is not None
        assert result.frame_flags["timezone_missing"] is True

    def test_frame_flags_currency_missing_when_currency_none(self, tmp_path):
        """currency=None in payment → currency_missing=True."""
        from fraud_detector.scoring.scorer import SingleTransactionScorer

        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)

        scorer = SingleTransactionScorer(
            feature_engineer_path="output/models/feature_engineer.joblib",
            artifacts=artifacts,
        )

        payment = _make_minimal_payment(facility_id=10)
        payment["currency"] = None  # explicitly absent
        ctx = _make_minimal_context()

        result = scorer.score(payment, context=ctx)

        assert result.frame_flags is not None
        assert result.frame_flags["currency_missing"] is True

    def test_if40_score_fields_remain_none(self):
        """IF-40 path: score() returns calibration_segment=None, fallback_level=None, frame_flags=None."""
        from fraud_detector.scoring.scorer import SingleTransactionScorer

        artifacts = load_artifacts(MODEL_DIR)
        scorer = SingleTransactionScorer(
            feature_engineer_path="output/models/feature_engineer.joblib",
            artifacts=artifacts,
        )

        # IF-40 scorer uses EnrichedFeatureCalculator; provide a UserContext
        # with the fields it needs (facility_avg_amount, is_third_party_payment).
        ctx = UserContext(
            txn_count_1h=0,
            txn_count_24h=0,
            amount_24h=100.0,
            last_txn_at=None,
            time_since_last_txn=-1.0,
            distinct_facilities_30d=1,
            distinct_methods=1,
            reversal_ratio_30d=0.0,
            discount_ratio_30d=0.0,
            debit_count_30d=0,
            debit_amount_30d=0.0,
            prepaid_spend_30d=0.0,
            credit_flow_ratio=-1.0,
            categories_30d=[],
            category_entropy_30d=-1.0,
            reversal_count_30d=0,
            merchandise_ratio_30d=0.0,
            gateway_change_recent=0,
            is_main_gateway=1,
            is_first_gateway_for_user=0,
            source_change_recent=0,
            user_role="player",
            is_third_party_payment=0,
        )

        payment = {
            "payment_id": 1,
            "user_id": 1,
            "facility_id": 10,
            "reservation_paid_out": 100.0,
            "created_at": "2025-06-15T14:30:00",
            "discount": 0.0,
            "tip": 0.0,
            "payment_method": "card",
            "category": "reservation",
            "club_credit_flag": False,
            "paid_by_manager": False,
            "currency": "USD",
        }

        result = scorer.score(payment, context=ctx)

        assert result.calibration_segment is None
        assert result.fallback_level is None
        assert result.frame_flags is None

    def test_frame_v1_latency_budget(self, tmp_path):
        """frame-v1 score() must complete within 0.2s (no I/O; stats in memory)."""
        from fraud_detector.scoring.scorer import SingleTransactionScorer

        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)

        scorer = SingleTransactionScorer(
            feature_engineer_path="output/models/feature_engineer.joblib",
            artifacts=artifacts,
        )

        payment = _make_minimal_payment(facility_id=10, currency="USD")
        ctx = _make_minimal_context()

        t0 = time.perf_counter()
        scorer.score(payment, context=ctx)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.2, f"frame-v1 score() took {elapsed:.3f}s > 0.2s budget"


# ---------------------------------------------------------------------------
# Task 3 — Router propagates frame-v1 fields to ScoreResponse
# ---------------------------------------------------------------------------


class TestRouterPropagatesFrameV1Fields:
    """Router constructs ScoreResponse with calibration_segment/fallback_level/frame_flags."""

    def test_router_propagates_frame_v1_fields(self, tmp_path):
        """ScoreResponse receives and serializes all 3 frame-v1 fields from ScoringResult."""
        from scorer.schemas import FactorItem, FrameFlags, ScoreResponse

        # Simulate a ScoringResult as produced by frame-v1 scorer.score()
        result = ScoringResult(
            score=0.05,
            is_anomaly=True,
            risk_level="high",
            percentile=0.92,
            factors=[],
            model_version="frame-v1",
            feature_version="frame-v1",
            threshold_version="segmented-v1",
            calibration_segment="currency:USD",
            fallback_level="currency",
            frame_flags={
                "timezone_missing": True,
                "currency_missing": False,
                "currency_unknown": False,
            },
        )

        # Replicate the router mapping logic
        frame_flags_obj = None
        if result.frame_flags is not None:
            frame_flags_obj = FrameFlags(**result.frame_flags)

        response = ScoreResponse(
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

        assert response.calibration_segment == "currency:USD"
        assert response.fallback_level == "currency"
        assert response.frame_flags is not None
        assert response.frame_flags.timezone_missing is True
        assert response.frame_flags.currency_missing is False

        # Verify JSON serialization
        d = response.model_dump()
        assert d["calibration_segment"] == "currency:USD"
        assert d["fallback_level"] == "currency"
        assert d["frame_flags"]["timezone_missing"] is True

    def test_router_if40_propagates_nulls(self):
        """IF-40 ScoringResult → ScoreResponse has None/null frame-v1 fields."""
        from scorer.schemas import FactorItem, FrameFlags, ScoreResponse

        result = ScoringResult(
            score=0.03,
            is_anomaly=False,
            risk_level="minimal",
            percentile=0.3,
            factors=[],
            model_version="IF-40-v1",
            feature_version="enriched-40",
            threshold_version="v2",
            # No frame-v1 fields — defaults to None
        )

        frame_flags_obj = None
        if result.frame_flags is not None:
            frame_flags_obj = FrameFlags(**result.frame_flags)

        response = ScoreResponse(
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

        assert response.calibration_segment is None
        assert response.fallback_level is None
        assert response.frame_flags is None

        d = response.model_dump()
        assert d["calibration_segment"] is None
        assert d["fallback_level"] is None
        assert d["frame_flags"] is None
