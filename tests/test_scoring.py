"""Tests for SingleTransactionScorer — Fase 12 contracts."""
from __future__ import annotations

import numpy as np
import pytest

from fraud_detector.features.engineering import FEATURE_NAMES
from fraud_detector.scoring.classifier import ScoringResult, ThresholdClassifier
from fraud_detector.scoring.context import UserContext
from fraud_detector.scoring.features import SingleFeatureCalculator
from fraud_detector.scoring.scorer import SingleTransactionScorer

MODEL_PATH = "output/models/isolation_forest.joblib"
SCALER_PATH = "output/models/scaler.joblib"
FE_PATH = "output/models/feature_engineer.joblib"
THRESHOLDS_PATH = "output/models/thresholds.json"

SAMPLE_PAYMENT = {
    "user_id": 12345,
    "facility_id": 67,
    "reservation_paid_out": 150.00,
    "discount": 0,
    "tip": 5.0,
    "payment_method": "card",
    "category": "reservation",
    "club_credit_flag": False,
    "paid_by_manager": False,
    "currency": "USD",
    "created_at": "2025-10-15T14:30:00",
}

SAMPLE_CONTEXT = UserContext(
    txn_count_1h=2,
    txn_count_24h=5,
    amount_24h=300.0,
    distinct_facilities_30d=1,
    distinct_methods=2,
    reversal_ratio_30d=0.0,
    discount_ratio_30d=0.01,
    debit_count_30d=0,
    debit_amount_30d=0.0,
    prepaid_spend_30d=0.0,
    categories_30d=["reservation", "reservation", "debit"],
    reversal_count_30d=0,
    merchandise_ratio_30d=0.0,
    user_role="player",
)


@pytest.fixture
def calculator():
    return SingleFeatureCalculator(FE_PATH)


@pytest.fixture
def classifier():
    return ThresholdClassifier(THRESHOLDS_PATH)


@pytest.fixture
def scorer():
    return SingleTransactionScorer(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        feature_engineer_path=FE_PATH,
        thresholds_path=THRESHOLDS_PATH,
        ch_connector=None,
    )


# --- Feature calculation ---


def test_feature_count_is_31(calculator):
    features = calculator.calculate(SAMPLE_PAYMENT, SAMPLE_CONTEXT)
    assert features.shape == (31,)


def test_features_are_finite(calculator):
    features = calculator.calculate(SAMPLE_PAYMENT, SAMPLE_CONTEXT)
    assert np.isfinite(features).all()


def test_features_dtype_float32(calculator):
    features = calculator.calculate(SAMPLE_PAYMENT, SAMPLE_CONTEXT)
    assert features.dtype == np.float32


# --- Classifier ---


def test_classifier_high_score_is_anomaly(classifier):
    is_anom, level, pct = classifier.classify(999.0)
    assert is_anom is True
    assert level == "critical"


def test_classifier_low_score_is_normal(classifier):
    is_anom, level, pct = classifier.classify(-999.0)
    assert is_anom is False
    assert level == "minimal"


def test_classifier_percentile_range(classifier):
    _, _, pct = classifier.classify(0.0)
    assert 0.0 <= pct <= 1.0


# --- Full scorer ---


def test_scorer_returns_scoring_result(scorer):
    result = scorer.score(SAMPLE_PAYMENT, context=SAMPLE_CONTEXT)
    assert isinstance(result, ScoringResult)
    assert isinstance(result.score, float)
    assert isinstance(result.is_anomaly, bool)
    assert result.risk_level in ("minimal", "low", "medium", "high", "critical")
    assert 0.0 <= result.percentile <= 1.0


def test_scorer_factors_sorted_by_importance(scorer):
    result = scorer.score(SAMPLE_PAYMENT, context=SAMPLE_CONTEXT)
    z_scores = [abs(f["z_score"]) for f in result.factors]
    assert z_scores == sorted(z_scores, reverse=True)


def test_scorer_factors_have_feature_names(scorer):
    result = scorer.score(SAMPLE_PAYMENT, context=SAMPLE_CONTEXT)
    for f in result.factors:
        assert f["feature"] in FEATURE_NAMES
        assert "value" in f
        assert "z_score" in f
        assert f["direction"] in ("high", "low")


def test_scorer_max_5_factors(scorer):
    result = scorer.score(SAMPLE_PAYMENT, context=SAMPLE_CONTEXT)
    assert len(result.factors) <= 5


# --- Context with zeros ---


def test_empty_context_produces_valid_score(scorer):
    empty_ctx = UserContext()
    result = scorer.score(SAMPLE_PAYMENT, context=empty_ctx)
    assert isinstance(result.score, float)
    assert np.isfinite(result.score)


# --- UserContext dataclass ---


def test_user_context_defaults():
    ctx = UserContext()
    assert ctx.txn_count_1h == 0
    assert ctx.txn_count_24h == 0
    assert ctx.user_role == "player"
    assert ctx.categories_30d == []
