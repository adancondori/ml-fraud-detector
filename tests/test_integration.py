"""Integration smoke test — full pipeline with synthetic data (no ClickHouse)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detector.features.engineering import FEATURE_NAMES, FeatureEngineer
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
from fraud_detector.models.trainer import ModelTrainer
from fraud_detector.evaluation.metrics import evaluate_scores


def _make_synthetic_data(n=500, seed=42):
    """Create minimal synthetic dataset that passes through the full pipeline."""
    rng = np.random.default_rng(seed)
    statuses = rng.choice(
        ["captured", "totally_refunded", "refunded_to_credit", ""],
        size=n,
        p=[0.80, 0.05, 0.02, 0.13],
    )
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "user_id": rng.choice(range(1, 51), size=n),
            "facility_id": rng.choice(range(1, 6), size=n),
            "created_at": pd.date_range("2025-03-01", periods=n, freq="30min"),
            "amount": rng.lognormal(mean=3.0, sigma=1.0, size=n).round(2).astype(np.float32),
            "discount": (rng.random(n) * 5).round(2).astype(np.float32),
            "tip": (rng.random(n) * 3).round(2).astype(np.float32),
            "payment_method": rng.choice(["card", "wallet", "club_credit"], size=n),
            "status": statuses,
            "currency": "USD",
            "category": rng.choice(["reservation", "debit", "merchandise", "prepaid"], size=n),
            "club_credit_flag": rng.choice([True, False], size=n, p=[0.1, 0.9]),
            "paid_by_manager": rng.choice([0, 1], size=n, p=[0.85, 0.15]).astype(np.int8),
            "user_role": rng.choice(
                ["player", "court_manager", "teacher"], size=n, p=[0.8, 0.15, 0.05]
            ),
            "is_staff": np.zeros(n, dtype=np.int8),
            "user_created_at": pd.Timestamp("2024-06-01"),
        }
    )


@pytest.mark.slow
def test_full_pipeline_smoke():
    """End-to-end: synthetic data → features → preprocess → train → score → evaluate."""
    df = _make_synthetic_data(n=500)

    # Mark is_staff based on role
    df.loc[df["user_role"].isin(["court_manager", "teacher"]), "is_staff"] = 1

    # Feature engineering
    fe = FeatureEngineer()
    fe.fit(df)
    df_feat = fe.transform(df)
    assert len(df_feat) == 500
    assert all(f in df_feat.columns for f in FEATURE_NAMES)
    assert df_feat[FEATURE_NAMES].isna().sum().sum() == 0

    # Preprocessing
    prep = UnsupervisedPreprocessor(variant="full")
    X = prep.fit_transform(df_feat)
    assert X.shape == (500, 31)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()

    # Train IF
    trainer = ModelTrainer(
        model_type="isolation_forest",
        model_params={"n_estimators": 50, "max_samples": 64},
    )
    trainer.fit(X)
    scores = trainer.decision_function_scores(X)
    assert scores.shape == (500,)
    assert np.isfinite(scores).all()

    # Evaluate
    proxy = (
        df_feat["status"].isin(["totally_refunded", "refunded_to_credit"]).astype(np.int8).values
    )
    if proxy.sum() >= 2:
        metrics = evaluate_scores(proxy, scores)
        assert "auc_roc" in metrics
        assert "ap" in metrics
        assert 0 <= metrics["auc_roc"] <= 1


@pytest.mark.slow
def test_all_three_models_produce_scores():
    """IF, LOF, OC-SVM all produce finite scores on synthetic data."""
    df = _make_synthetic_data(n=300)
    df.loc[df["user_role"].isin(["court_manager", "teacher"]), "is_staff"] = 1

    fe = FeatureEngineer()
    fe.fit(df)
    df_feat = fe.transform(df)

    prep = UnsupervisedPreprocessor(variant="full")
    X = prep.fit_transform(df_feat)

    for model_type in ["isolation_forest", "lof", "ocsvm"]:
        params = {"n_estimators": 30} if model_type == "isolation_forest" else {}
        trainer = ModelTrainer(model_type=model_type, model_params=params)
        trainer.fit(X)
        scores = trainer.decision_function_scores(X)
        assert scores.shape == (300,), f"{model_type} wrong shape"
        assert np.isfinite(scores).all(), f"{model_type} has non-finite scores"
