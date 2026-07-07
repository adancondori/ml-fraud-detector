"""Tests for preprocessing, models, metrics and currency normalization."""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.evaluation.metrics import (
    bootstrap_ci,
    enrichment_factor,
    evaluate_scores,
    precision_at_k,
)
from fraud_detector.features.engineering import FeatureEngineer
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
from fraud_detector.models.trainer import ModelTrainer
from fraud_detector.utils.currency import CurrencyNormalizer


@pytest.fixture(scope="module")
def engineered_train_df():
    rows = []
    for i in range(60):
        rows.append(
            {
                "id": i + 1,
                "user_id": (i % 10) + 1,
                "facility_id": (i % 3) + 1,
                "created_at": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i),
                "amount": 50 + (i % 7) * 10,
                "discount": float(i % 5),
                "tip": float(i % 4 == 0) * 5,
                "payment_method": ["card", "wallet", "club_credit"][i % 3],
                "status": "totally_refunded" if i % 11 == 0 else "paid",
                "currency": "USD",
                "category": ["reservation", "debit", "merchandise", "prepaid"][i % 4],
                "club_credit_flag": bool(i % 3 == 2),
                "paid_by_manager": bool(i % 13 == 0),
                "user_role": "teacher" if i % 9 == 0 else "player",
                "is_staff": int(i % 9 == 0),
                "user_created_at": pd.Timestamp("2024-01-01"),
            }
        )
    df = pd.DataFrame(rows)
    engineer = FeatureEngineer()
    return engineer.fit_transform(df)


class TestCurrencyNormalizer:
    def test_from_direct_rate_csv(self, tmp_path):
        csv_path = tmp_path / "rates.csv"
        csv_path.write_text("year_month,currency,rate_to_usd,source\n2025-01,CAD,0.70,test\n")
        normalizer = CurrencyNormalizer.from_csv(csv_path)

        df = pd.DataFrame(
            {
                "amount": [100.0],
                "discount": [10.0],
                "tip": [5.0],
                "tax": [2.0],
                "currency": ["CAD"],
                "created_at": [pd.Timestamp("2025-01-15")],
            }
        )
        result = normalizer.normalize(df)
        assert np.isclose(result.loc[0, "amount"], 70.0)
        assert np.isclose(result.loc[0, "discount"], 7.0)

    def test_from_clickhouse_snapshot_csv(self, tmp_path):
        csv_path = tmp_path / "snapshot.csv"
        csv_path.write_text(
            "base_currency,target_currency,conversion_rate,timestamp\n"
            "USD,AED,3.6731,2026-03-20\n"
        )
        normalizer = CurrencyNormalizer.from_csv(csv_path)
        df = pd.DataFrame(
            {
                "amount": [36.731],
                "currency": ["AED"],
                "created_at": [pd.Timestamp("2026-03-20")],
            }
        )
        result = normalizer.normalize(df)
        assert np.isclose(result.loc[0, "amount"], 10.0, atol=1e-3)


class TestPreprocessor:
    def test_fit_transform_shape_and_dtype(self, engineered_train_df):
        preprocessor = UnsupervisedPreprocessor()
        X = preprocessor.fit_transform(engineered_train_df)
        assert X.shape == (len(engineered_train_df), 31)
        assert X.dtype == np.float32

    def test_missing_feature_column_raises(self, engineered_train_df):
        preprocessor = UnsupervisedPreprocessor()
        broken = engineered_train_df.drop(columns=["amount"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            preprocessor.fit(broken)

    def test_save_load_roundtrip(self, engineered_train_df, tmp_path):
        preprocessor = UnsupervisedPreprocessor()
        X1 = preprocessor.fit_transform(engineered_train_df)
        path = tmp_path / "preprocessor.joblib"
        preprocessor.save(str(path))
        loaded = UnsupervisedPreprocessor.load(str(path))
        X2 = loaded.transform(engineered_train_df)
        np.testing.assert_allclose(X1, X2)


class TestModelTrainer:
    @pytest.mark.parametrize("model_type", ["isolation_forest", "lof", "ocsvm"])
    def test_model_scores_shape(self, engineered_train_df, model_type):
        preprocessor = UnsupervisedPreprocessor()
        X = preprocessor.fit_transform(engineered_train_df)
        trainer = ModelTrainer(model_type=model_type)
        trainer.fit(X)
        scores = trainer.score_samples(X[:12])
        preds = trainer.predict(X[:12])
        assert scores.shape == (12,)
        assert preds.shape == (12,)
        assert np.isfinite(scores).all()


class TestMetrics:
    def test_precision_and_enrichment(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert precision_at_k(labels, scores, k_pct=0.5) == 1.0
        assert enrichment_factor(labels, scores, k_pct=0.5) == 2.0

    def test_evaluate_scores_keys(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = evaluate_scores(labels, scores, top_k_percents=[0.5])
        assert "auc_roc" in metrics
        assert "ap" in metrics
        assert "precision_at_50pct" in metrics
        assert "enrichment_factor_at_50pct" in metrics

    def test_bootstrap_ci_bounds(self):
        labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        ci = bootstrap_ci(
            labels, scores, metric_fn=lambda y, s: precision_at_k(y, s, 0.5), n_iterations=100
        )
        assert ci["lower"] <= ci["mean"] <= ci["upper"]
