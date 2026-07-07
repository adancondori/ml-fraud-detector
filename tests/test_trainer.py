"""Tests for ModelTrainer — Fase 6 contracts.

Validates: score convention, grid search checkpoint/resume,
multi-seed, OC-SVM subsample, save/load roundtrip, LOF novelty.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detector.models.trainer import ModelTrainer


@pytest.fixture
def synthetic_data():
    """Synthetic inliers + outliers for testing."""
    rng = np.random.default_rng(42)
    X_inliers = rng.standard_normal((500, 10)).astype(np.float32)
    X_outliers = (rng.standard_normal((30, 10)) + 8).astype(np.float32)
    X_train = X_inliers.copy()
    X_val = np.vstack([X_inliers[:100], X_outliers]).astype(np.float32)
    y_val = np.array([0] * 100 + [1] * 30, dtype=np.int8)
    return X_train, X_val, y_val


# --- Contract 1: Score convention (higher = more anomalous) ---


def test_score_convention_higher_is_more_anomalous(synthetic_data):
    X_train, X_val, y_val = synthetic_data
    trainer = ModelTrainer(model_type="isolation_forest", model_params={"n_estimators": 50})
    trainer.fit(X_train)
    scores = trainer.score_samples(X_val)
    mean_normal = scores[y_val == 0].mean()
    mean_anomaly = scores[y_val == 1].mean()
    assert mean_anomaly > mean_normal, "Anomalies must have higher scores"


# --- Contract 2: IF trains without error ---


def test_if_trains_without_error(synthetic_data):
    X_train, _, _ = synthetic_data
    trainer = ModelTrainer(model_type="isolation_forest", model_params={"n_estimators": 50})
    trainer.fit(X_train)
    assert trainer.model is not None


# --- Contract 3: LOF requires novelty=True ---


def test_lof_has_novelty_true(synthetic_data):
    X_train, _, _ = synthetic_data
    trainer = ModelTrainer(model_type="lof")
    trainer.fit(X_train)
    assert trainer.model.novelty is True


# --- Contract 4: OC-SVM subsample ---


def test_ocsvm_subsample_size():
    rng = np.random.default_rng(42)
    X_big = rng.standard_normal((1000, 5)).astype(np.float32)
    subsample = ModelTrainer._subsample_temporal(X_big, n=200)
    assert subsample.shape[0] == 200
    assert subsample.shape[1] == 5


def test_ocsvm_subsample_preserves_small():
    rng = np.random.default_rng(42)
    X_small = rng.standard_normal((50, 5)).astype(np.float32)
    subsample = ModelTrainer._subsample_temporal(X_small, n=200)
    assert subsample.shape[0] == 50


# --- Contract 5: Grid search checkpoint resumes ---


def test_grid_search_checkpoint_resumes(synthetic_data):
    X_train, X_val, y_val = synthetic_data
    param_grid = {
        "n_estimators": [50, 100],
        "max_samples": [64],
        "max_features": [1.0],
        "contamination": [0.05],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = str(Path(tmpdir) / "gs_checkpoint.csv")
        # Run first 1 combo (simulate partial)
        results1 = ModelTrainer.grid_search_if(
            X_train,
            X_val,
            y_val,
            param_grid,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
            random_state=42,
            max_combos=1,
        )
        assert Path(checkpoint).exists()
        df1 = pd.read_csv(checkpoint)
        assert len(df1) == 1

        # Resume — should complete remaining
        results2 = ModelTrainer.grid_search_if(
            X_train,
            X_val,
            y_val,
            param_grid,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
            random_state=42,
        )
        df2 = pd.read_csv(checkpoint)
        assert len(df2) == 2


# --- Contract 6: Save/load roundtrip ---


def test_save_load_produces_same_scores(synthetic_data):
    X_train, X_val, _ = synthetic_data
    trainer = ModelTrainer(model_type="isolation_forest", model_params={"n_estimators": 50})
    trainer.fit(X_train)
    scores_before = trainer.score_samples(X_val)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "model.joblib")
        trainer.save_model(path)
        trainer2 = ModelTrainer(model_type="isolation_forest")
        trainer2.load_model(path)
        scores_after = trainer2.score_samples(X_val)

    np.testing.assert_array_equal(scores_before, scores_after)


# --- Contract 7: Multi-seed returns variability report ---


def test_multi_seed_reports_variability(synthetic_data):
    X_train, X_val, y_val = synthetic_data
    best_params = {"n_estimators": 50, "max_samples": 64, "max_features": 1.0}
    df = ModelTrainer.run_multi_seed(X_train, X_val, y_val, best_params, seeds=[42, 52])
    assert "seed" in df.columns
    assert "auc_roc" in df.columns
    assert len(df) == 2
    assert df["auc_roc"].notna().all()


# --- Score via decision_function for all models ---


def test_decision_function_scoring(synthetic_data):
    X_train, X_val, y_val = synthetic_data
    for model_type in ["isolation_forest", "lof", "ocsvm"]:
        params = {"n_estimators": 50} if model_type == "isolation_forest" else {}
        trainer = ModelTrainer(model_type=model_type, model_params=params)
        if model_type == "ocsvm":
            trainer.fit(X_train[:200])
        else:
            trainer.fit(X_train)
        scores = trainer.decision_function_scores(X_val)
        assert scores.shape == (len(X_val),)
        assert np.isfinite(scores).all()
        # Higher should be more anomalous
        assert scores[y_val == 1].mean() > scores[y_val == 0].mean()
