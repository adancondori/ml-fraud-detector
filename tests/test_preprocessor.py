"""Tests for UnsupervisedPreprocessor — Fase 5 contracts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detector.features.engineering import (
    FEATURE_NAMES,
    FEATURE_NAMES_21,
    FEATURE_NAMES_30,
)
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame with the 31 official features."""
    rng = np.random.default_rng(42)
    n = 200
    data = {name: rng.standard_normal(n).astype(np.float32) for name in FEATURE_NAMES}
    return pd.DataFrame(data)


@pytest.fixture
def fitted_preprocessor(sample_df: pd.DataFrame) -> UnsupervisedPreprocessor:
    prep = UnsupervisedPreprocessor(variant="full")
    prep.fit(sample_df)
    return prep


# --- Contract 1: correct shape ---


def test_fit_transform_produces_correct_shape(sample_df: pd.DataFrame):
    prep = UnsupervisedPreprocessor(variant="full")
    X = prep.fit_transform(sample_df)
    assert X.shape == (len(sample_df), len(FEATURE_NAMES))


# --- Contract 2: output is float32 ---


def test_output_is_float32(sample_df: pd.DataFrame):
    prep = UnsupervisedPreprocessor(variant="full")
    X = prep.fit_transform(sample_df)
    assert X.dtype == np.float32


# --- Contract 3: NaN input raises ValueError ---


def test_nan_input_raises_valueerror(fitted_preprocessor: UnsupervisedPreprocessor):
    bad_df = pd.DataFrame({name: [np.nan] * 5 for name in FEATURE_NAMES})
    with pytest.raises(ValueError, match="NaN|infinite"):
        fitted_preprocessor.transform(bad_df)


# --- Contract 4: missing columns raises ValueError ---


def test_missing_columns_raises_valueerror(
    fitted_preprocessor: UnsupervisedPreprocessor,
):
    incomplete_df = pd.DataFrame({"amount": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Missing"):
        fitted_preprocessor.transform(incomplete_df)


# --- Contract 5: save/load roundtrip produces identical output ---


def test_save_load_produces_identical_output(
    sample_df: pd.DataFrame,
    fitted_preprocessor: UnsupervisedPreprocessor,
):
    X_original = fitted_preprocessor.transform(sample_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "scaler.joblib")
        fitted_preprocessor.save(path)
        loaded = UnsupervisedPreprocessor.load(path)
        X_loaded = loaded.transform(sample_df)

    np.testing.assert_array_equal(X_original, X_loaded)


# --- Contract 6: transform without fit raises error ---


def test_transform_without_fit_raises_error(sample_df: pd.DataFrame):
    prep = UnsupervisedPreprocessor(variant="full")
    with pytest.raises((ValueError, RuntimeError)):
        prep.transform(sample_df)


# --- Additional: variant support ---


def test_variant_sensitivity_uses_30_features(sample_df: pd.DataFrame):
    # Add the 30 feature columns (sample_df already has all 31)
    prep = UnsupervisedPreprocessor(variant="sensitivity")
    X = prep.fit_transform(sample_df)
    assert X.shape[1] == 30
    assert "user_reversal_ratio_30d" not in prep.get_feature_names()


def test_variant_core_uses_21_features(sample_df: pd.DataFrame):
    prep = UnsupervisedPreprocessor(variant="core")
    X = prep.fit_transform(sample_df)
    assert X.shape[1] == 21


# --- Scaler fit-on-train only ---


def test_scaler_fitted_on_train_only(sample_df: pd.DataFrame):
    """Verify that scaler parameters come from fit data, not transform data."""
    rng = np.random.default_rng(99)
    train_df = sample_df.copy()
    # Create a different distribution for "other" data
    other_data = {
        name: (rng.standard_normal(50) * 100 + 500).astype(np.float32) for name in FEATURE_NAMES
    }
    other_df = pd.DataFrame(other_data)

    prep = UnsupervisedPreprocessor(variant="full")
    prep.fit(train_df)

    # Scaler mean should match train, not other
    train_means = train_df[FEATURE_NAMES].mean().values
    # rtol=1e-3 because scaler fits float64 internally but input is float32
    np.testing.assert_allclose(prep.scaler.mean_, train_means, rtol=1e-3)

    # Transform other should still work (but with train's parameters)
    X_other = prep.transform(other_df)
    assert X_other.shape == (50, 31)
    # Values should be far from 0 since other has different distribution
    assert np.abs(X_other.mean(axis=0)).mean() > 1.0
