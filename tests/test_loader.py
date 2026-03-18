"""Tests for DataManager — TDD contracts for Fase 1.

Written FIRST (red), then loader.py is implemented to pass them (green).
These tests use synthetic data; no ClickHouse required.
"""
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking raw ClickHouse extract."""
    n = 200
    rng = np.random.RandomState(42)
    statuses = ["captured"] * 170 + ["totally_refunded"] * 15 + \
               ["refunded_to_credit"] * 8 + ["partially_refunded"] * 7
    return pd.DataFrame({
        "id": range(1, n + 1),
        "user_id": rng.randint(1, 50, n),
        "facility_id": rng.randint(100, 110, n),
        "facility_name": ["Facility A"] * n,
        "created_at": pd.date_range("2025-01-01", periods=n, freq="h"),
        "captured_at": pd.date_range("2025-01-01", periods=n, freq="h"),
        "payment_method": ["card"] * n,
        "gateway": ["stripe"] * n,
        "source_enum": ["pbp_web"] * n,
        "status": statuses,
        "reservation_paid_out": rng.uniform(5, 500, n).round(2),
        "discount": rng.uniform(0, 10, n).round(2),
        "tax": rng.uniform(0, 5, n).round(2),
        "tip": rng.uniform(0, 20, n).round(2),
        "card_brand": ["visa"] * n,
        "currency": ["USD"] * n,
        "paid_by_manager": [False] * n,
        "reversed_id": [0] * n,
        "debit_refund": [False] * n,
        "_peerdb_version": [1] * n,
    })


# ── ProxyLabeler tests ───────────────────────────────────────────

class TestProxyLabeler:
    """Proxy label assignment matches thesis definitions."""

    def test_strict_proxy_labels(self, sample_df):
        from fraud_detector.data.loader import DataManager
        labels = DataManager.assign_proxy_labels(sample_df, "strict")
        assert labels.dtype == bool or labels.dtype == np.int8 or labels.dtype == int
        # 15 totally_refunded + 8 refunded_to_credit = 23
        assert labels.sum() == 23

    def test_wide_proxy_labels(self, sample_df):
        from fraud_detector.data.loader import DataManager
        labels = DataManager.assign_proxy_labels(sample_df, "wide")
        # 15 + 8 + 7 = 30
        assert labels.sum() == 30

    def test_invalid_proxy_type_raises(self, sample_df):
        from fraud_detector.data.loader import DataManager
        with pytest.raises(ValueError, match="proxy_type"):
            DataManager.assign_proxy_labels(sample_df, "invalid")


# ── DataValidator tests ──────────────────────────────────────────

class TestDataValidation:
    """Validation logic catches data quality issues."""

    def test_validates_required_columns(self, sample_df):
        from fraud_detector.data.loader import DataManager
        # Should pass — all required columns present
        dm = DataManager.__new__(DataManager)
        dm._validate_extraction(sample_df, "test_split")  # no error

    def test_validates_required_columns_missing(self, sample_df):
        from fraud_detector.data.loader import DataManager
        bad_df = sample_df.drop(columns=["user_id"])
        dm = DataManager.__new__(DataManager)
        with pytest.raises(ValueError, match="user_id"):
            dm._validate_extraction(bad_df, "test_split")

    def test_validates_user_id_positive(self, sample_df):
        from fraud_detector.data.loader import DataManager
        bad_df = sample_df.copy()
        bad_df.loc[0, "user_id"] = 0
        dm = DataManager.__new__(DataManager)
        with pytest.raises(ValueError, match="user_id.*0"):
            dm._validate_extraction(bad_df, "test_split")


# ── Downcast tests ───────────────────────────────────────────────

class TestDowncast:
    """Selective downcast preserves data integrity."""

    def test_downcast_preserves_large_ids(self, sample_df):
        from fraud_detector.data.loader import DataManager
        df = sample_df.copy()
        # Set a large ID that exceeds int32
        df.loc[0, "id"] = 3_000_000_000
        dm = DataManager.__new__(DataManager)
        result = dm._downcast(df)
        assert result.loc[0, "id"] == 3_000_000_000  # NOT truncated

    def test_downcast_float32_amounts(self, sample_df):
        from fraud_detector.data.loader import DataManager
        dm = DataManager.__new__(DataManager)
        result = dm._downcast(sample_df.copy())
        assert result["reservation_paid_out"].dtype == np.float32

    def test_downcast_int32_user_id(self, sample_df):
        from fraud_detector.data.loader import DataManager
        dm = DataManager.__new__(DataManager)
        result = dm._downcast(sample_df.copy())
        assert result["user_id"].dtype == np.int32


# ── Manifest tests ───────────────────────────────────────────────

class TestManifest:
    """Manifest JSON contains all required fields."""

    def test_manifest_fields(self, sample_df, tmp_path):
        from fraud_detector.data.loader import DataManager
        dm = DataManager.__new__(DataManager)
        manifest_path = tmp_path / "test_manifest.json"
        dm._save_manifest("test", "2025-09-01", "2026-01-01", sample_df, manifest_path)

        with open(manifest_path) as f:
            manifest = json.load(f)

        required_fields = ["name", "start_date", "end_date", "row_count",
                           "extracted_at", "columns", "status_distribution"]
        for field in required_fields:
            assert field in manifest, f"Missing field: {field}"
        assert manifest["row_count"] == len(sample_df)
        assert manifest["name"] == "test"


# ── Load/Save tests ─────────────────────────────────────────────

class TestLoadSave:
    """Parquet I/O roundtrip."""

    def test_save_and_load_split(self, sample_df, tmp_path):
        from fraud_detector.data.loader import DataManager
        from config.config import Settings

        s = Settings(project_root=tmp_path, data_dir="data")
        (tmp_path / "data" / "processed").mkdir(parents=True)

        path = tmp_path / "data" / "processed" / "test_raw.parquet"
        sample_df.to_parquet(path, engine="pyarrow", compression="snappy")

        dm = DataManager(s)
        loaded = dm.load_split("test")
        assert len(loaded) == len(sample_df)
        assert list(loaded.columns) == list(sample_df.columns)

    def test_load_missing_split_raises(self, tmp_path):
        from fraud_detector.data.loader import DataManager
        from config.config import Settings

        s = Settings(project_root=tmp_path, data_dir="data")
        (tmp_path / "data" / "processed").mkdir(parents=True)

        dm = DataManager(s)
        with pytest.raises(FileNotFoundError):
            dm.load_split("nonexistent")
