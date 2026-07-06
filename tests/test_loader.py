"""Tests for the thesis-aligned DataManager."""
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    n = 120
    rng = np.random.RandomState(42)
    statuses = ["captured"] * 96 + ["totally_refunded"] * 12 + ["refunded_to_credit"] * 8 + ["partially_refunded"] * 4
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "user_id": rng.randint(1, 30, n),
            "effective_user_id": rng.randint(1, 30, n),
            "facility_id": rng.randint(100, 105, n),
            "facility_name": ["Facility A"] * n,
            "created_at": pd.date_range("2025-01-01", periods=n, freq="h"),
            "captured_at": pd.date_range("2025-01-01", periods=n, freq="h"),
            "payment_method": ["card"] * n,
            "gateway": ["stripe"] * n,
            "source_enum": ["pbp_web"] * n,
            "status": statuses,
            "amount": rng.uniform(5, 500, n).round(2),
            "discount": rng.uniform(0, 10, n).round(2),
            "tax": rng.uniform(0, 5, n).round(2),
            "tip": rng.uniform(0, 20, n).round(2),
            "card_brand": ["visa"] * n,
            "currency": ["USD"] * n,
            "paid_by_manager": [False] * n,
            "reversed_id": [0] * n,
            "debit_refund": [False] * n,
            "category": ["reservation"] * n,
            "club_credit_flag": [False] * n,
            "_peerdb_version": [1] * n,
            "is_staff": [0] * n,
            "user_role": ["player"] * n,
            "user_created_at": [pd.Timestamp("2024-01-01")] * n,
        }
    )


class TestCanonicalSql:
    def test_sql_contains_required_joins(self):
        from fraud_detector.data.loader import CANONICAL_SQL

        assert "LEFT ANY JOIN" in CANONICAL_SQL
        assert "facilities_users" in CANONICAL_SQL
        assert "users" in CANONICAL_SQL
        assert "club_credit_flag" in CANONICAL_SQL
        assert "user_role" in CANONICAL_SQL


class TestProxyLabels:
    def test_strict_proxy_labels(self, sample_df):
        from fraud_detector.data.loader import DataManager

        labels = DataManager.assign_proxy_labels(sample_df, "strict")
        assert labels.sum() == 20

    def test_wide_proxy_labels(self, sample_df):
        from fraud_detector.data.loader import DataManager

        labels = DataManager.assign_proxy_labels(sample_df, "wide")
        assert labels.sum() == 24

    def test_invalid_proxy_type_raises(self, sample_df):
        from fraud_detector.data.loader import DataManager

        with pytest.raises(ValueError, match="proxy_type"):
            DataManager.assign_proxy_labels(sample_df, "invalid")


class TestValidation:
    def test_valid_dataframe_passes(self, sample_df):
        from fraud_detector.data.loader import DataManager

        dm = DataManager.__new__(DataManager)
        dm._validate_extraction(sample_df, "train")

    def test_missing_required_column_raises(self, sample_df):
        from fraud_detector.data.loader import DataManager

        broken = sample_df.drop(columns=["category"])
        dm = DataManager.__new__(DataManager)
        with pytest.raises(ValueError, match="category"):
            dm._validate_extraction(broken, "train")

    def test_user_id_must_be_positive(self, sample_df):
        from fraud_detector.data.loader import DataManager

        broken = sample_df.copy()
        broken.loc[0, "user_id"] = 0
        dm = DataManager.__new__(DataManager)
        with pytest.raises(ValueError, match="user_id"):
            dm._validate_extraction(broken, "train")


class TestPostprocessAndDowncast:
    def test_postprocess_renames_amount_and_runs_normalizer(self):
        from fraud_detector.data.loader import DataManager

        class StubNormalizer:
            def normalize(self, df, amount_col, currency_col, timestamp_col):
                result = df.copy()
                result["exchange_rate_applied"] = 1.0
                result[amount_col] = result[amount_col] * 2
                return result

        raw = pd.DataFrame(
            {
                "id": [1],
                "user_id": [1],
                "effective_user_id": [1],
                "facility_id": [10],
                "facility_name": ["A"],
                "created_at": ["2025-01-01 00:00:00"],
                "captured_at": ["2025-01-01 00:00:00"],
                "payment_method": ["card"],
                "gateway": ["stripe"],
                "source_enum": ["pbp_web"],
                "status": ["paid"],
                "reservation_paid_out": [100.0],
                "discount": [0.0],
                "tax": [0.0],
                "tip": [0.0],
                "card_brand": ["visa"],
                "currency": ["USD"],
                "paid_by_manager": [False],
                "reversed_id": [0],
                "debit_refund": [False],
                "category": ["reservation"],
                "club_credit_flag": [False],
                "_peerdb_version": [1],
                "is_staff": [0],
                "user_role": ["player"],
                "user_created_at": ["2024-01-01 00:00:00"],
            }
        )

        dm = DataManager.__new__(DataManager)
        dm._normalizer = StubNormalizer()
        processed = dm._postprocess_extraction(raw)

        assert "amount" in processed.columns
        assert "reservation_paid_out" not in processed.columns
        assert processed.loc[0, "amount"] == 200.0
        assert processed.loc[0, "category"] == "reservation"

    def test_downcast_preserves_large_ids(self, sample_df):
        from fraud_detector.data.loader import DataManager

        df = sample_df.copy()
        df.loc[0, "id"] = 3_000_000_000
        dm = DataManager.__new__(DataManager)
        result = dm._downcast(df)
        assert result.loc[0, "id"] == 3_000_000_000
        assert result["user_id"].dtype == np.int32
        assert result["amount"].dtype == np.float32


class TestManifestAndLoad:
    def test_manifest_fields(self, sample_df, tmp_path):
        from fraud_detector.data.loader import DataManager

        dm = DataManager.__new__(DataManager)
        manifest_path = tmp_path / "manifest.json"
        dm._save_manifest("train", "2025-01-01", "2025-07-01", sample_df, manifest_path)
        manifest = json.loads(manifest_path.read_text())

        for field in [
            "name",
            "start_date",
            "end_date",
            "row_count",
            "extracted_at",
            "columns",
            "status_distribution",
            "proxy_strict_rate",
            "proxy_wide_rate",
        ]:
            assert field in manifest

    def test_load_missing_split_raises(self, tmp_path):
        from config.config import Settings
        from fraud_detector.data.loader import DataManager

        settings = Settings(project_root=tmp_path, data_dir="data")
        (tmp_path / "data" / "processed").mkdir(parents=True)
        dm = DataManager(settings)

        with pytest.raises(FileNotFoundError):
            dm.load_split("test")


class TestAmountSanityGuard:
    """Tests for compute_amount_sanity_thresholds and sanitize_amount_df."""

    @pytest.fixture
    def amount_df(self):
        """Small DataFrame with two currencies and one extreme outlier each."""
        return pd.DataFrame(
            {
                "amount": [10.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 5_000.0,  # USD normal
                           100_000_000.0,                                          # USD corrupt
                           5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0,           # CAD normal
                           50_000_000.0],                                          # CAD corrupt
                "currency": (["USD"] * 7 + ["USD"]
                             + ["CAD"] * 7 + ["CAD"]),
            }
        )

    def test_compute_thresholds_per_currency(self, amount_df):
        from fraud_detector.data.loader import DataManager

        thresholds = DataManager.compute_amount_sanity_thresholds(amount_df)

        assert "USD" in thresholds
        assert "CAD" in thresholds
        # p99.99 must be below the extreme outlier values
        assert thresholds["USD"] < 100_000_000.0
        assert thresholds["CAD"] < 50_000_000.0

    def test_sanitize_drop_removes_corrupt_rows(self, amount_df):
        from fraud_detector.data.loader import DataManager

        thresholds = DataManager.compute_amount_sanity_thresholds(amount_df)
        cleaned = DataManager.sanitize_amount_df(
            amount_df, thresholds, split_name="train", drop=True
        )

        # Corrupt rows (100M USD and 50M CAD) must be gone
        assert cleaned["amount"].max() < 50_000_000.0
        assert len(cleaned) == len(amount_df) - 2

    def test_sanitize_flag_marks_corrupt_rows(self, amount_df):
        from fraud_detector.data.loader import DataManager

        thresholds = DataManager.compute_amount_sanity_thresholds(amount_df)
        flagged = DataManager.sanitize_amount_df(
            amount_df, thresholds, split_name="train", drop=False
        )

        assert "amount_corrupted" in flagged.columns
        assert int(flagged["amount_corrupted"].sum()) == 2
        assert len(flagged) == len(amount_df)

    def test_sanitize_no_op_when_no_thresholds(self, amount_df):
        from fraud_detector.data.loader import DataManager

        original_len = len(amount_df)
        result = DataManager.sanitize_amount_df(
            amount_df, thresholds=None, split_name="train"
        )
        assert len(result) == original_len

        result_empty = DataManager.sanitize_amount_df(
            amount_df, thresholds={}, split_name="train"
        )
        assert len(result_empty) == original_len

    def test_sanitize_normal_data_untouched(self):
        from fraud_detector.data.loader import DataManager

        # Build a dataset where the hard-coded threshold is clearly above all values.
        # Using a threshold from a reference set with a much higher max prevents the
        # p99.99 of the reference coinciding with the maximum of the validation data.
        reference = pd.DataFrame(
            {"amount": list(range(1, 101)) + [1_000_000.0], "currency": ["USD"] * 101}
        )
        thresholds = DataManager.compute_amount_sanity_thresholds(reference)

        # Normal data: all amounts well below the p99.99 of the reference
        normal = pd.DataFrame(
            {"amount": [10.0, 50.0, 100.0, 200.0, 300.0], "currency": ["USD"] * 5}
        )
        cleaned = DataManager.sanitize_amount_df(normal, thresholds, split_name="train")
        assert len(cleaned) == len(normal)
