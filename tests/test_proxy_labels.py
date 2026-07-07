"""Tests for proxy taxonomy (5 types + unified).

TDD contract: ProxyLabeler must support Tipos A-E and proxy unificado
as defined in PLAN-FINAL/01_CONTRATO_ALCANCE.md (v2.1, 2026-04-15).
"""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.data.loader import DataManager


@pytest.fixture
def sample_df():
    """DataFrame with columns needed for all proxy types."""
    return pd.DataFrame(
        {
            "status": [
                "captured",
                "totally_refunded",
                "refunded_to_credit",
                "partially_refunded",
                "captured",
                "captured",
                "captured",
            ],
            "user_discount_ratio_30d": [0.1, 0.5, 1.5, 0.3, 2.0, 0.0, 0.8],
            "user_txn_count_24h": [5, 10, 50, 20, 200, 3, 99],
        }
    )


class TestTipoA:
    def test_tipo_a_marks_refunds(self, sample_df):
        labels = DataManager.assign_proxy_labels(sample_df, "tipo_a")
        assert labels.sum() == 2  # totally_refunded + refunded_to_credit

    def test_tipo_a_equals_strict(self, sample_df):
        a = DataManager.assign_proxy_labels(sample_df, "tipo_a")
        s = DataManager.assign_proxy_labels(sample_df, "strict")
        pd.testing.assert_series_equal(a, s)

    def test_tipo_a_dtype_int8(self, sample_df):
        labels = DataManager.assign_proxy_labels(sample_df, "tipo_a")
        assert labels.dtype == np.int8


class TestTipoB:
    def test_tipo_b_returns_zeros_without_columns(self, sample_df):
        """Tipo B requires pre-computed aggregates not in raw parquets."""
        labels = DataManager.assign_proxy_labels(sample_df, "tipo_b")
        assert labels.sum() == 0

    def test_tipo_b_detects_circuit_closure(self):
        df = pd.DataFrame(
            {
                "status": ["captured", "captured", "captured"],
                "circuit_closure_ratio_30d": [0.5, 0.9, 0.85],
                "cash_loaded_30d": [100.0, 600.0, 400.0],
            }
        )
        labels = DataManager.assign_proxy_labels(df, "tipo_b")
        # Only row 1: closure > 0.80 AND cash > 500
        assert labels.tolist() == [0, 1, 0]


class TestTipoC:
    def test_tipo_c_detects_discount_anomaly(self, sample_df):
        labels = DataManager.assign_proxy_labels(sample_df, "tipo_c")
        # user_discount_ratio_30d > 1.0: row 2 (1.5) and row 4 (2.0)
        assert labels.sum() == 2

    def test_tipo_c_returns_zeros_without_column(self):
        df = pd.DataFrame({"status": ["captured"]})
        labels = DataManager.assign_proxy_labels(df, "tipo_c")
        assert labels.sum() == 0


class TestTipoD:
    def test_tipo_d_detects_extreme_velocity(self, sample_df):
        labels = DataManager.assign_proxy_labels(sample_df, "tipo_d")
        # user_txn_count_24h + 1 > 100: row 4 (200+1=201), row 6 (99+1=100, NOT >100)
        assert labels.sum() == 1

    def test_tipo_d_returns_zeros_without_column(self):
        df = pd.DataFrame({"status": ["captured"]})
        labels = DataManager.assign_proxy_labels(df, "tipo_d")
        assert labels.sum() == 0


class TestTipoE:
    def test_tipo_e_always_zero_in_depurated_universe(self, sample_df):
        """payment_method='free' excluded from universe -> Tipo E = 0 always."""
        labels = DataManager.assign_proxy_labels(sample_df, "tipo_e")
        assert labels.sum() == 0


class TestUnified:
    def test_unified_is_superset_of_tipo_a(self, sample_df):
        a = DataManager.assign_proxy_labels(sample_df, "tipo_a")
        u = DataManager.assign_proxy_labels(sample_df, "unified")
        # Every tipo_a positive must also be unified positive
        assert ((a == 1) & (u == 0)).sum() == 0

    def test_unified_includes_all_types(self, sample_df):
        u = DataManager.assign_proxy_labels(sample_df, "unified")
        # tipo_a=2 (rows 1,2), tipo_c=2 (rows 2,4), tipo_d=1 (row 4)
        # Union: rows 1, 2, 4 = at least 3 (row 2 counted once)
        assert u.sum() >= 3

    def test_unified_gte_tipo_a(self, sample_df):
        a = DataManager.assign_proxy_labels(sample_df, "tipo_a")
        u = DataManager.assign_proxy_labels(sample_df, "unified")
        assert u.sum() >= a.sum()

    def test_unified_dtype_int8(self, sample_df):
        labels = DataManager.assign_proxy_labels(sample_df, "unified")
        assert labels.dtype == np.int8


class TestInvalidType:
    def test_invalid_type_raises(self, sample_df):
        with pytest.raises(ValueError, match="Invalid proxy_type"):
            DataManager.assign_proxy_labels(sample_df, "nonexistent")


class TestConfigProxyThresholds:
    def test_tipo_b_thresholds_exist(self):
        from config.config import Settings

        s = Settings()
        assert s.tipo_b_circuit_closure_threshold == 0.80
        assert s.tipo_b_cash_loaded_threshold == 500.0

    def test_tipo_c_threshold_exists(self):
        from config.config import Settings

        s = Settings()
        assert s.tipo_c_discount_ratio_threshold == 1.00

    def test_tipo_d_threshold_exists(self):
        from config.config import Settings

        s = Settings()
        assert s.tipo_d_txn_count_1d_threshold == 100

    def test_tipo_e_thresholds_exist(self):
        from config.config import Settings

        s = Settings()
        assert s.tipo_e_free_pct_threshold == 0.25
        assert s.tipo_e_free_count_threshold == 10

    def test_tipo_a_list_alias(self):
        from config.config import Settings

        s = Settings()
        assert s.tipo_a_list == s.strict_proxy_list
