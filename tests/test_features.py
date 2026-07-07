"""Tests for the 31-feature anomaly-engineering contract."""

import numpy as np
import pandas as pd
import pytest

from fraud_detector.features.engineering import (
    FEATURE_NAMES,
    FEATURE_NAMES_21,
    FEATURE_NAMES_30,
    FeatureEngineer,
)


def _make_txn(
    txn_id: int,
    user_id: int,
    created_at: str,
    *,
    facility_id: int = 1,
    amount: float = 100.0,
    discount: float = 0.0,
    tip: float = 0.0,
    payment_method: str = "card",
    status: str = "paid",
    currency: str = "USD",
    category: str = "reservation",
    club_credit_flag: bool = False,
    paid_by_manager: bool = False,
    user_role: str = "player",
    is_staff: int = 0,
    user_created_at: str = "2024-12-01 00:00:00",
) -> dict:
    return {
        "id": txn_id,
        "user_id": user_id,
        "facility_id": facility_id,
        "created_at": pd.Timestamp(created_at),
        "amount": amount,
        "discount": discount,
        "tip": tip,
        "payment_method": payment_method,
        "status": status,
        "currency": currency,
        "category": category,
        "club_credit_flag": club_credit_flag,
        "paid_by_manager": paid_by_manager,
        "user_role": user_role,
        "is_staff": is_staff,
        "user_created_at": pd.Timestamp(user_created_at),
    }


@pytest.fixture(scope="module")
def sample_data():
    rows = [
        _make_txn(1, 1, "2025-01-01 08:00:00", amount=100, payment_method="card"),
        _make_txn(
            2,
            1,
            "2025-01-01 08:30:00",
            amount=20,
            category="debit",
            payment_method="wallet",
            club_credit_flag=True,
        ),
        _make_txn(
            3,
            1,
            "2025-01-02 09:00:00",
            amount=50,
            category="prepaid",
            payment_method="club_credit",
            club_credit_flag=True,
        ),
        _make_txn(4, 1, "2025-01-10 10:00:00", amount=70, category="merchandise", discount=5),
        _make_txn(5, 2, "2025-01-03 11:00:00", amount=90),
        _make_txn(6, 2, "2025-01-04 12:00:00", amount=200, status="totally_refunded"),
        _make_txn(7, 2, "2025-01-04 13:00:00", amount=110, category="merchandise"),
        _make_txn(
            8,
            2,
            "2025-01-20 14:00:00",
            amount=25,
            category="debit",
            payment_method="wallet",
            club_credit_flag=True,
        ),
        _make_txn(
            9,
            3,
            "2025-01-05 09:00:00",
            amount=60,
            tip=10,
            category="lesson",
            user_role="teacher",
            is_staff=1,
            paid_by_manager=True,
            user_created_at="2024-01-01 00:00:00",
        ),
        _make_txn(
            10,
            3,
            "2025-01-06 10:00:00",
            amount=55,
            status="refunded_to_credit",
            user_role="teacher",
            is_staff=1,
            user_created_at="2024-01-01 00:00:00",
        ),
        _make_txn(
            11,
            3,
            "2025-01-07 11:00:00",
            amount=45,
            category="prepaid",
            payment_method="wallet",
            user_role="teacher",
            is_staff=1,
            user_created_at="2024-01-01 00:00:00",
        ),
        _make_txn(12, 5, "2025-01-08 08:00:00", amount=0, category="reservation"),
        _make_txn(13, 1, "2025-07-03 09:00:00", amount=120, payment_method="wallet"),
        _make_txn(
            14,
            4,
            "2025-07-03 10:00:00",
            amount=80,
            payment_method="card",
            user_created_at="2025-07-03 10:00:00",
        ),
    ]
    df = pd.DataFrame(rows)
    return df.sort_values(["user_id", "created_at", "id"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def train_val_split(sample_data):
    train = sample_data[sample_data["created_at"] < "2025-07-01"].copy()
    val = sample_data[sample_data["created_at"] >= "2025-07-01"].copy()
    return train, val


@pytest.fixture(scope="module")
def engineered_features(train_val_split):
    train, val = train_val_split
    engineer = FeatureEngineer()
    train_features = engineer.fit_transform(train)
    val_features, state = engineer.transform_with_warm_history(
        val,
        train,
        method_state=engineer.get_feature_state(),
        return_state=True,
    )
    return engineer, train_features, val_features, state


class TestFeatureCatalog:
    def test_feature_counts(self):
        assert len(FEATURE_NAMES) == 31
        assert len(set(FEATURE_NAMES)) == 31
        assert len(FEATURE_NAMES_30) == 30
        assert len(FEATURE_NAMES_21) == 21

    def test_feature_order_prefix(self):
        assert FEATURE_NAMES[:5] == [
            "amount",
            "log_amount",
            "amount_usd_ratio",
            "discount_ratio",
            "has_tip",
        ]


class TestFeatureOutputs:
    def test_output_columns_match_catalog(self, engineered_features):
        _, train_features, _, _ = engineered_features
        for feature in FEATURE_NAMES:
            assert feature in train_features.columns

    def test_no_nans_in_train(self, engineered_features):
        _, train_features, _, _ = engineered_features
        assert not train_features[FEATURE_NAMES].isna().any().any()

    def test_no_nans_in_val(self, engineered_features):
        _, _, val_features, _ = engineered_features
        assert not val_features[FEATURE_NAMES].isna().any().any()

    def test_amount_zero_is_finite(self, engineered_features):
        _, train_features, _, _ = engineered_features
        zero_rows = train_features[train_features["amount"] == 0]
        assert np.isfinite(zero_rows["log_amount"]).all()
        assert np.isfinite(zero_rows["discount_ratio"]).all()
        assert np.isfinite(zero_rows["amount_facility_ratio"]).all()

    def test_staff_amount_zscore_is_finite(self, engineered_features):
        _, train_features, _, _ = engineered_features
        assert np.isfinite(train_features["staff_amount_zscore"]).all()


class TestAntiLeakage:
    def test_first_transaction_counts_zero(self, engineered_features):
        _, train_features, _, _ = engineered_features
        first_txns = train_features.groupby("user_id", sort=False).first()
        assert (first_txns["user_txn_count_1h"] == 0).all()
        assert (first_txns["user_txn_count_24h"] == 0).all()
        assert (first_txns["user_debit_count_30d"] == 0).all()
        assert (first_txns["user_reversal_count_30d"] == 0).all()

    def test_first_transaction_distinct_facilities_zero(self, engineered_features):
        _, train_features, _, _ = engineered_features
        first_txns = train_features.groupby("user_id", sort=False).first()
        assert (first_txns["user_distinct_facilities_30d"] == 0).all()

    def test_cold_start_user_in_val_has_neutral_history(self, engineered_features):
        _, _, val_features, _ = engineered_features
        cold_start = val_features[val_features["user_id"] == 4].iloc[0]
        assert cold_start["user_txn_count_1h"] == 0
        assert cold_start["user_account_age_days"] == 0

    def test_method_history_carries_across_splits(self, engineered_features):
        _, _, val_features, _ = engineered_features
        existing_user = val_features[val_features["user_id"] == 1].iloc[0]
        assert existing_user["user_distinct_methods"] >= 3


class TestFitTransformContract:
    def test_transform_requires_fit(self, sample_data):
        engineer = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fit"):
            engineer.transform(sample_data)

    def test_missing_required_column_raises(self, sample_data):
        engineer = FeatureEngineer()
        broken = sample_data.drop(columns=["category"])
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.fit(broken)

    def test_facility_avg_comes_from_train(self, train_val_split, engineered_features):
        train, _ = train_val_split
        engineer, train_features, _, _ = engineered_features
        contextual_group = next(
            group for group in engineer._groups if group.__class__.__name__ == "ContextualFeatures"
        )
        for facility_id, expected in train.groupby("facility_id")["amount"].mean().items():
            assert np.isclose(contextual_group._facility_avg_amount[facility_id], expected)
        assert train_features["facility_avg_amount"].notna().all()

    def test_save_load_roundtrip(self, engineered_features, tmp_path, train_val_split):
        engineer, train_features, _, _ = engineered_features
        path = tmp_path / "feature_engineer.joblib"
        engineer.save(str(path))
        loaded = FeatureEngineer.load(str(path))
        train, _ = train_val_split
        loaded_features = loaded.transform(train)
        pd.testing.assert_frame_equal(
            train_features[FEATURE_NAMES].reset_index(drop=True),
            loaded_features[FEATURE_NAMES].reset_index(drop=True),
        )
