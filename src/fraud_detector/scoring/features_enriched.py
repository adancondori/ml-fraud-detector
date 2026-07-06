"""IF-40 enriched feature calculator.

The final Isolation Forest model is trained on 40 features:
base non-circular features, engineered interactions, and raw-derived signals.
This calculator keeps the feature order delegated to final_feature_list.json.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from fraud_detector.features.engineering import FEATURE_NAMES as BASE_FEATURE_NAMES
from fraud_detector.scoring.context import UserContext
from fraud_detector.scoring.features import SingleFeatureCalculator
from fraud_detector.utils.currency import normalize_amount_value


class EnrichedFeatureCalculator:
    """Compute the IF-40 online feature vector in model feature-list order."""

    def __init__(
        self,
        feature_engineer_path: str = "output/models/feature_engineer.joblib",
        feature_list: Iterable[str] | None = None,
    ):
        self._base_calc = SingleFeatureCalculator(feature_engineer_path)
        self._feature_list = list(feature_list or [])

    @property
    def feature_list(self) -> list[str]:
        return self._feature_list

    def calculate(self, payment: Dict, context: UserContext) -> np.ndarray:
        """Compute IF-40 features from a live payment payload and user context."""
        payment_usd = self._payment_with_usd_amounts(payment)
        base_values = self._base_calc.calculate(payment_usd, context)
        features = dict(zip(BASE_FEATURE_NAMES, base_values.tolist()))

        log_amount = float(features["log_amount"])
        amount_facility_ratio = float(features["amount_facility_ratio"])
        account_age = float(features["user_account_age_days"])

        features.update(
            {
                "is_new_user": float(account_age < 14),
                "is_very_new_user": float(account_age < 3),
                "new_user_first_facility": float(
                    account_age < 14 and context.distinct_facilities_30d == 0
                ),
                "rapid_burst": float(
                    features["time_since_last_txn"] > 0
                    and features["time_since_last_txn"] < 60
                    and features["user_txn_count_1h"] > 3
                ),
                "small_amount_at_facility": float(amount_facility_ratio < 0.2),
                "very_small_amount_at_facility": float(amount_facility_ratio < 0.05),
                "off_hours_high_value": float(features["is_off_hours"] > 0 and log_amount > 8),
                "is_third_party_payment": self._third_party_signal(payment, context),
                "same_amount_count_1h": float(context.same_amount_count_1h),
                "same_amount_count_24h": float(context.same_amount_count_24h),
                "gateway_change_recent": float(context.gateway_change_recent),
                "capture_delay_seconds": self._capture_delay_seconds(payment),
                "is_main_gateway": float(context.is_main_gateway),
                "is_first_gateway_for_user": float(context.is_first_gateway_for_user),
                "source_change_recent": float(context.source_change_recent),
            }
        )

        missing = [name for name in self._feature_list if name not in features]
        if missing:
            raise ValueError(f"Missing IF-40 feature values: {missing}")

        return np.array([features[name] for name in self._feature_list], dtype=np.float32)

    def calculate_from_feature_row(self, row) -> np.ndarray:
        """Return the exact IF-40 vector from an already enriched parquet row."""
        missing = [name for name in self._feature_list if name not in row]
        if missing:
            raise ValueError(f"Enriched row missing IF-40 features: {missing}")
        return np.array([row[name] for name in self._feature_list], dtype=np.float32)

    @staticmethod
    def _payment_with_usd_amounts(payment: Dict) -> Dict:
        currency = payment.get("currency")
        out = dict(payment)
        out["reservation_paid_out"] = normalize_amount_value(
            payment.get("reservation_paid_out"), currency
        )
        out["discount"] = normalize_amount_value(payment.get("discount"), currency)
        out["tip"] = normalize_amount_value(payment.get("tip"), currency)
        return out

    @staticmethod
    def _third_party_signal(payment: Dict, context: UserContext) -> float:
        effective_user_id = payment.get("effective_user_id")
        if effective_user_id is None:
            return float(context.is_third_party_payment)
        return float(int(effective_user_id) != int(payment["user_id"]))

    @staticmethod
    def _capture_delay_seconds(payment: Dict) -> float:
        captured_at = payment.get("captured_at")
        created_at = payment.get("created_at")
        if not captured_at or not created_at:
            return 0.0

        try:
            captured = pd.Timestamp(captured_at)
            created = pd.Timestamp(created_at)
        except (ValueError, TypeError):
            return 0.0
        if pd.isnull(captured) or pd.isnull(created):
            return 0.0

        delay = (captured.to_pydatetime() - created.to_pydatetime()).total_seconds()
        return float(np.clip(delay, -86400, 86400))
