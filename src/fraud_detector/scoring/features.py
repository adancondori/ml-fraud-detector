"""Single-transaction feature calculator — computes 31 features for one payment."""
from __future__ import annotations

from collections import Counter
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from fraud_detector.features.engineering import FEATURE_NAMES
from fraud_detector.scoring.context import UserContext
from fraud_detector.utils.logger import logger


class SingleFeatureCalculator:
    """Computes 31 features for a single transaction using pre-learned parameters.

    Loads global_avg_amount, facility_avgs, and staff_stats from the
    FeatureEngineer trained in Fase 4.
    """

    def __init__(self, feature_engineer_path: str = "output/models/feature_engineer.joblib"):
        fe = joblib.load(feature_engineer_path)
        # Extract learned parameters from FeatureGroups
        self._global_avg_amount = fe._groups[0]._global_avg_amount
        self._facility_avgs = fe._groups[4]._facility_avg_amount
        self._staff_stats = fe._groups[6]._role_currency_stats
        self._staff_currency_stats = fe._groups[6]._currency_stats
        self._staff_global_mean = fe._groups[6]._global_mean
        self._staff_global_std = fe._groups[6]._global_std
        assert len(self._facility_avgs) > 0, (
            f"_facility_avg_amount vacío (grupo={type(fe._groups[4]).__name__}) — "
            "artefacto corrupto o nombre de atributo incorrecto"
        )
        assert len(self._staff_stats) > 0, (
            f"_role_currency_stats vacío (grupo={type(fe._groups[6]).__name__}) — "
            "artefacto corrupto o nombre de atributo incorrecto"
        )
        logger.info(
            f"SingleFeatureCalculator loaded: global_avg={self._global_avg_amount:.2f}, "
            f"facilities={len(self._facility_avgs)}, staff_role_currency={len(self._staff_stats)}"
        )

    def calculate(self, payment: Dict, context: UserContext) -> np.ndarray:
        """Compute 31 features in FEATURE_NAMES order.

        Args:
            payment: Dict with transaction fields (reservation_paid_out, discount,
                     tip, created_at, payment_method, category, club_credit_flag,
                     paid_by_manager, facility_id, currency).
            context: UserContext with rolling aggregates from ClickHouse.

        Returns:
            np.ndarray of shape (31,) in FEATURE_NAMES order.
        """
        amount = float(payment.get("reservation_paid_out", 0) or 0)
        discount = float(payment.get("discount", 0) or 0)
        tip = float(payment.get("tip", 0) or 0)
        ts = pd.Timestamp(payment["created_at"])
        hour = ts.hour
        dow = ts.dayofweek + 1  # 1=Mon, 7=Sun
        fid = payment.get("facility_id", 0)

        facility_avg = self._facility_avgs.get(fid, self._global_avg_amount)
        is_staff = context.user_role in ("court_manager", "court_operator", "teacher")
        role_key = context.user_role if is_staff else "player"
        currency = (payment.get("currency") or "USD").upper()
        currency_key = (role_key, currency)
        if currency_key in self._staff_stats:
            _s = self._staff_stats[currency_key]
        elif (role_key, "USD") in self._staff_stats:
            _s = self._staff_stats[(role_key, "USD")]
        elif currency in self._staff_currency_stats:
            _s = self._staff_currency_stats[currency]
        else:
            _s = {"mean": self._staff_global_mean, "std": self._staff_global_std}
        staff_mean = _s["mean"]
        staff_std = _s["std"] or 1.0

        # time_since_last_txn
        if context.last_txn_at is not None:
            time_since = (ts - pd.Timestamp(context.last_txn_at)).total_seconds()
            time_since = max(time_since, 0.0)
        else:
            time_since = 0.0

        # category_entropy_30d
        cat_entropy = self._shannon_entropy(context.categories_30d)

        # credit_flow_ratio
        prepaid_spend = max(float(context.prepaid_spend_30d or 0), 0.01)
        credit_flow = float(context.debit_amount_30d or 0) / prepaid_spend

        # user_account_age_days
        if context.user_created_at is not None:
            account_age = (ts - pd.Timestamp(context.user_created_at)).days
            account_age = max(account_age, 0)
        else:
            account_age = 0

        features = np.array([
            # Group A: Transactional
            amount,                                              # F01
            np.log1p(amount),                                    # F02
            amount / max(self._global_avg_amount, 1e-8),         # F03
            discount / max(amount, 0.01),                        # F04
            1.0 if tip > 0 else 0.0,                             # F05
            # Group B: Temporal
            np.sin(2 * np.pi * hour / 24),                      # F07
            np.cos(2 * np.pi * hour / 24),                      # F08
            float(dow),                                          # F09
            1.0 if dow >= 6 else 0.0,                            # F10
            1.0 if hour >= 23 or hour <= 6 else 0.0,            # F11
            # Group C: Velocity
            float(context.txn_count_1h),                         # F12
            float(context.txn_count_24h),                        # F13
            time_since,                                          # F14
            float(context.amount_24h),                           # F15
            # Group D: Behavior
            float(context.distinct_facilities_30d),              # F16
            float(context.distinct_methods),                     # F17
            float(context.reversal_ratio_30d),                   # F18
            float(account_age),                                  # F19
            float(context.discount_ratio_30d),                   # F20
            # Group E: Contextual
            facility_avg,                                        # F22
            amount / max(facility_avg, 1e-8),                    # F23
            # Group F: Credit/Flow
            1.0 if payment.get("club_credit_flag") else 0.0,     # F24
            float(context.debit_count_30d),                      # F25
            float(context.debit_amount_30d),                     # F26
            credit_flow,                                         # F27
            # Group G: Staff/Role
            1.0 if is_staff else 0.0,                            # F28
            1.0 if payment.get("paid_by_manager") else 0.0,      # F29
            (amount - staff_mean) / max(staff_std, 1e-8),        # F30
            # Group H: Diversity
            cat_entropy,                                         # F31
            float(context.reversal_count_30d),                   # F32
            float(context.merchandise_ratio_30d),                # F33
        ], dtype=np.float32)

        return features

    @staticmethod
    def _shannon_entropy(categories: list) -> float:
        if not categories:
            return 0.0
        counts = Counter(categories)
        total = sum(counts.values())
        return -sum(
            (c / total) * np.log2(c / total) for c in counts.values() if c > 0
        )
