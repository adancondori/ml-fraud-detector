"""SingleTransactionScorer — facade that scores one payment end-to-end.

Usage:
    scorer = SingleTransactionScorer()
    result = scorer.score({
        "user_id": 12345, "facility_id": 67,
        "reservation_paid_out": 150.00, "discount": 0, "tip": 5.0,
        "payment_method": "card", "category": "reservation",
        "club_credit_flag": False, "paid_by_manager": False,
        "currency": "USD", "created_at": "2026-04-28T14:30:00",
    })
    # result.score = 0.78, result.is_anomaly = True, result.risk_level = "high"
"""
from __future__ import annotations

from typing import Optional

import joblib
import numpy as np

from fraud_detector.features.engineering import FEATURE_NAMES
from fraud_detector.scoring.classifier import ScoringResult, ThresholdClassifier
from fraud_detector.scoring.context import UserContext, UserContextProvider
from fraud_detector.scoring.features import SingleFeatureCalculator
from fraud_detector.utils.logger import logger


class SingleTransactionScorer:
    """Facade: payment dict → score + is_anomaly + risk_level + factors.

    Loads model, scaler, feature parameters, and threshold once.
    Reuses ClickHouse connection for context queries.
    """

    def __init__(
        self,
        model_path: str = "output/models/isolation_forest.joblib",
        scaler_path: str = "output/models/scaler.joblib",
        feature_engineer_path: str = "output/models/feature_engineer.joblib",
        thresholds_path: str = "output/models/thresholds.json",
        ch_connector=None,
    ):
        self._model = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self._feature_calc = SingleFeatureCalculator(feature_engineer_path)
        self._classifier = ThresholdClassifier(thresholds_path)
        self._context_provider = UserContextProvider(ch_connector)

        logger.info("SingleTransactionScorer initialized")

    def score(self, payment: dict, context: Optional[UserContext] = None) -> ScoringResult:
        """Score a single transaction.

        Args:
            payment: Dict with minimum fields: user_id, facility_id,
                     reservation_paid_out, created_at.
                     Optional: discount, tip, payment_method, category,
                     club_credit_flag, paid_by_manager, currency.
            context: Pre-computed UserContext (if None, fetches from ClickHouse).

        Returns:
            ScoringResult with score, is_anomaly, risk_level, percentile, factors.
        """
        # 1. Get user context
        if context is None:
            import pandas as pd
            context = self._context_provider.get_context(
                user_id=payment["user_id"],
                facility_id=payment["facility_id"],
                timestamp=pd.Timestamp(payment["created_at"]),
            )

        # 2. Calculate 31 features
        features = self._feature_calc.calculate(payment, context)

        # 3. Scale (use internal scaler directly — preprocessor expects DataFrame)
        X_scaled = self._scaler.scaler.transform(features.reshape(1, -1)).astype(np.float32)

        # 4. Score (higher = more anomalous)
        raw_score = float(-self._model.score_samples(X_scaled)[0])

        # 5. Classify
        is_anomaly, risk_level, percentile = self._classifier.classify(raw_score)

        # 6. Explain (top features by z-score magnitude)
        factors = self._explain_top_factors(features, X_scaled[0], top_n=5)

        return ScoringResult(
            score=raw_score,
            is_anomaly=is_anomaly,
            risk_level=risk_level,
            percentile=percentile,
            factors=factors,
        )

    @staticmethod
    def _explain_top_factors(
        raw_features: np.ndarray,
        scaled_features: np.ndarray,
        top_n: int = 5,
    ) -> list:
        """Top features by absolute z-score as explanation proxy."""
        abs_scaled = np.abs(scaled_features)
        top_indices = np.argsort(abs_scaled)[-top_n:][::-1]
        factors = []
        for idx in top_indices:
            factors.append({
                "feature": FEATURE_NAMES[idx],
                "value": float(raw_features[idx]),
                "z_score": float(scaled_features[idx]),
                "direction": "high" if scaled_features[idx] > 0 else "low",
            })
        return factors
