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
from fraud_detector.scoring.features_enriched import EnrichedFeatureCalculator
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
        artifacts=None,
    ):
        if artifacts is not None:
            self._model = artifacts.model
            self._scaler = artifacts.scaler
            self._feature_names = list(artifacts.feature_list)
            self._metadata = dict(artifacts.metadata)
            self._classifier = ThresholdClassifier(config=artifacts.thresholds)
        else:
            self._model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
            self._feature_names = list(FEATURE_NAMES)
            self._metadata = {
                "model_version": "IF-31-v1",
                "feature_version": "base-31",
                "score_function": "score_samples",
                "threshold_version": "v1",
            }
            self._classifier = ThresholdClassifier(thresholds_path)

        if len(self._feature_names) == 40:
            self._feature_calc = EnrichedFeatureCalculator(
                feature_engineer_path=feature_engineer_path,
                feature_list=self._feature_names,
            )
        else:
            self._feature_calc = SingleFeatureCalculator(feature_engineer_path)

        self._context_provider = UserContextProvider(ch_connector)
        self._score_function = self._metadata.get("score_function", "score_samples")
        self._model_version = self._metadata.get("model_version", "IF-31-v1")
        self._feature_version = self._metadata.get("feature_version", "base-31")
        self._threshold_version = self._metadata.get(
            "threshold_version",
            getattr(self._classifier, "_threshold_version", "v1"),
        )

        logger.info(
            "SingleTransactionScorer initialized: "
            f"model={self._model_version}, features={len(self._feature_names)}, "
            f"score_function={self._score_function}"
        )

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
                payment=payment,
            )

        # 2. Calculate feature vector in the loaded model's order.
        features = self._feature_calc.calculate(payment, context)

        raw_score, X_scaled = self.score_features(features)

        is_anomaly, risk_level, percentile = self._classifier.classify(raw_score)

        factors = self._explain_top_factors(
            features,
            X_scaled[0],
            top_n=5,
            feature_names=self._feature_names,
        )

        return ScoringResult(
            score=raw_score,
            is_anomaly=is_anomaly,
            risk_level=risk_level,
            percentile=percentile,
            factors=factors,
            model_version=self._model_version,
            feature_version=self._feature_version,
            threshold_version=self._threshold_version,
        )

    def score_features(self, features: np.ndarray) -> tuple[float, np.ndarray]:
        """Scale and score an already ordered feature vector."""
        transformer = getattr(self._scaler, "scaler", self._scaler)
        X_scaled = transformer.transform(features.reshape(1, -1)).astype(np.float32)
        X_scaled = np.clip(X_scaled, -10, 10)

        if self._score_function == "decision_function":
            raw_score = float(-self._model.decision_function(X_scaled)[0])
        elif self._score_function == "score_samples":
            raw_score = float(-self._model.score_samples(X_scaled)[0])
        else:
            raise ValueError(f"Unsupported score function: {self._score_function}")

        return raw_score, X_scaled

    @staticmethod
    def _explain_top_factors(
        raw_features: np.ndarray,
        scaled_features: np.ndarray,
        top_n: int = 5,
        feature_names: list[str] | None = None,
    ) -> list:
        """Top features by absolute z-score as explanation proxy."""
        names = feature_names or list(FEATURE_NAMES)
        abs_scaled = np.abs(scaled_features)
        top_indices = np.argsort(abs_scaled)[-top_n:][::-1]
        factors = []
        for idx in top_indices:
            factors.append(
                {
                    "feature": names[idx] if idx < len(names) else f"feature_{idx}",
                    "value": float(raw_features[idx]),
                    "z_score": float(scaled_features[idx]),
                    "direction": "high" if scaled_features[idx] > 0 else "low",
                }
            )
        return factors
