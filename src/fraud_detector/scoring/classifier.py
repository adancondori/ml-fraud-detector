"""Threshold classifier — converts continuous anomaly score to binary decision + risk level."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

import numpy as np

from fraud_detector.utils.logger import logger

RISK_LEVELS = {
    "minimal": (0.0, 0.50),
    "low": (0.50, 0.70),
    "medium": (0.70, 0.85),
    "high": (0.85, 0.95),
    "critical": (0.95, 1.01),
}


@dataclass
class ScoringResult:
    """Result of scoring a single transaction."""

    score: float
    is_anomaly: bool
    risk_level: str
    percentile: float
    factors: List[dict] = field(default_factory=list)
    model_version: str = "IF-31-v1"
    feature_version: str = "base-31"
    threshold_version: str = "v1"


class ThresholdClassifier:
    """Converts a continuous anomaly score into a binary decision and risk level.

    Uses the threshold calibrated in Fase 7 (percentile 95 of test set scores).
    Percentiles are pre-computed from the test set score distribution.
    """

    def __init__(
        self,
        thresholds_path: str = "output/models/thresholds.json",
        config: dict | None = None,
    ):
        if config is None:
            with open(thresholds_path) as f:
                config = json.load(f)
        self._threshold = config["binary_threshold"]
        self._score_percentiles = np.array(config["score_percentiles"], dtype=np.float32)
        self._threshold_version = config.get("threshold_version", "v1")
        logger.info(
            f"ThresholdClassifier loaded: threshold={self._threshold:.6f}, "
            f"percentile_bins={len(self._score_percentiles)}"
        )

    @property
    def threshold(self) -> float:
        return self._threshold

    def classify(self, score: float) -> tuple:
        """Classify a score into (is_anomaly, risk_level, percentile).

        Args:
            score: Anomaly score (higher = more anomalous).

        Returns:
            (is_anomaly: bool, risk_level: str, percentile: float)
        """
        is_anomaly = score > self._threshold
        percentile = self._compute_percentile(score)
        risk_level = self._assign_risk_level(percentile)
        return is_anomaly, risk_level, percentile

    def _compute_percentile(self, score: float) -> float:
        if len(self._score_percentiles) == 0:
            return 0.5
        idx = np.searchsorted(self._score_percentiles, score)
        return min(idx / len(self._score_percentiles), 1.0)

    @staticmethod
    def _assign_risk_level(percentile: float) -> str:
        for level, (low, high) in RISK_LEVELS.items():
            if low <= percentile < high:
                return level
        return "critical"
