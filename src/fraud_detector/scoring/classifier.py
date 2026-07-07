"""Threshold classifier — converts continuous anomaly score to binary decision + risk level.

Classes:
    ThresholdClassifier         — legacy classifier for IF-40 path (do NOT remove).
    SegmentedThresholdClassifier — new classifier with facility→currency→global fallback chain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from fraud_detector.utils.logger import logger

RISK_LEVELS = {
    "minimal": (0.0, 0.50),
    "low": (0.50, 0.70),
    "medium": (0.70, 0.85),
    "high": (0.85, 0.95),
    "critical": (0.95, 1.01),
}


# ---------------------------------------------------------------------------
# Module-level helper shared by both classifiers
# ---------------------------------------------------------------------------


def _compute_percentile(score: float, lut: np.ndarray) -> float:
    """Map a score to [0, 1] percentile using a pre-computed LUT.

    Args:
        score: Anomaly score to map.
        lut: Pre-sorted percentile LUT (result of np.percentile(scores, ...)).

    Returns:
        Percentile in [0.0, 1.0].
    """
    if len(lut) == 0:
        return 0.5
    idx = np.searchsorted(lut, score)
    return min(idx / len(lut), 1.0)


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
    # Fase 3 — populated only in the frame-v1 path; None in IF-40 legacy.
    # Using dict (not FrameFlags) to avoid coupling classifier to Pydantic.
    calibration_segment: Optional[str] = None
    fallback_level: Optional[str] = None
    frame_flags: Optional[Dict] = None


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
        return _compute_percentile(score, self._score_percentiles)

    @staticmethod
    def _assign_risk_level(percentile: float) -> str:
        for level, (low, high) in RISK_LEVELS.items():
            if low <= percentile < high:
                return level
        return "critical"


# ---------------------------------------------------------------------------
# SegmentedThresholdClassifier
# ---------------------------------------------------------------------------


class SegmentedThresholdClassifier:
    """Classifies anomaly scores using a per-segment threshold with fallback chain.

    Fallback chain: facility -> currency -> global.

    This classifier is NOT connected to SingleTransactionScorer (that is Phase 3).
    ThresholdClassifier (above) remains the live scorer path.

    Usage::

        clf = SegmentedThresholdClassifier(config)
        is_anomaly, risk_level, percentile, fallback_level, segment = clf.classify(
            score=0.05, facility_id=1234, currency="USD"
        )
    """

    def __init__(self, config: dict) -> None:
        self._global_threshold: float = float(config["binary_threshold"])
        self._global_lut: np.ndarray = np.array(config["score_percentiles"], dtype=np.float32)
        self._by_facility: dict = config.get("by_facility", {})
        self._by_currency: dict = config.get("by_currency", {})

    def classify(
        self,
        score: float,
        facility_id: int,
        currency: str,
    ) -> Tuple[bool, str, float, str, str]:
        """Classify a score using the tightest available segment.

        Resolution order:
          1. If facility_id is in by_facility → use facility threshold.
          2. elif currency is in by_currency → use currency threshold.
          3. else → use global threshold.

        Args:
            score: Anomaly score (higher = more anomalous).
            facility_id: Integer facility identifier.
            currency: ISO currency code string.

        Returns:
            Tuple of (is_anomaly, risk_level, percentile, fallback_level, calibration_segment).
        """
        fid_key = str(facility_id)

        if fid_key in self._by_facility:
            seg = self._by_facility[fid_key]
            threshold = float(seg["binary_threshold"])
            lut = np.array(seg["score_percentiles"], dtype=np.float32)
            fallback_level = "facility"
            calibration_segment = f"facility:{facility_id}"
        elif currency in self._by_currency:
            seg = self._by_currency[currency]
            threshold = float(seg["binary_threshold"])
            lut = np.array(seg["score_percentiles"], dtype=np.float32)
            fallback_level = "currency"
            calibration_segment = f"currency:{currency}"
        else:
            threshold = self._global_threshold
            lut = self._global_lut
            fallback_level = "global"
            calibration_segment = "global"

        is_anomaly: bool = bool(score > threshold)
        percentile: float = _compute_percentile(score, lut)
        risk_level: str = self._assign_risk_level(percentile)

        return is_anomaly, risk_level, percentile, fallback_level, calibration_segment

    @staticmethod
    def _assign_risk_level(percentile: float) -> str:
        for level, (low, high) in RISK_LEVELS.items():
            if low <= percentile < high:
                return level
        return "critical"
