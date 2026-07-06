"""Segmented threshold calibrator for anomaly scores.

Calibrates binary classification thresholds per-segment (facility or currency)
using a held-out validation set. Segments with fewer than MIN_N observations
are excluded and their transactions fall back to a broader segment.

Fallback chain: facility -> currency -> global.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

import numpy as np


class SegmentedThresholdCalibrator:
    """Calibrates anomaly score thresholds for each segment with sufficient data.

    Segments are defined by facility_id (finest grain) and currency (middle grain).
    Segments with n < MIN_N are not assigned their own threshold; they fall back
    to the next level in the chain.

    Usage::

        calibrator = SegmentedThresholdCalibrator()
        thresholds = calibrator.fit(
            scores=scores_array,          # 1-D float array, higher = more anomalous
            facility_ids=fid_array,       # 1-D int array
            currencies=currency_array,    # 1-D str array
            percentile=95.0,
        )
        # thresholds is a dict ready for json.dump
    """

    #: Minimum segment size to receive its own threshold (non-negotiable guard).
    MIN_N: int = 200

    def fit(
        self,
        scores: np.ndarray,
        facility_ids: np.ndarray,
        currencies: np.ndarray,
        percentile: float = 95.0,
        model_version: str = "frame-v1",
        feature_version: str = "frame-v1",
        calibration_source: str = "validation_set",
    ) -> dict:
        """Compute segmented thresholds from a set of validation-set scores.

        Args:
            scores: 1-D array of anomaly scores (higher = more anomalous).
            facility_ids: 1-D integer array aligned with scores.
            currencies: 1-D string array aligned with scores.
            percentile: Percentile to use as binary_threshold (default 95.0).
            model_version: Identifier for the scoring model (metadata only).
            feature_version: Identifier for the feature set (metadata only).
            calibration_source: Where these scores came from (metadata only).

        Returns:
            Dict with root-level global thresholds (backward-compat with
            _validate_artifacts) plus by_facility and by_currency sections.
        """
        scores = np.asarray(scores, dtype=np.float64)
        facility_ids = np.asarray(facility_ids)
        currencies = np.asarray(currencies, dtype=str)

        n_total = len(scores)

        # --- Global threshold (always computed; root-level for backward-compat) ---
        global_threshold = float(np.percentile(scores, percentile))
        global_lut = np.percentile(scores, np.linspace(0, 100, 1001)).tolist()

        # --- By-facility ---
        by_facility: dict = {}
        unique_fids = np.unique(facility_ids)
        for fid in unique_fids:
            mask = facility_ids == fid
            n = int(mask.sum())
            if n < self.MIN_N:
                continue
            seg_scores = scores[mask]
            by_facility[str(fid)] = {
                "binary_threshold": float(np.percentile(seg_scores, percentile)),
                "n": n,
                "fallback_level": "facility",
                "score_percentiles": np.percentile(
                    seg_scores, np.linspace(0, 100, 201)
                ).tolist(),
            }

        # --- By-currency ---
        by_currency: dict = {}
        unique_curs = np.unique(currencies)
        for cur in unique_curs:
            mask = currencies == cur
            n = int(mask.sum())
            if n < self.MIN_N:
                continue
            seg_scores = scores[mask]
            by_currency[str(cur)] = {
                "binary_threshold": float(np.percentile(seg_scores, percentile)),
                "n": n,
                "fallback_level": "currency",
                "score_percentiles": np.percentile(
                    seg_scores, np.linspace(0, 100, 201)
                ).tolist(),
            }

        return {
            # Root-level global threshold — backward-compat with _validate_artifacts
            "binary_threshold": global_threshold,
            "score_percentiles": global_lut,
            # Segmented sections
            "by_facility": by_facility,
            "by_currency": by_currency,
            # Metadata
            "schema_version": "thresholds-segmented-v1",
            "model_version": model_version,
            "feature_version": feature_version,
            "calibration_source": calibration_source,
            "calibration_rows": n_total,
            "min_n_threshold": self.MIN_N,
            "percentile": percentile,
            "threshold_version": "segmented-v1",
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
