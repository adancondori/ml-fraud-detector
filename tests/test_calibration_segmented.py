"""Tests for SegmentedThresholdCalibrator and SegmentedThresholdClassifier.

TDD approach:
  - RED: tests written before implementation.
  - GREEN: implementation makes them pass.
  - All tests use deterministic synthetic data (no model loading).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SegmentedThresholdCalibrator tests
# ---------------------------------------------------------------------------

from fraud_detector.calibration.segmented import SegmentedThresholdCalibrator


def _make_calibrator_inputs(n_big: int = 250, n_small: int = 150, seed: int = 42):
    """Build synthetic scores, facility_ids, and currencies for calibrator tests."""
    rng = np.random.default_rng(seed)

    # Segment A: facility 1, currency "USD" — n=250 (above MIN_N)
    scores_a = rng.uniform(0.0, 0.1, n_big)
    fids_a = np.full(n_big, 1, dtype=np.int64)
    curs_a = np.array(["USD"] * n_big, dtype=str)

    # Segment B: facility 2, currency "MYR" — n=150 (below MIN_N)
    scores_b = rng.uniform(0.05, 0.15, n_small)
    fids_b = np.full(n_small, 2, dtype=np.int64)
    curs_b = np.array(["MYR"] * n_small, dtype=str)

    # Segment C: facility 3, currency "CAD" — n=250 (above MIN_N)
    scores_c = rng.uniform(0.02, 0.08, n_big)
    fids_c = np.full(n_big, 3, dtype=np.int64)
    curs_c = np.array(["CAD"] * n_big, dtype=str)

    scores = np.concatenate([scores_a, scores_b, scores_c])
    facility_ids = np.concatenate([fids_a, fids_b, fids_c])
    currencies = np.concatenate([curs_a, curs_b, curs_c])
    return scores, facility_ids, currencies


class TestSegmentedThresholdCalibrator:
    """Tests for SegmentedThresholdCalibrator."""

    def test_min_n_is_200(self):
        """MIN_N must be 200 — non-negotiable guard."""
        assert SegmentedThresholdCalibrator.MIN_N == 200

    def test_segment_above_min_n_has_entry(self):
        """A segment with n=250 must appear in by_facility."""
        scores, fids, curs = _make_calibrator_inputs(n_big=250, n_small=150)
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs, percentile=95.0)
        assert "1" in result["by_facility"], "facility 1 (n=250) must have entry"
        assert "3" in result["by_facility"], "facility 3 (n=250) must have entry"

    def test_segment_below_min_n_has_no_entry(self):
        """A segment with n=150 must NOT appear in by_facility."""
        scores, fids, curs = _make_calibrator_inputs(n_big=250, n_small=150)
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs, percentile=95.0)
        assert "2" not in result["by_facility"], "facility 2 (n=150) must be excluded"

    def test_all_facility_entries_have_n_ge_min_n(self):
        """Every entry in by_facility must have n >= MIN_N."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        for fid_str, entry in result["by_facility"].items():
            assert entry["n"] >= SegmentedThresholdCalibrator.MIN_N, (
                f"facility {fid_str} has n={entry['n']} < MIN_N"
            )

    def test_all_currency_entries_have_n_ge_min_n(self):
        """Every entry in by_currency must have n >= MIN_N."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        for cur_str, entry in result["by_currency"].items():
            assert entry["n"] >= SegmentedThresholdCalibrator.MIN_N, (
                f"currency {cur_str} has n={entry['n']} < MIN_N"
            )

    def test_root_keys_binary_threshold_and_score_percentiles_exist(self):
        """Root keys binary_threshold and score_percentiles must exist (backward-compat)."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        assert "binary_threshold" in result, "binary_threshold must be at root level"
        assert "score_percentiles" in result, "score_percentiles must be at root level"

    def test_global_lut_has_1001_points(self):
        """Global score_percentiles LUT must have exactly 1001 points."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        assert len(result["score_percentiles"]) == 1001, (
            f"Expected 1001 global LUT points, got {len(result['score_percentiles'])}"
        )

    def test_segment_lut_has_201_points(self):
        """Each segment LUT must have exactly 201 points (compact)."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        for fid_str, entry in result["by_facility"].items():
            assert len(entry["score_percentiles"]) == 201, (
                f"facility {fid_str} LUT has {len(entry['score_percentiles'])} points, expected 201"
            )
        for cur_str, entry in result["by_currency"].items():
            assert len(entry["score_percentiles"]) == 201, (
                f"currency {cur_str} LUT has {len(entry['score_percentiles'])} points, expected 201"
            )

    def test_global_threshold_matches_np_percentile(self):
        """Global binary_threshold must equal np.percentile(scores, percentile)."""
        rng = np.random.default_rng(0)
        scores = rng.uniform(0.0, 1.0, 1000)
        fids = np.ones(1000, dtype=np.int64)
        curs = np.array(["USD"] * 1000)
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs, percentile=95.0)
        expected = float(np.percentile(scores, 95.0))
        assert abs(result["binary_threshold"] - expected) < 1e-9, (
            f"binary_threshold={result['binary_threshold']} != np.percentile={expected}"
        )

    def test_fallback_level_facility(self):
        """Entries in by_facility must have fallback_level='facility'."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        for fid_str, entry in result["by_facility"].items():
            assert entry["fallback_level"] == "facility", (
                f"facility {fid_str} fallback_level={entry['fallback_level']!r}"
            )

    def test_fallback_level_currency(self):
        """Entries in by_currency must have fallback_level='currency'."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        for cur_str, entry in result["by_currency"].items():
            assert entry["fallback_level"] == "currency", (
                f"currency {cur_str} fallback_level={entry['fallback_level']!r}"
            )

    def test_metadata_keys_present(self):
        """Required metadata keys must be present at root level."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        required_meta = {
            "schema_version", "model_version", "feature_version",
            "calibration_source", "calibration_rows", "min_n_threshold",
            "percentile", "threshold_version", "built_at",
        }
        missing = required_meta - set(result.keys())
        assert not missing, f"Missing metadata keys: {missing}"

    def test_calibration_rows_matches_input_length(self):
        """calibration_rows must equal len(scores)."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        assert result["calibration_rows"] == len(scores)

    def test_schema_version(self):
        """schema_version must be 'thresholds-segmented-v1'."""
        scores, fids, curs = _make_calibrator_inputs()
        cal = SegmentedThresholdCalibrator()
        result = cal.fit(scores, fids, curs)
        assert result["schema_version"] == "thresholds-segmented-v1"


# ---------------------------------------------------------------------------
# SegmentedThresholdClassifier tests
# ---------------------------------------------------------------------------

from fraud_detector.scoring.classifier import SegmentedThresholdClassifier


def _make_classifier_config():
    """Build a minimal config dict for SegmentedThresholdClassifier tests."""
    global_lut = np.linspace(0.0, 0.2, 1001).tolist()
    seg_lut = np.linspace(0.0, 0.15, 201).tolist()
    facility_lut = np.linspace(0.0, 0.12, 201).tolist()

    return {
        "binary_threshold": 0.05,
        "score_percentiles": global_lut,
        "by_facility": {
            "123": {
                "binary_threshold": 0.06,
                "n": 300,
                "fallback_level": "facility",
                "score_percentiles": facility_lut,
            }
        },
        "by_currency": {
            "MYR": {
                "binary_threshold": 0.07,
                "n": 250,
                "fallback_level": "currency",
                "score_percentiles": seg_lut,
            }
        },
    }


class TestSegmentedThresholdClassifier:
    """Tests for SegmentedThresholdClassifier."""

    def test_facility_fallback_level_when_facility_known(self):
        """classify(score, 123, 'USD') must use facility segment (fallback_level='facility')."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        _, _, _, fallback_level, segment = clf.classify(0.03, 123, "USD")
        assert fallback_level == "facility"
        assert segment == "facility:123"

    def test_currency_fallback_level_when_facility_unknown(self):
        """classify(score, 999, 'MYR') — facility unknown, currency known → 'currency'."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        _, _, _, fallback_level, segment = clf.classify(0.03, 999, "MYR")
        assert fallback_level == "currency"
        assert segment == "currency:MYR"

    def test_global_fallback_when_both_unknown(self):
        """classify(score, 999, 'ZZZ') — both unknown → 'global'."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        _, _, _, fallback_level, segment = clf.classify(0.03, 999, "ZZZ")
        assert fallback_level == "global"
        assert segment == "global"

    def test_is_anomaly_true_when_score_above_facility_threshold(self):
        """Score above facility threshold → is_anomaly=True."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        # facility 123 threshold = 0.06; score 0.07 > 0.06
        is_anomaly, _, _, fallback_level, _ = clf.classify(0.07, 123, "USD")
        assert is_anomaly is True
        assert fallback_level == "facility"

    def test_is_anomaly_false_when_score_below_facility_threshold(self):
        """Score below facility threshold → is_anomaly=False."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        # facility 123 threshold = 0.06; score 0.05 < 0.06
        is_anomaly, _, _, fallback_level, _ = clf.classify(0.05, 123, "USD")
        assert is_anomaly is False
        assert fallback_level == "facility"

    def test_is_anomaly_true_when_score_above_currency_threshold(self):
        """Score above currency threshold → is_anomaly=True."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        # MYR threshold = 0.07; score 0.08 > 0.07
        is_anomaly, _, _, fallback_level, _ = clf.classify(0.08, 999, "MYR")
        assert is_anomaly is True
        assert fallback_level == "currency"

    def test_is_anomaly_false_when_score_below_global_threshold(self):
        """Score below global threshold → is_anomaly=False (global fallback)."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        # global threshold = 0.05; score 0.03 < 0.05
        is_anomaly, _, _, fallback_level, _ = clf.classify(0.03, 999, "ZZZ")
        assert is_anomaly is False
        assert fallback_level == "global"

    def test_classify_returns_5_tuple(self):
        """classify() must return a tuple of exactly 5 elements."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        result = clf.classify(0.03, 123, "USD")
        assert len(result) == 5, f"Expected 5-tuple, got {len(result)}"

    def test_risk_level_is_string(self):
        """risk_level must be a string."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        _, risk_level, _, _, _ = clf.classify(0.03, 123, "USD")
        assert isinstance(risk_level, str)

    def test_percentile_is_float_in_range(self):
        """percentile must be a float in [0.0, 1.0]."""
        clf = SegmentedThresholdClassifier(_make_classifier_config())
        _, _, percentile, _, _ = clf.classify(0.03, 123, "USD")
        assert 0.0 <= percentile <= 1.0

    def test_facility_id_as_int_or_str_both_work(self):
        """facility_id lookup must work for int 123 (config key is '123')."""
        config = _make_classifier_config()
        clf = SegmentedThresholdClassifier(config)
        _, _, _, fallback_level, _ = clf.classify(0.03, 123, "USD")
        assert fallback_level == "facility", "int facility_id must resolve via str key"

    def test_legacy_threshold_classifier_still_present(self):
        """ThresholdClassifier (legacy) must still be importable from classifier.py."""
        from fraud_detector.scoring.classifier import ThresholdClassifier  # noqa: F401
        assert ThresholdClassifier is not None
