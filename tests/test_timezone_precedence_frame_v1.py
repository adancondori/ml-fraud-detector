"""Precedencia de timezone frame-v1 (SDD frame-normalization-v1, design D1/D5).

El scorer resuelve la zona horaria para features temporales con precedencia:
  (1) facility_time_zone_iana del payload si es IANA válida (ZoneInfo — tzdata
      es la autoridad, sin tablas/mapeos propios),
  (2) iana_tz del artefacto de facility stats,
  (3) Etc/UTC.

El nivel usado es observable vía frame_flags:
  - timezone_from_artifact=True cuando NO se usó el payload pero SÍ el artefacto,
  - timezone_missing=True cuando se cayó a Etc/UTC.

Scenarios (specs/scorer-frame): precedencia/payload-prioridad,
precedencia/inválida-artefacto, precedencia/utc-último, precedencia/dst-payload.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from fraud_detector.scoring.context import UserContext
from fraud_detector.scoring.features_frame_v1 import (
    FRAME_V1_FEATURE_NAMES,
    FrameV1FeatureCalculator,
)
from scorer.schemas import FrameFlags

ARTIFACT_TZ = "America/New_York"

_IDX = {name: i for i, name in enumerate(FRAME_V1_FEATURE_NAMES)}

SYNTH_STATS = {
    "facilities": {
        # Facility conocida con IANA en el artefacto
        "7": {
            "n": 500,
            "median": 50.0,
            "mean": 60.0,
            "iqr": 20.0,
            "iqr_guarded": 20.0,
            "iana_tz": ARTIFACT_TZ,
            "fallback_level": "facility",
        },
        # Facility conocida pero con iana_tz nulo (builder degradado, design D6)
        "8": {
            "n": 500,
            "median": 50.0,
            "mean": 60.0,
            "iqr": 20.0,
            "iqr_guarded": 20.0,
            "iana_tz": None,
            "fallback_level": "facility",
        },
    },
    "global_fallback": {
        "median": 20.0,
        "mean": 30.0,
        "iqr": 10.0,
        "iqr_guarded": 10.0,
        "n": 1000,
        "fallback_level": "global",
    },
}


def _make_calc() -> FrameV1FeatureCalculator:
    """FrameV1FeatureCalculator con stats sintéticas, sin joblib (CI-safe)."""
    calc = FrameV1FeatureCalculator.__new__(FrameV1FeatureCalculator)
    calc._stats = SYNTH_STATS
    calc._staff_role_currency = {}
    calc._staff_currency = {}
    calc._staff_global_mean = 0.0
    calc._staff_global_std = 1.0
    return calc


def _payment(facility_id=7, created_at="2025-06-15T12:00:00Z", iana=None) -> dict:
    return {
        "payment_id": 1,
        "user_id": 1,
        "facility_id": facility_id,
        "reservation_paid_out": 100.0,
        "created_at": created_at,
        "discount": 0.0,
        "tip": 0.0,
        "payment_method": "card",
        "category": "reservation",
        "club_credit_flag": False,
        "paid_by_manager": False,
        "currency": "USD",
        "facility_time_zone_iana": iana,
    }


# ---------------------------------------------------------------------------
# Unidad: resolve_timezone — cadena payload → artefacto → Etc/UTC
# ---------------------------------------------------------------------------


class TestResolveTimezone:
    def test_valid_payload_iana_wins_over_artifact(self):
        calc = _make_calc()
        tz, source = calc.resolve_timezone(7, "America/Los_Angeles")
        assert tz == "America/Los_Angeles"
        assert source == "payload"

    def test_invalid_payload_iana_falls_to_artifact(self):
        """'Mars/Olympus' no es IANA válida — cae al artefacto sin excepción."""
        calc = _make_calc()
        tz, source = calc.resolve_timezone(7, "Mars/Olympus")
        assert tz == ARTIFACT_TZ
        assert source == "artifact"

    def test_no_payload_uses_artifact(self):
        calc = _make_calc()
        tz, source = calc.resolve_timezone(7, None)
        assert tz == ARTIFACT_TZ
        assert source == "artifact"

    def test_no_payload_unknown_facility_falls_to_utc(self):
        calc = _make_calc()
        tz, source = calc.resolve_timezone(999999, None)
        assert tz == "Etc/UTC"
        assert source == "utc"

    def test_empty_string_payload_treated_as_absent(self):
        """El LEFT JOIN batch produce '' cuando no hay match — es ausencia."""
        calc = _make_calc()
        tz, source = calc.resolve_timezone(7, "")
        assert tz == ARTIFACT_TZ
        assert source == "artifact"

    def test_artifact_null_iana_degrades_to_utc(self):
        """iana_tz nulo en el artefacto degrada por la cadena normal (D6)."""
        calc = _make_calc()
        tz, source = calc.resolve_timezone(8, None)
        assert tz == "Etc/UTC"
        assert source == "utc"


# ---------------------------------------------------------------------------
# Vector frame-v1: la fuente efectiva gobierna las features temporales
# ---------------------------------------------------------------------------


class TestCalculateTimezonePrecedence:
    def test_payload_iana_wins_over_artifact(self):
        """2025-06-15T12:00Z: LA=05:00 (off-hours), NY=08:00 (no) — gana LA."""
        calc = _make_calc()
        vec = calc.calculate(_payment(iana="America/Los_Angeles"), UserContext())
        assert vec[_IDX["is_off_hours_loc"]] == pytest.approx(1.0), (
            "con payload America/Los_Angeles la hora local es 05:00 (off-hours); "
            "si sale 0.0 el calculator ignoró el payload y usó el artefacto NY"
        )

    def test_invalid_payload_falls_back_to_artifact_tz(self):
        """'Mars/Olympus' cae al artefacto (NY 08:00, no off-hours) sin excepción."""
        calc = _make_calc()
        vec = calc.calculate(_payment(iana="Mars/Olympus"), UserContext())
        assert vec[_IDX["is_off_hours_loc"]] == pytest.approx(0.0)

    def test_dst_fall_back_with_payload_tz(self):
        """2025-11-02T05:30Z + payload America/New_York = 01:30 EDT (fall-back).

        A las 05:30Z todavía rige EDT (UTC-4): hora local 1, sin hora imposible.
        """
        calc = _make_calc()
        vec = calc.calculate(
            _payment(created_at="2025-11-02T05:30:00Z", iana="America/New_York"),
            UserContext(),
        )
        expected_sin = math.sin(2 * math.pi * 1 / 24)
        expected_cos = math.cos(2 * math.pi * 1 / 24)
        assert vec[_IDX["hour_sin_loc"]] == pytest.approx(expected_sin, abs=1e-6)
        assert vec[_IDX["hour_cos_loc"]] == pytest.approx(expected_cos, abs=1e-6)
        assert vec[_IDX["is_off_hours_loc"]] == pytest.approx(1.0)

    def test_utc_last_resort_for_unknown_facility_without_payload(self):
        """Sin payload y facility desconocida → Etc/UTC (hora 12)."""
        calc = _make_calc()
        vec = calc.calculate(_payment(facility_id=999999, iana=None), UserContext())
        assert vec[_IDX["hour_sin_loc"]] == pytest.approx(0.0, abs=1e-6)
        assert vec[_IDX["hour_cos_loc"]] == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# FrameFlags — extensión aditiva timezone_from_artifact
# ---------------------------------------------------------------------------


class TestFrameFlagsTimezoneFromArtifact:
    def test_defaults_false(self):
        flags = FrameFlags()
        assert flags.timezone_from_artifact is False

    def test_roundtrip(self):
        flags = FrameFlags(timezone_from_artifact=True)
        d = flags.model_dump()
        assert d["timezone_from_artifact"] is True
        assert d["timezone_missing"] is False


# ---------------------------------------------------------------------------
# SingleTransactionScorer — frame_flags reflejan el nivel usado
# ---------------------------------------------------------------------------


def _make_frame_scorer():
    """SingleTransactionScorer frame-v1 sin artefactos pesados (CI-safe)."""
    from fraud_detector.scoring.scorer import SingleTransactionScorer

    scorer = SingleTransactionScorer.__new__(SingleTransactionScorer)
    scorer._is_frame_v1 = True
    scorer._feature_calc = _make_calc()
    classifier = MagicMock()
    classifier.classify.return_value = (False, "low", 50.0, "facility", "facility:7")
    scorer._classifier = classifier
    scorer._feature_names = list(FRAME_V1_FEATURE_NAMES)
    scorer._model_version = "frame-v1"
    scorer._feature_version = "frame-v1"
    scorer._threshold_version = "segmented-v1"
    scorer.score_features = lambda features: (0.5, np.zeros((1, 30), dtype=np.float32))
    return scorer


class TestScorerFrameFlagsPrecedence:
    def test_payload_used_no_fallback_flags(self):
        """Payload IANA válida: sin flags de fallback de timezone."""
        scorer = _make_frame_scorer()
        result = scorer.score(_payment(iana="America/Los_Angeles"), context=UserContext())
        assert result.frame_flags["timezone_from_artifact"] is False
        assert result.frame_flags["timezone_missing"] is False

    def test_invalid_payload_flags_artifact(self):
        """IANA inválida → artefacto usado → timezone_from_artifact=True."""
        scorer = _make_frame_scorer()
        result = scorer.score(_payment(iana="Mars/Olympus"), context=UserContext())
        assert result.frame_flags["timezone_from_artifact"] is True
        assert result.frame_flags["timezone_missing"] is False

    def test_utc_flags_missing_not_artifact(self):
        """Sin payload y facility desconocida → timezone_missing=True,
        timezone_from_artifact=False."""
        scorer = _make_frame_scorer()
        result = scorer.score(_payment(facility_id=999999, iana=None), context=UserContext())
        assert result.frame_flags["timezone_missing"] is True
        assert result.frame_flags["timezone_from_artifact"] is False

    def test_flags_parse_into_schema(self):
        """El dict de flags valida contra FrameFlags (path del router)."""
        scorer = _make_frame_scorer()
        result = scorer.score(_payment(iana="Mars/Olympus"), context=UserContext())
        flags = FrameFlags(**result.frame_flags)
        assert flags.timezone_from_artifact is True
