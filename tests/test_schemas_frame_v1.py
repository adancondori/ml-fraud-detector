"""Tests for frame-v1 Pydantic contract: ScoreRequest/ScoreResponse/FrameFlags."""

from __future__ import annotations

from datetime import datetime

import pytest

from scorer.schemas import FrameFlags, ScoreRequest, ScoreResponse


# ---------------------------------------------------------------------------
# ScoreRequest — optional fields with no silent defaults
# ---------------------------------------------------------------------------


class TestScoreRequestOptionalFields:
    """Validate that currency/facility_time_zone_iana/amount_local are Optional=None."""

    def _base_request(self) -> dict:
        return dict(
            user_id=1,
            facility_id=2,
            reservation_paid_out=10.0,
            created_at=datetime(2025, 1, 1, 12, 0, 0),
        )

    def test_minimal_request_validates_ok(self):
        req = ScoreRequest(**self._base_request())
        assert req.currency is None
        assert req.facility_time_zone_iana is None
        assert req.amount_local is None

    def test_currency_is_none_by_default(self):
        """currency must NOT default to 'USD' silently."""
        req = ScoreRequest(**self._base_request())
        assert req.currency is None, (
            "currency defaulting to 'USD' would silence absence — must be None"
        )

    def test_facility_time_zone_iana_is_none_by_default(self):
        """facility_time_zone_iana must NOT default to 'UTC' silently."""
        req = ScoreRequest(**self._base_request())
        assert req.facility_time_zone_iana is None, (
            "facility_time_zone_iana defaulting to 'UTC' is the exact bias being corrected"
        )

    def test_amount_local_is_none_by_default(self):
        req = ScoreRequest(**self._base_request())
        assert req.amount_local is None

    def test_explicit_currency_preserved(self):
        req = ScoreRequest(**self._base_request(), currency="MXN")
        assert req.currency == "MXN"

    def test_explicit_iana_timezone_preserved(self):
        req = ScoreRequest(**self._base_request(), facility_time_zone_iana="America/La_Paz")
        assert req.facility_time_zone_iana == "America/La_Paz"

    def test_explicit_amount_local_preserved(self):
        req = ScoreRequest(**self._base_request(), amount_local=125.50)
        assert req.amount_local == pytest.approx(125.50)

    def test_no_utc_default_anywhere_in_request(self):
        """Explicit assertion that no UTC default sneaks in."""
        req = ScoreRequest(**self._base_request())
        assert req.facility_time_zone_iana != "UTC"
        assert req.facility_time_zone_iana is None


# ---------------------------------------------------------------------------
# FrameFlags
# ---------------------------------------------------------------------------


class TestFrameFlags:
    def test_default_all_false(self):
        flags = FrameFlags()
        assert flags.timezone_missing is False
        assert flags.currency_missing is False
        assert flags.currency_unknown is False

    def test_set_timezone_missing(self):
        flags = FrameFlags(timezone_missing=True)
        assert flags.timezone_missing is True
        assert flags.currency_missing is False

    def test_roundtrip_model_dump(self):
        flags = FrameFlags(timezone_missing=True, currency_unknown=True)
        d = flags.model_dump()
        assert d["timezone_missing"] is True
        assert d["currency_unknown"] is True
        assert d["currency_missing"] is False


# ---------------------------------------------------------------------------
# ScoreResponse — backward compat + enriched fields
# ---------------------------------------------------------------------------


class TestScoreResponseBackwardCompat:
    """Existing Rails clients only send the original fields — must still validate."""

    def _base_response(self) -> dict:
        return dict(
            raw_score=-0.05,
            percentile=95.0,
            risk_level="high",
            is_anomaly=True,
            factors=[],
            model_version="IF-40-v1",
            feature_version="enriched-40",
            threshold_version="v2",
        )

    def test_legacy_response_validates_ok(self):
        resp = ScoreResponse(**self._base_response())
        assert resp.raw_score == pytest.approx(-0.05)

    def test_new_optional_fields_default_to_none(self):
        resp = ScoreResponse(**self._base_response())
        assert resp.calibration_segment is None
        assert resp.fallback_level is None
        assert resp.frame_flags is None

    def test_enriched_response_validates_ok(self):
        resp = ScoreResponse(
            **self._base_response(),
            calibration_segment="currency:MYR",
            fallback_level="currency",
            frame_flags=FrameFlags(timezone_missing=True),
        )
        assert resp.calibration_segment == "currency:MYR"
        assert resp.fallback_level == "currency"
        assert resp.frame_flags is not None
        assert resp.frame_flags.timezone_missing is True

    def test_enriched_response_roundtrip(self):
        resp = ScoreResponse(
            **self._base_response(),
            calibration_segment="facility:1234",
            fallback_level="facility",
            frame_flags=FrameFlags(currency_missing=True),
        )
        d = resp.model_dump()
        assert d["calibration_segment"] == "facility:1234"
        assert d["fallback_level"] == "facility"
        assert d["frame_flags"]["currency_missing"] is True
        assert d["frame_flags"]["timezone_missing"] is False
