"""Contract tests frame-v1 — fixtures canónicos request/response (SDD frame-normalization-v1).

Fuente única del contrato scorer↔Rails (design D3): los fixtures en
tests/fixtures/contract/frame_v1/ son el shape canónico; platform vendoriza
una copia. Cada fixture embebe feature_frame_version: "frame-v1" como gate
anti-drift.

Scenarios cubiertos (specs/scorer-frame):
- contrato/payload-nuevo: ScoreRequest acepta el payload Rails completo
  (incluye facility_time_zone + facility_time_zone_iana + amount_local).
- contrato/payload-legacy: payload pre-cambio parsea con None (sin defaults
  "USD"/"UTC" que disfracen ausencia).
- contrato/campo-desconocido: campos extra se ignoran sin error.
- contrato/amount-accidental: un campo `amount` no sobreescribe
  reservation_paid_out (design D9).
- respuesta/rt-sin-tasas: la respuesta RT ecoa amount_local/currency y OMITE
  amount_usd_display (no se inventa USD sin tasas — decisión humana 1).
- respuesta/batch-con-conversión: la entrada de critical_alerts incluye
  amount_usd_display además de amount_local y currency.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fraud_detector.scoring.classifier import ScoringResult
from scorer.schemas import CriticalAlert, ScoreRequest, ScoreResponse

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "contract" / "frame_v1"
FIXTURE_NAMES = [
    "request_full.json",
    "request_legacy.json",
    "response_rt.json",
    "response_batch.json",
]

CONTRACT_VERSION = "frame-v1"


def _load_fixture(name: str) -> dict:
    path = FIXTURE_DIR / name
    assert path.exists(), f"Fixture canónico faltante: {path}"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Gate anti-drift (design D3): fixtures existen y declaran su versión
# ---------------------------------------------------------------------------


class TestContractFixturesVersioned:
    @pytest.mark.parametrize("name", FIXTURE_NAMES)
    def test_fixture_exists_and_embeds_frame_version(self, name):
        fixture = _load_fixture(name)
        assert fixture.get("feature_frame_version") == CONTRACT_VERSION, (
            f"{name} debe embeber feature_frame_version={CONTRACT_VERSION!r} "
            f"(gate anti-drift, design D3)"
        )


# ---------------------------------------------------------------------------
# ScoreRequest — contrato retrocompatible bidireccional
# ---------------------------------------------------------------------------


class TestScoreRequestContract:
    def test_full_rails_payload_parses_and_populates_frame_fields(self):
        """Scenario contrato/payload-nuevo: payload completo post-cambio."""
        fixture = _load_fixture("request_full.json")
        req = ScoreRequest.model_validate(fixture)
        assert req.amount_local == pytest.approx(fixture["amount_local"])
        assert req.currency == fixture["currency"]
        assert req.facility_time_zone == fixture["facility_time_zone"]
        assert req.facility_time_zone_iana == fixture["facility_time_zone_iana"]
        # El campo legacy sigue siendo la fuente del monto base (design D9).
        assert req.reservation_paid_out == pytest.approx(fixture["reservation_paid_out"])

    def test_legacy_payload_parses_with_none_not_defaults(self):
        """Scenario contrato/payload-legacy: sin defaults que disfracen ausencia."""
        fixture = _load_fixture("request_legacy.json")
        assert "amount_local" not in fixture
        assert "currency" not in fixture
        assert "facility_time_zone" not in fixture
        assert "facility_time_zone_iana" not in fixture

        req = ScoreRequest.model_validate(fixture)
        assert req.currency is None, "currency legacy debe quedar None, nunca 'USD'"
        assert req.facility_time_zone is None, "facility_time_zone debe quedar None"
        assert (
            req.facility_time_zone_iana is None
        ), "facility_time_zone_iana debe quedar None, nunca 'UTC'"
        assert req.amount_local is None

    def test_facility_time_zone_defaults_none_without_utc(self):
        """facility_time_zone (nombre Rails) es Optional sin default."""
        req = ScoreRequest(
            user_id=1,
            facility_id=2,
            reservation_paid_out=10.0,
            created_at="2026-07-01T18:30:00",
        )
        assert req.facility_time_zone is None
        assert req.facility_time_zone != "UTC"

    def test_unknown_field_is_ignored(self):
        """Scenario contrato/campo-desconocido: extra no rompe ni se cuela."""
        fixture = _load_fixture("request_full.json")
        payload = dict(fixture)
        payload["facility_primary_currency"] = "CAD"

        req = ScoreRequest.model_validate(payload)
        dumped = req.model_dump()
        assert "facility_primary_currency" not in dumped
        assert req.currency == fixture["currency"]

    def test_accidental_amount_does_not_override_reservation_paid_out(self):
        """Scenario contrato/amount-accidental (design D9)."""
        fixture = _load_fixture("request_full.json")
        payload = dict(fixture)
        payload["amount"] = 999.0

        req = ScoreRequest.model_validate(payload)
        assert req.reservation_paid_out == pytest.approx(fixture["reservation_paid_out"])
        assert req.amount_local == pytest.approx(fixture["amount_local"])
        assert "amount" not in req.model_dump(), (
            "un campo genérico `amount` en el payload real-time no debe entrar "
            "al contrato ni sobreescribir reservation_paid_out"
        )


# ---------------------------------------------------------------------------
# Respuesta frame-v1 — eco de metadata monetaria con USD opcional
# ---------------------------------------------------------------------------


def _frame_v1_scoring_result() -> ScoringResult:
    """ScoringResult como lo produce el path frame-v1 de SingleTransactionScorer."""
    return ScoringResult(
        score=0.4123,
        is_anomaly=True,
        risk_level="high",
        percentile=97.5,
        factors=[{"feature": "log_amount_fac", "value": 2.31, "z_score": 3.1, "direction": "high"}],
        model_version="frame-v1",
        feature_version="frame-v1",
        threshold_version="segmented-v1",
        calibration_segment="facility:7",
        fallback_level="facility",
        frame_flags={
            "timezone_missing": False,
            "currency_missing": False,
            "currency_unknown": False,
        },
    )


class TestScoreResponseEcho:
    """Scenario respuesta/rt-sin-tasas — el fixture response_rt.json es el shape canónico."""

    def test_response_rt_fixture_parses_with_echo_fields(self):
        fixture = _load_fixture("response_rt.json")
        resp = ScoreResponse.model_validate(fixture)
        assert resp.amount_local == pytest.approx(120.0)
        assert resp.currency == "CAD"
        # Campos de calibración/versionado completos
        assert resp.calibration_segment == "facility:7"
        assert resp.fallback_level == "facility"
        assert resp.frame_flags is not None
        assert resp.model_version == "frame-v1"
        assert resp.feature_version == "frame-v1"
        assert resp.threshold_version == "segmented-v1"

    def test_response_rt_fixture_omits_amount_usd_display(self):
        """RT sin tasas: amount_usd_display AUSENTE, no null ni inventado."""
        fixture = _load_fixture("response_rt.json")
        assert "amount_usd_display" not in fixture

    def test_api_score_echoes_amount_local_and_currency(self):
        """POST /score con payload Rails completo ecoa amount_local/currency
        y NO incluye amount_usd_display (el proceso RT no tiene tasas)."""
        from fastapi.testclient import TestClient

        from scorer.dependencies import get_scorer
        from scorer.main import app

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = _frame_v1_scoring_result()

        payload = _load_fixture("request_full.json")
        prev = app.dependency_overrides.get(get_scorer)
        app.dependency_overrides[get_scorer] = lambda: mock_scorer
        try:
            client = TestClient(app, raise_server_exceptions=True)
            response = client.post("/api/v1/score", json=payload)
        finally:
            if prev is None:
                app.dependency_overrides.pop(get_scorer, None)
            else:
                app.dependency_overrides[get_scorer] = prev

        assert response.status_code == 200
        body = response.json()
        # Eco de metadata monetaria
        assert body["amount_local"] == pytest.approx(120.0)
        assert body["currency"] == "CAD"
        # USD opcional: MUST omitirse cuando no hay tasas en el proceso RT
        assert "amount_usd_display" not in body
        # Calibración/versionado completos
        assert body["calibration_segment"] == "facility:7"
        assert body["fallback_level"] == "facility"
        assert body["frame_flags"]["timezone_missing"] is False
        assert body["model_version"] == "frame-v1"
        assert body["feature_version"] == "frame-v1"
        assert body["threshold_version"] == "segmented-v1"


class TestBatchCriticalAlertContract:
    """Scenario respuesta/batch-con-conversión — entrada de critical_alerts."""

    def test_response_batch_fixture_parses_with_usd_display(self):
        fixture = _load_fixture("response_batch.json")
        alert = CriticalAlert.model_validate(fixture)
        assert alert.amount_local == pytest.approx(120.0)
        assert alert.currency == "CAD"
        assert alert.amount_usd_display == pytest.approx(88.42)

    def test_batch_critical_alert_includes_local_and_usd_display(self):
        """_score_all produce entradas con amount_local/currency/amount_usd_display."""
        from fraud_detector.utils.currency import normalize_amount_value
        from scorer.batch.scorer import BatchScorer

        mock_scorer = MagicMock()
        mock_scorer._feature_names = [f"feature_{i}" for i in range(31)]
        mock_scorer._feature_version = "frame-v1"
        mock_scorer._threshold_version = "segmented-v1"
        mock_scorer._feature_calc.calculate.return_value = np.zeros(31)
        mock_scorer.score_features.return_value = (0.97, np.zeros((1, 31), dtype=np.float32))
        mock_scorer._classifier.classify.return_value = (True, "critical", 0.98)

        batch_scorer = BatchScorer(
            scorer=mock_scorer,
            read_ch_client=MagicMock(),
            write_ch_client=MagicMock(),
            read_fingerprint=("prod", 8443, "db", True, "ro"),
            write_fingerprint=("clickhouse", 8123, "db", False, "default"),
            write_host="clickhouse",
        )

        payment = {
            "payment_id": 987654,
            "user_id": 42,
            "facility_id": 7,
            "reservation_paid_out": 120.0,
            "created_at": "2026-07-01T18:30:00",
            "currency": "CAD",
            "payment_method": "card",
        }
        with patch(
            "fraud_detector.scoring.scorer.SingleTransactionScorer._explain_top_factors",
            return_value=[],
        ):
            _, critical_alerts = batch_scorer._score_all([payment], {}, "frame-v1")

        assert len(critical_alerts) == 1
        alert = critical_alerts[0]
        assert alert["amount_local"] == pytest.approx(120.0)
        assert alert["currency"] == "CAD"
        expected_usd = normalize_amount_value(120.0, "CAD")
        assert alert["amount_usd_display"] == pytest.approx(expected_usd)
        # La llave analítica existente no se renombra (design D8)
        assert alert["amount_usd"] == pytest.approx(expected_usd)
        # Y la entrada valida contra el schema del contrato
        CriticalAlert.model_validate(alert)
