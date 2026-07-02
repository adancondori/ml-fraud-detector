"""Integration tests for the ML Scorer HTTP API.

Uses FastAPI TestClient with dependency_overrides + direct _state injection
so no model files are required (CI-safe).

The scorer now uses two ClickHouse clients (READ prod / WRITE local). Health
reports clickhouse_connected = read_ok AND write_ok, and /score/batch injects
both clients plus the guardrail metadata from _state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from scorer.main import app
from scorer.dependencies import (
    get_scorer,
    get_ch_client,
    get_read_ch_client,
    get_write_ch_client,
    _state as scorer_state,
)
from scorer.batch.scorer import BatchScorer
from fraud_detector.scoring.classifier import ScoringResult


# ---------------------------------------------------------------------------
# Shared mock objects
# ---------------------------------------------------------------------------


def _make_mock_scorer():
    mock = MagicMock()
    mock._feature_names = [f"feature_{i}" for i in range(31)]
    mock._model_version = "IF-31-v1-test"
    mock._feature_version = "base-31"
    mock._threshold_version = "v1-test"
    mock._score_function = "score_samples"
    mock._classifier._threshold = 0.123
    mock.score.return_value = ScoringResult(
        score=0.75,
        is_anomaly=True,
        risk_level="high",
        percentile=0.92,
        factors=[{"feature": "log_amount", "value": 5.5, "z_score": 3.2, "direction": "high"}],
        model_version="IF-31-v1-test",
        feature_version="base-31",
        threshold_version="v1-test",
    )
    return mock


def _make_mock_ch_client():
    mock = MagicMock()
    mock.command.return_value = 1  # SELECT 1
    return mock


MOCK_SCORER = _make_mock_scorer()
MOCK_READ_CLIENT = _make_mock_ch_client()
MOCK_WRITE_CLIENT = _make_mock_ch_client()

# Override DI for the whole test module. get_ch_client is a READ alias.
app.dependency_overrides[get_scorer] = lambda: MOCK_SCORER
app.dependency_overrides[get_ch_client] = lambda: MOCK_READ_CLIENT
app.dependency_overrides[get_read_ch_client] = lambda: MOCK_READ_CLIENT
app.dependency_overrides[get_write_ch_client] = lambda: MOCK_WRITE_CLIENT


# ---------------------------------------------------------------------------
# Autouse fixture: populate _state before each test, clear after
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_lifespan():
    """Populate _state so endpoints that read it directly behave correctly."""
    scorer_state["scorer"] = MOCK_SCORER
    scorer_state["read_ch_client"] = MOCK_READ_CLIENT
    scorer_state["write_ch_client"] = MOCK_WRITE_CLIENT
    scorer_state["ch_client"] = MOCK_READ_CLIENT  # backward-compat alias
    scorer_state["model_loaded"] = True
    scorer_state["model_version"] = "IF-31-v1-test"
    scorer_state["last_batch_at"] = None
    scorer_state["anomaly_scores_table"] = "pbp_productionDB_optimized.anomaly_scores"
    scorer_state["read_fingerprint"] = ("prod-host", 8443, "pbp_productionDB_optimized", True, "ro")
    scorer_state["write_fingerprint"] = ("clickhouse", 8123, "pbp_productionDB_optimized", False, "default")
    scorer_state["write_host"] = "clickhouse"
    scorer_state["allow_nonlocal_write"] = False
    # Reset command side effects between tests
    MOCK_READ_CLIENT.command.side_effect = None
    MOCK_READ_CLIENT.command.return_value = 1
    MOCK_WRITE_CLIENT.command.side_effect = None
    MOCK_WRITE_CLIENT.command.return_value = 1
    yield
    scorer_state.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client():
    """Return a TestClient that does NOT trigger lifespan (state is pre-populated)."""
    return TestClient(app, raise_server_exceptions=True)


def _valid_score_payload():
    return {
        "user_id": 42,
        "facility_id": 7,
        "reservation_paid_out": 150.0,
        "created_at": "2026-04-28T14:30:00",
        "discount": 0.0,
        "tip": 5.0,
        "payment_method": "card",
        "category": "reservation",
        "club_credit_flag": False,
        "paid_by_manager": False,
        "currency": "USD",
    }


# ---------------------------------------------------------------------------
# Test 1: GET /api/v1/health — both clients healthy
# ---------------------------------------------------------------------------


def test_health_returns_200():
    client = _client()
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["clickhouse_connected"] is True  # read_ok AND write_ok
    assert body["model_version"] == "IF-31-v1-test"
    assert "last_batch_at" in body


# ---------------------------------------------------------------------------
# Test 1b: health is false if WRITE client probe fails
# ---------------------------------------------------------------------------


def test_health_false_when_write_client_down():
    MOCK_WRITE_CLIENT.command.side_effect = Exception("write CH down")
    client = _client()
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["clickhouse_connected"] is False


# ---------------------------------------------------------------------------
# Test 1c: health is false if READ client probe fails
# ---------------------------------------------------------------------------


def test_health_false_when_read_client_down():
    MOCK_READ_CLIENT.command.side_effect = Exception("read CH down")
    client = _client()
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["clickhouse_connected"] is False


# ---------------------------------------------------------------------------
# Test 2: GET /api/v1/model/info
# ---------------------------------------------------------------------------


def test_model_info_returns_metadata():
    client = _client()
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "IF-31-v1-test"
    assert body["feature_count"] == 31
    assert "threshold" in body
    assert body["threshold"] == 0.123
    assert "risk_levels" in body
    assert "critical" in body["risk_levels"]


# ---------------------------------------------------------------------------
# Test 3: POST /api/v1/model/reload — rebuilds scorer on the READ client
# ---------------------------------------------------------------------------


def test_model_reload_returns_success():
    from unittest.mock import patch, MagicMock as MM

    new_scorer = _make_mock_scorer()
    captured = {}

    def _capture_scorer(*args, **kwargs):
        captured["ch_connector"] = kwargs.get("ch_connector")
        return new_scorer

    with patch("scorer.routers.model.load_artifacts", return_value=MM()):
        with patch("scorer.routers.model.SingleTransactionScorer", side_effect=_capture_scorer):
            client = _client()
            response = client.post("/api/v1/model/reload")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "reloaded"
    assert "model_version" in body
    # Reload must wire the READ client into the new scorer.
    assert captured["ch_connector"] is MOCK_READ_CLIENT


# ---------------------------------------------------------------------------
# Test 4: POST /api/v1/score — valid request
# ---------------------------------------------------------------------------


def test_score_single_valid_request():
    client = _client()
    response = client.post("/api/v1/score", json=_valid_score_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["raw_score"] == 0.75
    assert body["is_anomaly"] is True
    assert body["risk_level"] == "high"
    assert body["percentile"] == 0.92
    assert body["model_version"] == "IF-31-v1-test"
    assert body["feature_version"] == "base-31"
    assert body["threshold_version"] == "v1-test"
    assert isinstance(body["factors"], list)
    assert len(body["factors"]) == 1
    assert body["factors"][0]["feature"] == "log_amount"


# ---------------------------------------------------------------------------
# Test 5: POST /api/v1/score — malformed request → 422
# ---------------------------------------------------------------------------


def test_score_single_malformed_request_422():
    client = _client()
    response = client.post("/api/v1/score", json={"user_id": 42})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 6: POST /api/v1/score/batch — valid request injects both clients
# ---------------------------------------------------------------------------


def test_score_batch_valid_request():
    from unittest.mock import patch
    from datetime import datetime, timedelta

    cursor_end = datetime(2026, 4, 28, 15, 0, 0)
    fake_result = {
        "processed": 10,
        "scored": 10,
        "critical_alerts": [
            {
                "payment_id": 999,
                "user_id": 42,
                "facility_id": 7,
                "raw_score": 0.97,
                "risk_level": "critical",
                "amount_usd": 500.0,
            }
        ],
        "next_cursor": cursor_end + timedelta(seconds=1),
    }

    captured = {}
    real_init = BatchScorer.__init__

    def _spy_init(self, *args, **kwargs):
        captured["kwargs"] = kwargs
        captured["args"] = args
        return real_init(self, *args, **kwargs)

    with patch.object(BatchScorer, "__init__", _spy_init):
        with patch.object(BatchScorer, "score_batch", return_value=fake_result):
            client = _client()
            response = client.post(
                "/api/v1/score/batch",
                json={"cursor": "2026-04-28T00:00:00"},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["processed"] == 10
    assert body["scored"] == 10
    assert len(body["critical_alerts"]) == 1
    assert body["critical_alerts"][0]["payment_id"] == 999
    assert body["critical_alerts"][0]["risk_level"] == "critical"
    assert body["next_cursor"] is not None
    assert datetime.fromisoformat(body["next_cursor"]) == cursor_end + timedelta(seconds=1)

    # The router wired both clients and guardrail metadata into BatchScorer.
    kw = captured["kwargs"]
    assert kw["read_ch_client"] is MOCK_READ_CLIENT
    assert kw["write_ch_client"] is MOCK_WRITE_CLIENT
    assert kw["anomaly_scores_table"] == "pbp_productionDB_optimized.anomaly_scores"
    assert kw["write_host"] == "clickhouse"


# ---------------------------------------------------------------------------
# Test 7: POST /api/v1/score/batch — malformed request → 422
# ---------------------------------------------------------------------------


def test_score_batch_malformed_request_422():
    client = _client()
    response = client.post("/api/v1/score/batch", json={"not_cursor": "foo"})
    assert response.status_code == 422
