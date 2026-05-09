"""Integration tests for the ML Scorer HTTP API.

Uses FastAPI TestClient with dependency_overrides + direct _state injection
so no model files are required (CI-safe).
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from scorer.main import app
from scorer.dependencies import get_scorer, get_ch_client, _state as scorer_state
from scorer.batch.scorer import BatchScorer
from fraud_detector.scoring.classifier import ScoringResult


# ---------------------------------------------------------------------------
# Shared mock objects
# ---------------------------------------------------------------------------

def _make_mock_scorer():
    mock = MagicMock()
    mock._classifier._threshold = 0.123
    mock.score.return_value = ScoringResult(
        score=0.75,
        is_anomaly=True,
        risk_level="high",
        percentile=0.92,
        factors=[
            {"feature": "log_amount", "value": 5.5, "z_score": 3.2, "direction": "high"}
        ],
    )
    return mock


def _make_mock_ch_client():
    mock = MagicMock()
    mock.command.return_value = 1  # SELECT 1
    return mock


MOCK_SCORER = _make_mock_scorer()
MOCK_CH_CLIENT = _make_mock_ch_client()

# Override DI for the whole test module
app.dependency_overrides[get_scorer] = lambda: MOCK_SCORER
app.dependency_overrides[get_ch_client] = lambda: MOCK_CH_CLIENT


# ---------------------------------------------------------------------------
# Autouse fixture: populate _state before each test, clear after
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_lifespan():
    """Populate _state so endpoints that read it directly behave correctly."""
    scorer_state["scorer"] = MOCK_SCORER
    scorer_state["ch_client"] = MOCK_CH_CLIENT
    scorer_state["model_loaded"] = True
    scorer_state["model_version"] = "IF-31-v1-test"
    scorer_state["last_batch_at"] = None
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
# Test 1: GET /api/v1/health
# ---------------------------------------------------------------------------

def test_health_returns_200():
    client = _client()
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["clickhouse_connected"] is True  # mock.command returns 1 (truthy)
    assert body["model_version"] == "IF-31-v1-test"
    assert "last_batch_at" in body


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
# Test 3: POST /api/v1/model/reload
# ---------------------------------------------------------------------------

def test_model_reload_returns_success(monkeypatch):
    """Reload patches SingleTransactionScorer constructor so no files are needed."""
    from unittest.mock import patch, MagicMock

    new_scorer = _make_mock_scorer()

    with patch("scorer.routers.model.SingleTransactionScorer", return_value=new_scorer):
        # Also patch open() so _load_version doesn't fail on missing thresholds.json
        with patch("builtins.open", MagicMock(side_effect=FileNotFoundError)):
            client = _client()
            response = client.post("/api/v1/model/reload")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "reloaded"
    assert "model_version" in body


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
    assert isinstance(body["factors"], list)
    assert len(body["factors"]) == 1
    assert body["factors"][0]["feature"] == "log_amount"


# ---------------------------------------------------------------------------
# Test 5: POST /api/v1/score — malformed request → 422
# ---------------------------------------------------------------------------

def test_score_single_malformed_request_422():
    """Missing required fields should produce a 422 Pydantic validation error."""
    client = _client()
    response = client.post("/api/v1/score", json={"user_id": 42})  # missing required fields
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 6: POST /api/v1/score/batch — valid request
# ---------------------------------------------------------------------------

def test_score_batch_valid_request(monkeypatch):
    """Mock BatchScorer.score_batch to avoid real ClickHouse calls."""
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
    # next_cursor must be present and parseable as an ISO8601 datetime
    assert body["next_cursor"] is not None
    assert datetime.fromisoformat(body["next_cursor"]) == cursor_end + timedelta(seconds=1)


# ---------------------------------------------------------------------------
# Test 7: POST /api/v1/score/batch — malformed request → 422
# ---------------------------------------------------------------------------

def test_score_batch_malformed_request_422():
    """Missing cursor field should produce a 422 Pydantic validation error."""
    client = _client()
    response = client.post("/api/v1/score/batch", json={"not_cursor": "foo"})
    assert response.status_code == 422
