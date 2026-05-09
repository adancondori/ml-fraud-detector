"""Unit tests for BatchContextProvider and BatchScorer.

All ClickHouse interactions are mocked — no real connection required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from scorer.batch.context_provider import BatchContextProvider
from scorer.batch.scorer import BatchScorer
from fraud_detector.scoring.context import UserContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_ch_client(result_rows=None):
    """Return a mock ClickHouse client whose query() returns result_rows."""
    mock = MagicMock()
    query_result = MagicMock()
    query_result.result_rows = result_rows or []
    mock.query.return_value = query_result
    return mock


def _sample_payments(n=2):
    """Return a list of minimal payment dicts for testing."""
    return [
        {
            "payment_id": 100 + i,
            "user_id": 42 + i,
            "facility_id": 7,
            "reservation_paid_out": 100.0 + i * 10,
            "created_at": datetime(2026, 4, 28, 14, 30, 0),
            "discount": 0.0,
            "tip": 0.0,
            "payment_method": "card",
            "category": "reservation",
            "club_credit_flag": False,
            "paid_by_manager": False,
            "currency": "USD",
            "status": "paid",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# BatchContextProvider: Test 1 — get_batch_context returns dict
# ---------------------------------------------------------------------------

def test_get_batch_context_returns_dict():
    """get_batch_context returns a dict keyed by (user_id, facility_id)."""
    # Mock client that returns empty result for all 6 queries
    mock_client = _make_mock_ch_client(result_rows=[])
    provider = BatchContextProvider(mock_client)

    payments = _sample_payments(2)
    result = provider.get_batch_context(payments)

    # Should return a dict with one entry per unique (user_id, facility_id) pair
    assert isinstance(result, dict)
    assert len(result) == 2
    for key, ctx in result.items():
        uid, fid = key
        assert isinstance(uid, int)
        assert isinstance(fid, int)
        assert isinstance(ctx, UserContext)


# ---------------------------------------------------------------------------
# BatchContextProvider: Test 2 — VALUES clause construction
# ---------------------------------------------------------------------------

def test_values_clause_construction():
    """_build_values_str produces correctly formatted tuples."""
    chunk = [
        ((42, 7), datetime(2026, 4, 28, 14, 30, 0)),
        ((99, 3), datetime(2026, 3, 15, 9, 0, 0)),
    ]
    values_str = BatchContextProvider._build_values_str(chunk)
    assert "(42, 7, '2026-04-28 14:30:00')" in values_str
    assert "(99, 3, '2026-03-15 09:00:00')" in values_str


# ---------------------------------------------------------------------------
# BatchContextProvider: Test 3 — empty batch returns empty dict
# ---------------------------------------------------------------------------

def test_empty_batch_returns_empty_dict():
    """Empty payments list returns {} without executing any queries."""
    mock_client = _make_mock_ch_client()
    provider = BatchContextProvider(mock_client)

    result = provider.get_batch_context([])

    assert result == {}
    mock_client.query.assert_not_called()


# ---------------------------------------------------------------------------
# BatchScorer: Test 4 — score_batch with 0 payments
# ---------------------------------------------------------------------------

def test_score_batch_empty_cursor():
    """When fetch returns 0 payments, score_batch returns the zero summary."""
    mock_scorer = MagicMock()
    mock_ch = _make_mock_ch_client(result_rows=[])

    batch_scorer = BatchScorer(scorer=mock_scorer, ch_client=mock_ch)
    cursor = datetime(2026, 4, 28, 0, 0, 0)
    result = batch_scorer.score_batch(cursor)

    assert result["processed"] == 0
    assert result["scored"] == 0
    assert result["critical_alerts"] == []


# ---------------------------------------------------------------------------
# BatchScorer: Test 5 — score_batch with 2 payments
# ---------------------------------------------------------------------------

def test_score_batch_with_payments():
    """Mocked scorer scores 2 payments and inserts once with dedup token."""
    # Build the payment rows the fetch query would return
    cursor = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
    fetch_rows = [
        (101, 42, 7, 100.0, datetime(2026, 4, 28, 14, 30, 0), 0.0, 0.0, "card", "reservation", False, False, "USD", "paid"),
        (102, 43, 7, 120.0, datetime(2026, 4, 28, 15, 0, 0), 0.0, 0.0, "card", "reservation", False, False, "USD", "paid"),
    ]

    mock_ch = MagicMock()
    # First query is _fetch_payments, rest are context queries (6), last is insert
    fetch_result = MagicMock()
    fetch_result.result_rows = fetch_rows
    context_result = MagicMock()
    context_result.result_rows = []
    mock_ch.query.side_effect = [fetch_result] + [context_result] * 6

    # Mock scorer internals
    mock_scorer = MagicMock()
    mock_scorer._feature_calc.calculate.return_value = np.zeros(31)
    mock_scorer._scaler.scaler.transform.return_value = np.zeros((1, 31), dtype=np.float32)
    mock_scorer._model.score_samples.return_value = np.array([-0.75])
    mock_scorer._classifier.classify.return_value = (True, "high", 0.92)
    mock_scorer._model._version = "if-31-v1"

    with patch.object(
        BatchScorer, "_explain_top_factors" if hasattr(BatchScorer, "_explain_top_factors") else "_score_all",
        wraps=None,
    ):
        pass  # We'll let _score_all run but patch SingleTransactionScorer._explain_top_factors

    from fraud_detector.scoring.scorer import SingleTransactionScorer
    with patch.object(
        SingleTransactionScorer,
        "_explain_top_factors",
        return_value=[{"feature": "log_amount", "value": 5.5, "z_score": 3.2, "direction": "high"}],
    ):
        batch_scorer = BatchScorer(scorer=mock_scorer, ch_client=mock_ch)
        result = batch_scorer.score_batch(cursor)

    assert result["processed"] == 2
    assert result["scored"] == 2
    assert isinstance(result["critical_alerts"], list)

    # Verify INSERT was called once with insert_deduplication_token in settings
    mock_ch.insert.assert_called_once()
    call_kwargs = mock_ch.insert.call_args
    assert "settings" in call_kwargs.kwargs or len(call_kwargs.args) >= 4
    # Check the settings contain the dedup token
    settings = call_kwargs.kwargs.get("settings") or {}
    assert "insert_deduplication_token" in settings
    token = settings["insert_deduplication_token"]
    assert token.startswith("batch-")
    assert "chunk-0" in token


# ---------------------------------------------------------------------------
# BatchScorer: Test 6 — critical alerts collected
# ---------------------------------------------------------------------------

def test_critical_alerts_collected():
    """Payments with risk_level='critical' appear in critical_alerts."""
    cursor = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
    fetch_rows = [
        (999, 42, 7, 500.0, datetime(2026, 4, 28, 14, 30, 0), 0.0, 0.0, "card", "reservation", False, False, "USD", "paid"),
    ]

    mock_ch = MagicMock()
    fetch_result = MagicMock()
    fetch_result.result_rows = fetch_rows
    context_result = MagicMock()
    context_result.result_rows = []
    mock_ch.query.side_effect = [fetch_result] + [context_result] * 6

    mock_scorer = MagicMock()
    mock_scorer._feature_calc.calculate.return_value = np.zeros(31)
    mock_scorer._scaler.scaler.transform.return_value = np.zeros((1, 31), dtype=np.float32)
    mock_scorer._model.score_samples.return_value = np.array([-0.97])
    # Classifier returns critical risk level
    mock_scorer._classifier.classify.return_value = (True, "critical", 0.98)
    mock_scorer._model._version = "if-31-v1"

    from fraud_detector.scoring.scorer import SingleTransactionScorer
    with patch.object(
        SingleTransactionScorer,
        "_explain_top_factors",
        return_value=[],
    ):
        batch_scorer = BatchScorer(scorer=mock_scorer, ch_client=mock_ch)
        result = batch_scorer.score_batch(cursor)

    assert result["processed"] == 1
    assert result["scored"] == 1
    assert len(result["critical_alerts"]) == 1
    alert = result["critical_alerts"][0]
    assert alert["payment_id"] == 999
    assert alert["risk_level"] == "critical"
    assert alert["user_id"] == 42
    assert alert["facility_id"] == 7
