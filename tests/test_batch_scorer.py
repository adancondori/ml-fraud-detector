"""Unit tests for BatchContextProvider and BatchScorer.

All ClickHouse interactions are mocked — no real connection required.

BatchScorer now uses two separate ClickHouse clients:
  - read_ch_client  -> cursor resolution, payment fetch, batch context (prod read-only)
  - write_ch_client -> anomaly_scores INSERT (local)
A guardrail blocks INSERT when the WRITE target matches the READ fingerprint
or points to a non-local host without an explicit bypass.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scorer.batch.context_provider import BatchContextProvider
from scorer.batch.scorer import BatchScorer, assert_write_target_is_safe
from fraud_detector.scoring.context import UserContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Distinct fingerprints: READ = prod-like (secure, remote), WRITE = local docker.
READ_FP = ("prod-clickhouse-host", 8443, "pbp_productionDB_optimized", True, "readonly")
WRITE_FP = ("clickhouse", 8123, "pbp_productionDB_optimized", False, "default")
WRITE_HOST = "clickhouse"
TABLE = "pbp_productionDB_optimized.anomaly_scores"


def _make_mock_ch_client(result_rows=None):
    """Return a mock ClickHouse client whose query() returns result_rows."""
    mock = MagicMock()
    query_result = MagicMock()
    query_result.result_rows = result_rows or []
    mock.query.return_value = query_result
    return mock


def _make_batch_scorer(scorer, read_client, write_client, **overrides):
    """Construct a BatchScorer with safe local defaults for tests."""
    kwargs = dict(
        anomaly_scores_table=TABLE,
        read_fingerprint=READ_FP,
        write_fingerprint=WRITE_FP,
        write_host=WRITE_HOST,
        allow_nonlocal_write=False,
    )
    kwargs.update(overrides)
    return BatchScorer(
        scorer=scorer,
        read_ch_client=read_client,
        write_ch_client=write_client,
        **kwargs,
    )


def _fetch_row(payment_id, user_id, amount, created_at):
    """Build one fetch result row in _FETCH_SQL column order."""
    return (
        payment_id,
        user_id,
        7,
        "Test Facility",
        amount,
        created_at,
        0.0,
        0.0,
        "card",
        "reservation",
        False,
        False,
        "USD",
        "paid",
        user_id,
        created_at,
        "stripe",
        "pbp_web",
    )


def _mock_scorer(risk_level="high", percentile=0.92):
    mock = MagicMock()
    mock._feature_names = [f"feature_{i}" for i in range(31)]
    mock._model_version = "IF-31-v1"
    mock._feature_version = "base-31"
    mock._threshold_version = "v1"
    mock._feature_calc.calculate.return_value = np.zeros(31)
    mock.score_features.return_value = (0.75, np.zeros((1, 31), dtype=np.float32))
    mock._classifier.classify.return_value = (True, risk_level, percentile)
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
    mock_client = _make_mock_ch_client(result_rows=[])
    provider = BatchContextProvider(mock_client)

    payments = _sample_payments(2)
    result = provider.get_batch_context(payments)

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
    """When _resolve_cursor_end returns None, score_batch returns the zero summary.

    The cursor probe must hit the READ client; the WRITE client must not insert.
    """
    mock_scorer = MagicMock()

    cursor_end_result = MagicMock()
    cursor_end_result.result_rows = [(None,)]
    mock_read = MagicMock()
    mock_read.query.return_value = cursor_end_result
    mock_write = MagicMock()

    batch_scorer = _make_batch_scorer(mock_scorer, mock_read, mock_write)
    cursor = datetime(2026, 4, 28, 0, 0, 0)
    result = batch_scorer.score_batch(cursor)

    assert result["processed"] == 0
    assert result["scored"] == 0
    assert result["critical_alerts"] == []
    assert result["next_cursor"] is None

    # Cursor resolution used READ; nothing was inserted via WRITE.
    mock_read.query.assert_called_once()
    mock_write.insert.assert_not_called()


# ---------------------------------------------------------------------------
# BatchScorer: Test 5 — score_batch with 2 payments (READ/WRITE split)
# ---------------------------------------------------------------------------


def test_score_batch_with_payments():
    """Mocked scorer scores 2 payments; verifies READ for fetch and WRITE for insert."""
    cursor = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
    cursor_end = datetime(2026, 4, 28, 15, 0, 0)

    fetch_rows = [
        _fetch_row(101, 42, 100.0, datetime(2026, 4, 28, 14, 30, 0)),
        _fetch_row(102, 43, 120.0, datetime(2026, 4, 28, 15, 0, 0)),
    ]

    # READ client: cursor_end, fetch, then 6 context queries
    mock_read = MagicMock()
    cursor_end_result = MagicMock()
    cursor_end_result.result_rows = [(cursor_end,)]
    fetch_result = MagicMock()
    fetch_result.result_rows = fetch_rows
    context_result = MagicMock()
    context_result.result_rows = []
    mock_read.query.side_effect = [cursor_end_result, fetch_result] + [context_result] * 6

    # WRITE client: only used for insert
    mock_write = MagicMock()

    mock_scorer = _mock_scorer()

    from fraud_detector.scoring.scorer import SingleTransactionScorer

    with patch.object(
        SingleTransactionScorer,
        "_explain_top_factors",
        return_value=[{"feature": "log_amount", "value": 5.5, "z_score": 3.2, "direction": "high"}],
    ):
        batch_scorer = _make_batch_scorer(mock_scorer, mock_read, mock_write)
        result = batch_scorer.score_batch(cursor)

    assert result["processed"] == 2
    assert result["scored"] == 2
    assert isinstance(result["critical_alerts"], list)
    assert result["next_cursor"] == cursor_end + timedelta(seconds=1)

    # INSERT went to the WRITE client only, never the READ client.
    mock_write.insert.assert_called_once()
    mock_read.insert.assert_not_called()

    # INSERT target table is taken from anomaly_scores_table.
    insert_args = mock_write.insert.call_args
    assert insert_args.args[0] == TABLE

    settings = insert_args.kwargs.get("settings") or {}
    assert "insert_deduplication_token" in settings
    token = settings["insert_deduplication_token"]
    assert token.startswith("batch-")
    assert cursor_end.isoformat() in token
    assert "IF-31-v1" in token
    assert "chunk-0" in token


# ---------------------------------------------------------------------------
# BatchScorer: Test 6 — critical alerts collected
# ---------------------------------------------------------------------------


def test_critical_alerts_collected():
    """Payments with risk_level='critical' appear in critical_alerts."""
    cursor = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
    cursor_end = datetime(2026, 4, 28, 14, 30, 0)
    fetch_rows = [_fetch_row(999, 42, 500.0, datetime(2026, 4, 28, 14, 30, 0))]

    mock_read = MagicMock()
    cursor_end_result = MagicMock()
    cursor_end_result.result_rows = [(cursor_end,)]
    fetch_result = MagicMock()
    fetch_result.result_rows = fetch_rows
    context_result = MagicMock()
    context_result.result_rows = []
    mock_read.query.side_effect = [cursor_end_result, fetch_result] + [context_result] * 6
    mock_write = MagicMock()

    mock_scorer = _mock_scorer(risk_level="critical", percentile=0.98)
    mock_scorer.score_features.return_value = (0.97, np.zeros((1, 31), dtype=np.float32))

    from fraud_detector.scoring.scorer import SingleTransactionScorer

    with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
        batch_scorer = _make_batch_scorer(mock_scorer, mock_read, mock_write)
        result = batch_scorer.score_batch(cursor)

    assert result["processed"] == 1
    assert result["scored"] == 1
    assert len(result["critical_alerts"]) == 1
    alert = result["critical_alerts"][0]
    assert alert["payment_id"] == 999
    assert alert["risk_level"] == "critical"
    assert alert["user_id"] == 42
    assert alert["facility_id"] == 7


# ---------------------------------------------------------------------------
# BatchScorer: Test 7 — context provider is built on the READ client
# ---------------------------------------------------------------------------


def test_context_provider_uses_read_client():
    """BatchContextProvider must receive the READ client, never the WRITE client."""
    cursor = datetime(2026, 4, 28, 0, 0, 0)
    cursor_end = datetime(2026, 4, 28, 15, 0, 0)
    fetch_rows = [_fetch_row(101, 42, 100.0, datetime(2026, 4, 28, 14, 30, 0))]

    mock_read = MagicMock()
    cursor_end_result = MagicMock()
    cursor_end_result.result_rows = [(cursor_end,)]
    fetch_result = MagicMock()
    fetch_result.result_rows = fetch_rows
    context_result = MagicMock()
    context_result.result_rows = []
    mock_read.query.side_effect = [cursor_end_result, fetch_result] + [context_result] * 6
    mock_write = MagicMock()

    mock_scorer = _mock_scorer()

    captured = {}

    real_init = BatchContextProvider.__init__

    def _spy_init(self, client, *args, **kwargs):
        captured["client"] = client
        return real_init(self, client, *args, **kwargs)

    from fraud_detector.scoring.scorer import SingleTransactionScorer

    with patch.object(BatchContextProvider, "__init__", _spy_init):
        with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
            batch_scorer = _make_batch_scorer(mock_scorer, mock_read, mock_write)
            batch_scorer.score_batch(cursor)

    assert captured["client"] is mock_read
    assert captured["client"] is not mock_write


# ---------------------------------------------------------------------------
# Guardrail tests
# ---------------------------------------------------------------------------


def test_guardrail_blocks_identical_fingerprint():
    """WRITE fingerprint identical to READ must raise before any insert."""
    with pytest.raises(ValueError, match="same ClickHouse"):
        assert_write_target_is_safe(
            read_fingerprint=READ_FP,
            write_fingerprint=READ_FP,  # identical -> dangerous
            write_host="prod-clickhouse-host",
            allow_nonlocal_write=False,
        )


def test_guardrail_blocks_nonlocal_write_host():
    """A non-local WRITE host without bypass must raise."""
    with pytest.raises(ValueError, match="non-local"):
        assert_write_target_is_safe(
            read_fingerprint=READ_FP,
            write_fingerprint=("prod-clickhouse-host", 8443, "db", True, "writer"),
            write_host="prod-clickhouse-host",
            allow_nonlocal_write=False,
        )


def test_guardrail_allows_local_write():
    """Local WRITE host with a distinct fingerprint passes."""
    # Should not raise.
    assert_write_target_is_safe(
        read_fingerprint=READ_FP,
        write_fingerprint=WRITE_FP,
        write_host=WRITE_HOST,
        allow_nonlocal_write=False,
    )


def test_guardrail_allows_nonlocal_with_explicit_bypass():
    """Non-local WRITE host is permitted only with the explicit bypass flag."""
    assert_write_target_is_safe(
        read_fingerprint=READ_FP,
        write_fingerprint=("other-host", 8443, "db", True, "writer"),
        write_host="other-host",
        allow_nonlocal_write=True,
    )


def test_fetch_sql_excludes_user_id_zero():
    """Paso 4b: el universo del batch debe coincidir con el del loader.

    El loader de entrenamiento excluye `user_id != 0` (loader.py); sin el
    mismo filtro, el batch puntuaría ~47K pagos/año con historial de usuario
    vacío — una superficie distinta a la del modelo.
    """
    from scorer.batch.scorer import _FETCH_SQL

    assert "user_id != 0" in _FETCH_SQL


def test_cursor_end_sql_excludes_user_id_zero():
    """Paso 4b: el cursor no debe avanzar apoyado en pagos fuera del universo.

    Si `_CURSOR_END_SQL` resuelve cursor_end con un pago user_id=0 más
    reciente que el último pago del universo, ese tramo queda marcado como
    procesado sin haberse puntuado.
    """
    from scorer.batch.scorer import _CURSOR_END_SQL

    assert "user_id != 0" in _CURSOR_END_SQL


def test_fetch_sql_aliases_real_payment_id_column():
    """The prod `payments` PK column is `id`; the fetch SQL must alias it.

    Regression: selecting a bare `payment_id` column raises
    `Code: 47 Unknown expression identifier payment_id` against prod.
    """
    from scorer.batch.scorer import _FETCH_SQL

    assert "id AS payment_id" in _FETCH_SQL
    # Must not select a bare payment_id column from payments.
    assert "\n    payment_id,\n" not in _FETCH_SQL


def test_fetch_sql_and_insert_carry_facility_name():
    """facility_name is fetched from payments and carried into anomaly_scores.

    This lets the dashboard fall back to the score-time facility name when the
    local MySQL facilities table has no matching record.
    """
    from scorer.batch.scorer import _FETCH_SQL, _INSERT_COLUMNS

    assert "facility_name" in _FETCH_SQL
    assert "facility_name" in _INSERT_COLUMNS
    # Carried right after facility_id for a stable, readable column order.
    assert _INSERT_COLUMNS.index("facility_name") == _INSERT_COLUMNS.index("facility_id") + 1


def test_score_batch_aborts_when_write_equals_read():
    """End-to-end: identical READ/WRITE targets abort before inserting."""
    cursor = datetime(2026, 4, 28, 0, 0, 0)
    cursor_end = datetime(2026, 4, 28, 15, 0, 0)
    fetch_rows = [_fetch_row(101, 42, 100.0, datetime(2026, 4, 28, 14, 30, 0))]

    mock_read = MagicMock()
    cursor_end_result = MagicMock()
    cursor_end_result.result_rows = [(cursor_end,)]
    fetch_result = MagicMock()
    fetch_result.result_rows = fetch_rows
    context_result = MagicMock()
    context_result.result_rows = []
    mock_read.query.side_effect = [cursor_end_result, fetch_result] + [context_result] * 6
    mock_write = MagicMock()

    mock_scorer = _mock_scorer()

    from fraud_detector.scoring.scorer import SingleTransactionScorer

    with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[]):
        batch_scorer = _make_batch_scorer(
            mock_scorer,
            mock_read,
            mock_write,
            read_fingerprint=READ_FP,
            write_fingerprint=READ_FP,  # identical -> must abort
            write_host="prod-clickhouse-host",
        )
        with pytest.raises(ValueError):
            batch_scorer.score_batch(cursor)

    # Nothing was inserted.
    mock_write.insert.assert_not_called()
