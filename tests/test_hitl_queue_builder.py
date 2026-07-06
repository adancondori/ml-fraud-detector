"""Tests for hitl_queue_builder.py.

All tests use a mocked ClickHouse client — no live ClickHouse connection
required.  The mock distinguishes query types by inspecting the SQL text
(p50 query vs top-k vs below-p50), following the same pattern established
in test_batch_scorer.py and test_artifact_loader.py.

Synthetic data: 3 top-k rows (percentile 0.95 / 0.92 / 0.88) and 1 below-p50
row (percentile 0.30).  p50 resolves to 0.50.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

# Ensure scripts/ is importable (mirrors shadow_gate/monitor test pattern)
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from hitl_queue_builder import (  # noqa: E402
    build_hitl_queue,
    compute_counts,
    resolve_p50,
    write_output,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic ClickHouse mock
# ---------------------------------------------------------------------------

# Column order returned by the mock must match _SELECT_COLS in the builder.
_COL_NAMES = [
    "payment_id",
    "facility_id",
    "facility_name",
    "user_id",
    "scored_at",
    "payment_created_at",
    "amount_usd",
    "raw_score",
    "percentile",
    "risk_level",
    "is_anomaly",
    "model_version",
    "top_factors",
]


def _make_row(payment_id: str, percentile: float, top_factors: str) -> list:
    return [
        payment_id,  # payment_id
        "facility_1",  # facility_id
        "Test Venue",  # facility_name
        "user_1",  # user_id
        "2026-07-01 10:00:00",  # scored_at
        "2026-07-01 09:55:00",  # payment_created_at
        42.0,  # amount_usd
        -0.1,  # raw_score
        percentile,  # percentile
        "high",  # risk_level
        True,  # is_anomaly
        "frame-v1-1.0",  # model_version
        top_factors,  # top_factors
    ]


_TOP_FACTORS_JSON = json.dumps(
    [{"feature": "amount_usd", "value": 200.0, "z_score": 3.1, "direction": "high"}]
)

# Three top-k rows (percentile 0.95, 0.92, 0.88) and one below-p50 (0.30)
_TOP_ROWS = [
    _make_row("pay_001", 0.95, _TOP_FACTORS_JSON),
    _make_row("pay_002", 0.92, _TOP_FACTORS_JSON),
    _make_row("pay_003", 0.88, _TOP_FACTORS_JSON),
]
_BELOW_ROWS = [
    _make_row("pay_004", 0.30, "[]"),
]


class _MockQueryResult:
    """Minimal stub that exposes .result_rows and .column_names."""

    def __init__(self, rows, columns):
        self.result_rows = rows
        self.column_names = columns


def _make_mock_client(p50_empty: bool = False) -> MagicMock:
    """Return a MagicMock ch_client whose .query() dispatches by SQL content."""

    def _query_side_effect(sql: str):
        sql_lower = sql.lower()
        if "quantile" in sql_lower:
            # p50 query
            if p50_empty:
                return _MockQueryResult([], ["p50"])
            return _MockQueryResult([[0.50]], ["p50"])
        elif "order by percentile desc" in sql_lower:
            # top-k query
            return _MockQueryResult(_TOP_ROWS, _COL_NAMES)
        elif "order by rand()" in sql_lower:
            # below-p50 query
            return _MockQueryResult(_BELOW_ROWS, _COL_NAMES)
        else:
            return _MockQueryResult([], _COL_NAMES)

    client = MagicMock()
    client.query.side_effect = _query_side_effect
    return client


# ---------------------------------------------------------------------------
# Tests 1-2: compute_counts (pure function, no client needed)
# ---------------------------------------------------------------------------


def test_compute_counts_absolute():
    """Absolute mode (no capacity): top_k=100, pct=0.20 => (100, 20)."""
    top_k_count, below_k_count = compute_counts(100, 0.20, capacity=None)
    assert top_k_count == 100
    assert below_k_count == 20


def test_compute_counts_capacity():
    """Capacity mode: capacity=10, pct=0.20 => floor(8), 2."""
    top_k_count, below_k_count = compute_counts(100, 0.20, capacity=10)
    assert top_k_count == 8
    assert below_k_count == 2


def test_compute_counts_capacity_minimum_below():
    """Capacity=1, pct=0.20: top=floor(0.8)=0, below=1 (remainder).

    When capacity is small, the remainder logic still allocates 1 row to
    below-p50 (capacity - top_k_count = 1 - 0 = 1).
    """
    top_k_count, below_k_count = compute_counts(100, 0.20, capacity=1)
    # floor(1 * 0.80) = 0; below = 1 - 0 = 1
    assert top_k_count == 0
    assert below_k_count == 1
    # Sum must equal capacity
    assert top_k_count + below_k_count == 1


# ---------------------------------------------------------------------------
# Tests 3-8: build_hitl_queue (mocked client)
# ---------------------------------------------------------------------------


def test_build_queue_filters_shadow_new():
    """All SQL queries must filter on scoring_mode = 'shadow_new'."""
    client = _make_mock_client()
    build_hitl_queue(client, top_k=3, below_p50_pct=0.20)

    for call in client.query.call_args_list:
        sql = call.args[0]
        assert "scoring_mode = 'shadow_new'" in sql, f"SQL missing shadow_new filter: {sql}"


def test_build_queue_top_k_ordered():
    """Top-k query must use ORDER BY percentile DESC and LIMIT."""
    client = _make_mock_client()
    build_hitl_queue(client, top_k=3, below_p50_pct=0.20)

    top_k_sql = next(
        call.args[0]
        for call in client.query.call_args_list
        if "order by percentile desc" in call.args[0].lower()
    )
    assert "ORDER BY percentile DESC" in top_k_sql
    assert "LIMIT 3" in top_k_sql


def test_build_queue_below_uses_p50():
    """Below-p50 query must filter percentile < p50 and use ORDER BY rand()."""
    client = _make_mock_client()
    build_hitl_queue(client, top_k=3, below_p50_pct=0.20)

    below_sql = next(
        call.args[0]
        for call in client.query.call_args_list
        if "order by rand()" in call.args[0].lower()
    )
    assert "percentile < 0.5" in below_sql
    assert "ORDER BY rand()" in below_sql


def test_build_queue_marks_source():
    """Output DataFrame must have hitl_queue_source with correct values/counts."""
    client = _make_mock_client()
    df = build_hitl_queue(client, top_k=3, below_p50_pct=0.20)

    assert "hitl_queue_source" in df.columns
    sources = set(df["hitl_queue_source"].unique())
    assert "top_k" in sources
    assert "below_p50" in sources

    n_top = (df["hitl_queue_source"] == "top_k").sum()
    n_below = (df["hitl_queue_source"] == "below_p50").sum()
    assert n_top == 3, f"Expected 3 top_k rows, got {n_top}"
    assert n_below == 1, f"Expected 1 below_p50 row, got {n_below}"


def test_build_queue_includes_top_factors():
    """Result DataFrame must include the top_factors column."""
    client = _make_mock_client()
    df = build_hitl_queue(client, top_k=3, below_p50_pct=0.20)

    assert "top_factors" in df.columns
    top_k_rows = df[df["hitl_queue_source"] == "top_k"]
    assert top_k_rows["top_factors"].notna().all()


def test_build_queue_capacity_mode():
    """With capacity=5, pct=0.20: top_k_count=4, below=1."""
    client = _make_mock_client()
    build_hitl_queue(client, top_k=100, below_p50_pct=0.20, capacity=5)

    # Verify the LIMIT in the top-k SQL is 4 (floor(5*0.80))
    top_k_sql = next(
        call.args[0]
        for call in client.query.call_args_list
        if "order by percentile desc" in call.args[0].lower()
    )
    assert "LIMIT 4" in top_k_sql

    below_sql = next(
        call.args[0]
        for call in client.query.call_args_list
        if "order by rand()" in call.args[0].lower()
    )
    assert "LIMIT 1" in below_sql


# ---------------------------------------------------------------------------
# Test 9: resolve_p50 fallback
# ---------------------------------------------------------------------------


def test_resolve_p50_fallback():
    """When p50 query returns no rows, resolve_p50 falls back to 0.5."""
    client = _make_mock_client(p50_empty=True)
    p50 = resolve_p50(client)
    assert p50 == 0.5


# ---------------------------------------------------------------------------
# Test 10: write_output CSV and JSON
# ---------------------------------------------------------------------------


def test_write_output_csv_and_json(tmp_path):
    """write_output produces readable .csv and .json files with expected columns."""
    client = _make_mock_client()
    df = build_hitl_queue(client, top_k=3, below_p50_pct=0.20)

    # CSV
    csv_path = str(tmp_path / "queue.csv")
    write_output(df, csv_path)
    df_read_csv = pd.read_csv(csv_path)
    assert "payment_id" in df_read_csv.columns
    assert "top_factors" in df_read_csv.columns
    assert "hitl_queue_source" in df_read_csv.columns
    assert len(df_read_csv) == len(df)

    # JSON
    json_path = str(tmp_path / "queue.json")
    write_output(df, json_path)
    df_read_json = pd.read_json(json_path, orient="records")
    assert "payment_id" in df_read_json.columns
    assert len(df_read_json) == len(df)


def test_write_output_creates_parent_dir(tmp_path):
    """write_output creates missing parent directories."""
    client = _make_mock_client()
    df = build_hitl_queue(client, top_k=3, below_p50_pct=0.20)

    deep_path = str(tmp_path / "nested" / "dir" / "queue.csv")
    write_output(df, deep_path)
    assert Path(deep_path).exists()
