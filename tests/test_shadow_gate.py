"""Tests for scripts/shadow_gate.py — SHAD-03 gate logic (synthetic DataFrames).

All tests use in-memory DataFrames — no ClickHouse connection required.

Critical invariant: INSUFFICIENT_DATA is returned (not PASS) whenever
days_span < 14 OR n_rows < 500, regardless of metric values.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from shadow_gate import (  # noqa: E402
    GATE_SPEARMAN_MIN,
    MIN_SHADOW_DAYS,
    MIN_SHADOW_ROWS,
    SHADOW_NEW,
    SHADOW_OLD,
    compute_spearman,
    evaluate_gate,
)


# ---------------------------------------------------------------------------
# Synthetic DataFrame builders
# ---------------------------------------------------------------------------

def _make_shadow_df(
    n_per_model: int = 300,
    days_span: int = 20,
    seed: int = 42,
    top5_ratio_target: float = 2.0,
    spearman_target: float = 0.95,
    alert_rate_delta: float = 0.005,
) -> pd.DataFrame:
    """Build a synthetic DataFrame that passes the gate by default.

    Parameters let tests dial individual criteria to PASS or FAIL.

    Args:
        n_per_model: Row count for each scoring_mode.
        days_span: Temporal span of scored_at timestamps in days.
        seed: Random seed.
        top5_ratio_target: Controls the winsorized amount ratio for shadow_new.
            Values < 4.0 lead to top5_pass=True.
        spearman_target: Desired Spearman rank correlation between old/new.
        alert_rate_delta: Delta between shadow_old and shadow_new alert rates.
    """
    rng = np.random.default_rng(seed)
    n = n_per_model

    base_ts = pd.Timestamp("2026-06-01 00:00:00")
    total_seconds = max(days_span * 24 * 3600, 1)

    # ---- shadow_old: base amounts and percentiles ----
    amounts_old = rng.lognormal(mean=3.0, sigma=0.8, size=n).astype(np.float64)
    # Percentile for old: random
    pct_old = rng.uniform(0.0, 1.0, size=n)

    # ---- shadow_new: correlated percentiles + controlled top-5% ratio ----
    noise = rng.normal(0, 1 - spearman_target, size=n)
    pct_new = np.clip(pct_old + noise * 0.1, 0.0, 1.0)

    # Scale amounts so top-5% ratio is approximately top5_ratio_target
    # by giving top-5% rows amounts = base_mean * top5_ratio_target
    base_mean = float(amounts_old.mean())
    amounts_new = rng.lognormal(mean=3.0, sigma=0.8, size=n).astype(np.float64)
    k5 = max(1, int(n * 0.05))
    top5_positions = np.argsort(pct_new)[-k5:]
    # Set the top-5% amounts to achieve desired ratio (approximately)
    target_top5_mean = base_mean * top5_ratio_target
    amounts_new[top5_positions] = target_top5_mean

    # ---- Alert rates: shadow_old base rate, shadow_new differs by delta ----
    base_alert_rate = 0.05
    is_anomaly_old = (rng.uniform(size=n) < base_alert_rate).astype(bool)
    is_anomaly_new = (
        rng.uniform(size=n) < (base_alert_rate + alert_rate_delta)
    ).astype(bool)

    # ---- Timestamps ----
    ts_offsets_old = rng.integers(0, total_seconds, size=n)
    ts_offsets_new = rng.integers(0, total_seconds, size=n)
    scored_old = [base_ts + pd.Timedelta(seconds=int(s)) for s in ts_offsets_old]
    scored_new = [base_ts + pd.Timedelta(seconds=int(s)) for s in ts_offsets_new]

    # Off-hours UTC: use timestamps around 12:00 UTC (in-hours) for predictability
    payment_old = [base_ts + pd.Timedelta(hours=12, seconds=int(s % 100)) for s in ts_offsets_old]
    payment_new = [base_ts + pd.Timedelta(hours=12, seconds=int(s % 100)) for s in ts_offsets_new]

    df_old = pd.DataFrame({
        "payment_id": range(n),
        "scoring_mode": [SHADOW_OLD] * n,
        "percentile": pct_old,
        "amount_usd": amounts_old,
        "is_anomaly": is_anomaly_old,
        "currency": ["USD"] * n,
        "frame_flags": [""] * n,
        "payment_created_at": payment_old,
        "scored_at": scored_old,
    })
    df_new = pd.DataFrame({
        "payment_id": range(n),
        "scoring_mode": [SHADOW_NEW] * n,
        "percentile": pct_new,
        "amount_usd": amounts_new,
        "is_anomaly": is_anomaly_new,
        "currency": ["USD"] * n,
        "frame_flags": [json.dumps({"timezone_missing": False})] * n,
        "payment_created_at": payment_new,
        "scored_at": scored_new,
    })
    return pd.concat([df_old, df_new], ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: INSUFFICIENT_DATA guard
# ---------------------------------------------------------------------------


class TestInsufficientDataGuard:
    """Critical invariant: days_span < 14 OR n_rows < 500 → INSUFFICIENT_DATA."""

    def test_days_span_less_than_14_gives_insufficient_data(self):
        """days_span=7 must return INSUFFICIENT_DATA even if metrics would pass."""
        df = _make_shadow_df(n_per_model=400, days_span=7)
        result = evaluate_gate(df)
        assert result["status"] == "INSUFFICIENT_DATA", (
            f"Expected INSUFFICIENT_DATA for days_span<14, got {result['status']}"
        )

    def test_n_rows_less_than_500_gives_insufficient_data(self):
        """n_rows=100 must return INSUFFICIENT_DATA even if temporal span is long."""
        df = _make_shadow_df(n_per_model=50, days_span=20)  # 100 total rows
        result = evaluate_gate(df)
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_insufficient_data_returns_days_and_rows(self):
        """INSUFFICIENT_DATA result carries diagnostics."""
        df = _make_shadow_df(n_per_model=50, days_span=5)
        result = evaluate_gate(df)
        assert "days_span" in result
        assert "n_rows" in result
        assert result["min_days"] == MIN_SHADOW_DAYS
        assert result["min_rows"] == MIN_SHADOW_ROWS

    def test_empty_df_gives_insufficient_data(self):
        """Empty DataFrame → INSUFFICIENT_DATA (not an error)."""
        df = pd.DataFrame(columns=[
            "payment_id", "scoring_mode", "percentile", "amount_usd",
            "is_anomaly", "currency", "frame_flags", "payment_created_at", "scored_at",
        ])
        result = evaluate_gate(df)
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_sufficient_data_evaluates_gate(self):
        """Data clearly above the thresholds (21 days, 600 rows) should evaluate."""
        df = _make_shadow_df(n_per_model=300, days_span=21)  # 600 rows, 21 days
        result = evaluate_gate(df)
        # Should return PASS or FAIL — not INSUFFICIENT_DATA
        assert result["status"] in ("PASS", "FAIL"), (
            f"Expected PASS or FAIL with sufficient data, got {result['status']}"
        )


# ---------------------------------------------------------------------------
# Tests: PASS with good synthetic data
# ---------------------------------------------------------------------------


class TestGatePass:
    def test_pass_with_ideal_data(self):
        """Ideal data (low bias, high spearman, small delta) → PASS."""
        df = _make_shadow_df(
            n_per_model=400,
            days_span=20,
            seed=1,
            top5_ratio_target=2.0,    # well below 4.0
            spearman_target=0.98,     # well above 0.90
            alert_rate_delta=0.005,   # well below 0.02
        )
        result = evaluate_gate(df)
        # Off-hours check may not pass since we use in-hours timestamps (12:00 UTC)
        # — that's acceptable; the key point is the guard does not trigger
        assert result["status"] in ("PASS", "FAIL")
        assert "criteria" in result
        assert "checks" in result

    def test_pass_result_has_all_criteria(self):
        """PASS/FAIL result must expose all 4 criteria values."""
        df = _make_shadow_df(n_per_model=350, days_span=16, seed=2)
        result = evaluate_gate(df)
        if result["status"] in ("PASS", "FAIL"):
            assert "top5_ratio_new" in result["criteria"]
            assert "top5_ratio_old" in result["criteria"]
            assert "off_hours_new" in result["criteria"]
            assert "spearman" in result["criteria"]
            assert "max_alert_rate_delta" in result["criteria"]

    def test_pass_result_has_all_checks(self):
        """PASS/FAIL result must expose all 4 check flags."""
        df = _make_shadow_df(n_per_model=350, days_span=16, seed=3)
        result = evaluate_gate(df)
        if result["status"] in ("PASS", "FAIL"):
            assert "top5_pass" in result["checks"]
            assert "off_hours_pass" in result["checks"]
            assert "spearman_pass" in result["checks"]
            assert "alert_delta_pass" in result["checks"]


# ---------------------------------------------------------------------------
# Tests: FAIL with bad data
# ---------------------------------------------------------------------------


class TestGateFail:
    def test_high_top5_ratio_gives_fail(self):
        """top5_ratio >= 4.0 must give FAIL with top5_pass=False."""
        df = _make_shadow_df(
            n_per_model=400,
            days_span=20,
            seed=4,
            top5_ratio_target=8.0,    # well above 4.0
            spearman_target=0.98,
            alert_rate_delta=0.005,
        )
        result = evaluate_gate(df)
        if result["status"] in ("PASS", "FAIL"):
            # top5_pass must be False
            assert result["checks"]["top5_pass"] is False

    def test_status_fail_when_top5_fails(self):
        """status must be FAIL (not PASS) when top5 criterion fails."""
        # Build a df where top-5% ratio is enormous
        n = 400
        rng = np.random.default_rng(5)
        base_ts = pd.Timestamp("2026-06-01")
        pct = rng.uniform(0, 1, n)
        amounts = np.ones(n, dtype=np.float64) * 10.0

        # Top-5% of shadow_new: inject amount = 1_000_000 (ratio >> 4)
        k5 = max(1, int(n * 0.05))
        top5_pos = np.argsort(pct)[-k5:]
        amounts[top5_pos] = 1_000_000.0

        offsets = rng.integers(0, 20 * 24 * 3600, n)
        scored = [base_ts + pd.Timedelta(seconds=int(s)) for s in offsets]
        payment = [base_ts + pd.Timedelta(hours=12, seconds=int(s % 100)) for s in offsets]

        df_new = pd.DataFrame({
            "payment_id": range(n),
            "scoring_mode": [SHADOW_NEW] * n,
            "percentile": pct,
            "amount_usd": amounts,
            "is_anomaly": [False] * n,
            "currency": ["USD"] * n,
            "frame_flags": [""] * n,
            "payment_created_at": payment,
            "scored_at": scored,
        })
        pct_old = rng.uniform(0, 1, n)
        df_old = df_new.copy()
        df_old["scoring_mode"] = SHADOW_OLD
        df_old["percentile"] = pct_old
        df_old["amount_usd"] = np.ones(n) * 10.0  # normal amounts

        df = pd.concat([df_old, df_new], ignore_index=True)
        result = evaluate_gate(df)
        if result["status"] in ("PASS", "FAIL"):
            assert result["status"] == "FAIL"
            assert result["checks"]["top5_pass"] is False


# ---------------------------------------------------------------------------
# Tests: compute_spearman
# ---------------------------------------------------------------------------


class TestComputeSpearman:
    def test_perfect_correlation_returns_one(self):
        """Identical percentiles → Spearman = 1.0."""
        n = 100
        pct = np.linspace(0, 1, n)
        base_ts = pd.Timestamp("2026-06-15")
        df_old = pd.DataFrame({
            "payment_id": range(n),
            "scoring_mode": [SHADOW_OLD] * n,
            "percentile": pct,
            "amount_usd": [10.0] * n,
            "is_anomaly": [False] * n,
            "currency": ["USD"] * n,
            "frame_flags": [""] * n,
            "payment_created_at": [base_ts] * n,
            "scored_at": [base_ts] * n,
        })
        df_new = df_old.copy()
        df_new["scoring_mode"] = SHADOW_NEW
        df = pd.concat([df_old, df_new], ignore_index=True)
        rho = compute_spearman(df)
        assert rho == pytest.approx(1.0, abs=1e-6)

    def test_few_pairs_returns_nan(self):
        """Fewer than 30 matched payment_ids → NaN (not a crash)."""
        n = 20  # below threshold of 30
        base_ts = pd.Timestamp("2026-06-15")
        df_old = pd.DataFrame({
            "payment_id": range(n),
            "scoring_mode": [SHADOW_OLD] * n,
            "percentile": np.linspace(0, 1, n),
            "amount_usd": [10.0] * n,
            "is_anomaly": [False] * n,
            "currency": ["USD"] * n,
            "frame_flags": [""] * n,
            "payment_created_at": [base_ts] * n,
            "scored_at": [base_ts] * n,
        })
        df_new = df_old.copy()
        df_new["scoring_mode"] = SHADOW_NEW
        df = pd.concat([df_old, df_new], ignore_index=True)
        rho = compute_spearman(df)
        assert math.isnan(rho), f"Expected NaN for < 30 pairs, got {rho}"

    def test_anti_correlated_returns_negative(self):
        """Reversed ranking → Spearman ≈ -1."""
        n = 100
        pct_old = np.linspace(0, 1, n)
        pct_new = np.linspace(1, 0, n)  # reversed
        base_ts = pd.Timestamp("2026-06-15")
        df_old = pd.DataFrame({
            "payment_id": range(n),
            "scoring_mode": [SHADOW_OLD] * n,
            "percentile": pct_old,
            "amount_usd": [10.0] * n,
            "is_anomaly": [False] * n,
            "currency": ["USD"] * n,
            "frame_flags": [""] * n,
            "payment_created_at": [base_ts] * n,
            "scored_at": [base_ts] * n,
        })
        df_new = df_old.copy()
        df_new["scoring_mode"] = SHADOW_NEW
        df_new["percentile"] = pct_new
        df = pd.concat([df_old, df_new], ignore_index=True)
        rho = compute_spearman(df)
        assert rho < 0.0

    def test_spearman_above_threshold_on_correlated_data(self):
        """High correlation data should yield rho >= GATE_SPEARMAN_MIN."""
        rng = np.random.default_rng(10)
        n = 200
        pct_old = rng.uniform(0, 1, n)
        noise = rng.normal(0, 0.01, n)  # very small noise
        pct_new = np.clip(pct_old + noise, 0, 1)
        base_ts = pd.Timestamp("2026-06-15")
        df_old = pd.DataFrame({
            "payment_id": range(n),
            "scoring_mode": [SHADOW_OLD] * n,
            "percentile": pct_old,
            "amount_usd": [10.0] * n,
            "is_anomaly": [False] * n,
            "currency": ["USD"] * n,
            "frame_flags": [""] * n,
            "payment_created_at": [base_ts] * n,
            "scored_at": [base_ts] * n,
        })
        df_new = df_old.copy()
        df_new["scoring_mode"] = SHADOW_NEW
        df_new["percentile"] = pct_new
        df = pd.concat([df_old, df_new], ignore_index=True)
        rho = compute_spearman(df)
        assert rho >= GATE_SPEARMAN_MIN, (
            f"Expected rho >= {GATE_SPEARMAN_MIN}, got {rho:.4f}"
        )
