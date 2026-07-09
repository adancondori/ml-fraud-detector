"""Tests for scripts/shadow_monitor.py — SHAD-02 metrics (synthetic DataFrames).

No ClickHouse connection required.  All tests use in-memory DataFrames.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import helpers — add scripts/ to path so the module is importable without
# installing it as a package.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from shadow_monitor import (  # noqa: E402
    SHADOW_NEW,
    SHADOW_OLD,
    compute_alert_rate_by_segment,
    compute_jaccard_at_k,
    compute_off_hours,
    compute_top5_bias,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(
    n_old: int = 200,
    n_new: int = 200,
    seed: int = 42,
    heavy_tail: bool = False,
) -> pd.DataFrame:
    """Build a synthetic shadow DataFrame with both models."""
    rng = np.random.default_rng(seed)
    total = n_old + n_new
    modes = [SHADOW_OLD] * n_old + [SHADOW_NEW] * n_new

    # Scores: random percentiles in [0, 1]
    percentiles = rng.uniform(0, 1, total)

    # Amounts: log-normal base; optionally inject a heavy tail
    amounts = rng.lognormal(mean=3.0, sigma=1.0, size=total).astype(np.float64)
    if heavy_tail:
        # Two extreme outliers at the end of shadow_new (positions n_old and n_old+1)
        amounts[n_old] = 1_000_000.0
        amounts[n_old + 1] = 500_000.0

    is_anomaly = (percentiles > 0.95).astype(bool)
    currencies = rng.choice(["USD", "MXN", "CAD"], total)

    # payment_created_at: spread over 15 days
    base_ts = pd.Timestamp("2026-06-15 00:00:00")
    payment_ts = [
        base_ts + pd.Timedelta(seconds=int(s)) for s in rng.integers(0, 15 * 24 * 3600, total)
    ]
    scored_ts = [t + pd.Timedelta(seconds=10) for t in payment_ts]

    # frame_flags: empty for shadow_old; JSON with timezone_missing for shadow_new
    frame_flags = []
    for m in modes:
        if m == SHADOW_NEW:
            frame_flags.append(json.dumps({"timezone_missing": False}))
        else:
            frame_flags.append("")

    return pd.DataFrame(
        {
            "payment_id": list(range(total)),
            "scoring_mode": modes,
            "percentile": percentiles,
            "amount_usd": amounts,
            "is_anomaly": is_anomaly,
            "currency": currencies,
            "frame_flags": frame_flags,
            "payment_created_at": payment_ts,
            "scored_at": scored_ts,
        }
    )


# ---------------------------------------------------------------------------
# SHAD-02-B: compute_top5_bias
# ---------------------------------------------------------------------------


class TestComputeTop5Bias:
    def test_winsorized_ratio_less_than_raw_ratio_with_heavy_tail(self):
        """Winsorized ratio < raw mean ratio when outliers are in top-5%.

        Constructs a controlled DataFrame where two extreme outliers explicitly
        have the highest percentile scores (guaranteed to land in top-5%),
        so winsorization provably reduces the ratio.
        """
        n = 200
        rng = np.random.default_rng(7)
        # Base amounts: log-normal; percentiles all below 0.9
        amounts = rng.lognormal(mean=3.0, sigma=1.0, size=n).astype(np.float64)
        percentiles = rng.uniform(0.0, 0.9, size=n)

        # Inject two extreme outliers with the highest percentiles (top-5%)
        amounts[0] = 1_000_000.0
        amounts[1] = 500_000.0
        percentiles[0] = 0.999
        percentiles[1] = 0.998

        base_ts = pd.Timestamp("2026-06-15 12:00:00")
        df_new = pd.DataFrame(
            {
                "payment_id": range(n),
                "scoring_mode": [SHADOW_NEW] * n,
                "percentile": percentiles,
                "amount_usd": amounts,
                "is_anomaly": [False] * n,
                "currency": ["USD"] * n,
                "frame_flags": [json.dumps({"timezone_missing": False})] * n,
                "payment_created_at": [base_ts] * n,
                "scored_at": [base_ts] * n,
            }
        )
        df_old = df_new.copy()
        df_old["scoring_mode"] = SHADOW_OLD
        df = pd.concat([df_old, df_new], ignore_index=True)

        result = compute_top5_bias(df, SHADOW_NEW)
        assert not math.isnan(result["top5_wins_ratio"]), "ratio should not be NaN"

        # Compute raw ratio independently (reset index to avoid iloc/loc mismatch)
        sub = df[df["scoring_mode"] == SHADOW_NEW].copy().reset_index(drop=True)
        k5 = max(1, int(len(sub) * 0.05))
        top5_positions = sub["percentile"].nlargest(k5).index.to_numpy()
        raw_amounts = sub["amount_usd"].to_numpy(dtype=np.float64)
        raw_ratio = float(raw_amounts[top5_positions].mean() / raw_amounts.mean())

        assert result["top5_wins_ratio"] < raw_ratio, (
            f"Winsorized ratio {result['top5_wins_ratio']:.4f} should be less "
            f"than raw ratio {raw_ratio:.4f} when extreme outliers are in top-5%"
        )

    def test_ratio_is_positive(self):
        """top5_wins_ratio should be > 1 (top-5% amounts are above average)."""
        df = _make_df(seed=42)
        result = compute_top5_bias(df, SHADOW_OLD)
        assert result["top5_wins_ratio"] > 1.0

    def test_n_equals_subset_length(self):
        """n field should match the number of rows for that model."""
        df = _make_df(n_old=150, n_new=250)
        result_old = compute_top5_bias(df, SHADOW_OLD)
        result_new = compute_top5_bias(df, SHADOW_NEW)
        assert result_old["n"] == 150
        assert result_new["n"] == 250

    def test_empty_model_returns_nan(self):
        """Empty subset should return NaN ratio without raising."""
        df = _make_df(n_old=0, n_new=50)
        result = compute_top5_bias(df, SHADOW_OLD)
        assert math.isnan(result["top5_wins_ratio"])

    def test_model_field_matches_argument(self):
        """Returned dict should carry the model name passed in."""
        df = _make_df()
        for model in (SHADOW_OLD, SHADOW_NEW):
            result = compute_top5_bias(df, model)
            assert result["model"] == model

    def test_p999_threshold_present(self):
        """p999 key should be a valid non-negative number."""
        df = _make_df()
        result = compute_top5_bias(df, SHADOW_NEW)
        assert result["p999"] >= 0.0


# ---------------------------------------------------------------------------
# SHAD-02-D: compute_jaccard_at_k
# ---------------------------------------------------------------------------


class TestComputeJaccardAtK:
    def test_perfect_overlap_returns_one(self):
        """If both models rank the same top-k payment_ids first, Jaccard = 1."""
        # Build deterministic df where percentile matches payment_id order
        n = 200
        payment_ids = list(range(n))
        percentiles = np.linspace(0, 1, n)

        df_old = pd.DataFrame(
            {
                "payment_id": payment_ids,
                "scoring_mode": [SHADOW_OLD] * n,
                "percentile": percentiles,
                "amount_usd": [10.0] * n,
                "is_anomaly": [False] * n,
                "currency": ["USD"] * n,
                "frame_flags": [""] * n,
                "payment_created_at": [pd.Timestamp("2026-06-15")] * n,
                "scored_at": [pd.Timestamp("2026-06-15")] * n,
            }
        )
        df_new = df_old.copy()
        df_new["scoring_mode"] = SHADOW_NEW

        df = pd.concat([df_old, df_new], ignore_index=True)
        j = compute_jaccard_at_k(df, k=50)
        assert j == pytest.approx(1.0)

    def test_zero_overlap_returns_zero(self):
        """If top-k sets are completely disjoint, Jaccard = 0."""
        # shadow_old top-100: payment_ids 100-199 have high percentile
        # shadow_new top-100: payment_ids 0-99 have high percentile
        df_old = pd.DataFrame(
            {
                "payment_id": list(range(200)),
                "scoring_mode": [SHADOW_OLD] * 200,
                "percentile": np.concatenate([np.zeros(100), np.ones(100)]),
                "amount_usd": [10.0] * 200,
                "is_anomaly": [False] * 200,
                "currency": ["USD"] * 200,
                "frame_flags": [""] * 200,
                "payment_created_at": [pd.Timestamp("2026-06-15")] * 200,
                "scored_at": [pd.Timestamp("2026-06-15")] * 200,
            }
        )
        df_new = pd.DataFrame(
            {
                "payment_id": list(range(200)),
                "scoring_mode": [SHADOW_NEW] * 200,
                "percentile": np.concatenate([np.ones(100), np.zeros(100)]),
                "amount_usd": [10.0] * 200,
                "is_anomaly": [False] * 200,
                "currency": ["USD"] * 200,
                "frame_flags": [""] * 200,
                "payment_created_at": [pd.Timestamp("2026-06-15")] * 200,
                "scored_at": [pd.Timestamp("2026-06-15")] * 200,
            }
        )
        df = pd.concat([df_old, df_new], ignore_index=True)
        j = compute_jaccard_at_k(df, k=100)
        assert j == pytest.approx(0.0)

    def test_partial_overlap_correct_value(self):
        """50% overlap in top-10 gives Jaccard = 0.5 / 1.5 ≈ 0.333."""
        # 10 payments each; old ranks 0-9 top-10; new ranks 5-14 top-10 → overlap 5-9 = 5
        df_old = pd.DataFrame(
            {
                "payment_id": list(range(20)),
                "scoring_mode": [SHADOW_OLD] * 20,
                "percentile": np.concatenate([np.linspace(0.5, 1.0, 10), np.zeros(10)]),
                "amount_usd": [10.0] * 20,
                "is_anomaly": [False] * 20,
                "currency": ["USD"] * 20,
                "frame_flags": [""] * 20,
                "payment_created_at": [pd.Timestamp("2026-06-15")] * 20,
                "scored_at": [pd.Timestamp("2026-06-15")] * 20,
            }
        )
        # new: payment_ids 5-14 have high percentile (overlap with old: 5,6,7,8,9)
        percs_new = np.zeros(20)
        percs_new[5:15] = np.linspace(0.5, 1.0, 10)
        df_new = pd.DataFrame(
            {
                "payment_id": list(range(20)),
                "scoring_mode": [SHADOW_NEW] * 20,
                "percentile": percs_new,
                "amount_usd": [10.0] * 20,
                "is_anomaly": [False] * 20,
                "currency": ["USD"] * 20,
                "frame_flags": [""] * 20,
                "payment_created_at": [pd.Timestamp("2026-06-15")] * 20,
                "scored_at": [pd.Timestamp("2026-06-15")] * 20,
            }
        )
        df = pd.concat([df_old, df_new], ignore_index=True)
        # top-10 old: payment_ids 0-9; top-10 new: payment_ids 5-14
        # intersection = {5,6,7,8,9} = 5; union = {0..14} = 15
        j = compute_jaccard_at_k(df, k=10)
        assert j == pytest.approx(5 / 15, abs=1e-6)

    def test_empty_df_returns_one(self):
        """Empty DataFrame returns 1.0 (vacuous Jaccard)."""
        df = _make_df(n_old=0, n_new=0)
        j = compute_jaccard_at_k(df, k=100)
        assert j == 1.0


# ---------------------------------------------------------------------------
# SHAD-02-A: compute_alert_rate_by_segment
# ---------------------------------------------------------------------------


class TestComputeAlertRateBySegment:
    def test_aggregates_correctly(self):
        """Alert rates match manually computed values."""
        df = pd.DataFrame(
            {
                "payment_id": range(6),
                "scoring_mode": [
                    SHADOW_OLD,
                    SHADOW_OLD,
                    SHADOW_NEW,
                    SHADOW_NEW,
                    SHADOW_OLD,
                    SHADOW_NEW,
                ],
                "currency": ["USD", "USD", "USD", "MXN", "MXN", "MXN"],
                "is_anomaly": [True, False, True, True, True, False],
                "percentile": [0.9, 0.5, 0.95, 0.8, 0.7, 0.3],
                "amount_usd": [100.0] * 6,
                "frame_flags": [""] * 6,
                "payment_created_at": [pd.Timestamp("2026-06-15")] * 6,
                "scored_at": [pd.Timestamp("2026-06-15")] * 6,
            }
        )
        result = compute_alert_rate_by_segment(df)
        # USD / shadow_old: 2 rows, 1 alert -> 0.5
        usd_old = result[(result["currency"] == "USD") & (result["scoring_mode"] == SHADOW_OLD)]
        assert len(usd_old) == 1
        assert usd_old.iloc[0]["alert_rate"] == pytest.approx(0.5)
        # MXN / shadow_new: 2 rows (ids 3 and 5), is_anomaly=[True, False] -> rate=0.5
        mxn_new = result[(result["currency"] == "MXN") & (result["scoring_mode"] == SHADOW_NEW)]
        assert len(mxn_new) == 1
        assert mxn_new.iloc[0]["alert_rate"] == pytest.approx(0.5)

    def test_empty_df_returns_empty(self):
        df = _make_df(n_old=0, n_new=0)
        result = compute_alert_rate_by_segment(df)
        assert result.empty

    def test_total_column_correct(self):
        df = _make_df(n_old=100, n_new=150)
        result = compute_alert_rate_by_segment(df)
        assert result["total"].sum() == 250


# ---------------------------------------------------------------------------
# SHAD-02-C: compute_off_hours
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SHAD-02-E: compute_currency_breakdown (frame-normalization-v1, grupo 4)
# ---------------------------------------------------------------------------


def _currency_df(spec: dict, seed: int = 11) -> pd.DataFrame:
    """Build a shadow_new DataFrame from {currency: dict(n=..., ...)} spec.

    Each currency spec supports:
      n: number of rows
      amount: base amount (default lognormal)
      top5_amount: amount injected in the top-5% percentile rows (bias)
      alert_frac: fraction of is_anomaly=True rows
      fallback_levels: list to cycle through (default ['facility'])
    """
    rng = np.random.default_rng(seed)
    frames = []
    pid = 0
    for currency, cfg in spec.items():
        n = cfg["n"]
        percentiles = np.linspace(0.0, 0.94, n)
        amounts = np.full(n, float(cfg.get("amount", 10.0)))
        if "top5_amount" in cfg:
            k5 = max(1, int(n * 0.05))
            # give the last k5 rows the highest percentiles and biased amounts
            percentiles[-k5:] = np.linspace(0.95, 0.999, k5)
            amounts[-k5:] = float(cfg["top5_amount"])
        alert_frac = cfg.get("alert_frac", 0.0)
        n_alerts = int(round(n * alert_frac))
        is_anomaly = np.array([True] * n_alerts + [False] * (n - n_alerts))
        levels = cfg.get("fallback_levels", ["facility"])
        fallback = [levels[i % len(levels)] for i in range(n)]
        frames.append(
            pd.DataFrame(
                {
                    "payment_id": range(pid, pid + n),
                    "scoring_mode": [SHADOW_NEW] * n,
                    "percentile": percentiles,
                    "amount_usd": amounts,
                    "is_anomaly": is_anomaly,
                    "currency": [currency] * n,
                    "fallback_level": fallback,
                    "frame_flags": [""] * n,
                    "payment_created_at": [pd.Timestamp("2026-06-15 12:00:00")] * n,
                    "scored_at": [pd.Timestamp("2026-06-15 12:00:00")] * n,
                }
            )
        )
        pid += n
    _ = rng
    return pd.concat(frames, ignore_index=True)


class TestComputeCurrencyBreakdown:
    def _breakdown(self, df, **kwargs):
        from shadow_monitor import compute_currency_breakdown

        return compute_currency_breakdown(df, **kwargs)

    def test_row_per_currency_with_required_metrics(self):
        """Cada moneda con volumen propio tiene fila con sesgo, tasa de alertas y fallback."""
        df = _currency_df(
            {
                "USD": {"n": 2000, "alert_frac": 0.05},
                "CAD": {"n": 1500, "alert_frac": 0.02},
                "NIO": {"n": 50, "alert_frac": 0.10},
                "HNL": {"n": 30},
            }
        )
        result = self._breakdown(df)
        currencies = set(result["currency"])
        assert {"USD", "CAD", "NIO", "HNL"} <= currencies
        for col in ("n", "alert_rate", "top5_amount_x_avg", "fallback_levels", "exceeds_gate"):
            assert col in result.columns, f"columna '{col}' ausente"

    def test_no_currency_above_1000_payments_under_otros(self):
        """Ninguna moneda con >1000 pagos queda agregada bajo 'otros'."""
        df = _currency_df(
            {
                "USD": {"n": 3000},
                "MXN": {"n": 1200},  # >1000 -> fila propia obligatoria
                "EUR": {"n": 200},  # ni mandatoria ni >1000 -> 'otros'
            }
        )
        result = self._breakdown(df)
        assert "MXN" in set(result["currency"])
        assert "EUR" not in set(result["currency"])
        otros = result[result["currency"] == "otros"]
        assert len(otros) == 1
        assert int(otros.iloc[0]["n"]) == 200

    def test_mandatory_currencies_present_with_any_volume(self):
        """NIO/HNL/PKR tienen fila propia cuando tienen volumen, aunque n<=1000."""
        df = _currency_df({"USD": {"n": 2000}, "PKR": {"n": 10}, "NIO": {"n": 5}})
        result = self._breakdown(df)
        assert {"PKR", "NIO"} <= set(result["currency"])

    def test_mandatory_currency_absent_without_volume(self):
        df = _currency_df({"USD": {"n": 2000}})
        result = self._breakdown(df)
        assert "PKR" not in set(result["currency"])

    def test_currency_exceeding_gate_is_flagged(self):
        """Moneda con ratio de sesgo > 4x queda señalada como excedida."""
        df = _currency_df(
            {
                "USD": {"n": 2000, "amount": 10.0},
                "PKR": {"n": 1200, "amount": 10.0, "top5_amount": 1000.0},
            }
        )
        result = self._breakdown(df)
        pkr = result[result["currency"] == "PKR"].iloc[0]
        usd = result[result["currency"] == "USD"].iloc[0]
        assert pkr["top5_amount_x_avg"] > 4.0
        assert bool(pkr["exceeds_gate"]) is True
        assert bool(usd["exceeds_gate"]) is False

    def test_alert_rate_computed_per_currency(self):
        df = _currency_df({"USD": {"n": 2000, "alert_frac": 0.05}, "NIO": {"n": 100,
                                                                           "alert_frac": 0.10}})
        result = self._breakdown(df)
        nio = result[result["currency"] == "NIO"].iloc[0]
        assert nio["alert_rate"] == pytest.approx(0.10, abs=1e-9)

    def test_fallback_level_distribution(self):
        df = _currency_df(
            {"USD": {"n": 2000, "fallback_levels": ["facility", "facility", "currency",
                                                    "global"]}}
        )
        result = self._breakdown(df)
        usd = result[result["currency"] == "USD"].iloc[0]
        dist = usd["fallback_levels"]
        assert dist["facility"] == 1000
        assert dist["currency"] == 500
        assert dist["global"] == 500

    def test_empty_df_returns_empty(self):
        df = _make_df(n_old=0, n_new=0)
        df["fallback_level"] = pd.Series(dtype=str)
        result = self._breakdown(df)
        assert result.empty

    def test_load_sql_includes_fallback_level(self):
        """La query de carga del shadow monitor trae fallback_level por fila."""
        import shadow_monitor

        assert "fallback_level" in shadow_monitor._LOAD_SQL


class TestComputeOffHours:
    def test_off_hours_rate_in_range(self):
        """Off-hours rate should be between 0 and 1."""
        df = _make_df()
        oh = compute_off_hours(df, SHADOW_OLD)
        assert 0.0 <= oh["off_hours_utc_rate"] <= 1.0

    def test_tz_missing_rate_zero_for_shadow_old(self):
        """shadow_old has no frame_flags — tz_missing_rate should be 0.0."""
        df = _make_df()
        oh = compute_off_hours(df, SHADOW_OLD)
        assert oh["tz_missing_rate"] == 0.0

    def test_tz_missing_rate_nonzero_when_flags_set(self):
        """shadow_new rows with timezone_missing=True raise tz_missing_rate."""
        n = 100
        flags = [json.dumps({"timezone_missing": i % 2 == 0}) for i in range(n)]
        df = pd.DataFrame(
            {
                "payment_id": range(n),
                "scoring_mode": [SHADOW_NEW] * n,
                "percentile": [0.5] * n,
                "amount_usd": [10.0] * n,
                "is_anomaly": [False] * n,
                "currency": ["USD"] * n,
                "frame_flags": flags,
                "payment_created_at": [pd.Timestamp("2026-06-15 12:00:00")] * n,
                "scored_at": [pd.Timestamp("2026-06-15 12:00:00")] * n,
            }
        )
        oh = compute_off_hours(df, SHADOW_NEW)
        assert oh["tz_missing_rate"] == pytest.approx(0.5, abs=1e-6)

    def test_empty_model_returns_nan(self):
        df = _make_df(n_old=0, n_new=50)
        oh = compute_off_hours(df, SHADOW_OLD)
        assert math.isnan(oh["off_hours_utc_rate"])
