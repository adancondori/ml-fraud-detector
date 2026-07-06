"""shadow_gate.py — SHAD-03: go/no-go gate for frame-v1 promotion.

Evaluates 4 criteria over the dual-run shadow data and emits:
  PASS            (exit 0) — all 4 criteria met; frame-v1 can be promoted
  FAIL            (exit 1) — criteria not met; do NOT promote
  INSUFFICIENT_DATA (exit 2) — < 14 days OR < 500 rows; defer evaluation

The gate NEVER forces a PASS with insufficient data.  The structural separation
of code availability from gate evaluation is intentional (pitfall #5 from the
SHAD-03 design).

Gate thresholds (locked from 01-03 bias report):
  GATE_TOP5_MAX_RATIO    = 4.0   top-5% winsorized < 4x (baseline: 11.79x)
  GATE_OFF_HOURS_MIN     = 0.03  off-hours >= 3% (UTC proxy)
  GATE_OFF_HOURS_MAX     = 0.07  off-hours <= 7% (band ~4-5%)
  GATE_SPEARMAN_MIN      = 0.90  Spearman ranking between old/new >= 0.90
  GATE_ALERT_RATE_DELTA  = 0.02  max delta alert rate per segment <= 2pp
  MIN_SHADOW_DAYS        = 14    minimum accrual period
  MIN_SHADOW_ROWS        = 500   minimum row count (covers Jaccard@100 + Spearman)

Connects ONLY to the WRITE ClickHouse local (ANOMALY_SCORES_CH_* env vars).
Never touches the production READ target.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import pandas as pd

# scipy is imported lazily inside compute_spearman to avoid hard import errors
# when running tests that mock the function; however scipy IS required at runtime.

# ---------------------------------------------------------------------------
# Constants — hardcoded gate thresholds (locked)
# ---------------------------------------------------------------------------
GATE_TOP5_MAX_RATIO: float = 4.0
GATE_OFF_HOURS_MIN: float = 0.03
GATE_OFF_HOURS_MAX: float = 0.07
GATE_SPEARMAN_MIN: float = 0.90
GATE_ALERT_RATE_DELTA_MAX: float = 0.02
MIN_SHADOW_DAYS: int = 14
MIN_SHADOW_ROWS: int = 500

SHADOW_OLD = "shadow_old"
SHADOW_NEW = "shadow_new"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INSUFFICIENT_DATA = 2


# ---------------------------------------------------------------------------
# Spearman ranking correlation
# ---------------------------------------------------------------------------


def compute_spearman(df: pd.DataFrame) -> float:
    """Compute Spearman rank correlation between shadow_old and shadow_new percentiles.

    Merges the two model rows on payment_id and computes the Spearman
    correlation between their percentile scores.

    Args:
        df: Shadow DataFrame with both scoring_mode values.

    Returns:
        Spearman rho in [-1, 1], or NaN if fewer than 30 matched pairs.
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    old = df[df["scoring_mode"] == SHADOW_OLD][["payment_id", "percentile"]].rename(
        columns={"percentile": "pct_old"}
    )
    new = df[df["scoring_mode"] == SHADOW_NEW][["payment_id", "percentile"]].rename(
        columns={"percentile": "pct_new"}
    )
    merged = old.merge(new, on="payment_id", how="inner")
    if len(merged) < 30:
        return float("nan")

    pct_old = merged["pct_old"].to_numpy()
    pct_new = merged["pct_new"].to_numpy()
    rho, _ = spearmanr(pct_old, pct_new)
    return float(rho)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluate_gate(df: pd.DataFrame) -> dict:
    """Evaluate the SHAD-03 go/no-go gate.

    Args:
        df: Shadow DataFrame with columns: payment_id, scoring_mode, percentile,
            amount_usd, is_anomaly, currency, frame_flags, payment_created_at,
            scored_at.

    Returns:
        dict with 'status' key set to one of 'PASS', 'FAIL', 'INSUFFICIENT_DATA'.
        INSUFFICIENT_DATA also includes 'days_span', 'n_rows', 'min_days', 'min_rows'.
        PASS/FAIL includes 'criteria' and 'checks' sub-dicts.
    """
    # ------------------------------------------------------------------
    # Guard: temporal span and row count (structural separation — SHAD-03)
    # ------------------------------------------------------------------
    n_rows = len(df)

    if n_rows == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "days_span": 0,
            "n_rows": 0,
            "min_days": MIN_SHADOW_DAYS,
            "min_rows": MIN_SHADOW_ROWS,
        }

    ts_col = "scored_at" if "scored_at" in df.columns else "payment_created_at"
    ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
    days_span = int((ts.max() - ts.min()).days) if len(ts) >= 2 else 0

    if days_span < MIN_SHADOW_DAYS or n_rows < MIN_SHADOW_ROWS:
        return {
            "status": "INSUFFICIENT_DATA",
            "days_span": days_span,
            "n_rows": n_rows,
            "min_days": MIN_SHADOW_DAYS,
            "min_rows": MIN_SHADOW_ROWS,
        }

    # ------------------------------------------------------------------
    # Compute criteria using shadow_monitor functions
    # ------------------------------------------------------------------
    # Import here to avoid circular dependency if shadow_gate is imported first
    import sys as _sys
    from pathlib import Path as _Path

    _scripts_dir = str(_Path(__file__).parent)
    if _scripts_dir not in _sys.path:
        _sys.path.insert(0, _scripts_dir)

    from shadow_monitor import (  # noqa: PLC0415
        compute_alert_rate_by_segment,
        compute_off_hours,
        compute_top5_bias,
    )

    # Criterion 1: top-5% winsorized amount ratio for frame-v1 < 4x
    bias_new = compute_top5_bias(df, SHADOW_NEW)
    bias_old = compute_top5_bias(df, SHADOW_OLD)
    top5_ratio_new = bias_new["top5_wins_ratio"]
    top5_ratio_old = bias_old["top5_wins_ratio"]
    top5_pass = (not math.isnan(top5_ratio_new)) and (top5_ratio_new < GATE_TOP5_MAX_RATIO)

    # Criterion 2: off-hours UTC rate within [3%, 7%] for shadow_new
    oh_new = compute_off_hours(df, SHADOW_NEW)
    off_hours_new = oh_new["off_hours_utc_rate"]
    off_hours_pass = (
        (not math.isnan(off_hours_new))
        and GATE_OFF_HOURS_MIN <= off_hours_new <= GATE_OFF_HOURS_MAX
    )

    # Criterion 3: Spearman ranking correlation >= 0.90
    rho = compute_spearman(df)
    spearman_pass = (not math.isnan(rho)) and (rho >= GATE_SPEARMAN_MIN)

    # Criterion 4: max delta alert rate per (currency, scoring_mode) pair <= 2pp
    seg = compute_alert_rate_by_segment(df)
    max_alert_delta = float("nan")
    alert_delta_pass = False
    if not seg.empty:
        pivot = seg.pivot_table(
            index="currency", columns="scoring_mode", values="alert_rate"
        )
        if SHADOW_OLD in pivot.columns and SHADOW_NEW in pivot.columns:
            delta = (pivot[SHADOW_NEW] - pivot[SHADOW_OLD]).abs().dropna()
            if len(delta) > 0:
                max_alert_delta = float(delta.max())
                alert_delta_pass = max_alert_delta <= GATE_ALERT_RATE_DELTA_MAX

    all_pass = top5_pass and off_hours_pass and spearman_pass and alert_delta_pass

    return {
        "status": "PASS" if all_pass else "FAIL",
        "criteria": {
            "top5_ratio_new": top5_ratio_new,
            "top5_ratio_old": top5_ratio_old,
            "off_hours_new": off_hours_new,
            "spearman": rho,
            "max_alert_rate_delta": max_alert_delta,
        },
        "checks": {
            "top5_pass": top5_pass,
            "off_hours_pass": off_hours_pass,
            "spearman_pass": spearman_pass,
            "alert_delta_pass": alert_delta_pass,
        },
    }


# ---------------------------------------------------------------------------
# ClickHouse client (WRITE local only)
# ---------------------------------------------------------------------------


def _build_ch_client():
    """Build ClickHouse client for the WRITE local target (ANOMALY_SCORES_CH_*)."""
    try:
        import clickhouse_connect
    except ImportError as exc:
        print("ERROR: clickhouse-connect not installed.", file=sys.stderr)
        raise exc

    host = os.environ.get("ANOMALY_SCORES_CH_HOST", "clickhouse")
    port = int(os.environ.get("ANOMALY_SCORES_CH_PORT", "8123"))
    user = os.environ.get("ANOMALY_SCORES_CH_USER", "default")
    password = os.environ.get("ANOMALY_SCORES_CH_PASSWORD", "")
    database = os.environ.get(
        "ANOMALY_SCORES_CH_DATABASE", "pbp_productionDB_optimized"
    )
    secure_str = os.environ.get("ANOMALY_SCORES_CH_SECURE", "false").lower()
    secure = secure_str in ("1", "true", "yes")

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        database=database,
        secure=secure,
        autogenerate_session_id=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SHAD-03: shadow go/no-go gate. "
            "Exit 0=PASS, 1=FAIL, 2=INSUFFICIENT_DATA."
        )
    )
    parser.parse_args(argv)

    ch_client = _build_ch_client()

    print("\n=== SHAD-03 Shadow Gate ===\n")

    # Load ALL available shadow data (no day limit — gate needs full history)
    from shadow_monitor import load_shadow_df  # noqa: PLC0415

    df = load_shadow_df(ch_client, days=9999)
    n_old = len(df[df["scoring_mode"] == SHADOW_OLD])
    n_new = len(df[df["scoring_mode"] == SHADOW_NEW])
    print(f"Shadow rows:  shadow_old={n_old}  shadow_new={n_new}")

    result = evaluate_gate(df)
    status = result["status"]

    if status == "INSUFFICIENT_DATA":
        print(f"\nRESULT: INSUFFICIENT_DATA (exit {EXIT_INSUFFICIENT_DATA})")
        print(f"  days_span : {result['days_span']} (need >= {result['min_days']})")
        print(f"  n_rows    : {result['n_rows']} (need >= {result['min_rows']})")
        print("\nRun again after >=14 days of shadow scoring with >=500 rows.")
        sys.exit(EXIT_INSUFFICIENT_DATA)

    crit = result["criteria"]
    checks = result["checks"]

    print(f"\n  Criterion 1 — top-5% ratio (frame-v1)    : "
          f"{crit['top5_ratio_new']:.3f}x  "
          f"[{'PASS' if checks['top5_pass'] else 'FAIL'}  threshold < {GATE_TOP5_MAX_RATIO}x]")
    print(f"  Criterion 1 — top-5% ratio (champion)    : "
          f"{crit['top5_ratio_old']:.3f}x  [baseline: 11.79x]")
    print(f"  Criterion 2 — off-hours UTC (frame-v1)   : "
          f"{crit['off_hours_new']*100:.2f}%  "
          f"[{'PASS' if checks['off_hours_pass'] else 'FAIL'}  "
          f"band {GATE_OFF_HOURS_MIN*100:.0f}–{GATE_OFF_HOURS_MAX*100:.0f}%]")
    rho_str = f"{crit['spearman']:.4f}" if not math.isnan(crit["spearman"]) else "N/A"
    print(f"  Criterion 3 — Spearman ranking            : "
          f"{rho_str}  "
          f"[{'PASS' if checks['spearman_pass'] else 'FAIL'}  "
          f"threshold >= {GATE_SPEARMAN_MIN}]")
    delta_str = (
        f"{crit['max_alert_rate_delta']*100:.3f}pp"
        if not math.isnan(crit["max_alert_rate_delta"])
        else "N/A"
    )
    print(f"  Criterion 4 — max alert rate delta        : "
          f"{delta_str}  "
          f"[{'PASS' if checks['alert_delta_pass'] else 'FAIL'}  "
          f"threshold <= {GATE_ALERT_RATE_DELTA_MAX*100:.0f}pp]")

    exit_code = EXIT_PASS if status == "PASS" else EXIT_FAIL
    print(f"\nRESULT: {status} (exit {exit_code})")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
