"""shadow_monitor.py — SHAD-02: 4 shadow scoring metrics.

Connects to the WRITE ClickHouse local (ANOMALY_SCORES_CH_* env vars) and
computes metrics over the dual-run rows (scoring_mode IN ('shadow_old',
'shadow_new')):

  SHAD-02-A  alert_rate_by_segment  — per (currency, scoring_mode)
  SHAD-02-B  top5_bias              — top-5% winsorized p99.9 amount ratio
  SHAD-02-C  off_hours              — UTC hour approximation + tz_missing rate
  SHAD-02-D  jaccard_at_k           — Jaccard overlap of top-k payment sets

Writes ONLY to local ClickHouse (ANOMALY_SCORES_CH_*).  Never touches the
production READ target (CLICKHOUSE_*).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DAYS = 30
SHADOW_OLD = "shadow_old"
SHADOW_NEW = "shadow_new"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_LOAD_SQL = """
SELECT
    payment_id,
    scoring_mode,
    percentile,
    amount_usd,
    is_anomaly,
    currency,
    frame_flags,
    payment_created_at,
    scored_at
FROM {table}
WHERE scoring_mode IN ('shadow_old', 'shadow_new')
  AND scored_at >= now() - INTERVAL {days} DAY
"""


_DEFAULT_ANOMALY_SCORES_TABLE = "pbp_productionDB_optimized.anomaly_scores"


def load_shadow_df(
    ch_client,
    days: int = DEFAULT_DAYS,
    table: str = _DEFAULT_ANOMALY_SCORES_TABLE,
) -> pd.DataFrame:
    """Fetch shadow rows from the WRITE ClickHouse local into a DataFrame.

    Args:
        ch_client: A clickhouse_connect client pointed at the WRITE target.
        days: How many days back to query.
        table: Full table name (default = pbp_productionDB_optimized.anomaly_scores).

    Returns:
        DataFrame with columns: payment_id, scoring_mode, percentile,
        amount_usd, is_anomaly, currency, frame_flags, payment_created_at,
        scored_at.  May be empty if no shadow rows exist yet.
    """
    sql = _LOAD_SQL.format(table=table, days=days)
    result = ch_client.query(sql)
    if not result.result_rows:
        return pd.DataFrame(
            columns=[
                "payment_id",
                "scoring_mode",
                "percentile",
                "amount_usd",
                "is_anomaly",
                "currency",
                "frame_flags",
                "payment_created_at",
                "scored_at",
            ]
        )
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0.0)
    df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce").fillna(0.0)
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# SHAD-02-A: alert rate by segment
# ---------------------------------------------------------------------------


def compute_alert_rate_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """SHAD-02-A: alert rate per (currency, scoring_mode).

    Args:
        df: Shadow DataFrame from load_shadow_df.

    Returns:
        DataFrame with columns: currency, scoring_mode, total, alerts,
        alert_rate.  Empty if df is empty.
    """
    if df.empty:
        return pd.DataFrame(columns=["currency", "scoring_mode", "total", "alerts", "alert_rate"])

    grp = (
        df.groupby(["currency", "scoring_mode"], as_index=False)
        .agg(total=("is_anomaly", "count"), alerts=("is_anomaly", "sum"))
    )
    grp["alert_rate"] = grp["alerts"] / grp["total"].replace(0, np.nan)
    return grp.sort_values(["currency", "scoring_mode"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SHAD-02-B: top-5% amount bias (winsorized p99.9)
# ---------------------------------------------------------------------------


def compute_top5_bias(df: pd.DataFrame, model: str) -> dict:
    """SHAD-02-B: top-5% amount bias for one scoring model.

    Uses the winsorized p99.9 metric (decision locked in 01-03 — raw mean ratio
    is invalid due to heavy tail; this matches retrain_frame_v1.py lines 525-545).

    Args:
        df: Shadow DataFrame.
        model: One of 'shadow_old' or 'shadow_new'.

    Returns:
        dict with keys: model, top5_wins_ratio, p999, n.
        top5_wins_ratio is NaN when there are fewer than 2 rows.
    """
    sub = df[df["scoring_mode"] == model].copy()
    n = len(sub)
    if n < 2:
        return {"model": model, "top5_wins_ratio": float("nan"), "p999": float("nan"), "n": n}

    k5 = max(1, int(n * 0.05))
    top5_idx = sub["percentile"].nlargest(k5).index

    amounts = sub["amount_usd"].to_numpy(dtype=np.float64)
    p999 = float(np.percentile(amounts, 99.9))

    # Winsorize the full distribution and the top-5% subset
    wins_all = np.clip(amounts, None, p999)
    top5_amounts = sub.loc[top5_idx, "amount_usd"].to_numpy(dtype=np.float64)
    wins_top5 = np.clip(top5_amounts, None, p999)

    global_wins_mean = wins_all.mean()
    if global_wins_mean <= 0:
        return {"model": model, "top5_wins_ratio": float("nan"), "p999": p999, "n": n}

    ratio = float(wins_top5.mean() / global_wins_mean)
    return {"model": model, "top5_wins_ratio": ratio, "p999": p999, "n": n}


# ---------------------------------------------------------------------------
# SHAD-02-C: off-hours (UTC approximation + tz_missing rate)
# ---------------------------------------------------------------------------


def compute_off_hours(df: pd.DataFrame, model: str) -> dict:
    """SHAD-02-C: off-hours fraction and timezone-missing rate.

    Off-hours UTC approximation: hours 0-8 and 22-23 (UTC).  Facilities span
    UTC-3..UTC-8 so UTC off-hours hour is a reasonable proxy when local tz data
    is unavailable in the shadow row.

    tz_missing_rate: fraction of shadow_new rows where frame_flags JSON contains
    "timezone_missing": true.  shadow_old rows have frame_flags='' so this is
    always 0 for shadow_old.

    Args:
        df: Shadow DataFrame.
        model: One of 'shadow_old' or 'shadow_new'.

    Returns:
        dict with keys: model, off_hours_utc_rate, tz_missing_rate, n.
    """
    sub = df[df["scoring_mode"] == model].copy()
    n = len(sub)
    if n == 0:
        return {
            "model": model,
            "off_hours_utc_rate": float("nan"),
            "tz_missing_rate": float("nan"),
            "n": 0,
        }

    # UTC off-hours: hours 0-8 and 22-23
    hours = pd.to_datetime(sub["payment_created_at"], errors="coerce").dt.hour
    off_hours_mask = hours.isin(range(0, 9)) | hours.isin([22, 23])
    off_hours_rate = float(off_hours_mask.mean())

    # tz_missing_rate from frame_flags JSON (only meaningful for shadow_new)
    tz_missing_rate = 0.0
    if model == SHADOW_NEW:
        def _extract_tz_missing(flags_str: str) -> bool:
            if not flags_str:
                return False
            try:
                d = json.loads(flags_str)
                return bool(d.get("timezone_missing", False))
            except (json.JSONDecodeError, TypeError, ValueError):
                return False

        tz_missing_vals = sub["frame_flags"].apply(_extract_tz_missing)
        tz_missing_rate = float(tz_missing_vals.mean())

    return {
        "model": model,
        "off_hours_utc_rate": off_hours_rate,
        "tz_missing_rate": tz_missing_rate,
        "n": n,
    }


# ---------------------------------------------------------------------------
# SHAD-02-D: Jaccard@k
# ---------------------------------------------------------------------------


def compute_jaccard_at_k(df: pd.DataFrame, k: int = 100) -> float:
    """SHAD-02-D: Jaccard similarity of top-k payment sets.

    Computes the Jaccard index between the top-k payment_ids by percentile
    for shadow_old and shadow_new.

    Args:
        df: Shadow DataFrame containing both shadow_old and shadow_new rows.
        k: Number of top payments to compare (default 100).

    Returns:
        Jaccard index in [0, 1].  Returns 1.0 if either set is empty (vacuous).
    """
    old = df[df["scoring_mode"] == SHADOW_OLD]
    new = df[df["scoring_mode"] == SHADOW_NEW]

    if old.empty or new.empty:
        return 1.0

    k_old = min(k, len(old))
    k_new = min(k, len(new))

    top_old = set(old.nlargest(k_old, "percentile")["payment_id"].tolist())
    top_new = set(new.nlargest(k_new, "percentile")["payment_id"].tolist())

    union = top_old | top_new
    if not union:
        return 1.0

    intersection = top_old & top_new
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_ch_client():
    """Build a ClickHouse client for the WRITE local target (ANOMALY_SCORES_CH_*)."""
    try:
        import clickhouse_connect
    except ImportError as exc:
        print("ERROR: clickhouse-connect not installed.", file=sys.stderr)
        raise exc

    host = os.environ.get("ANOMALY_SCORES_CH_HOST", "clickhouse")
    port = int(os.environ.get("ANOMALY_SCORES_CH_PORT", "8123"))
    user = os.environ.get("ANOMALY_SCORES_CH_USER", "default")
    password = os.environ.get("ANOMALY_SCORES_CH_PASSWORD", "")
    database = os.environ.get("ANOMALY_SCORES_CH_DATABASE", "pbp_productionDB_optimized")
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


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SHAD-02: shadow scoring metrics "
            "(alert rate, top-5% bias, off-hours, Jaccard@100)."
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Window in days to query (default: {DEFAULT_DAYS}).",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        default=False,
        help="Also dump metrics as JSON to stdout after the report.",
    )
    args = parser.parse_args(argv)

    ch_client = _build_ch_client()

    print(f"\n=== SHAD-02 Shadow Monitor (last {args.days} days) ===\n")

    df = load_shadow_df(ch_client, days=args.days)
    n_old = len(df[df["scoring_mode"] == SHADOW_OLD])
    n_new = len(df[df["scoring_mode"] == SHADOW_NEW])
    print(f"Rows loaded:  shadow_old={n_old}  shadow_new={n_new}  total={len(df)}")

    if df.empty:
        print("\nNo shadow data available yet.  Run the dual-run batch scorer first.")
        return

    # SHAD-02-A
    print("\n--- SHAD-02-A: Alert rate by segment (currency × scoring_mode) ---")
    seg = compute_alert_rate_by_segment(df)
    if seg.empty:
        print("  (no data)")
    else:
        print(seg.to_string(index=False))

    # SHAD-02-B
    print("\n--- SHAD-02-B: Top-5% amount bias (winsorized p99.9) ---")
    bias_old = compute_top5_bias(df, SHADOW_OLD)
    bias_new = compute_top5_bias(df, SHADOW_NEW)
    for b in (bias_old, bias_new):
        ratio_str = f"{b['top5_wins_ratio']:.3f}x" if not np.isnan(b["top5_wins_ratio"]) else "N/A"
        p999_str = f"{b['p999']:.2f}" if not np.isnan(b["p999"]) else "N/A"
        print(f"  {b['model']:<12}  top5_wins_ratio={ratio_str}  p99.9={p999_str}  n={b['n']}")
    if not np.isnan(bias_old["top5_wins_ratio"]) and not np.isnan(bias_new["top5_wins_ratio"]):
        gate_label = "PASS" if bias_new["top5_wins_ratio"] < 4.0 else "FAIL"
        print(f"  gate (<4x for shadow_new): {gate_label}  [baseline: 11.79x]")

    # SHAD-02-C
    print("\n--- SHAD-02-C: Off-hours UTC approximation + tz_missing rate ---")
    oh_old = compute_off_hours(df, SHADOW_OLD)
    oh_new = compute_off_hours(df, SHADOW_NEW)
    for oh in (oh_old, oh_new):
        if not np.isnan(oh["off_hours_utc_rate"]):
            rate_str = f"{oh['off_hours_utc_rate']*100:.2f}%"
        else:
            rate_str = "N/A"
        tz_str = f"{oh['tz_missing_rate']*100:.2f}%"
        print(f"  {oh['model']:<12}  off_hours_utc={rate_str}  tz_missing={tz_str}  n={oh['n']}")

    # SHAD-02-D
    print("\n--- SHAD-02-D: Jaccard@100 (top-100 payment overlap) ---")
    j = compute_jaccard_at_k(df, k=100)
    print(f"  Jaccard@100 = {j:.4f}")

    metrics = {
        "days": args.days,
        "n_shadow_old": n_old,
        "n_shadow_new": n_new,
        "top5_bias_old": bias_old,
        "top5_bias_new": bias_new,
        "off_hours_old": oh_old,
        "off_hours_new": oh_new,
        "jaccard_at_100": j,
    }

    if args.output_json:
        print("\n--- JSON dump ---")
        print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
