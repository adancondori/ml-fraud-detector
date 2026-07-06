"""hitl_queue_builder.py — HITL-01 (scorer side): parametrized HITL queue export.

Operational exporter that assembles the HITL review queue from the live
ClickHouse WRITE target (scoring_mode='shadow_new').  Produces a CSV or JSON
file with top-k anomalies ordered by percentile DESC plus a defensive
below-p50 sample (>=20% of capacity) for false-negative estimation.

**This script is OPERATIONAL infrastructure** — it reads from anomaly_scores
(ClickHouse WRITE local, ANOMALY_SCORES_CH_* env vars) and writes a flat
file for the human review team.

It is intentionally **separate** from the thesis offline pipeline
(hitl_export_alerts.py / hitl_ingest_labels.py), which operates on
pre-extracted Parquet splits.  Both pipelines coexist without overlap.

Queue composition (replicates HitlQueueQuery logic from 05-01):
  - If --capacity is given:
      top_k_count = floor(capacity * (1 - below_p50_pct))
      below_k_count = capacity - top_k_count
  - Otherwise (absolute mode):
      top_k_count = top_k
      below_k_count = max(1, ceil(top_k * below_p50_pct))

Usage:
    python scripts/hitl_queue_builder.py \\
        --top-k 100 --below-p50-pct 0.20 --capacity 50 \\
        --output output/hitl_queue_2026-07-06.csv

Environment variables:
    ANOMALY_SCORES_CH_HOST      (default: clickhouse)
    ANOMALY_SCORES_CH_PORT      (default: 8123)
    ANOMALY_SCORES_CH_USER      (default: default)
    ANOMALY_SCORES_CH_PASSWORD  (default: "")
    ANOMALY_SCORES_CH_DATABASE  (default: pbp_productionDB_optimized)
    ANOMALY_SCORES_CH_SECURE    (default: false)
    ANOMALY_SCORES_CH_TABLE     (default: pbp_productionDB_optimized.anomaly_scores)
    HITL_TOP_K                  (default: 100)  — CLI overrides env
    HITL_BELOW_P50_PCT          (default: 0.20) — CLI overrides env
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_TABLE = "pbp_productionDB_optimized.anomaly_scores"

# Columns fetched from anomaly_scores for every queue row
_SELECT_COLS = (
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
)


# ---------------------------------------------------------------------------
# ClickHouse client (WRITE local — same pattern as shadow_gate.py)
# ---------------------------------------------------------------------------


def _build_ch_client():
    """Build ClickHouse client for the WRITE local target (ANOMALY_SCORES_CH_*).

    Credentials are read exclusively from environment variables — never
    hardcoded.  The password defaults to an empty string (no-auth dev env).
    """
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
# p50 resolver
# ---------------------------------------------------------------------------


def resolve_p50(ch_client, table: str = _DEFAULT_TABLE) -> float:
    """Query the median percentile among shadow_new rows.

    Args:
        ch_client: ClickHouse client (WRITE local).
        table: Full table name.

    Returns:
        Median percentile as a float in [0, 1].  Falls back to 0.5 when
        shadow_new has no rows yet (SHAD-03 PENDING_DATA phase).
    """
    sql = (
        f"SELECT quantile(0.5)(percentile) AS p50 "
        f"FROM {table} "
        f"WHERE scoring_mode = 'shadow_new'"
    )
    result = ch_client.query(sql)
    if not result.result_rows:
        return 0.5
    val = result.result_rows[0][0]
    if val is None:
        return 0.5
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Queue count computation (pure, replicates HitlQueueQuery — 05-01)
# ---------------------------------------------------------------------------


def compute_counts(
    top_k: int,
    below_p50_pct: float,
    capacity: Optional[int] = None,
) -> tuple[int, int]:
    """Compute top-k and below-p50 row counts for the HITL queue.

    Replicates the reparto logic from HitlQueueQuery (Ruby, 05-01):
      - With capacity:
          top_k_count  = floor(capacity * (1 - below_p50_pct))
          below_k_count = capacity - top_k_count
      - Without capacity (absolute mode):
          top_k_count  = top_k
          below_k_count = max(1, ceil(top_k * below_p50_pct))

    Args:
        top_k: Maximum top-k rows (used in absolute mode).
        below_p50_pct: Fraction of capacity/top_k reserved for below-p50.
        capacity: Total review capacity.  None = absolute mode.

    Returns:
        (top_k_count, below_k_count) as a tuple of non-negative ints.
    """
    if capacity is not None:
        top_k_count = math.floor(capacity * (1.0 - below_p50_pct))
        below_k_count = capacity - top_k_count
    else:
        top_k_count = top_k
        below_k_count = max(1, math.ceil(top_k * below_p50_pct))
    return int(top_k_count), int(below_k_count)


# ---------------------------------------------------------------------------
# Queue builder
# ---------------------------------------------------------------------------


def build_hitl_queue(
    ch_client,
    top_k: int = 100,
    below_p50_pct: float = 0.20,
    capacity: Optional[int] = None,
    table: str = _DEFAULT_TABLE,
) -> pd.DataFrame:
    """Build the HITL review queue DataFrame from ClickHouse shadow_new rows.

    Assembles:
      1. top_k_count rows with the highest percentile (ORDER BY percentile DESC).
      2. below_k_count rows sampled pseudo-randomly from below the median
         percentile (ORDER BY rand()), for false-negative coverage.

    Each row is tagged with `hitl_queue_source` ('top_k' | 'below_p50').

    Args:
        ch_client: ClickHouse client (WRITE local, ANOMALY_SCORES_CH_*).
        top_k: Maximum top-k rows for absolute mode (ignored when capacity
            is set).
        below_p50_pct: Fraction of the queue reserved for below-p50 sampling.
            Must be in (0, 1).  Minimum recommended: 0.20 (see
            docs/hitl_false_negative_methodology.md).
        capacity: Optional total review capacity (rows).  When provided,
            overrides top_k for the top-k side of the queue.
        table: Full ClickHouse table name.

    Returns:
        DataFrame with columns from _SELECT_COLS plus `hitl_queue_source`.
        May be empty if shadow_new has no rows yet.
    """
    top_k_count, below_k_count = compute_counts(top_k, below_p50_pct, capacity)
    p50 = resolve_p50(ch_client, table=table)

    cols_sql = ", ".join(_SELECT_COLS)

    # --- Top-k query ---
    sql_top = (
        f"SELECT {cols_sql} "
        f"FROM {table} "
        f"WHERE scoring_mode = 'shadow_new' "
        f"ORDER BY percentile DESC "
        f"LIMIT {top_k_count}"
    )
    result_top = ch_client.query(sql_top)
    if result_top.result_rows:
        df_top = pd.DataFrame(result_top.result_rows, columns=result_top.column_names)
    else:
        df_top = pd.DataFrame(columns=list(_SELECT_COLS))

    # --- Below-p50 query ---
    sql_below = (
        f"SELECT {cols_sql} "
        f"FROM {table} "
        f"WHERE scoring_mode = 'shadow_new' "
        f"  AND percentile < {p50} "
        f"ORDER BY rand() "
        f"LIMIT {below_k_count}"
    )
    result_below = ch_client.query(sql_below)
    if result_below.result_rows:
        df_below = pd.DataFrame(
            result_below.result_rows, columns=result_below.column_names
        )
    else:
        df_below = pd.DataFrame(columns=list(_SELECT_COLS))

    df_top = df_top.copy()
    df_below = df_below.copy()
    df_top["hitl_queue_source"] = "top_k"
    df_below["hitl_queue_source"] = "below_p50"

    return pd.concat([df_top, df_below], ignore_index=True)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def write_output(df: pd.DataFrame, path: str) -> None:
    """Write the queue DataFrame to a file, inferring format from extension.

    Supports .csv and .json.  Creates parent directories as needed.

    Args:
        df: HITL queue DataFrame.
        path: Output file path.  Extension must be .csv or .json.

    Raises:
        ValueError: If the extension is not supported.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    if ext == ".csv":
        df.to_csv(out, index=False)
    elif ext == ".json":
        df.to_json(out, orient="records", indent=2)
    else:
        raise ValueError(
            f"Unsupported output extension '{ext}'. Use .csv or .json."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "HITL-01 (scorer): export the HITL review queue from ClickHouse "
            "shadow_new rows.  Produces top-k rows ordered by percentile DESC "
            "plus a below-p50 sample (>=20%% of capacity) for false-negative "
            "coverage.  See docs/hitl_false_negative_methodology.md."
        )
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("HITL_TOP_K", "100")),
        help="Maximum top-k rows for absolute mode (env: HITL_TOP_K, default: 100).",
    )
    parser.add_argument(
        "--below-p50-pct",
        type=float,
        default=float(os.environ.get("HITL_BELOW_P50_PCT", "0.20")),
        help=(
            "Fraction of capacity/top-k reserved for below-p50 sampling "
            "(env: HITL_BELOW_P50_PCT, default: 0.20).  Minimum recommended: 0.20."
        ),
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help=(
            "Total review capacity (rows).  When set, overrides --top-k for "
            "the top-k side and distributes rows as "
            "floor(capacity*(1-pct)) top-k + remainder below-p50."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output file path (.csv or .json).  Defaults to "
            "output/hitl_queue_<timestamp>.csv."
        ),
    )

    args = parser.parse_args(argv)

    # Validate below_p50_pct
    if not (0.0 < args.below_p50_pct < 1.0):
        print(
            f"ERROR: --below-p50-pct must be in (0, 1), got {args.below_p50_pct}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve output path
    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = str(
            Path(__file__).resolve().parent.parent / "output" / f"hitl_queue_{ts}.csv"
        )
    else:
        output_path = args.output

    ch_client = _build_ch_client()

    top_k_count, below_k_count = compute_counts(
        args.top_k, args.below_p50_pct, args.capacity
    )
    p50 = resolve_p50(ch_client)

    print(f"\n=== HITL-01 Queue Builder ===\n")
    print(f"  top_k          : {args.top_k}")
    print(f"  below_p50_pct  : {args.below_p50_pct:.0%}")
    print(f"  capacity       : {args.capacity if args.capacity is not None else '(absolute)'}")
    print(f"  -> top_k_count : {top_k_count}")
    print(f"  -> below_count : {below_k_count}")
    print(f"  p50 (shadow_new): {p50:.4f}")

    df = build_hitl_queue(
        ch_client,
        top_k=args.top_k,
        below_p50_pct=args.below_p50_pct,
        capacity=args.capacity,
    )

    n_top = int((df["hitl_queue_source"] == "top_k").sum())
    n_below = int((df["hitl_queue_source"] == "below_p50").sum())
    print(f"\n  Rows exported  : {len(df)}")
    print(f"  top_k rows     : {n_top}")
    print(f"  below_p50 rows : {n_below}")

    write_output(df, output_path)
    print(f"\n  Output         : {output_path}\n")


if __name__ == "__main__":
    main()
