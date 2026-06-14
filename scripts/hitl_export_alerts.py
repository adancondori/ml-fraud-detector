#!/usr/bin/env python
"""Export top-K scored alerts with operative context for HITL human review.

Implements the export step of the Human-in-the-Loop (HITL) protocol described in
the thesis Chapter 3 (`sec:sistema-entregable-hitl`). Produces a CSV with the
schema documented in `tab:hitl-csv-esquema`.

Pipeline:
  1. Load features-enriched parquet for the requested test period.
  2. Score every row with the final IF-40 model via `PaymentScorer`.
  3. Filter top-K by percentile of score (`--top-pct`).
  4. Join with the raw parquet to bring `gateway`, `payment_method`,
     `source_enum`, `card_brand` (columns absent from the enriched parquet).
  5. Compute the two proxy columns (`proxy_tipo_a`, `proxy_anomalias_operativas`).
  6. Add empty `comentario_revisor` and `categoria_revisor` columns.
  7. Emit CSV + a `template_revision.csv` (empty rows with the same schema).

Usage:
    python scripts/hitl_export_alerts.py --top-pct 1 --out output/hitl/alertas.csv
    python scripts/hitl_export_alerts.py --top-pct 1 --period 2025-12 \
        --out output/hitl/alertas_diciembre.csv  # filtra a diciembre 2025
    python scripts/hitl_export_alerts.py --top-pct 5 \
        --features data/processed/test_features_enriched.parquet \
        --raw      data/processed/test_raw.parquet \
        --out      output/hitl/alertas_top5.csv

The `--period` argument, when provided, filters the enriched parquet to rows whose
`created_at` falls inside that period before scoring. Accepted formats:
  YYYY        (e.g. 2025          → entire year)
  YYYY-MM     (e.g. 2025-12       → that single month)
  YYYY-MM-DD/YYYY-MM-DD (custom inclusive range)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fraud_detector.utils.logger import logger  # noqa: E402
from score_payment import PaymentScorer  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
HITL_DIR = PROJECT_ROOT / "output" / "hitl"

CONTEXT_COLS_ENRICHED = [
    "created_at",
    "user_id",
    "facility_id",
    "amount",
    "currency",
    "status",
    "user_account_age_days",
    "user_txn_count_1h",
    "same_amount_count_1h",
    "is_third_party_payment",
]
CONTEXT_COLS_RAW = ["gateway", "payment_method", "source_enum", "card_brand"]
REVIEWER_COLS = ["comentario_revisor", "categoria_revisor"]


def build_proxy_anomalias_operativas(df: pd.DataFrame) -> np.ndarray:
    """Proxy operativo estricto de anomalía (card-testing + new-user-burst + 3rd-party-burst)."""
    cols = df.columns
    card_test = (
        (df["same_amount_count_1h"] >= 3).to_numpy()
        if "same_amount_count_1h" in cols
        else np.zeros(len(df), dtype=bool)
    )
    new_burst = (
        (df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)
    ).to_numpy()
    third_party_burst = (
        (df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)
    ).to_numpy() if "is_third_party_payment" in cols else np.zeros(len(df), dtype=bool)
    return (card_test | new_burst | third_party_burst).astype(np.int8)


def build_proxy_tipo_a(df: pd.DataFrame) -> np.ndarray:
    """Proxy Tipo A: status in {totally_refunded, refunded_to_credit}."""
    return df["status"].isin(["totally_refunded", "refunded_to_credit"]).astype(np.int8).to_numpy()


def _resolve_period(period: str | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not period:
        return None
    if "/" in period:
        start_s, end_s = period.split("/", 1)
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    elif len(period) == 4:
        start = pd.Timestamp(f"{period}-01-01")
        end = pd.Timestamp(f"{int(period) + 1}-01-01") - pd.Timedelta(seconds=1)
    elif len(period) == 7:
        start = pd.Timestamp(period + "-01")
        end = (start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
    else:
        raise ValueError(
            f"Unsupported --period format: {period!r}. "
            "Accepted: YYYY, YYYY-MM, YYYY-MM-DD/YYYY-MM-DD"
        )
    return start, end


def export_alerts(
    features_path: Path,
    raw_path: Path,
    top_pct: float,
    out_path: Path,
    period: str | None = None,
) -> pd.DataFrame:
    logger.info(f"Loading enriched features from {features_path}")
    df_feat = pd.read_parquet(features_path)
    logger.info(f"  {len(df_feat):,} rows × {len(df_feat.columns)} cols")

    window = _resolve_period(period)
    if window is not None:
        start, end = window
        if "created_at" not in df_feat.columns:
            raise ValueError(
                "Cannot filter by --period: enriched parquet has no `created_at` column"
            )
        created_at = pd.to_datetime(df_feat["created_at"])
        mask = (created_at >= start) & (created_at <= end)
        df_feat = df_feat.loc[mask].reset_index(drop=True)
        logger.info(
            f"Filtered by --period {period} ({start.date()}..{end.date()}): "
            f"{len(df_feat):,} rows remain"
        )

    logger.info("Scoring with IF-40 model")
    scorer = PaymentScorer()
    df_scored = scorer.score_frame(df_feat)

    # Filter top-K by percentile
    threshold_pct = top_pct / 100.0
    mask = df_scored["percentile_rank"] <= threshold_pct
    df_top = df_scored.loc[mask].copy()
    logger.info(
        f"Selected {len(df_top):,} alerts (top {top_pct}% of {len(df_scored):,})"
    )

    # Join enriched context (id is the key)
    df_top = df_top.merge(
        df_feat[["id"] + CONTEXT_COLS_ENRICHED],
        on="id",
        how="left",
        validate="one_to_one",
    )

    # Join raw context
    logger.info(f"Joining raw context from {raw_path}")
    df_raw = pd.read_parquet(raw_path, columns=["id"] + CONTEXT_COLS_RAW)
    df_top = df_top.merge(df_raw, on="id", how="left", validate="one_to_one")

    # Computed proxies (do NOT overwrite anything in the raw parquet)
    df_top["proxy_tipo_a"] = build_proxy_tipo_a(df_top)
    df_top["proxy_anomalias_operativas"] = build_proxy_anomalias_operativas(df_top)

    # Empty reviewer columns
    for col in REVIEWER_COLS:
        df_top[col] = ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_top.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(df_top):,} alerts to {out_path}")

    # Template (empty rows) for the analyst's offline use
    template_path = out_path.parent / "template_revision.csv"
    df_top.iloc[0:0].to_csv(template_path, index=False)
    logger.info(f"Wrote template to {template_path}")

    return df_top


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-K HITL alerts CSV")
    parser.add_argument("--top-pct", type=float, default=1.0,
                        help="Percentile cut-off for the top-K (default 1%%)")
    parser.add_argument("--period", type=str, default=None,
                        help="Optional filter on created_at (YYYY, YYYY-MM, YYYY-MM-DD/YYYY-MM-DD)")
    parser.add_argument("--features", type=str,
                        default=str(DATA_DIR / "test_features_enriched.parquet"))
    parser.add_argument("--raw", type=str,
                        default=str(DATA_DIR / "test_raw.parquet"))
    parser.add_argument("--out", type=str,
                        default=str(HITL_DIR / "alertas.csv"))
    args = parser.parse_args()

    export_alerts(
        features_path=Path(args.features),
        raw_path=Path(args.raw),
        top_pct=args.top_pct,
        out_path=Path(args.out),
        period=args.period,
    )


if __name__ == "__main__":
    main()
