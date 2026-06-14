#!/usr/bin/env python
"""Ingest reviewed HITL CSVs into an append-only label registry.

Implements the ingest step of the HITL protocol (`tab:hitl-protocolo` in the
thesis). The registry is the input for any future migration to a supervised
or semi-supervised model.

Append-only contract:
  * Each ingested row is timestamped with `ingested_at` (UTC) and tagged with
    the reviewer-provided columns (`id_revisor`, `comentario_revisor`,
    `categoria_revisor`).
  * Rows are appended; no row is ever deleted from the registry. To correct a
    label, ingest a new row with `categoria_revisor='_correccion_'` and a
    pointer to the original row id in `comentario_revisor`.

This script intentionally stays as a skeleton: validation is strict and the
write is atomic, but the surrounding governance (κ inter-anotador, audit
sampling, drift dashboards) lives outside this entry point and is described
in the Recomendaciones chapter.

Usage:
    python scripts/hitl_ingest_labels.py --csv output/hitl/alertas_revisadas.csv \
                                         --reviewer-id ana.lopez
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detector.utils.logger import logger  # noqa: E402

REGISTRY_DIR = PROJECT_ROOT / "output" / "hitl"
REGISTRY_PATH = REGISTRY_DIR / "labels_acumulados.parquet"

REQUIRED_COLS = {
    "id",
    "score",
    "decile",
    "is_top_1pct",
    "is_top_5pct",
    "created_at",
    "user_id",
    "facility_id",
    "amount",
    "currency",
    "status",
    "gateway",
    "payment_method",
    "comentario_revisor",
    "categoria_revisor",
}
VALID_CATEGORIES = {
    "sospecha_fraude",
    "anomalia_operativa",
    "falso_positivo",
    "indeterminado",
    "_correccion_",  # correction marker; see module docstring
}


def validate_csv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if df["categoria_revisor"].isna().any() or (df["categoria_revisor"] == "").any():
        raise ValueError(
            "categoria_revisor must be populated on every row before ingest"
        )
    invalid = set(df["categoria_revisor"].unique()) - VALID_CATEGORIES
    if invalid:
        raise ValueError(
            f"Unknown categoria_revisor values: {sorted(invalid)}. "
            f"Allowed: {sorted(VALID_CATEGORIES)}"
        )
    if df["id"].duplicated().any():
        raise ValueError("Duplicate `id` rows in CSV; deduplicate before ingest")


def ingest(csv_path: Path, reviewer_id: str) -> Path:
    logger.info(f"Loading reviewed CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    validate_csv(df)

    df["id_revisor"] = reviewer_id
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        prior = pd.read_parquet(REGISTRY_PATH)
        combined = pd.concat([prior, df], ignore_index=True)
        logger.info(
            f"Append: {len(prior):,} prior rows + {len(df):,} new = {len(combined):,}"
        )
    else:
        combined = df
        logger.info(f"Creating new registry with {len(df):,} rows")

    # Atomic write via temp file + rename
    tmp = REGISTRY_PATH.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(REGISTRY_PATH)
    logger.info(f"Registry updated: {REGISTRY_PATH}")
    return REGISTRY_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest reviewed HITL CSV into registry")
    parser.add_argument("--csv", required=True, type=str,
                        help="Path to a reviewed CSV (output of hitl_export_alerts + manual review)")
    parser.add_argument("--reviewer-id", required=True, type=str,
                        help="Identifier of the human reviewer (e.g. 'ana.lopez')")
    args = parser.parse_args()
    ingest(Path(args.csv), args.reviewer_id)


if __name__ == "__main__":
    main()
