"""Offline script to build and materialize the facility stats artifact.

frame-normalization-v1 (design D6): iana_tz proviene de la columna replicada
``facilities.tzinfo_identifier`` en ClickHouse (misma fuente que el payload de
Rails). El diccionario Rails->IANA fue retirado.

Reads:
  data/processed/train_features_enriched.parquet  (universo canónico, ya filtrado
      por el loader con FINAL + _peerdb_is_deleted=0 + NOT IN ('reversal','free')
      + user_id != 0)
  output/revision/facility_iana.parquet           (snapshot {facility_id,
      tzinfo_identifier}; se refresca desde ClickHouse con --fetch-iana)

Produces:
  output/models/facility_stats_v1.json (o la ruta pasada en --out para
  artefactos candidatos, p. ej. output/models/candidates/...)

Run from the project root with the project venv:
  ./venv/bin/python scripts/build_facility_stats.py [--fetch-iana] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
TRAIN_PARQUET = ROOT / "data" / "processed" / "train_features_enriched.parquet"
IANA_PARQUET = ROOT / "output" / "revision" / "facility_iana.parquet"
DEFAULT_OUT_JSON = ROOT / "output" / "models" / "facility_stats_v1.json"

# Ventana canónica (config/config.py: train = [2025-01-01, 2025-07-01);
# shadow/evaluación inicia con el split de val el 2025-07-01).
DEFAULT_WINDOW_START = "2025-01-01"
DEFAULT_WINDOW_END = "2025-06-30"
DEFAULT_SHADOW_START = "2025-07-01"

# Ensure src/ is importable when running directly (without pip install -e).
sys.path.insert(0, str(ROOT / "src"))

_IANA_SQL = """
SELECT id AS facility_id, tzinfo_identifier
FROM {database}.facilities FINAL
WHERE _peerdb_is_deleted = 0
ORDER BY id
"""


def fetch_iana_snapshot() -> pd.DataFrame:
    """Read {facility_id, tzinfo_identifier} from ClickHouse (READ prod) and
    persist the snapshot parquet for reproducibility."""
    from config.config import Settings
    from fraud_detector.data.clickhouse_connector import ClickHouseConnector

    settings = Settings()
    with ClickHouseConnector(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
        secure=settings.clickhouse_secure,
    ) as ch:
        df = ch.query_to_dataframe(_IANA_SQL.format(database=settings.clickhouse_database))
    df["facility_id"] = df["facility_id"].astype(int)
    df["tzinfo_identifier"] = df["tzinfo_identifier"].astype(str)
    IANA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(IANA_PARQUET, index=False)
    print(f"  snapshot escrito: {IANA_PARQUET} ({len(df):,} facilities)")
    return df


def main() -> None:
    from fraud_detector.stats.builder import FacilityStatsBuilder
    from fraud_detector.stats.validator import validate_universe_filter

    parser = argparse.ArgumentParser(description="Build facility_stats artifact")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument(
        "--fetch-iana",
        action="store_true",
        help="Refrescar facility_iana.parquet desde ClickHouse antes de construir",
    )
    parser.add_argument("--stats-window-start", default=DEFAULT_WINDOW_START)
    parser.add_argument("--stats-window-end", default=DEFAULT_WINDOW_END)
    parser.add_argument("--shadow-period-start", default=DEFAULT_SHADOW_START)
    args = parser.parse_args()

    print(f"Reading train parquet: {TRAIN_PARQUET}")
    train_df = pd.read_parquet(
        TRAIN_PARQUET, columns=["amount", "facility_id", "currency", "created_at"]
    )
    print(f"  train rows: {len(train_df):,}")

    if args.fetch_iana:
        print("Fetching facilities.tzinfo_identifier from ClickHouse (READ prod)...")
        iana_df = fetch_iana_snapshot()
    else:
        print(f"Reading iana snapshot parquet: {IANA_PARQUET}")
        iana_df = pd.read_parquet(IANA_PARQUET)
    print(f"  facilities in snapshot: {iana_df['facility_id'].nunique():,}")

    # {facility_id -> tzinfo_identifier} — columna replicada tal cual.
    iana_map: dict[int, str] = {
        int(row["facility_id"]): str(row["tzinfo_identifier"]) for _, row in iana_df.iterrows()
    }

    # Dominant currency per facility (mode of currency column in train)
    fid_currency: dict[int, str] = (
        train_df.groupby("facility_id")["currency"].agg(lambda s: s.mode().iloc[0]).to_dict()
    )
    fid_currency = {int(k): str(v) for k, v in fid_currency.items()}

    print("Building facility stats artifact...")
    stats = FacilityStatsBuilder().build(
        train_df,
        iana_map,
        fid_currency,
        stats_window_start=args.stats_window_start,
        stats_window_end=args.stats_window_end,
        shadow_period_start=args.shadow_period_start,
    )

    print("Validating artifact before writing...")
    validate_universe_filter(stats, train_df, iana_df)
    print("  Validation passed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Written: {args.out}")

    # --- Summary ---
    facilities = stats["facilities"]
    n_total = len(facilities)
    n_facility = sum(1 for e in facilities.values() if e["fallback_level"] == "facility")
    n_currency = sum(1 for e in facilities.values() if e["fallback_level"] == "currency")
    n_global = sum(1 for e in facilities.values() if e["fallback_level"] == "global")
    n_iqr_zero = sum(1 for e in facilities.values() if e.get("iqr") == 0.0)
    n_iqr_guarded = sum(1 for e in facilities.values() if e.get("iqr_guarded") == 1.0)
    n_null_iana = sum(1 for e in facilities.values() if not e.get("iana_tz"))

    print()
    print("=== Artifact Summary ===")
    print(f"  schema_version : {stats['schema_version']}")
    print(f"  built_at       : {stats['built_at']}")
    print(f"  universe_filter: {stats['universe_filter']}")
    print(f"  stats_window   : {stats['stats_window_start']} .. {stats['stats_window_end']}")
    print(f"  shadow_start   : {stats['shadow_period_start']}")
    print(f"  amount_source  : {stats['amount_source']}")
    print(f"  train_rows     : {stats['train_rows']:,}")
    print(f"  n_facilities   : {n_total:,}")
    print(f"  fallback=facility : {n_facility:,}")
    print(f"  fallback=currency : {n_currency:,}")
    print(f"  fallback=global   : {n_global:,}")
    print(f"  iqr=0.0           : {n_iqr_zero:,}")
    print(f"  iqr_guarded=1.0   : {n_iqr_guarded:,}")
    print(f"  iana_tz nulo      : {n_null_iana:,}")
    print(f"  currency_fallbacks: {list(stats['currency_fallbacks'].keys())}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
