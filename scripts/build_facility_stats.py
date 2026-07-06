"""Offline script to build and materialize output/models/facility_stats_v1.json.

Reads:
  data/processed/train_features_enriched.parquet (scorer universe, already filtered)
  output/revision/facility_tz.parquet            (1876 facilities, 64 Rails tz names)

Produces:
  output/models/facility_stats_v1.json           (facility-stats-v1 artifact)

Run from the project root with the project venv:
  ./venv/bin/python scripts/build_facility_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
TRAIN_PARQUET = ROOT / "data" / "processed" / "train_features_enriched.parquet"
TZ_PARQUET = ROOT / "output" / "revision" / "facility_tz.parquet"
OUT_JSON = ROOT / "output" / "models" / "facility_stats_v1.json"

# Ensure src/ is importable when running directly (without pip install -e).
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from fraud_detector.stats.builder import FacilityStatsBuilder
    from fraud_detector.stats.validator import validate_universe_filter

    print(f"Reading train parquet: {TRAIN_PARQUET}")
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=["amount", "facility_id", "currency"])
    print(f"  train rows: {len(train_df):,}")

    print(f"Reading facility_tz parquet: {TZ_PARQUET}")
    tz_df = pd.read_parquet(TZ_PARQUET)
    print(f"  facilities in tz parquet: {tz_df['facility_id'].nunique():,}")

    # Build lookup dicts expected by FacilityStatsBuilder.build()
    tz_map: dict[int, str] = {
        int(row["facility_id"]): str(row["time_zone"])
        for _, row in tz_df.iterrows()
    }

    # Dominant currency per facility (mode of currency column in train)
    fid_currency: dict[int, str] = (
        train_df.groupby("facility_id")["currency"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )
    # Convert keys to int (groupby may return numpy int types)
    fid_currency = {int(k): str(v) for k, v in fid_currency.items()}

    print("Building facility stats artifact...")
    stats = FacilityStatsBuilder().build(train_df, tz_map, fid_currency)

    print("Validating artifact before writing...")
    validate_universe_filter(stats, train_df, tz_df)
    print("  Validation passed.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Written: {OUT_JSON}")

    # --- Summary ---
    facilities = stats["facilities"]
    n_total = len(facilities)
    n_facility = sum(1 for e in facilities.values() if e["fallback_level"] == "facility")
    n_currency = sum(1 for e in facilities.values() if e["fallback_level"] == "currency")
    n_global = sum(1 for e in facilities.values() if e["fallback_level"] == "global")
    n_iqr_zero = sum(1 for e in facilities.values() if e.get("iqr") == 0.0)
    n_iqr_guarded = sum(1 for e in facilities.values() if e.get("iqr_guarded") == 1.0)

    print()
    print("=== Artifact Summary ===")
    print(f"  schema_version : {stats['schema_version']}")
    print(f"  built_at       : {stats['built_at']}")
    print(f"  train_rows     : {stats['train_rows']:,}")
    print(f"  n_facilities   : {n_total:,}  (expected 1876)")
    print(f"  fallback=facility : {n_facility:,}  (expected ~689)")
    print(f"  fallback=currency : {n_currency:,}")
    print(f"  fallback=global   : {n_global:,}")
    print(f"  iqr=0.0           : {n_iqr_zero:,}  (expected ~116)")
    print(f"  iqr_guarded=1.0   : {n_iqr_guarded:,}  (includes iqr=0 and cold-start facilities)")
    print(f"  currency_fallbacks: {list(stats['currency_fallbacks'].keys())}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
