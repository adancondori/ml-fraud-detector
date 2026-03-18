#!/usr/bin/env python3
"""
Extrae el dataset completo de ClickHouse y lo guarda localmente como parquet.
Splits temporales definidos en .env:
  - warm_start : 2024-12-01 → 2025-01-01  (~887K filas)
  - train      : 2025-01-01 → 2025-07-01  (~6.2M filas)
  - validation : 2025-07-01 → 2025-09-01  (~2.1M filas)
  - test       : 2025-09-01 → 2026-01-01  (~4.7M filas)

Uso:
    cd ml-fraud-detector
    source venv/bin/activate
    python scripts/extract_full_dataset.py [--split all|train|validation|test|warm_start]
"""
import argparse
import sys
from pathlib import Path
from time import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from config.config import settings
from fraud_detector.data.clickhouse_connector import ClickHouseConnector, FraudDataExtractor
from fraud_detector.utils.logger import logger


SPLITS = {
    "warm_start": ("2024-12-01", "2025-01-01"),
    "train":      ("2025-01-01", "2025-07-01"),
    "validation": ("2025-07-01", "2025-09-01"),
    "test":       ("2025-09-01", "2026-01-01"),
}

CHUNK_SIZE = 200_000  # filas por chunk (ajustar si hay problemas de memoria)


def extract_split(extractor: FraudDataExtractor, name: str, start: str, end: str, out_dir: Path):
    out_path = out_dir / f"{name}.parquet"

    if out_path.exists():
        import pandas as pd
        existing = pd.read_parquet(out_path, columns=["id"])
        logger.info(f"[{name}] Ya existe con {len(existing):,} filas → omitiendo. "
                    f"Elimina {out_path.name} para re-extraer.")
        return len(existing)

    logger.info(f"[{name}] Extrayendo {start} → {end} en chunks de {CHUNK_SIZE:,}...")
    t0 = time()

    chunks = list(extractor.extract_period(start, end, chunked=True, chunk_size=CHUNK_SIZE))

    if not chunks:
        logger.warning(f"[{name}] Sin datos para el período.")
        return 0

    import pandas as pd
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(out_path, index=False, compression="snappy")

    elapsed = time() - t0
    anomaly_pct = df["is_fraud"].mean() * 100
    size_mb = out_path.stat().st_size / 1_048_576

    logger.success(
        f"[{name}] {len(df):,} filas | "
        f"{anomaly_pct:.2f}% anomalías | "
        f"{size_mb:.1f} MB | "
        f"{elapsed:.0f}s → {out_path}"
    )
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Extrae dataset de ClickHouse a parquet local.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all"] + list(SPLITS.keys()),
        help="Qué split extraer (default: all)",
    )
    args = parser.parse_args()

    out_dir = project_root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_secure = settings.clickhouse_secure or settings.clickhouse_port == 8443
    connector = ClickHouseConnector(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
        secure=use_secure,
    )

    splits_to_run = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}

    print("=" * 70)
    print("EXTRACCIÓN DATASET — Anomaly Detection Thesis")
    print(f"Destino: {out_dir}")
    print("=" * 70)

    total_rows = 0
    with connector:
        extractor = FraudDataExtractor(
            connector,
            database=settings.clickhouse_database,
            table=settings.clickhouse_table,
        )
        for name, (start, end) in splits_to_run.items():
            total_rows += extract_split(extractor, name, start, end, out_dir)

    print("=" * 70)
    print(f"LISTO. Total filas extraídas: {total_rows:,}")
    print(f"Archivos en: {out_dir}/")
    for p in sorted(out_dir.glob("*.parquet")):
        size_mb = p.stat().st_size / 1_048_576
        print(f"  {p.name:30s}  {size_mb:7.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
