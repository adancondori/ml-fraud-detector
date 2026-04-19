"""
verify_counts.py — Gate A: verifica conteos del snapshot contra ClickHouse.

Compara los Parquets locales (o consulta ClickHouse directamente) contra
los valores esperados del plan. Falla con sys.exit(1) si algún criterio
supera la tolerancia del ±1%.

Uso:
    python scripts/verify_counts.py              # Verifica Parquets locales
    python scripts/verify_counts.py --live       # Consulta ClickHouse en tiempo real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Bootstrap path ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from config.config import settings

# ── Valores objetivo (del plan maestro, validados 2026-03-17) ────────────────

EXPECTED = {
    "train": 3_137_086,
    "val":   1_130_118,
    "test":  2_517_491,
    "warm":    419_820,
}
EXPECTED_TOTAL = 6_784_695
EXPECTED_PROXY_STRICT_RATE = 0.0633   # 6.33%
EXPECTED_PROXY_WIDE_RATE   = 0.0755   # 7.55%
EXPECTED_PROXY_UNIFIED_RATE = 0.1023  # 10.23% (OR of types A+C+D; B=0, E=0)
TOLERANCE = 0.01  # ±1%

STRICT_STATUSES = {"totally_refunded", "refunded_to_credit"}
WIDE_STATUSES   = {"totally_refunded", "refunded_to_credit", "partially_refunded"}


def _pct_diff(actual: int, expected: int) -> float:
    return abs(actual - expected) / expected


def _check(name: str, actual: int, expected: int) -> bool:
    diff = _pct_diff(actual, expected)
    ok = diff <= TOLERANCE
    symbol = "✓" if ok else "✗"
    logger.info(
        f"  {symbol} {name:10s}: actual={actual:>10,}  expected={expected:>10,}  diff={diff*100:.2f}%"
    )
    return ok


# ── Verificación desde Parquets locales ──────────────────────────────────────

def verify_from_parquets() -> bool:
    logger.info("═" * 60)
    logger.info("Gate A — Verificación desde Parquets locales")
    logger.info("═" * 60)

    all_ok = True
    total_rows = 0
    proxy_strict_sum = 0
    proxy_wide_sum   = 0

    for name, expected in EXPECTED.items():
        path = settings.processed_dir / f"{name}_raw.parquet"
        if not path.exists():
            logger.error(f"  ✗ {name}: Parquet no encontrado en {path}")
            all_ok = False
            continue

        df = pd.read_parquet(path, engine="pyarrow")
        n = len(df)
        ok = _check(name, n, expected)
        all_ok = all_ok and ok

        if name != "warm":
            total_rows += n
            proxy_strict_sum += df["status"].isin(STRICT_STATUSES).sum()
            proxy_wide_sum   += df["status"].isin(WIDE_STATUSES).sum()

        # Gate A checks per split
        _run_per_split_checks(df, name)

    # Total universe (train+val+test)
    logger.info("")
    ok = _check("total", total_rows, EXPECTED_TOTAL)
    all_ok = all_ok and ok

    # Proxy rates (over train+val+test universe)
    if total_rows > 0:
        strict_rate = proxy_strict_sum / total_rows
        wide_rate   = proxy_wide_sum   / total_rows
        strict_ok = abs(strict_rate - EXPECTED_PROXY_STRICT_RATE) <= TOLERANCE
        wide_ok   = abs(wide_rate   - EXPECTED_PROXY_WIDE_RATE)   <= TOLERANCE
        logger.info(f"  {'✓' if strict_ok else '✗'} proxy_strict : {proxy_strict_sum:>10,}  rate={strict_rate*100:.2f}%  (expected ~6.33%)")
        logger.info(f"  {'✓' if wide_ok   else '✗'} proxy_wide   : {proxy_wide_sum:>10,}  rate={wide_rate*100:.2f}%  (expected ~7.55%)")
        all_ok = all_ok and strict_ok and wide_ok

    # Proxy unificado (requires feature parquets for tipo_c and tipo_d)
    from fraud_detector.data.loader import DataManager

    unified_sum = 0
    feature_available = True
    for name in ("train", "val", "test"):
        feat_path = settings.processed_dir / f"{name}_features.parquet"
        if not feat_path.exists():
            logger.warning(f"  ⚠ Feature parquet not found for {name}; skipping proxy unificado check")
            feature_available = False
            break
        df_feat = pd.read_parquet(feat_path, engine="pyarrow")
        unified_sum += int(DataManager.assign_proxy_labels(df_feat, "unified").sum())
        del df_feat

    if feature_available and total_rows > 0:
        unified_rate = unified_sum / total_rows
        unified_ok = abs(unified_rate - EXPECTED_PROXY_UNIFIED_RATE) <= TOLERANCE
        logger.info(
            f"  {'✓' if unified_ok else '✗'} proxy_unified: {unified_sum:>10,}  "
            f"rate={unified_rate*100:.2f}%  (expected ~10.23%)"
        )
        all_ok = all_ok and unified_ok

    logger.info("")
    logger.info(f"Gate A: {'PASS ✓' if all_ok else 'FAIL ✗'}")
    return all_ok


def _run_per_split_checks(df: pd.DataFrame, name: str) -> None:
    """10 criterios de validación del Gate A."""
    issues = []

    # 1. Columnas requeridas no nulas
    for col in ["id", "user_id", "facility_id", "amount", "created_at", "status"]:
        if col not in df.columns:
            issues.append(f"{col} falta en columnas")
        elif df[col].isna().sum() > 0:
            issues.append(f"{col} tiene {df[col].isna().sum()} NULLs")

    # 2. user_id > 0
    if "user_id" in df.columns and (df["user_id"] <= 0).any():
        issues.append(f"user_id <= 0: {(df['user_id'] <= 0).sum()} filas")

    # 3. Sin duplicados en id
    dup = df["id"].duplicated().sum() if "id" in df.columns else 0
    if dup > 0:
        issues.append(f"id duplicados: {dup}")

    # 4. is_fraud no debe estar presente
    if "is_fraud" in df.columns:
        issues.append("columna is_fraud presente (debe eliminarse)")

    # 5. Rango temporal
    if "created_at" in df.columns:
        min_ts, max_ts = df["created_at"].min(), df["created_at"].max()
        logger.debug(f"    {name}: created_at [{min_ts} .. {max_ts}]")

    if issues:
        for issue in issues:
            logger.warning(f"    ⚠ {name}: {issue}")
    else:
        logger.debug(f"    {name}: todos los checks OK")


# ── Verificación en vivo contra ClickHouse ───────────────────────────────────

def verify_live() -> bool:
    logger.info("═" * 60)
    logger.info("Gate A — Verificación en vivo contra ClickHouse")
    logger.info("═" * 60)

    import clickhouse_connect
    client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        secure=settings.clickhouse_secure,
    )

    splits = {
        "warm":  (settings.warm_start,  settings.train_start),
        "train": (settings.train_start, settings.train_end),
        "val":   (settings.train_end,   settings.val_end),
        "test":  (settings.val_end,     settings.test_end),
    }

    all_ok = True
    total_rows = 0
    proxy_strict_sum = 0
    proxy_wide_sum   = 0

    db    = settings.clickhouse_database
    table = settings.clickhouse_table

    for name, (start, end) in splits.items():
        sql = f"""
        SELECT
            count() AS n,
            countIf(status IN ('totally_refunded','refunded_to_credit')) AS proxy_strict,
            countIf(status IN ('totally_refunded','refunded_to_credit','partially_refunded')) AS proxy_wide
        FROM {db}.{table} FINAL
        WHERE created_at >= '{start}'
          AND created_at < '{end}'
          AND payment_method != 'reversal'
          AND payment_method != 'free'
          AND user_id != 0
          AND _peerdb_is_deleted = 0
        """
        row = client.query(sql).first_row
        n, strict, wide = int(row[0]), int(row[1]), int(row[2])
        ok = _check(name, n, EXPECTED.get(name, n))
        all_ok = all_ok and ok

        if name != "warm":
            total_rows += n
            proxy_strict_sum += strict
            proxy_wide_sum   += wide

    client.close()

    logger.info("")
    ok = _check("total", total_rows, EXPECTED_TOTAL)
    all_ok = all_ok and ok

    if total_rows > 0:
        strict_rate = proxy_strict_sum / total_rows
        wide_rate   = proxy_wide_sum   / total_rows
        strict_ok = abs(strict_rate - EXPECTED_PROXY_STRICT_RATE) <= TOLERANCE
        wide_ok   = abs(wide_rate   - EXPECTED_PROXY_WIDE_RATE)   <= TOLERANCE
        logger.info(f"  {'✓' if strict_ok else '✗'} proxy_strict : {proxy_strict_sum:>10,}  rate={strict_rate*100:.2f}%  (expected ~6.33%)")
        logger.info(f"  {'✓' if wide_ok   else '✗'} proxy_wide   : {proxy_wide_sum:>10,}  rate={wide_rate*100:.2f}%  (expected ~7.55%)")
        all_ok = all_ok and strict_ok and wide_ok

    logger.info("")
    logger.info(f"Gate A: {'PASS ✓' if all_ok else 'FAIL ✗'}")
    return all_ok


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Gate A verification script")
    parser.add_argument("--live", action="store_true", help="Query ClickHouse directly")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="DEBUG")

    ok = verify_live() if args.live else verify_from_parquets()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
