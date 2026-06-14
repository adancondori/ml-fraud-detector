#!/usr/bin/env python
"""Extract user_tokens info and join to 2025 payments.

Produces:
  data/processed/payment_token_features.parquet
    columns: payment_id, user_token_id, token_age_days_at_payment,
             is_new_token (< 7d), is_default_token, n_tokens_user_at_payment

Joins via:
  - ClickHouse: payments(id, user_token_id, created_at) for 2025
  - MySQL:      user_tokens(id, user_id, created_at, is_default)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import clickhouse_connect
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def fetch_payments_token_map() -> pd.DataFrame:
    """ClickHouse: id, user_id, created_at, user_token_id for warm + 2025."""
    logger.info("  Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        secure=settings.clickhouse_secure,
        connect_timeout=10,
        send_receive_timeout=60,
    )
    query = """
        SELECT id, user_id, created_at, user_token_id
        FROM pbp_productionDB_optimized.payments FINAL
        WHERE created_at >= '2024-12-01' AND created_at < '2026-01-01'
          AND payment_method NOT IN ('reversal','free')
          AND user_id != 0
          AND _peerdb_is_deleted = 0
    """
    logger.info("  Fetching payment→token mapping (this may take a minute)...")
    t0 = time.perf_counter()
    df = client.query_df(query)
    logger.info(f"    fetched {len(df):,} rows ({time.perf_counter() - t0:.1f}s)")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["user_token_id"] = df["user_token_id"].astype("Int64")
    return df.rename(columns={"id": "payment_id"})


def fetch_user_tokens() -> pd.DataFrame:
    """MySQL: id, user_id, created_at, is_default for all user_tokens."""
    logger.info("  Connecting to MySQL replica...")
    import os
    import urllib.parse
    host = os.environ.get("MYSQL_HOST", "172.31.30.217")
    port = os.environ.get("MYSQL_PORT", "3306")
    user = os.environ.get("MYSQL_USER", "ironman")
    pw = urllib.parse.quote_plus(
        os.environ.get("MYSQL_PASSWORD", "_((**k<>10dk6c9n9x267x32gm9780caph34t72")
    )
    db = os.environ.get("MYSQL_DATABASE", "paybycourtDB")
    url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}"
    # SSL config: empty dict triggers default SSL context (matches MYSQL_SSL=true)
    engine = create_engine(url, connect_args={"ssl": {}})
    logger.info("  Fetching user_tokens...")
    t0 = time.perf_counter()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT id, user_id, created_at, is_default FROM user_tokens WHERE created_at < '2026-01-01'"),
            conn,
        )
    logger.info(f"    fetched {len(df):,} tokens ({time.perf_counter() - t0:.1f}s)")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["is_default"] = df["is_default"].fillna(0).astype(np.int8)
    return df.rename(columns={
        "id": "user_token_id",
        "created_at": "token_created_at",
        "is_default": "is_default_token",
    })


def build_features(df_pay: pd.DataFrame, df_tok: pd.DataFrame) -> pd.DataFrame:
    """Join payments with tokens; compute features."""
    logger.info("  Joining payments with tokens...")
    # Some payments have user_token_id = 0 or NaN — they have no token
    df_pay["has_token"] = (df_pay["user_token_id"].fillna(0) > 0).astype(np.int8)

    merged = df_pay.merge(
        df_tok[["user_token_id", "token_created_at", "is_default_token"]],
        on="user_token_id", how="left",
    )

    # token_age_days at payment time
    merged["token_age_days_at_payment"] = (
        (merged["created_at"] - merged["token_created_at"]).dt.total_seconds() / 86400.0
    ).clip(lower=-1000, upper=10000).fillna(-1).astype(np.float32)

    # is_new_token: token < 7d at payment time
    merged["is_new_token"] = (
        (merged["token_age_days_at_payment"] >= 0) &
        (merged["token_age_days_at_payment"] < 7)
    ).astype(np.int8)

    # is_very_new_token: < 1d
    merged["is_very_new_token"] = (
        (merged["token_age_days_at_payment"] >= 0) &
        (merged["token_age_days_at_payment"] < 1)
    ).astype(np.int8)

    merged["is_default_token"] = merged["is_default_token"].fillna(0).astype(np.int8)

    # n_tokens_per_user — count tokens user had BEFORE payment time
    # Vectorized: for each payment, count tokens of user with token_created_at < payment_created_at
    logger.info("  Computing n_tokens_per_user_at_payment...")
    t0 = time.perf_counter()
    # Build per-user sorted list of token creation times
    tok_by_user = df_tok.groupby("user_id")["token_created_at"].apply(
        lambda s: np.sort(s.to_numpy())
    ).to_dict()

    def n_tokens_at(row):
        times = tok_by_user.get(row["user_id"])
        if times is None:
            return 0
        return int(np.searchsorted(times, row["created_at"], side="right"))

    # Loop in chunks for speed; this is fast since searchsorted is O(log n)
    n_tokens = np.zeros(len(merged), dtype=np.int32)
    uids = merged["user_id"].to_numpy()
    times = merged["created_at"].to_numpy()
    for i in range(len(merged)):
        toks = tok_by_user.get(int(uids[i]))
        if toks is None:
            continue
        n_tokens[i] = int(np.searchsorted(toks, times[i], side="right"))
    merged["n_tokens_user_at_payment"] = n_tokens.astype(np.float32)
    logger.info(f"    done ({time.perf_counter() - t0:.1f}s)")

    cols = [
        "payment_id", "user_token_id", "has_token",
        "token_age_days_at_payment", "is_new_token", "is_very_new_token",
        "is_default_token", "n_tokens_user_at_payment",
    ]
    return merged[cols].copy()


def main():
    logger.info("=" * 60)
    logger.info("Extract user_tokens features → payment_token_features.parquet")
    logger.info("=" * 60)

    t_total = time.perf_counter()
    df_pay = fetch_payments_token_map()
    df_tok = fetch_user_tokens()

    out = build_features(df_pay, df_tok)
    logger.info(f"  Output: {out.shape}")
    logger.info(
        f"  has_token rate: {out['has_token'].mean():.4f}; "
        f"is_new_token rate: {out['is_new_token'].mean():.4f}; "
        f"is_very_new_token rate: {out['is_very_new_token'].mean():.4f}"
    )
    logger.info(
        f"  token_age_days stats (only paid w/ token, age >= 0): "
        f"median={out.loc[out['has_token'] == 1, 'token_age_days_at_payment'].median():.1f}d, "
        f"p25={out.loc[out['has_token'] == 1, 'token_age_days_at_payment'].quantile(0.25):.1f}d, "
        f"p75={out.loc[out['has_token'] == 1, 'token_age_days_at_payment'].quantile(0.75):.1f}d"
    )

    out_path = DATA_DIR / "payment_token_features.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"  Written {out_path}")
    logger.info(f"  Total elapsed: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
