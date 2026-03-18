"""
DataManager — Facade for data extraction, validation, proxy labeling and I/O.

SOLID decomposition (SRP):
  - DataManager: facade/orchestrator
  - _validate_extraction: validation logic
  - _downcast: memory optimization
  - _save_manifest: manifest writing
  - assign_proxy_labels: static proxy labeling

DIP: ClickHouse extractor injected via constructor for testability.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from config.config import Settings

# ── Constants ────────────────────────────────────────────────────

REQUIRED_NON_NULL = ["id", "user_id", "facility_id", "reservation_paid_out", "created_at", "status"]
SAFE_INT32_COLS = ["user_id", "facility_id"]
SAFE_FLOAT32_COLS = ["reservation_paid_out", "tax", "tip", "discount"]

CANONICAL_SQL = """
SELECT
    id,
    user_id,
    facility_id,
    facility_name,
    created_at,
    captured_at,
    payment_method,
    gateway,
    source_enum,
    status,
    reservation_paid_out,
    discount,
    tax,
    tip,
    card_brand,
    currency,
    paid_by_manager,
    reversed_id,
    debit_refund,
    _peerdb_version
FROM {database}.{table} FINAL
WHERE created_at >= %(start)s
  AND created_at < %(end)s
  AND payment_method != 'reversal'
  AND payment_method != 'free'
  AND user_id != 0
  AND _peerdb_is_deleted = 0
ORDER BY created_at, id
"""

SPLIT_DEFINITIONS = {
    "warm":  ("warm_start",  "train_start"),
    "train": ("train_start", "train_end"),
    "val":   ("train_end",   "val_end"),
    "test":  ("val_end",     "test_end"),
}


class DataManager:
    """Orchestrates data extraction, validation, and I/O.

    Usage:
        dm = DataManager(settings)
        dm.extract_from_clickhouse()
        train, val, test = dm.load_splits()
    """

    def __init__(self, settings: "Settings", extractor=None):
        self._settings = settings
        self._extractor = extractor

    def _get_extractor(self):
        if self._extractor is not None:
            return self._extractor
        from fraud_detector.data.clickhouse_connector import ClickHouseConnector
        self._extractor = ClickHouseConnector(
            host=self._settings.clickhouse_host,
            port=self._settings.clickhouse_port,
            user=self._settings.clickhouse_user,
            password=self._settings.clickhouse_password,
            database=self._settings.clickhouse_database,
            secure=self._settings.clickhouse_secure,
        )
        return self._extractor

    # ── Extraction ───────────────────────────────────────────────

    def extract_from_clickhouse(self, splits: Optional[List[str]] = None) -> None:
        if splits is None:
            splits = list(SPLIT_DEFINITIONS.keys())

        self._settings.ensure_directories()
        extractor = self._get_extractor()
        db = self._settings.clickhouse_database
        table = self._settings.clickhouse_table
        query = CANONICAL_SQL.format(database=db, table=table)

        for name in splits:
            start_attr, end_attr = SPLIT_DEFINITIONS[name]
            start = getattr(self._settings, start_attr)
            end = getattr(self._settings, end_attr)

            logger.info(f"Extracting {name}: {start} to {end}")
            df = extractor.query_df(query, parameters={"start": start, "end": end})

            df = df[df["user_id"] > 0]
            if "_peerdb_is_deleted" in df.columns:
                df = df[df["_peerdb_is_deleted"] == 0]
                df = df.drop(columns=["_peerdb_is_deleted"], errors="ignore")
            df = df.drop(columns=["is_fraud"], errors="ignore")

            self._validate_extraction(df, name)
            df = self._downcast(df)

            target = self._settings.processed_dir / f"{name}_raw.parquet"
            tmp = target.with_suffix(".tmp.parquet")
            df.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
            tmp.rename(target)
            logger.info(f"Saved {name}: {len(df):,} rows -> {target}")

            manifest_path = self._settings.manifests_dir / f"{name}_manifest.json"
            self._save_manifest(name, start, end, df, manifest_path)

        sql_path = self._settings.manifests_dir / "query_snapshot.sql"
        sql_path.write_text(query)

    # ── Validation ───────────────────────────────────────────────

    def _validate_extraction(self, df: pd.DataFrame, name: str) -> None:
        missing = [c for c in REQUIRED_NON_NULL if c not in df.columns]
        if missing:
            raise ValueError(f"Split '{name}': missing required columns: {missing}")

        for col in REQUIRED_NON_NULL:
            null_count = df[col].isna().sum()
            if null_count > 0:
                raise ValueError(f"Split '{name}': {null_count} NULLs in column '{col}'")

        bad_users = (df["user_id"] <= 0).sum()
        if bad_users > 0:
            raise ValueError(f"Split '{name}': {bad_users} rows with user_id <= 0")

        dup_count = df["id"].duplicated().sum()
        if dup_count > 0:
            dup_pct = dup_count / len(df) * 100
            if dup_pct > 0.01:
                raise ValueError(f"Split '{name}': {dup_count} duplicate IDs ({dup_pct:.4f}%)")
            logger.warning(f"Split '{name}': {dup_count} duplicate IDs — deduplicating")
            df.drop_duplicates(subset=["id"], keep="last", inplace=True)

        logger.info(f"Validated {name}: {len(df):,} rows, memory={df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    # ── Downcast ─────────────────────────────────────────────────

    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in SAFE_FLOAT32_COLS:
            if col in df.columns and df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        for col in SAFE_INT32_COLS:
            if col in df.columns:
                max_val = df[col].max()
                if max_val < 2**31 - 1:
                    df[col] = df[col].astype(np.int32)
        return df

    # ── Proxy labeling ───────────────────────────────────────────

    @staticmethod
    def assign_proxy_labels(df: pd.DataFrame, proxy_type: str) -> pd.Series:
        if proxy_type == "strict":
            statuses = ["totally_refunded", "refunded_to_credit"]
        elif proxy_type == "wide":
            statuses = ["totally_refunded", "refunded_to_credit", "partially_refunded"]
        else:
            raise ValueError(f"Invalid proxy_type='{proxy_type}'. Must be 'strict' or 'wide'.")
        return df["status"].isin(statuses).astype(np.int8)

    # ── Manifest ─────────────────────────────────────────────────

    def _save_manifest(self, name: str, start: str, end: str, df: pd.DataFrame, path: Path) -> None:
        manifest = {
            "name": name,
            "start_date": start,
            "end_date": end,
            "row_count": len(df),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "columns": list(df.columns),
            "status_distribution": df["status"].value_counts().to_dict() if "status" in df.columns else {},
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        }
        path.write_text(json.dumps(manifest, indent=2, default=str))

    # ── Loading ──────────────────────────────────────────────────

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.load_split("train"), self.load_split("val"), self.load_split("test")

    def load_split(self, name: str) -> pd.DataFrame:
        path = self._settings.processed_dir / f"{name}_raw.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Split '{name}' not found at {path}. Run extract_from_clickhouse() first.")
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Loaded {name}: {len(df):,} rows")
        return df

    def close(self):
        if self._extractor is not None and hasattr(self._extractor, "close"):
            self._extractor.close()
