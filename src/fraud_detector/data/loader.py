"""
Data extraction and split I/O for the anomaly-detection pipeline.

This module is intentionally thesis-aligned:
- extraction is temporal and offline,
- proxy labels are derived from `status`,
- currency normalization happens before feature engineering,
- the canonical SQL includes the joins required for the 31-feature catalog.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from config.config import Settings

REQUIRED_COLUMNS = [
    "id",
    "user_id",
    "facility_id",
    "created_at",
    "status",
    "amount",
    "currency",
    "category",
    "club_credit_flag",
    "paid_by_manager",
    "user_role",
]

SAFE_FLOAT32_COLS = [
    "amount",
    "discount",
    "tax",
    "tip",
    "amount_local",
    "discount_local",
    "tax_local",
    "tip_local",
    "exchange_rate_applied",
]
SAFE_INT32_COLS = ["user_id", "effective_user_id", "facility_id"]
SAFE_INT8_COLS = ["is_staff"]

CANONICAL_SQL = """
SELECT
    p.id,
    p.user_id,
    p.effective_user_id,
    p.facility_id,
    p.facility_name,
    p.created_at,
    p.captured_at,
    p.payment_method,
    p.gateway,
    p.source_enum,
    p.status,
    p.reservation_paid_out,
    p.discount,
    p.tax,
    p.tip,
    p.card_brand,
    p.currency,
    p.paid_by_manager,
    p.reversed_id,
    p.debit_refund,
    p.category,
    p.club_credit_flag,
    p._peerdb_version,
    CASE
        WHEN fu.role IN ('court_manager', 'court_operator', 'teacher') THEN 1
        ELSE 0
    END AS is_staff,
    coalesce(fu.role, 'player') AS user_role,
    u.created_at AS user_created_at
FROM {database}.{table} AS p FINAL
LEFT ANY JOIN (
    SELECT user_id, facility_id, role
    FROM {database}.facilities_users FINAL
    WHERE _peerdb_is_deleted = 0
) AS fu
    ON p.user_id = fu.user_id
   AND p.facility_id = fu.facility_id
LEFT ANY JOIN (
    SELECT id, created_at
    FROM {database}.users FINAL
    WHERE _peerdb_is_deleted = 0
) AS u
    ON p.user_id = u.id
WHERE p.created_at >= %(start)s
  AND p.created_at < %(end)s
  AND p.payment_method != 'reversal'
  AND p.payment_method != 'free'
  AND p.user_id != 0
  AND p._peerdb_is_deleted = 0
ORDER BY p.created_at, p.id
"""

SPLIT_DEFINITIONS = {
    "warm": ("warm_start", "train_start"),
    "train": ("train_start", "train_end"),
    "val": ("train_end", "val_end"),
    "test": ("val_end", "test_end"),
}


class DataManager:
    """Facade for extraction, validation and split I/O."""

    def __init__(self, settings: "Settings", extractor=None, normalizer=None):
        self._settings = settings
        self._extractor = extractor
        self._normalizer = normalizer

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

    def _get_normalizer(self):
        if self._normalizer is not None:
            return self._normalizer

        from fraud_detector.utils.currency import CurrencyNormalizer

        self._normalizer = CurrencyNormalizer.from_csv(self._settings.exchange_rates_file)
        return self._normalizer

    def extract_from_clickhouse(self, splits: Optional[List[str]] = None) -> None:
        """Extract the temporal splits as parquet snapshots."""
        if splits is None:
            splits = list(SPLIT_DEFINITIONS.keys())

        self._settings.ensure_directories()
        extractor = self._get_extractor()
        query = CANONICAL_SQL.format(
            database=self._settings.clickhouse_database,
            table=self._settings.clickhouse_table,
        )

        for split_name in splits:
            start_attr, end_attr = SPLIT_DEFINITIONS[split_name]
            start = getattr(self._settings, start_attr)
            end = getattr(self._settings, end_attr)

            logger.info(f"Extracting split={split_name} from {start} to {end}")
            df = extractor.query_to_dataframe(query, params={"start": start, "end": end})
            df = self._postprocess_extraction(df)
            self._validate_extraction(df, split_name)
            df = self._downcast(df)

            output_path = self._settings.processed_dir / f"{split_name}_raw.parquet"
            tmp_path = output_path.with_suffix(".tmp.parquet")
            df.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
            tmp_path.rename(output_path)
            logger.info(f"Saved {split_name}: {len(df):,} rows -> {output_path}")

            manifest_path = self._settings.manifests_dir / f"{split_name}_manifest.json"
            self._save_manifest(split_name, start, end, df, manifest_path)

        sql_path = self._settings.manifests_dir / "query_snapshot.sql"
        sql_path.write_text(query)

    def _postprocess_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize schema and monetary values after extraction."""
        out = df.copy()

        for col in ("created_at", "captured_at", "user_created_at"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")

        out = out[out["user_id"] > 0].copy()
        out = out.drop(columns=["_peerdb_is_deleted", "is_fraud"], errors="ignore")

        if "reservation_paid_out" in out.columns:
            out = out.rename(columns={"reservation_paid_out": "amount"})

        out["category"] = out.get("category", "unknown")
        out["category"] = out["category"].fillna("unknown").astype(str).str.lower()
        out["club_credit_flag"] = out.get("club_credit_flag", False).fillna(False)
        out["paid_by_manager"] = out.get("paid_by_manager", False).fillna(False)
        out["is_staff"] = out.get("is_staff", 0).fillna(0).astype(np.int8)
        out["user_role"] = out.get("user_role", "player").fillna("player").astype(str)

        if "currency" in out.columns:
            normalizer = self._get_normalizer()
            out = normalizer.normalize(
                out,
                amount_col="amount",
                currency_col="currency",
                timestamp_col="created_at",
            )

        return out

    def _validate_extraction(self, df: pd.DataFrame, split_name: str) -> None:
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Split '{split_name}': missing required columns: {missing}")

        for col in REQUIRED_COLUMNS:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                raise ValueError(f"Split '{split_name}': {null_count} NULLs in column '{col}'")

        duplicate_count = int(df["id"].duplicated().sum())
        if duplicate_count > 0:
            duplicate_pct = duplicate_count / len(df) * 100
            if duplicate_pct > 0.01:
                raise ValueError(
                    f"Split '{split_name}': {duplicate_count} duplicate IDs "
                    f"({duplicate_pct:.4f}%)"
                )
            logger.warning(
                f"Split '{split_name}': {duplicate_count} duplicate IDs; deduplicating"
            )
            df.drop_duplicates(subset=["id"], keep="last", inplace=True)

        if (df["user_id"] <= 0).any():
            raise ValueError(f"Split '{split_name}': found rows with user_id <= 0")

        logger.info(
            f"Validated split={split_name}: {len(df):,} rows, "
            f"memory={df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )

    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in SAFE_FLOAT32_COLS:
            if col in out.columns and pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].astype(np.float32)
        for col in SAFE_INT32_COLS:
            if col in out.columns and pd.api.types.is_integer_dtype(out[col]):
                if int(out[col].max()) < (2**31 - 1):
                    out[col] = out[col].astype(np.int32)
        for col in SAFE_INT8_COLS:
            if col in out.columns:
                out[col] = out[col].astype(np.int8)
        return out

    @staticmethod
    def assign_proxy_labels(df: pd.DataFrame, proxy_type: str) -> pd.Series:
        if proxy_type == "strict":
            statuses = {"totally_refunded", "refunded_to_credit"}
        elif proxy_type == "wide":
            statuses = {"totally_refunded", "refunded_to_credit", "partially_refunded"}
        else:
            raise ValueError("Invalid proxy_type. Expected 'strict' or 'wide'.")
        return df["status"].isin(statuses).astype(np.int8)

    def _save_manifest(
        self,
        split_name: str,
        start: str,
        end: str,
        df: pd.DataFrame,
        path: Path,
    ) -> None:
        manifest = {
            "name": split_name,
            "start_date": start,
            "end_date": end,
            "row_count": len(df),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "columns": list(df.columns),
            "status_distribution": df["status"].value_counts().to_dict(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
            "proxy_strict_rate": float(self.assign_proxy_labels(df, "strict").mean()),
            "proxy_wide_rate": float(self.assign_proxy_labels(df, "wide").mean()),
        }
        path.write_text(json.dumps(manifest, indent=2, default=str))

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.load_split("train"), self.load_split("val"), self.load_split("test")

    def load_split(self, split_name: str) -> pd.DataFrame:
        path = self._settings.processed_dir / f"{split_name}_raw.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Split '{split_name}' not found at {path}. Run extract_from_clickhouse() first."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Loaded split={split_name}: {len(df):,} rows")
        return df

    def close(self) -> None:
        if self._extractor is not None and hasattr(self._extractor, "close"):
            self._extractor.close()
