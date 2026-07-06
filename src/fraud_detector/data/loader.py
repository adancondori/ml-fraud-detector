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
    p.id AS id,
    p.user_id AS user_id,
    p.effective_user_id AS effective_user_id,
    p.facility_id AS facility_id,
    p.facility_name AS facility_name,
    p.created_at AS created_at,
    p.captured_at AS captured_at,
    p.payment_method AS payment_method,
    p.gateway AS gateway,
    p.source_enum AS source_enum,
    p.status AS status,
    p.reservation_paid_out AS reservation_paid_out,
    p.discount AS discount,
    p.tax AS tax,
    p.tip AS tip,
    p.card_brand AS card_brand,
    p.currency AS currency,
    p.paid_by_manager AS paid_by_manager,
    p.reversed_id AS reversed_id,
    p.debit_refund AS debit_refund,
    p.category AS category,
    p.club_credit_flag AS club_credit_flag,
    p._peerdb_version AS _peerdb_version,
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

    @staticmethod
    def _sanitize_currency(series: pd.Series) -> pd.Series:
        """Normalize a currency Series: fillna, uppercase, and replace EMPTY/'' with USD.

        This is a preventive guard for future ClickHouse extractions where
        currency can arrive as 'EMPTY' (currency.utils.fallback_rate returns 1.0
        for unknown codes, which would silently corrupt log_amount and
        amount_usd_ratio without this fix).

        Returns the sanitized series.  Logs a warning with the row count when
        any EMPTY/'' values are replaced (expected count: 0 in current parquets).
        """
        normalized = series.fillna("USD").astype(str).str.upper()
        empty_mask = normalized.isin(["EMPTY", ""])
        n_empty = int(empty_mask.sum())
        if n_empty > 0:
            logger.warning(f"Sanitized {n_empty} rows with currency EMPTY/'' -> USD")
        return normalized.replace({"EMPTY": "USD", "": "USD"})

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
        out["user_role"] = out.get("user_role", "player").fillna("player").astype(str).replace("", "player")

        if "currency" in out.columns:
            out["currency"] = DataManager._sanitize_currency(out["currency"])
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
    def assign_proxy_labels(
        df: pd.DataFrame,
        proxy_type: str,
        settings: Optional["Settings"] = None,
    ) -> pd.Series:
        """Assign binary proxy labels based on type.

        Parameters
        ----------
        df : DataFrame with raw or feature columns.
        proxy_type : One of "strict", "wide", "tipo_a", "tipo_b", "tipo_c",
            "tipo_d", "tipo_e", "unified".
        settings : Optional Settings for threshold values. Uses defaults if None.

        Returns
        -------
        pd.Series of int8 (0/1).
        """
        if settings is None:
            from config.config import settings as _default_settings
            settings = _default_settings

        if proxy_type in ("strict", "tipo_a"):
            statuses = set(settings.tipo_a_list)
            return df["status"].isin(statuses).astype(np.int8)

        if proxy_type == "wide":
            statuses = set(settings.wide_proxy_list)
            return df["status"].isin(statuses).astype(np.int8)

        if proxy_type == "tipo_b":
            # Circuito de credito: circuit_closure_ratio_30d > threshold AND
            # cash_loaded_30d > threshold. Requires pre-computed rolling columns.
            # If columns absent, returns all zeros (documented: requires aggregates).
            if "circuit_closure_ratio_30d" not in df.columns:
                logger.warning(
                    "Tipo B: column 'circuit_closure_ratio_30d' not found; "
                    "returning all zeros. Tipo B requires pre-computed aggregates."
                )
                return pd.Series(np.int8(0), index=df.index)
            return (
                (df["circuit_closure_ratio_30d"] > settings.tipo_b_circuit_closure_threshold)
                & (df["cash_loaded_30d"] > settings.tipo_b_cash_loaded_threshold)
            ).astype(np.int8)

        if proxy_type == "tipo_c":
            # Descuento anomalo: user_discount_ratio_30d > threshold.
            col = "user_discount_ratio_30d"
            if col not in df.columns:
                logger.warning(
                    f"Tipo C: column '{col}' not found; returning all zeros."
                )
                return pd.Series(np.int8(0), index=df.index)
            return (df[col] > settings.tipo_c_discount_ratio_threshold).astype(np.int8)

        if proxy_type == "tipo_d":
            # Velocidad extrema: txn_count_1d > threshold.
            # Uses user_txn_count_24h (shifted -1 for anti-leakage) + 1 to recover actual.
            col = "user_txn_count_24h"
            if col not in df.columns:
                logger.warning(
                    f"Tipo D: column '{col}' not found; returning all zeros."
                )
                return pd.Series(np.int8(0), index=df.index)
            # Feature stores count-1 (anti-leakage shift), so actual = value + 1
            return ((df[col] + 1) > settings.tipo_d_txn_count_1d_threshold).astype(np.int8)

        if proxy_type == "tipo_e":
            # Gratuitas sistematicas. Since payment_method='free' is excluded from
            # the depurated universe, this type will always be 0. Documented as
            # "tipo sin incidencia" in the thesis.
            logger.info(
                "Tipo E: payment_method='free' excluded from universe; "
                "returning all zeros (documented limitation)."
            )
            return pd.Series(np.int8(0), index=df.index)

        if proxy_type == "unified":
            tipo_a = DataManager.assign_proxy_labels(df, "tipo_a", settings)
            tipo_b = DataManager.assign_proxy_labels(df, "tipo_b", settings)
            tipo_c = DataManager.assign_proxy_labels(df, "tipo_c", settings)
            tipo_d = DataManager.assign_proxy_labels(df, "tipo_d", settings)
            tipo_e = DataManager.assign_proxy_labels(df, "tipo_e", settings)
            return (tipo_a | tipo_b | tipo_c | tipo_d | tipo_e).astype(np.int8)

        valid = "strict, wide, tipo_a, tipo_b, tipo_c, tipo_d, tipo_e, unified"
        raise ValueError(f"Invalid proxy_type '{proxy_type}'. Expected one of: {valid}")

    def _save_manifest(
        self,
        split_name: str,
        start: str,
        end: str,
        df: pd.DataFrame,
        path: Path,
    ) -> None:
        proxy_rates = {}
        s = getattr(self, "_settings", None)
        for ptype in ("tipo_a", "tipo_b", "tipo_c", "tipo_d", "tipo_e", "unified", "wide"):
            labels = self.assign_proxy_labels(df, ptype, s)
            proxy_rates[f"proxy_{ptype}_n"] = int(labels.sum())
            proxy_rates[f"proxy_{ptype}_rate"] = round(float(labels.mean()), 6)

        manifest = {
            "name": split_name,
            "start_date": start,
            "end_date": end,
            "row_count": len(df),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "columns": list(df.columns),
            "status_distribution": df["status"].value_counts().to_dict(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
            "proxy_strict_rate": proxy_rates["proxy_tipo_a_rate"],
            "proxy_wide_rate": proxy_rates["proxy_wide_rate"],
            **proxy_rates,
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
