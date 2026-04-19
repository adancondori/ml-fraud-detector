"""
Feature engineering for the payment anomaly-detection study.

The implementation follows the 31-feature contract:
- F06 and F21 are intentionally removed because the universe excludes `free`.
- fit() learns only training-set statistics.
- transform() is leakage-safe inside the provided frame.
- transform_with_warm_history() supports split boundaries and method-history carryover.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from fraud_detector.utils.logger import logger

FEATURE_NAMES: List[str] = [
    "amount",
    "log_amount",
    "amount_usd_ratio",
    "discount_ratio",
    "has_tip",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    "user_txn_count_1h",
    "user_txn_count_24h",
    "time_since_last_txn",
    "user_amount_24h",
    "user_distinct_facilities_30d",
    "user_distinct_methods",
    "user_reversal_ratio_30d",
    "user_account_age_days",
    "user_discount_ratio_30d",
    "facility_avg_amount",
    "amount_facility_ratio",
    "is_club_credit",
    "user_debit_count_30d",
    "user_debit_amount_30d",
    "credit_flow_ratio",
    "is_staff",
    "paid_by_manager",
    "staff_amount_zscore",
    "category_entropy_30d",
    "user_reversal_count_30d",
    "user_merchandise_ratio_30d",
]

FEATURE_NAMES_30: List[str] = [f for f in FEATURE_NAMES if f != "user_reversal_ratio_30d"]
FEATURE_NAMES_21: List[str] = FEATURE_NAMES[:21]

METADATA_COLS: List[str] = [
    "id",
    "user_id",
    "facility_id",
    "created_at",
    "status",
    "currency",
    "category",
    "user_role",
]

if len(FEATURE_NAMES) != 31:
    raise ValueError(f"FEATURE_NAMES must contain 31 elements, found {len(FEATURE_NAMES)}")
if len(FEATURE_NAMES_30) != 30:
    raise ValueError(f"FEATURE_NAMES_30 must contain 30 elements, found {len(FEATURE_NAMES_30)}")
if len(FEATURE_NAMES_21) != 21:
    raise ValueError(f"FEATURE_NAMES_21 must contain 21 elements, found {len(FEATURE_NAMES_21)}")


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Timestamp column contains invalid values")
    return timestamps


def _series_group_shift(series: pd.Series, user_ids: pd.Series) -> pd.Series:
    return series.groupby(user_ids).shift(1).fillna(0)


def _rolling_shifted_stat(
    df: pd.DataFrame,
    value_col: str,
    agg: str,
    window: str = "30D",
) -> pd.Series:
    raw = (
        df.groupby("user_id")
        .rolling(window, on="created_at")[value_col]
        .agg(agg)
    )
    raw = pd.Series(raw.droplevel(0).values, index=df.index)
    return _series_group_shift(raw.astype(np.float64), df["user_id"]).astype(np.float32)


class FeatureGroup(ABC):
    """Base interface for feature groups."""

    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> "FeatureGroup":
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def feature_names(self) -> List[str]:
        ...


class TransactionalFeatures(FeatureGroup):
    def __init__(self) -> None:
        self._global_avg_amount: Optional[float] = None

    def fit(self, df_train: pd.DataFrame) -> "TransactionalFeatures":
        self._global_avg_amount = float(df_train["amount"].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._global_avg_amount is None:
            raise RuntimeError("TransactionalFeatures requires fit() before transform()")
        out = df.copy()
        out["log_amount"] = np.log1p(out["amount"]).astype(np.float32)
        out["amount_usd_ratio"] = (
            out["amount"] / max(self._global_avg_amount, 1e-8)
        ).astype(np.float32)
        out["discount_ratio"] = (
            out["discount"] / (out["amount"] + 0.01)
        ).astype(np.float32)
        out["has_tip"] = (out["tip"] > 0).astype(np.int8)
        out["amount"] = out["amount"].astype(np.float32)
        return out

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES[:5]


class TemporalFeatures(FeatureGroup):
    def fit(self, df_train: pd.DataFrame) -> "TemporalFeatures":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hour = out["created_at"].dt.hour
        day_of_week = out["created_at"].dt.dayofweek + 1
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        out["day_of_week"] = day_of_week.astype(np.int8)
        out["is_weekend"] = (day_of_week >= 6).astype(np.int8)
        out["is_off_hours"] = hour.isin([23, 0, 1, 2, 3, 4, 5, 6]).astype(np.int8)
        return out

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES[5:10]


class VelocityFeatures(FeatureGroup):
    def fit(self, df_train: pd.DataFrame) -> "VelocityFeatures":
        return self

    @staticmethod
    def _rolling_count(df: pd.DataFrame, window: str) -> pd.Series:
        raw = (
            df.groupby("user_id")
            .rolling(window, on="created_at")["id"]
            .count()
        )
        values = raw.droplevel(0).values
        return pd.Series(values - 1, index=df.index).fillna(0).astype(np.float32)

    @staticmethod
    def _rolling_amount_sum(df: pd.DataFrame, window: str) -> pd.Series:
        raw = (
            df.groupby("user_id")
            .rolling(window, on="created_at")["amount"]
            .sum()
        )
        values = raw.droplevel(0).values
        return pd.Series(values - df["amount"].values, index=df.index).fillna(0).clip(lower=0).astype(np.float32)

    @staticmethod
    def _time_since_last(df: pd.DataFrame) -> pd.Series:
        return (
            df.groupby("user_id")["created_at"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
            .astype(np.float32)
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["user_txn_count_1h"] = self._rolling_count(out, "1h")
        out["user_txn_count_24h"] = self._rolling_count(out, "24h")
        out["time_since_last_txn"] = self._time_since_last(out)
        out["user_amount_24h"] = self._rolling_amount_sum(out, "24h")
        return out

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES[10:14]


class BehavioralFeatures(FeatureGroup):
    """Behavioral features plus method-history state management."""

    def __init__(self) -> None:
        self._user_created_at: Dict[int, pd.Timestamp] = {}
        self._train_method_history: Dict[int, set[str]] = {}

    def fit(self, df_train: pd.DataFrame) -> "BehavioralFeatures":
        created_map = (
            df_train.groupby("user_id")["user_created_at"].first()
            if "user_created_at" in df_train.columns
            else df_train.groupby("user_id")["created_at"].min()
        )
        self._user_created_at = created_map.to_dict()
        self._train_method_history = {
            int(user_id): set(group.dropna().astype(str))
            for user_id, group in df_train.groupby("user_id")["payment_method"]
        }
        return self

    def initial_method_history(self) -> Dict[int, set[str]]:
        return deepcopy(self._train_method_history)

    @staticmethod
    def _distinct_facilities_30d(df: pd.DataFrame) -> pd.Series:
        results = np.zeros(len(df), dtype=np.float32)
        for _, group in df.groupby("user_id", sort=False):
            created = group["created_at"].to_numpy()
            facilities = group["facility_id"].to_numpy()
            values = []
            for i in range(len(group)):
                cutoff = created[i] - np.timedelta64(30, "D")
                previous = facilities[:i][created[:i] >= cutoff]
                values.append(float(len(set(previous.tolist()))))
            results[group.index.to_numpy()] = values
        return pd.Series(results, index=df.index, dtype=np.float32)

    @staticmethod
    def _distinct_methods_with_history(
        user_ids: pd.Series,
        methods: pd.Series,
        baseline_history: Optional[Dict[int, set[str]]] = None,
    ) -> Tuple[pd.Series, Dict[int, set[str]]]:
        history = deepcopy(baseline_history or {})
        values = []
        for user_id, method in zip(user_ids.tolist(), methods.tolist()):
            key = int(user_id)
            seen = history.setdefault(key, set())
            values.append(float(len(seen)))
            if pd.notna(method):
                seen.add(str(method))
        return pd.Series(values, index=user_ids.index, dtype=np.float32), history

    def _account_age_days(self, df: pd.DataFrame) -> pd.Series:
        if "user_created_at" in df.columns:
            created_at = pd.to_datetime(df["user_created_at"], errors="coerce")
        else:
            created_at = df["user_id"].map(self._user_created_at)

        created_at = pd.to_datetime(created_at, errors="coerce")
        new_users = created_at.isna()
        if new_users.any():
            split_first = df.loc[new_users].groupby("user_id")["created_at"].transform("min")
            created_at.loc[new_users] = split_first

        return (
            (df["created_at"] - created_at)
            .dt.days
            .clip(lower=0)
            .fillna(0)
            .astype(np.int32)
        )

    @staticmethod
    def _discount_ratio_30d(df: pd.DataFrame) -> pd.Series:
        discount_30d = _rolling_shifted_stat(df, "discount", "sum", window="30D")
        amount_30d = _rolling_shifted_stat(df, "amount", "sum", window="30D")
        return (discount_30d / (amount_30d + 0.01)).astype(np.float32)

    @staticmethod
    def _reversal_ratio_30d(df: pd.DataFrame) -> pd.Series:
        tmp = df.copy()
        tmp["_is_reversal"] = tmp["status"].isin(
            ["totally_refunded", "refunded_to_credit"]
        ).astype(np.int8)
        return _rolling_shifted_stat(tmp, "_is_reversal", "mean", window="30D")

    def transform(
        self,
        df: pd.DataFrame,
        baseline_method_history: Optional[Dict[int, set[str]]] = None,
        return_method_history: bool = False,
    ):
        out = df.copy()
        out["user_distinct_facilities_30d"] = self._distinct_facilities_30d(out)
        distinct_methods, final_history = self._distinct_methods_with_history(
            out["user_id"],
            out["payment_method"],
            baseline_history=baseline_method_history,
        )
        out["user_distinct_methods"] = distinct_methods
        out["user_reversal_ratio_30d"] = self._reversal_ratio_30d(out)
        out["user_account_age_days"] = self._account_age_days(out)
        out["user_discount_ratio_30d"] = self._discount_ratio_30d(out)
        if return_method_history:
            return out, final_history
        return out

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES[14:19]


class ContextualFeatures(FeatureGroup):
    def __init__(self) -> None:
        self._facility_avg_amount: Dict[int, float] = {}
        self._global_avg_amount: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "ContextualFeatures":
        self._global_avg_amount = float(df_train["amount"].mean())
        self._facility_avg_amount = (
            df_train.groupby("facility_id")["amount"].mean().to_dict()
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["facility_avg_amount"] = (
            out["facility_id"]
            .map(self._facility_avg_amount)
            .fillna(self._global_avg_amount)
            .astype(np.float32)
        )
        out["amount_facility_ratio"] = (
            out["amount"] / (out["facility_avg_amount"] + 0.01)
        ).astype(np.float32)
        return out

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES[19:21]


class CreditFlowFeatures(FeatureGroup):
    def fit(self, df_train: pd.DataFrame) -> "CreditFlowFeatures":
        return self

    @staticmethod
    def _debit_features(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        tmp = df.copy()
        tmp["_is_debit"] = (tmp["category"] == "debit").astype(np.int8)
        tmp["_debit_amount"] = tmp["amount"] * tmp["_is_debit"]
        count_30d = _rolling_shifted_stat(tmp, "_is_debit", "sum", window="30D")
        amount_30d = _rolling_shifted_stat(tmp, "_debit_amount", "sum", window="30D")
        return count_30d.astype(np.float32), amount_30d.astype(np.float32)

    @staticmethod
    def _prepaid_spend_30d(df: pd.DataFrame) -> pd.Series:
        tmp = df.copy()
        tmp["_is_prepaid"] = (tmp["category"] == "prepaid").astype(np.int8)
        tmp["_prepaid_amount"] = tmp["amount"] * tmp["_is_prepaid"]
        return _rolling_shifted_stat(tmp, "_prepaid_amount", "sum", window="30D")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["is_club_credit"] = out["club_credit_flag"].fillna(False).astype(np.int8)
        debit_count, debit_amount = self._debit_features(out)
        prepaid_spend = self._prepaid_spend_30d(out)
        out["user_debit_count_30d"] = debit_count
        out["user_debit_amount_30d"] = debit_amount
        out["credit_flow_ratio"] = (
            debit_amount / (prepaid_spend + 0.01)
        ).astype(np.float32)
        return out

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES[21:25]


class StaffRoleFeatures(FeatureGroup):
    STAFF_ROLES = {"court_manager", "court_operator", "teacher"}

    def __init__(self) -> None:
        self._role_currency_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._currency_stats: Dict[str, Dict[str, float]] = {}
        self._global_mean: float = 0.0
        self._global_std: float = 1.0

    def fit(self, df_train: pd.DataFrame) -> "StaffRoleFeatures":
        train = df_train.copy()
        train["user_role"] = train["user_role"].fillna("player").astype(str)
        train["currency"] = train["currency"].fillna("USD").astype(str)
        self._global_mean = float(train["amount"].mean())
        self._global_std = float(train["amount"].std(ddof=0) or 1.0)

        role_grouped = train.groupby(["user_role", "currency"])["amount"]
        self._role_currency_stats = {
            (str(role), str(currency)): {
                "mean": float(group.mean()),
                "std": float(group.std(ddof=0) or 1.0),
            }
            for (role, currency), group in role_grouped
        }
        currency_grouped = train.groupby("currency")["amount"]
        self._currency_stats = {
            str(currency): {
                "mean": float(group.mean()),
                "std": float(group.std(ddof=0) or 1.0),
            }
            for currency, group in currency_grouped
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["user_role"] = out["user_role"].fillna("player").astype(str)
        out["currency"] = out["currency"].fillna("USD").astype(str)

        if "is_staff" in out.columns:
            out["is_staff"] = out["is_staff"].fillna(0).astype(np.int8)
        else:
            out["is_staff"] = out["user_role"].isin(self.STAFF_ROLES).astype(np.int8)

        out["paid_by_manager"] = out["paid_by_manager"].fillna(False).astype(np.int8)

        means = []
        stds = []
        for role, currency in zip(out["user_role"], out["currency"]):
            role_key = (str(role), str(currency))
            if role_key in self._role_currency_stats:
                stats = self._role_currency_stats[role_key]
            elif str(currency) in self._currency_stats:
                stats = self._currency_stats[str(currency)]
            else:
                stats = {"mean": self._global_mean, "std": self._global_std}
            means.append(stats["mean"])
            stds.append(stats["std"] or 1.0)

        mean_series = pd.Series(means, index=out.index, dtype=np.float64)
        std_series = pd.Series(stds, index=out.index, dtype=np.float64).replace(0, 1.0)
        out["staff_amount_zscore"] = (
            (out["amount"] - mean_series) / std_series
        ).fillna(0).astype(np.float32)
        return out

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES[25:28]


class OperationalDiversityFeatures(FeatureGroup):
    def fit(self, df_train: pd.DataFrame) -> "OperationalDiversityFeatures":
        return self

    @staticmethod
    def _category_entropy_30d(df: pd.DataFrame) -> pd.Series:
        results = np.zeros(len(df), dtype=np.float32)
        for _, group in df.groupby("user_id", sort=False):
            created = group["created_at"].to_numpy()
            categories = group["category"].astype(str).to_numpy()
            values = []
            for i in range(len(group)):
                cutoff = created[i] - np.timedelta64(30, "D")
                window = categories[:i][created[:i] >= cutoff]
                if len(window) == 0:
                    values.append(0.0)
                    continue
                _, counts = np.unique(window, return_counts=True)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log2(probs + 1e-12))
                values.append(float(entropy))
            results[group.index.to_numpy()] = values
        return pd.Series(results, index=df.index, dtype=np.float32)

    @staticmethod
    def _reversal_count_30d(df: pd.DataFrame) -> pd.Series:
        tmp = df.copy()
        tmp["_is_reversal"] = tmp["status"].isin(
            ["totally_refunded", "refunded_to_credit"]
        ).astype(np.int8)
        return _rolling_shifted_stat(tmp, "_is_reversal", "sum", window="30D")

    @staticmethod
    def _merchandise_ratio_30d(df: pd.DataFrame) -> pd.Series:
        tmp = df.copy()
        tmp["_is_merchandise"] = (tmp["category"] == "merchandise").astype(np.int8)
        return _rolling_shifted_stat(tmp, "_is_merchandise", "mean", window="30D")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["category_entropy_30d"] = self._category_entropy_30d(out)
        out["user_reversal_count_30d"] = self._reversal_count_30d(out)
        out["user_merchandise_ratio_30d"] = self._merchandise_ratio_30d(out)
        return out

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES[28:]


class FeatureEngineer:
    """Composes the 31-feature anomaly-detection catalog."""

    def __init__(self, groups: Optional[List[FeatureGroup]] = None) -> None:
        self._groups: List[FeatureGroup] = groups or [
            TransactionalFeatures(),
            TemporalFeatures(),
            VelocityFeatures(),
            BehavioralFeatures(),
            ContextualFeatures(),
            CreditFlowFeatures(),
            StaffRoleFeatures(),
            OperationalDiversityFeatures(),
        ]
        self._behavioral_group = next(
            group for group in self._groups if isinstance(group, BehavioralFeatures)
        )
        self._fitted = False

    def fit(self, df_train: pd.DataFrame) -> "FeatureEngineer":
        self._validate_required_columns(df_train)
        ordered = self._prepare_input(df_train)
        for group in self._groups:
            group.fit(ordered)
        self._fitted = True
        logger.info("FeatureEngineer fitted successfully")
        return self

    def _transform_internal(
        self,
        df: pd.DataFrame,
        baseline_method_history: Optional[Dict[int, set[str]]] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, set[str]]]:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        self._validate_required_columns(df)
        out = self._prepare_input(df)

        method_history = deepcopy(baseline_method_history or {})
        for group in self._groups:
            if isinstance(group, BehavioralFeatures):
                out, method_history = group.transform(
                    out,
                    baseline_method_history=method_history,
                    return_method_history=True,
                )
            else:
                out = group.transform(out)

        for feature in FEATURE_NAMES:
            if feature not in out.columns:
                raise ValueError(f"Missing engineered feature '{feature}'")
        if out[FEATURE_NAMES].isna().any().any():
            missing = out[FEATURE_NAMES].isna().sum()
            missing = missing[missing > 0].to_dict()
            raise ValueError(f"Engineered features contain NaN values: {missing}")

        available_meta = [col for col in METADATA_COLS if col in out.columns]
        return out[FEATURE_NAMES + available_meta].copy(), method_history

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed, _ = self._transform_internal(df, baseline_method_history={})
        return transformed

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(df_train)
        transformed, _ = self._transform_internal(df_train, baseline_method_history={})
        return transformed

    def transform_with_warm_history(
        self,
        df_split: pd.DataFrame,
        df_warm: pd.DataFrame,
        method_state: Optional[Dict[str, Dict[int, set[str]]]] = None,
        return_state: bool = False,
    ):
        if not self._fitted:
            raise RuntimeError("Call fit() before transform_with_warm_history()")

        marker = "_is_split"
        split_marked = df_split.assign(**{marker: True})
        warm_marked = df_warm.assign(**{marker: False})
        combined = pd.concat([warm_marked, split_marked], ignore_index=True)
        combined = self._prepare_input(combined)

        baseline = method_state["user_method_history"] if method_state else self.get_feature_state()["user_method_history"]
        transformed, final_state = self._transform_internal(
            combined,
            baseline_method_history=baseline,
        )

        split_index = combined.index[combined[marker]]
        result = transformed.loc[split_index].reset_index(drop=True)

        if return_state:
            return result, {"user_method_history": final_state}
        return result

    def get_feature_state(self) -> Dict[str, Dict[int, set[str]]]:
        return {"user_method_history": self._behavioral_group.initial_method_history()}

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"FeatureEngineer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        instance = joblib.load(path)
        logger.info(f"FeatureEngineer loaded from {path}")
        return instance

    @staticmethod
    def get_feature_names() -> List[str]:
        return FEATURE_NAMES.copy()

    @staticmethod
    def get_feature_names_30() -> List[str]:
        return FEATURE_NAMES_30.copy()

    @staticmethod
    def get_feature_names_21() -> List[str]:
        return FEATURE_NAMES_21.copy()

    @staticmethod
    def _prepare_input(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["created_at"] = _coerce_timestamp(out["created_at"])
        if "user_created_at" in out.columns:
            out["user_created_at"] = pd.to_datetime(out["user_created_at"], errors="coerce")
        out["category"] = out["category"].fillna("unknown").astype(str).str.lower()
        if "user_role" in out.columns:
            out["user_role"] = out["user_role"].fillna("player").astype(str)
        else:
            out["user_role"] = np.where(
                out["is_staff"].fillna(0).astype(bool),
                "staff",
                "player",
            )
        out["currency"] = out["currency"].fillna("USD").astype(str).str.upper()
        return out.sort_values(["user_id", "created_at", "id"]).reset_index(drop=True)

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> None:
        required = {
            "id",
            "user_id",
            "facility_id",
            "created_at",
            "status",
            "amount",
            "discount",
            "tip",
            "payment_method",
            "currency",
            "category",
            "club_credit_flag",
            "paid_by_manager",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if "user_role" not in df.columns and "is_staff" not in df.columns:
            raise ValueError("Missing required role information: provide 'user_role' or 'is_staff'")
