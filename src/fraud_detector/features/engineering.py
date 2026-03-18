"""
Feature Engineering — Catálogo oficial de 20 features.

Implementa el patrón Strategy (FeatureGroup) + Compositor (FeatureEngineer)
para separar responsabilidades por grupo y facilitar tests unitarios por grupo.

Catálogo:
  Grupo 1 Transaccionales (#1-5):   amount, log_amount, amount_usd_ratio,
                                     discount_ratio, has_tip
  Grupo 2 Temporales      (#6-10):  hour_sin, hour_cos, day_of_week,
                                     is_weekend, is_off_hours
  Grupo 3 Velocidad       (#11-14): user_txn_count_1h, user_txn_count_24h,
                                     time_since_last_txn, user_amount_24h
  Grupo 4 Comportamental  (#15-18): user_distinct_facilities_cumul,
                                     user_distinct_methods,
                                     user_reversal_ratio_30d,
                                     user_account_age_days
  Grupo 5 Contextual      (#19-20): facility_avg_amount, amount_facility_ratio

Regla anti-leakage obligatoria:
  - DataFrame ordenado por (user_id, created_at) antes de rolling/cumulative.
  - Ninguna ventana usa información de la fila actual ni del futuro.
  - Estadísticas fit() sólo en train; transform() las aplica sin reaprender.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from fraud_detector.utils.logger import logger

# ── Catálogo oficial ─────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    # Grupo 1: Transaccionales
    "amount",
    "log_amount",
    "amount_usd_ratio",
    "discount_ratio",
    "has_tip",
    # Grupo 2: Temporales
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    # Grupo 3: Velocidad
    "user_txn_count_1h",
    "user_txn_count_24h",
    "time_since_last_txn",
    "user_amount_24h",
    # Grupo 4: Comportamentales
    "user_distinct_facilities_cumul",
    "user_distinct_methods",
    "user_reversal_ratio_30d",
    "user_account_age_days",
    # Grupo 5: Contextuales
    "facility_avg_amount",
    "amount_facility_ratio",
]

if len(FEATURE_NAMES) != 20:
    raise ValueError(f"FEATURE_NAMES debe tener 20 elementos, tiene {len(FEATURE_NAMES)}")

FEATURE_NAMES_19: List[str] = [f for f in FEATURE_NAMES if f != "user_reversal_ratio_30d"]

if len(FEATURE_NAMES_19) != 19:
    raise ValueError(f"FEATURE_NAMES_19 debe tener 19 elementos, tiene {len(FEATURE_NAMES_19)}")

# Columnas de metadata que viajan junto a las features
METADATA_COLS: List[str] = ["id", "user_id", "facility_id", "created_at", "status"]


# ── Interfaz base ─────────────────────────────────────────────────────────────

class FeatureGroup(ABC):
    """Interfaz base para grupos de features."""

    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> "FeatureGroup":
        """Aprende estadísticas del training set."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega columnas de features al DataFrame. No muta el input."""
        ...

    @abstractmethod
    def feature_names(self) -> List[str]:
        """Retorna los nombres de features que genera este grupo."""
        ...


# ── Grupo 1: Transaccionales ─────────────────────────────────────────────────

class TransactionalFeatures(FeatureGroup):
    """Features #1-5: transformaciones aritméticas sobre montos."""

    def __init__(self) -> None:
        self._global_avg_amount: Optional[float] = None

    def fit(self, df_train: pd.DataFrame) -> "TransactionalFeatures":
        self._global_avg_amount = float(df_train["amount"].mean())
        logger.debug(f"TransactionalFeatures.fit — global_avg_amount={self._global_avg_amount:.4f}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._global_avg_amount is None:
            raise RuntimeError("Llamar fit() antes de transform()")
        out = df.copy()
        out["log_amount"] = np.log1p(out["amount"])
        out["amount_usd_ratio"] = out["amount"] / self._global_avg_amount
        out["discount_ratio"] = out["discount"] / (out["amount"] + 1e-8)
        out["has_tip"] = (out["tip"] > 0).astype(np.int8)
        return out

    def feature_names(self) -> List[str]:
        return ["amount", "log_amount", "amount_usd_ratio", "discount_ratio", "has_tip"]


# ── Grupo 2: Temporales ──────────────────────────────────────────────────────

class TemporalFeatures(FeatureGroup):
    """Features #6-10: codificación cíclica y flags temporales."""

    def fit(self, df_train: pd.DataFrame) -> "TemporalFeatures":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hour = out["created_at"].dt.hour
        dow = out["created_at"].dt.dayofweek + 1  # 1=Lunes … 7=Domingo (ISO)

        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["day_of_week"] = dow.astype(np.int8)
        out["is_weekend"] = (dow >= 6).astype(np.int8)
        out["is_off_hours"] = hour.isin([23, 0, 1, 2, 3, 4, 5, 6]).astype(np.int8)
        return out

    def feature_names(self) -> List[str]:
        return ["hour_sin", "hour_cos", "day_of_week", "is_weekend", "is_off_hours"]


# ── Grupo 3: Velocidad ────────────────────────────────────────────────────────

class VelocityFeatures(FeatureGroup):
    """Features #11-14: rolling windows con exclusión de la fila actual."""

    def fit(self, df_train: pd.DataFrame) -> "VelocityFeatures":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["user_txn_count_1h"] = self._rolling_count(out, "1h")
        out["user_txn_count_24h"] = self._rolling_count(out, "24h")
        out["time_since_last_txn"] = self._time_since_last(out)
        out["user_amount_24h"] = self._rolling_amount_sum(out, "24h")
        return out

    @staticmethod
    def _rolling_count(df: pd.DataFrame, window: str) -> pd.Series:
        """Conteo de transacciones del usuario en la ventana, excluyendo la fila actual."""
        result = (
            df.groupby("user_id")
            .rolling(window, on="created_at")["id"]
            .count()
            .droplevel(0)
        )
        return (result - 1).reindex(df.index).fillna(0).astype(np.float32)

    @staticmethod
    def _rolling_amount_sum(df: pd.DataFrame, window: str) -> pd.Series:
        """Suma de amount del usuario en la ventana, excluyendo el monto propio."""
        result = (
            df.groupby("user_id")
            .rolling(window, on="created_at")["amount"]
            .sum()
            .droplevel(0)
            .reset_index(drop=True)  # alinear por posición, no por timestamp
        )
        return pd.Series(
            result.values - df["amount"].values, index=df.index
        ).fillna(0).clip(lower=0).astype(np.float32)

    @staticmethod
    def _time_since_last(df: pd.DataFrame) -> pd.Series:
        """Segundos desde la última transacción del mismo usuario. Primera txn = 0."""
        return (
            df.groupby("user_id")["created_at"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
            .astype(np.float32)
        )

    def feature_names(self) -> List[str]:
        return [
            "user_txn_count_1h",
            "user_txn_count_24h",
            "time_since_last_txn",
            "user_amount_24h",
        ]


# ── Grupo 4: Comportamentales ─────────────────────────────────────────────────

class BehavioralFeatures(FeatureGroup):
    """Features #15-18: acumulados, ratio de reversión y antigüedad de cuenta."""

    def __init__(self) -> None:
        self._user_first_txn: Optional[Dict] = None

    def fit(self, df_train: pd.DataFrame) -> "BehavioralFeatures":
        self._user_first_txn = (
            df_train.groupby("user_id")["created_at"].min().to_dict()
        )
        logger.debug(f"BehavioralFeatures.fit — {len(self._user_first_txn)} usuarios en train")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._user_first_txn is None:
            raise RuntimeError("Llamar fit() antes de transform()")
        out = df.copy()
        out["user_distinct_facilities_cumul"] = self._cumulative_nunique(out, "facility_id")
        out["user_distinct_methods"] = self._cumulative_nunique(out, "payment_method")
        out["user_reversal_ratio_30d"] = self._reversal_ratio_30d(out)
        out["user_account_age_days"] = self._account_age_days(out)
        return out

    @staticmethod
    def _cumulative_nunique_fn(series: pd.Series) -> pd.Series:
        """Conteo acumulado de valores únicos PREVIOS — O(n), excluye el valor actual."""
        seen: set = set()
        result = []
        for val in series:
            result.append(len(seen))
            seen.add(val)
        return pd.Series(result, index=series.index, dtype=np.int32)

    def _cumulative_nunique(self, df: pd.DataFrame, col: str) -> pd.Series:
        return (
            df.groupby("user_id")[col]
            .transform(self._cumulative_nunique_fn)
            .astype(np.int32)
        )

    @staticmethod
    def _reversal_ratio_30d(df: pd.DataFrame) -> pd.Series:
        """Proporción rolling 30D de reversiones, shifted 1 dentro del grupo.

        ADVERTENCIA: usa `status` de forma derivada. Feature #17 requiere
        análisis de sensibilidad (Gate D).
        """
        df = df.copy()
        df["_is_reversal"] = df["status"].isin(
            ["totally_refunded", "refunded_to_credit"]
        ).astype(np.int8)

        df["_ratio_raw"] = (
            df.groupby("user_id")
            .rolling("30D", on="created_at")["_is_reversal"]
            .mean()
            .droplevel(0)
            .reindex(df.index)
        )
        shifted = df.groupby("user_id")["_ratio_raw"].shift(1).fillna(0)
        return shifted.astype(np.float32)

    def _account_age_days(self, df: pd.DataFrame) -> pd.Series:
        """Días desde la primera transacción del usuario (lookup en train).

        Usuarios no vistos en train usan su primera aparición en el split.
        """
        first_txn = df["user_id"].map(self._user_first_txn)
        new_users = first_txn.isna()
        if new_users.any():
            split_first = df.loc[new_users].groupby("user_id")["created_at"].transform("min")
            first_txn.loc[new_users] = split_first
        return (df["created_at"] - first_txn).dt.days.clip(lower=0).astype(np.int32)

    def feature_names(self) -> List[str]:
        return [
            "user_distinct_facilities_cumul",
            "user_distinct_methods",
            "user_reversal_ratio_30d",
            "user_account_age_days",
        ]


# ── Grupo 5: Contextuales ─────────────────────────────────────────────────────

class ContextualFeatures(FeatureGroup):
    """Features #19-20: lookup de estadísticas de facility calculadas en train."""

    def __init__(self) -> None:
        self._facility_avg_amount: Optional[Dict] = None
        self._global_avg_amount: Optional[float] = None

    def fit(self, df_train: pd.DataFrame) -> "ContextualFeatures":
        self._global_avg_amount = float(df_train["amount"].mean())
        self._facility_avg_amount = (
            df_train.groupby("facility_id")["amount"].mean().to_dict()
        )
        logger.debug(
            f"ContextualFeatures.fit — {len(self._facility_avg_amount)} facilities"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._facility_avg_amount is None:
            raise RuntimeError("Llamar fit() antes de transform()")
        out = df.copy()
        out["facility_avg_amount"] = (
            out["facility_id"]
            .map(self._facility_avg_amount)
            .fillna(self._global_avg_amount)
            .astype(np.float32)
        )
        out["amount_facility_ratio"] = (
            out["amount"] / (out["facility_avg_amount"] + 1e-8)
        ).astype(np.float32)
        return out

    def feature_names(self) -> List[str]:
        return ["facility_avg_amount", "amount_facility_ratio"]


# ── Compositor ────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """Genera las 20 features oficiales del catálogo.

    Patrón Compositor: delega a FeatureGroups individuales.
    Patrón fit/transform para evitar leakage.

    Uso:
        fe = FeatureEngineer()
        train_features = fe.fit_transform(df_train)       # con warm history ya prepended
        val_features   = fe.transform(df_val)
        test_features  = fe.transform(df_test)
    """

    def __init__(self, groups: Optional[List[FeatureGroup]] = None) -> None:
        self._groups: List[FeatureGroup] = groups or [
            TransactionalFeatures(),
            TemporalFeatures(),
            VelocityFeatures(),
            BehavioralFeatures(),
            ContextualFeatures(),
        ]
        self._fitted: bool = False

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame) -> "FeatureEngineer":
        """Aprende estadísticas del training set."""
        self._validate_required_columns(df_train)
        df_sorted = df_train.sort_values(["user_id", "created_at"]).reset_index(drop=True)
        for group in self._groups:
            group.fit(df_sorted)
        self._fitted = True
        logger.info("FeatureEngineer.fit() completado")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera las 20 features. Requiere fit() previo.

        Returns:
            DataFrame con FEATURE_NAMES + METADATA_COLS.
        """
        if not self._fitted:
            raise RuntimeError("Llamar fit() antes de transform()")
        self._validate_required_columns(df)

        # Prerequisito obligatorio: orden temporal por usuario
        out = df.sort_values(["user_id", "created_at"]).reset_index(drop=True)

        for group in self._groups:
            out = group.transform(out)

        available_meta = [c for c in METADATA_COLS if c in out.columns]
        return out[FEATURE_NAMES + available_meta].copy()

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform en un solo paso (solo para train)."""
        return self.fit(df_train).transform(df_train)

    def transform_with_warm_history(
        self,
        df_split: pd.DataFrame,
        df_warm: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute features prepending warm history, luego descarta filas warm.

        Uso:
          - Para train: df_warm = warm_history (Dic 2024)
          - Para val:   df_warm = últimas 30 días del train
          - Para test:  df_warm = últimas 30 días del val
        """
        if not self._fitted:
            raise RuntimeError("Llamar fit() antes de transform_with_warm_history()")

        _MARKER = "_is_split"
        df_split = df_split.assign(**{_MARKER: True})
        df_warm = df_warm.assign(**{_MARKER: False})

        combined = (
            pd.concat([df_warm, df_split], ignore_index=True)
            .sort_values(["user_id", "created_at"])
            .reset_index(drop=True)
        )

        # Transform sobre el combined (warm history ya ordena el contexto previo)
        combined_feat = self.transform(combined)

        # Recuperar el marcador del combined original (transform no lo retiene)
        split_idx = combined.loc[combined[_MARKER]].index
        result = combined_feat.loc[combined_feat.index.isin(split_idx)].reset_index(drop=True)
        return result

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serializar instancia con joblib (incluye dicts y estadísticas)."""
        joblib.dump(self, path)
        logger.info(f"FeatureEngineer guardado en {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        """Cargar instancia previamente guardada."""
        instance = joblib.load(path)
        logger.info(f"FeatureEngineer cargado desde {path}")
        return instance

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_feature_names() -> List[str]:
        return FEATURE_NAMES.copy()

    @staticmethod
    def get_feature_names_19() -> List[str]:
        return FEATURE_NAMES_19.copy()

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> None:
        required = {
            "id", "user_id", "facility_id", "created_at", "status",
            "amount", "discount", "tip", "payment_method",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
