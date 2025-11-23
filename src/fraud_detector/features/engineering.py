"""
Feature engineering específico para detección de fraude.

Este módulo implementa features críticos para detectar patrones de fraude:
- Temporal features
- Aggregation features
- Velocity features
- Behavioral features
"""
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from config.config import settings
from fraud_detector.utils.logger import logger


class FraudFeatureEngineer:
    """
    Ingeniero de features específico para detección de fraude.

    Features implementados:
    1. Temporal: hora del día, día de la semana, festivos
    2. Aggregations: stats por usuario en ventanas de tiempo
    3. Velocity: velocidad de transacciones
    4. Behavioral: desviaciones del comportamiento normal
    """

    def __init__(
        self,
        user_col: str = "user_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
        merchant_col: Optional[str] = "merchant_id",
    ):
        """
        Inicializa el feature engineer.

        Args:
            user_col: Nombre de columna de usuario
            timestamp_col: Nombre de columna de timestamp
            amount_col: Nombre de columna de monto
            merchant_col: Nombre de columna de comerciante
        """
        self.user_col = user_col
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        self.merchant_col = merchant_col

        logger.info("FraudFeatureEngineer inicializado")

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features temporales.

        Features:
        - hour_of_day: Hora del día (0-23)
        - day_of_week: Día de la semana (0-6)
        - is_weekend: Si es fin de semana
        - is_night: Si es de noche (10pm-6am)
        - is_early_morning: Si es madrugada (12am-6am)
        """
        df = df.copy()

        logger.info("Creando features temporales...")

        # Asegurar que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Hora del día
        df["hour_of_day"] = df[self.timestamp_col].dt.hour

        # Día de la semana (0=Lunes, 6=Domingo)
        df["day_of_week"] = df[self.timestamp_col].dt.dayofweek

        # Fin de semana
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Noche (10pm - 6am) - alto riesgo de fraude
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)).astype(
            int
        )

        # Madrugada (12am - 6am) - muy alto riesgo
        df["is_early_morning"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(
            int
        )

        # Día del mes
        df["day_of_month"] = df[self.timestamp_col].dt.day

        # Mes
        df["month"] = df[self.timestamp_col].dt.month

        logger.info(f"✅ Creadas 7 features temporales")

        return df

    def create_aggregation_features(
        self, df: pd.DataFrame, windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Crea features de agregación por usuario en ventanas de tiempo.

        Args:
            df: DataFrame con transacciones
            windows: Lista de ventanas en días (default: [1, 7, 30])

        Features por ventana:
        - n_transactions_{window}d: Número de transacciones
        - total_amount_{window}d: Monto total
        - mean_amount_{window}d: Monto promedio
        - std_amount_{window}d: Desviación estándar del monto
        - max_amount_{window}d: Monto máximo
        """
        df = df.copy()

        if windows is None:
            windows = settings.aggregation_windows_list

        logger.info(f"Creando features de agregación para ventanas: {windows} días")

        # Asegurar que está ordenado por timestamp
        df = df.sort_values([self.user_col, self.timestamp_col])

        # Asegurar que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        features_created = 0

        for window_days in windows:
            logger.debug(f"  Procesando ventana de {window_days} días...")

            # Timestamp de corte
            df[f"_cutoff_{window_days}d"] = df[self.timestamp_col] - timedelta(
                days=window_days
            )

            # Agrupar por usuario
            for user_id in df[self.user_col].unique():
                user_mask = df[self.user_col] == user_id
                user_data = df[user_mask].copy()

                for idx, row in user_data.iterrows():
                    # Transacciones en la ventana
                    window_mask = (
                        (user_data[self.timestamp_col] >= row[f"_cutoff_{window_days}d"])
                        & (user_data[self.timestamp_col] < row[self.timestamp_col])
                    )

                    window_txns = user_data[window_mask]

                    # Número de transacciones
                    df.loc[idx, f"n_transactions_{window_days}d"] = len(window_txns)

                    if len(window_txns) >= settings.min_transactions_for_aggregation:
                        # Monto total
                        df.loc[idx, f"total_amount_{window_days}d"] = window_txns[
                            self.amount_col
                        ].sum()

                        # Monto promedio
                        df.loc[idx, f"mean_amount_{window_days}d"] = window_txns[
                            self.amount_col
                        ].mean()

                        # Desviación estándar
                        df.loc[idx, f"std_amount_{window_days}d"] = window_txns[
                            self.amount_col
                        ].std()

                        # Monto máximo
                        df.loc[idx, f"max_amount_{window_days}d"] = window_txns[
                            self.amount_col
                        ].max()
                    else:
                        # Valores por defecto si no hay suficientes transacciones
                        df.loc[idx, f"total_amount_{window_days}d"] = 0
                        df.loc[idx, f"mean_amount_{window_days}d"] = 0
                        df.loc[idx, f"std_amount_{window_days}d"] = 0
                        df.loc[idx, f"max_amount_{window_days}d"] = 0

            # Limpiar columna temporal
            df = df.drop(columns=[f"_cutoff_{window_days}d"])

            features_created += 5  # 5 features por ventana

        # Fill NaN con 0
        agg_cols = [col for col in df.columns if any(f"_{w}d" in col for w in windows)]
        df[agg_cols] = df[agg_cols].fillna(0)

        logger.info(f"✅ Creadas {features_created} features de agregación")

        return df

    def create_velocity_features(
        self, df: pd.DataFrame, windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Crea features de velocidad de transacciones.

        Args:
            df: DataFrame con transacciones
            windows: Lista de ventanas en horas (default: [1, 6, 24])

        Features:
        - transactions_per_hour_{window}h: Transacciones por hora
        - amount_per_hour_{window}h: Monto por hora
        """
        df = df.copy()

        if windows is None:
            windows = settings.velocity_windows_list

        logger.info(f"Creando features de velocidad para ventanas: {windows} horas")

        # Asegurar que está ordenado por timestamp
        df = df.sort_values([self.user_col, self.timestamp_col])

        # Asegurar que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        features_created = 0

        for window_hours in windows:
            logger.debug(f"  Procesando ventana de {window_hours} horas...")

            # Timestamp de corte
            df[f"_cutoff_{window_hours}h"] = df[self.timestamp_col] - timedelta(
                hours=window_hours
            )

            # Agrupar por usuario
            for user_id in df[self.user_col].unique():
                user_mask = df[self.user_col] == user_id
                user_data = df[user_mask].copy()

                for idx, row in user_data.iterrows():
                    # Transacciones en la ventana
                    window_mask = (
                        (user_data[self.timestamp_col] >= row[f"_cutoff_{window_hours}h"])
                        & (user_data[self.timestamp_col] < row[self.timestamp_col])
                    )

                    window_txns = user_data[window_mask]
                    n_txns = len(window_txns)

                    # Transacciones por hora
                    df.loc[idx, f"transactions_per_hour_{window_hours}h"] = (
                        n_txns / window_hours if window_hours > 0 else 0
                    )

                    # Monto por hora
                    total_amount = window_txns[self.amount_col].sum()
                    df.loc[idx, f"amount_per_hour_{window_hours}h"] = (
                        total_amount / window_hours if window_hours > 0 else 0
                    )

            # Limpiar columna temporal
            df = df.drop(columns=[f"_cutoff_{window_hours}h"])

            features_created += 2  # 2 features por ventana

        # Fill NaN con 0
        velocity_cols = [
            col for col in df.columns if any(f"_{w}h" in col for w in windows)
        ]
        df[velocity_cols] = df[velocity_cols].fillna(0)

        logger.info(f"✅ Creadas {features_created} features de velocidad")

        return df

    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de comportamiento del usuario.

        Features:
        - amount_deviation: Desviación del monto respecto al promedio del usuario
        - is_new_merchant: Si el comerciante es nuevo para el usuario
        - time_since_last_transaction: Tiempo desde última transacción (horas)
        """
        df = df.copy()

        logger.info("Creando features de comportamiento...")

        # Asegurar que está ordenado
        df = df.sort_values([self.user_col, self.timestamp_col])

        # Asegurar que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Calcular monto promedio histórico por usuario
        user_avg_amount = (
            df.groupby(self.user_col)[self.amount_col]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Desviación del monto
        df["amount_deviation"] = (
            df[self.amount_col] - user_avg_amount
        ) / (user_avg_amount + 1e-6)

        # Tiempo desde última transacción
        df["time_since_last_transaction"] = (
            df.groupby(self.user_col)[self.timestamp_col]
            .diff()
            .dt.total_seconds()
            / 3600  # convertir a horas
        )

        # Fill NaN para primera transacción
        df["time_since_last_transaction"] = df["time_since_last_transaction"].fillna(0)

        # Si hay columna de merchant
        if self.merchant_col and self.merchant_col in df.columns:
            # Marcar si es un merchant nuevo para el usuario
            df["is_new_merchant"] = 0

            for user_id in df[self.user_col].unique():
                user_mask = df[self.user_col] == user_id
                user_data = df[user_mask].copy()

                seen_merchants = set()
                for idx, row in user_data.iterrows():
                    merchant = row[self.merchant_col]
                    if merchant not in seen_merchants:
                        df.loc[idx, "is_new_merchant"] = 1
                        seen_merchants.add(merchant)

            logger.info("✅ Creadas 3 features de comportamiento (con merchant)")
        else:
            logger.info("✅ Creadas 2 features de comportamiento (sin merchant)")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todos los feature engineering.

        Args:
            df: DataFrame con transacciones

        Returns:
            DataFrame con todos los features agregados
        """
        logger.info("=" * 60)
        logger.info("Iniciando Feature Engineering para Detección de Fraude")
        logger.info("=" * 60)

        df = df.copy()

        # 1. Features temporales
        df = self.create_temporal_features(df)

        # 2. Features de agregación
        df = self.create_aggregation_features(df)

        # 3. Features de velocidad
        df = self.create_velocity_features(df)

        # 4. Features de comportamiento
        df = self.create_behavioral_features(df)

        logger.info("=" * 60)
        logger.info(f"✅ Feature Engineering completado")
        logger.info(f"   Features originales: {len(df.columns) - self._count_new_features(df)}")
        logger.info(f"   Features nuevos: {self._count_new_features(df)}")
        logger.info(f"   Total features: {len(df.columns)}")
        logger.info("=" * 60)

        return df

    def _count_new_features(self, df: pd.DataFrame) -> int:
        """Cuenta features nuevos creados."""
        new_feature_patterns = [
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_night",
            "is_early_morning",
            "day_of_month",
            "month",
            "_transactions_",
            "_amount_",
            "amount_deviation",
            "time_since_last_transaction",
            "is_new_merchant",
        ]

        count = 0
        for col in df.columns:
            if any(pattern in col for pattern in new_feature_patterns):
                count += 1

        return count
