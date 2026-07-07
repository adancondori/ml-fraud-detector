"""Preprocessing for the unsupervised anomaly-detection pipeline."""

from __future__ import annotations

from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud_detector.features.engineering import (
    FEATURE_NAMES,
    FEATURE_NAMES_21,
    FEATURE_NAMES_30,
)
from fraud_detector.utils.logger import logger

FeatureVariant = Literal["full", "sensitivity", "core"]


class UnsupervisedPreprocessor:
    """Fit-on-train StandardScaler over numeric feature sets."""

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        variant: FeatureVariant = "full",
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.feature_names = feature_names or self._resolve_variant(variant)
        self.variant = variant
        self.scaler = scaler or StandardScaler()
        self._fitted = False

    @staticmethod
    def _resolve_variant(variant: FeatureVariant) -> List[str]:
        if variant == "full":
            return FEATURE_NAMES.copy()
        if variant == "sensitivity":
            return FEATURE_NAMES_30.copy()
        if variant == "core":
            return FEATURE_NAMES_21.copy()
        raise ValueError(f"Unknown feature variant '{variant}'")

    def fit(self, df: pd.DataFrame) -> "UnsupervisedPreprocessor":
        X = self._extract_matrix(df)
        self.scaler.fit(X)
        self._fitted = True
        logger.info(f"Preprocessor fitted on {X.shape[0]:,} rows and {X.shape[1]} features")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        X = self._extract_matrix(df)
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "UnsupervisedPreprocessor":
        instance = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return instance

    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()

    def _extract_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [feature for feature in self.feature_names if feature not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for preprocessing: {missing}")

        values = df[self.feature_names].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Preprocessor input contains NaN or infinite values")
        return values
