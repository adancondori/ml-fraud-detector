"""
Feature preprocessing and transformation.
Handles scaling, encoding, and feature transformations.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from fraud_detector.utils.logger import logger


class FeaturePreprocessor:
    """
    Handle feature preprocessing including scaling and encoding.

    Features:
    - Automatic detection of numeric and categorical features
    - Configurable scaling (standard, robust)
    - One-hot encoding for categorical features
    - Handles missing values
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scaler_type: str = "robust",
    ):
        """
        Initialize preprocessor.

        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.scaler_type = scaler_type
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_: Optional[List[str]] = None

        logger.info(f"Initialized preprocessor with {scaler_type} scaling")

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        """
        Fit preprocessor on training data.

        Args:
            X: Training features

        Returns:
            Self
        """
        logger.info("Fitting preprocessor...")

        # Auto-detect feature types if not provided
        if not self.numeric_features and not self.categorical_features:
            self._auto_detect_features(X)

        # Create transformers
        transformers = []

        if self.numeric_features:
            scaler = (
                StandardScaler() if self.scaler_type == "standard" else RobustScaler()
            )
            numeric_transformer = Pipeline(steps=[("scaler", scaler)])

            transformers.append(
                ("num", numeric_transformer, self.numeric_features)
            )
            logger.info(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features[:5]}...")

        if self.categorical_features:
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    )
                ]
            )

            transformers.append(
                ("cat", categorical_transformer, self.categorical_features)
            )
            logger.info(
                f"Categorical features ({len(self.categorical_features)}): "
                f"{self.categorical_features[:5]}..."
            )

        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )

        # Fit preprocessor
        self.preprocessor.fit(X)

        # Store feature names after transformation
        self.feature_names_ = self._get_feature_names()

        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names_)}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.

        Args:
            X: Features to transform

        Returns:
            Transformed feature array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        logger.debug(f"Transforming {len(X)} samples...")
        X_transformed = self.preprocessor.transform(X)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform features.

        Args:
            X: Features to fit and transform

        Returns:
            Transformed feature array
        """
        return self.fit(X).transform(X)

    def _auto_detect_features(self, X: pd.DataFrame) -> None:
        """Automatically detect numeric and categorical features."""
        self.numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        self.categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.info(
            f"Auto-detected {len(self.numeric_features)} numeric and "
            f"{len(self.categorical_features)} categorical features"
        )

    def _get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        if self.preprocessor is None:
            return []

        feature_names = []

        for name, transformer, features in self.preprocessor.transformers_:
            if name == "remainder":
                continue

            if name == "num":
                feature_names.extend(features)
            elif name == "cat":
                # Get one-hot encoded feature names
                onehot = transformer.named_steps["onehot"]
                cat_features = onehot.get_feature_names_out(features)
                feature_names.extend(cat_features)

        return feature_names

    def get_feature_names(self) -> List[str]:
        """Get output feature names."""
        if self.feature_names_ is None:
            raise ValueError("Preprocessor not fitted yet.")
        return self.feature_names_


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.

    Args:
        df: Input DataFrame
        numeric_strategy: Strategy for numeric columns ('mean', 'median', 'drop')
        categorical_strategy: Strategy for categorical columns ('mode', 'unknown', 'drop')

    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()

    missing_counts = df.isnull().sum()
    if missing_counts.sum() == 0:
        logger.info("No missing values found")
        return df

    logger.info(f"Handling missing values: {missing_counts[missing_counts > 0].to_dict()}")

    # Numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if numeric_strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif numeric_strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif numeric_strategy == "drop":
                df = df.dropna(subset=[col])

    # Categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            if categorical_strategy == "mode":
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col].fillna(mode_val, inplace=True)
            elif categorical_strategy == "unknown":
                df[col].fillna("unknown", inplace=True)
            elif categorical_strategy == "drop":
                df = df.dropna(subset=[col])

    logger.info("Missing values handled successfully")

    return df
