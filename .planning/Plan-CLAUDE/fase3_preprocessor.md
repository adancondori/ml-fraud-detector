# Fase 3: Preprocessor

## Archivo: `src/fraud_detector/features/preprocessor.py`

### Eliminar
Class `FeaturePreprocessor` and function `handle_missing_values()`.

### Nueva clase: `UnsupervisedPreprocessor`

```python
class UnsupervisedPreprocessor:
    """
    StandardScaler for unsupervised anomaly detection.
    Fits on training features, transforms val/test consistently.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._fitted = False

    def fit(self, X_train: pd.DataFrame, feature_names: List[str]) -> "UnsupervisedPreprocessor":
        self.feature_names = feature_names
        # Validate all features exist
        missing = set(feature_names) - set(X_train.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        self.scaler.fit(X_train[feature_names])
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        X = self.scaler.transform(df[self.feature_names])
        return X.astype(np.float32)

    def fit_transform(self, X_train: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        return self.fit(X_train, feature_names).transform(X_train)

    def save(self, path: Path) -> None:
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: Path) -> "UnsupervisedPreprocessor":
        data = joblib.load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj._fitted = True
        return obj
```

Why StandardScaler: OC-SVM with RBF kernel is sensitive to scale. StandardScaler enables fair comparison across all 3 models. log_amount and ratios already handle skew.

Output as float32: ~50% memory reduction vs float64.

NaN policy: Should NOT exist at this point (engineering.py fills with 0). If present, scaler raises ValueError (fail fast).

### Imports
```python
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fraud_detector.utils.logger import logger
```

### __init__.py update
```python
"""Feature engineering modules."""
from fraud_detector.features.engineering import FeatureEngineer, FEATURE_NAMES
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
__all__ = ["FeatureEngineer", "FEATURE_NAMES", "UnsupervisedPreprocessor"]
```
