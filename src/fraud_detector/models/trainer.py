"""Training helpers for unsupervised anomaly-detection models."""
from __future__ import annotations

from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from config.config import settings
from fraud_detector.evaluation.metrics import evaluate_scores
from fraud_detector.utils.logger import logger


class ModelTrainer:
    """Train and score unsupervised anomaly models."""

    MODEL_REGISTRY = {
        "isolation_forest": IsolationForest,
        "lof": LocalOutlierFactor,
        "ocsvm": OneClassSVM,
    }

    def __init__(
        self,
        model_type: str = "isolation_forest",
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(self.MODEL_REGISTRY)}"
            )
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None

    def _default_params(self) -> Dict[str, Any]:
        if self.model_type == "isolation_forest":
            return {
                "n_estimators": 200,
                "max_samples": 512,
                "max_features": 1.0,
                "contamination": "auto",
                "random_state": settings.random_seed,
                "n_jobs": settings.n_jobs,
            }
        if self.model_type == "lof":
            return {
                "n_neighbors": 20,
                "novelty": True,
                "metric": "minkowski",
            }
        return {
            "kernel": "rbf",
            "nu": 0.05,
            "gamma": "scale",
        }

    def fit(self, X_train: np.ndarray) -> "ModelTrainer":
        params = self._default_params()
        params.update(self.model_params)
        self.model = self.MODEL_REGISTRY[self.model_type](**params)
        self.model.fit(X_train)
        logger.info(
            f"Fitted {self.model_type} on {len(X_train):,} rows with params={params}"
        )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if not hasattr(self.model, "score_samples"):
            raise ValueError(f"Model '{self.model_type}' does not expose score_samples()")

        raw_scores = self.model.score_samples(X)
        anomaly_scores = -np.asarray(raw_scores, dtype=np.float64)
        return anomaly_scores.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        raw = np.asarray(self.model.predict(X))
        return np.where(raw == -1, 1, 0).astype(np.int8)

    @staticmethod
    def predict_top_k(scores: np.ndarray, k_pct: float = 0.05) -> np.ndarray:
        if not 0 < k_pct <= 1:
            raise ValueError("k_pct must be in (0, 1]")
        n = len(scores)
        k = max(1, int(np.ceil(n * k_pct)))
        threshold = np.partition(scores, -k)[-k]
        return (scores >= threshold).astype(np.int8)

    def evaluate(
        self,
        X: np.ndarray,
        proxy_labels: np.ndarray,
        top_k_percents: Optional[list[float]] = None,
    ) -> Dict[str, float]:
        scores = self.score_samples(X)
        return evaluate_scores(proxy_labels, scores, top_k_percents=top_k_percents)

    def save_model(self, output_path: str) -> None:
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        joblib.dump(self.model, output_path)
        logger.info(f"Saved {self.model_type} model to {output_path}")

    def load_model(self, model_path: str) -> None:
        self.model = joblib.load(model_path)
        logger.info(f"Loaded {self.model_type} model from {model_path}")
