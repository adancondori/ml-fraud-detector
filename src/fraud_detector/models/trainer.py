"""Training helpers for unsupervised anomaly-detection models.

Supports Isolation Forest, LOF (novelty=True), and One-Class SVM.
Score convention: higher = more anomalous (negate decision_function).
Grid search with checkpoint/resume. Multi-seed variability analysis.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from config.config import settings
from fraud_detector.evaluation.metrics import evaluate_scores
from fraud_detector.utils.logger import logger


class ModelTrainer:
    """Train and score unsupervised anomaly models.

    Score convention: higher = more anomalous.
    Uses -decision_function(X) as the canonical scoring method for all models.
    """

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
                f"Unknown model type '{model_type}'. " f"Available: {list(self.MODEL_REGISTRY)}"
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
                "n_jobs": settings.n_jobs,
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
        logger.info(f"Fitted {self.model_type} on {len(X_train):,} rows")
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """IF-specific scoring via -score_samples(). Higher = more anomalous."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if not hasattr(self.model, "score_samples"):
            raise ValueError(f"Model '{self.model_type}' does not expose score_samples()")
        raw_scores = self.model.score_samples(X)
        return (-np.asarray(raw_scores, dtype=np.float64)).astype(np.float32)

    def decision_function_scores(self, X: np.ndarray) -> np.ndarray:
        """Canonical scoring via -decision_function(). Works for all 3 models.

        Higher = more anomalous. This is the method used during grid search
        and evaluation for consistent comparison across IF, LOF, and OC-SVM.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        raw = self.model.decision_function(X)
        return (-np.asarray(raw, dtype=np.float64)).astype(np.float32)

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
        scores = self.decision_function_scores(X)
        return evaluate_scores(proxy_labels, scores, top_k_percents=top_k_percents)

    def save_model(self, output_path: str) -> None:
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        joblib.dump(self.model, output_path)
        logger.info(f"Saved {self.model_type} model to {output_path}")

    def load_model(self, model_path: str) -> None:
        self.model = joblib.load(model_path)
        logger.info(f"Loaded {self.model_type} model from {model_path}")

    # ── Static helpers ──────────────────────────────────────────────

    @staticmethod
    def _subsample_temporal(X: np.ndarray, n: int = 100_000) -> np.ndarray:
        """Equi-spaced temporal subsample (preserves distribution)."""
        if X.shape[0] <= n:
            return X
        indices = np.linspace(0, X.shape[0] - 1, n, dtype=int)
        return X[indices]

    # ── Grid Search: Isolation Forest ───────────────────────────────

    @staticmethod
    def grid_search_if(
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val_proxy: np.ndarray,
        param_grid: Dict[str, list],
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 20,
        random_state: int = 42,
        max_combos: Optional[int] = None,
    ) -> pd.DataFrame:
        """Grid search IF with checkpoint/resume.

        Scoring: -score_samples(X) for ranking (invariant to contamination).
        Metric: AUC-ROC on validation proxy.
        """
        all_combos = list(ParameterGrid(param_grid))
        if max_combos is not None:
            all_combos = all_combos[:max_combos]

        results: list[dict] = []
        completed = 0

        if checkpoint_path and os.path.exists(checkpoint_path):
            existing = pd.read_csv(checkpoint_path)
            results = existing.to_dict("records")
            completed = len(results)
            logger.info(f"Resuming IF grid search from combo {completed}/{len(all_combos)}")

        for i, params in enumerate(all_combos[completed:], start=completed):
            t0 = time.perf_counter()
            model = IsolationForest(random_state=random_state, n_jobs=-1, **params)
            model.fit(X_train)
            scores_val = -model.score_samples(X_val)
            auc = float(roc_auc_score(y_val_proxy, scores_val))
            elapsed = time.perf_counter() - t0

            results.append({**params, "auc_roc": auc, "time_seconds": round(elapsed, 2)})
            logger.info(
                f"IF [{i + 1}/{len(all_combos)}] "
                f"n_est={params.get('n_estimators')} "
                f"max_s={params.get('max_samples')} "
                f"max_f={params.get('max_features')} "
                f"cont={params.get('contamination')} "
                f"AUC={auc:.6f} ({elapsed:.1f}s)"
            )

            if checkpoint_path and (i + 1) % checkpoint_every == 0:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

        df = pd.DataFrame(results)
        if checkpoint_path:
            df.to_csv(checkpoint_path, index=False)
        return df

    # ── Grid Search: LOF ────────────────────────────────────────────

    @staticmethod
    def grid_search_lof(
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val_proxy: np.ndarray,
        n_neighbors_list: List[int],
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Grid search LOF (novelty=True). Scoring: -decision_function."""
        results: list[dict] = []
        for n_neighbors in n_neighbors_list:
            t0 = time.perf_counter()
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                novelty=True,
                metric="minkowski",
                n_jobs=-1,
            )
            model.fit(X_train)
            scores_val = -model.decision_function(X_val)
            auc = float(roc_auc_score(y_val_proxy, scores_val))
            elapsed = time.perf_counter() - t0

            results.append(
                {
                    "n_neighbors": n_neighbors,
                    "auc_roc": auc,
                    "time_seconds": round(elapsed, 2),
                }
            )
            logger.info(f"LOF n_neighbors={n_neighbors}: AUC={auc:.6f} ({elapsed:.1f}s)")

        return pd.DataFrame(results)

    # ── Grid Search: OC-SVM ─────────────────────────────────────────

    @staticmethod
    def grid_search_ocsvm(
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val_proxy: np.ndarray,
        param_grid: Dict[str, list],
        n_subsample: int = 100_000,
    ) -> pd.DataFrame:
        """Grid search OC-SVM with temporal subsample. Scoring: -decision_function."""
        X_sub = ModelTrainer._subsample_temporal(X_train, n=n_subsample)
        all_combos = list(ParameterGrid(param_grid))
        results: list[dict] = []

        for i, params in enumerate(all_combos):
            t0 = time.perf_counter()
            model = OneClassSVM(kernel="rbf", **params)
            model.fit(X_sub)
            scores_val = -model.decision_function(X_val)
            auc = float(roc_auc_score(y_val_proxy, scores_val))
            elapsed = time.perf_counter() - t0

            results.append({**params, "auc_roc": auc, "time_seconds": round(elapsed, 2)})
            logger.info(
                f"OC-SVM [{i + 1}/{len(all_combos)}] "
                f"nu={params.get('nu')} gamma={params.get('gamma')} "
                f"AUC={auc:.6f} ({elapsed:.1f}s)"
            )

        return pd.DataFrame(results)

    # ── Multi-seed ──────────────────────────────────────────────────

    @staticmethod
    def run_multi_seed(
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val_proxy: np.ndarray,
        best_params: Dict[str, Any],
        seeds: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Train IF with multiple seeds and report AUC variability."""
        seeds = seeds or settings.multi_seeds_list
        results: list[dict] = []

        for seed in seeds:
            # Filter out non-IF params like contamination from best_params
            model_params = {
                k: v
                for k, v in best_params.items()
                if k in ("n_estimators", "max_samples", "max_features")
            }
            model = IsolationForest(
                contamination="auto",
                random_state=seed,
                n_jobs=-1,
                **model_params,
            )
            model.fit(X_train)
            scores = -model.score_samples(X_val)
            auc = float(roc_auc_score(y_val_proxy, scores))
            results.append({"seed": seed, "auc_roc": auc})
            logger.info(f"Multi-seed IF seed={seed}: AUC={auc:.6f}")

        df = pd.DataFrame(results)
        mean_auc = df["auc_roc"].mean()
        range_auc = df["auc_roc"].max() - df["auc_roc"].min()
        logger.info(f"Multi-seed summary: mean={mean_auc:.6f}, range={range_auc:.6f}")
        if range_auc < 0.005:
            logger.info("Trivial dispersion — single seed (42) is sufficient")
        else:
            logger.warning(f"Non-trivial dispersion (range={range_auc:.6f})")
        return df
