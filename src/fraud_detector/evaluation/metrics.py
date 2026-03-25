"""Evaluation metrics for unsupervised anomaly scores."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from config.config import settings


def _as_arrays(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int8)
    y_scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("labels and scores must have the same length")
    return y_true, y_scores


def precision_at_k(labels, scores, k_pct: float = 0.05) -> float:
    y_true, y_scores = _as_arrays(labels, scores)
    if not 0 < k_pct <= 1:
        raise ValueError("k_pct must be in (0, 1]")

    k = max(1, int(np.ceil(len(y_true) * k_pct)))
    top_idx = np.argsort(y_scores)[-k:]
    return float(y_true[top_idx].mean())


def enrichment_factor(labels, scores, k_pct: float = 0.05) -> float:
    y_true, y_scores = _as_arrays(labels, scores)
    base_rate = float(y_true.mean())
    if base_rate == 0:
        return 0.0
    return precision_at_k(y_true, y_scores, k_pct=k_pct) / base_rate


def evaluate_scores(
    labels,
    scores,
    top_k_percents: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    y_true, y_scores = _as_arrays(labels, scores)
    metrics: Dict[str, float] = {
        "n_samples": float(len(y_true)),
        "base_rate": float(y_true.mean()),
    }

    if len(np.unique(y_true)) < 2:
        metrics["auc_roc"] = np.nan
        metrics["ap"] = np.nan
    else:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_scores))
        metrics["ap"] = float(average_precision_score(y_true, y_scores))

    for k_pct in (top_k_percents or settings.top_k_percents_list):
        label = int(round(k_pct * 100))
        metrics[f"precision_at_{label}pct"] = precision_at_k(y_true, y_scores, k_pct)
        metrics[f"enrichment_factor_at_{label}pct"] = enrichment_factor(
            y_true,
            y_scores,
            k_pct,
        )

    return metrics


def bootstrap_ci(
    labels,
    scores,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: Optional[int] = None,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    y_true, y_scores = _as_arrays(labels, scores)
    iterations = n_iterations or settings.bootstrap_n
    rng = np.random.RandomState(random_seed)
    values = []

    for _ in range(iterations):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        sample_value = metric_fn(y_true[idx], y_scores[idx])
        values.append(sample_value)

    sample = np.asarray(values, dtype=np.float64)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.nanmean(sample)),
        "lower": float(np.nanquantile(sample, alpha)),
        "upper": float(np.nanquantile(sample, 1 - alpha)),
        "std": float(np.nanstd(sample)),
        "n_iterations": int(iterations),
    }


class FraudMetrics:
    """Compatibility wrapper around anomaly-score evaluation."""

    def compute_all_metrics(self, y_true, y_pred=None, y_pred_proba=None) -> Dict[str, float]:
        scores = y_pred_proba if y_pred_proba is not None else y_pred
        if scores is None:
            raise ValueError("Provide anomaly scores via y_pred or y_pred_proba")
        return evaluate_scores(y_true, scores)
