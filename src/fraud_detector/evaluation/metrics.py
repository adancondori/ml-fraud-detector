"""Evaluation metrics for unsupervised anomaly scores."""

from __future__ import annotations

import inspect
from typing import Callable, Dict, Iterable, Literal, Optional

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

    for k_pct in top_k_percents or settings.top_k_percents_list:
        label = int(round(k_pct * 100))
        metrics[f"precision_at_{label}pct"] = precision_at_k(y_true, y_scores, k_pct)
        metrics[f"enrichment_factor_at_{label}pct"] = enrichment_factor(
            y_true,
            y_scores,
            k_pct,
        )

    return metrics


def _supports_sample_weight(metric_fn: Callable) -> bool:
    try:
        return "sample_weight" in inspect.signature(metric_fn).parameters
    except (TypeError, ValueError):
        return False


def bootstrap_ci(
    labels,
    scores,
    metric_fn: Callable[..., float],
    n_iterations: Optional[int] = None,
    ci: float = 0.95,
    random_seed: int = 42,
    user_ids: Optional[np.ndarray] = None,
    method: Literal["auto", "weighted", "concatenate"] = "auto",
) -> Dict[str, float]:
    """Bootstrap CI for a metric.

    When ``user_ids`` is provided, performs a clustered bootstrap that resamples
    users (not transactions), preserving within-user serial dependence induced
    by rolling features. Two routes are available:

    * ``method="weighted"``: resamples users, converts user frequencies into
      ``sample_weight`` per row, and calls ``metric_fn(y, s, sample_weight=w)``.
      Requires ``metric_fn`` to accept ``sample_weight`` (e.g.
      ``sklearn.metrics.roc_auc_score`` and ``average_precision_score``). Fast
      O(n) per iteration.
    * ``method="concatenate"``: resamples users and concatenates their row
      indices. Works with any ``metric_fn(y, s) -> float``.
    * ``method="auto"``: picks ``weighted`` if supported, else ``concatenate``.

    Back-compat: when ``user_ids is None`` the function behaves exactly as
    before (identical bit-for-bit output for the same ``random_seed``). The
    returned dict in that case does NOT contain the ``cluster_unit`` key.
    """
    y_true, y_scores = _as_arrays(labels, scores)
    iterations = n_iterations or settings.bootstrap_n
    rng = np.random.RandomState(random_seed)
    values = []

    if user_ids is None:
        # ── Legacy path: bootstrap por transacción, intacto ───────────
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

    # ── Clustered bootstrap by user ──────────────────────────────────
    uids = np.asarray(user_ids)
    if uids.shape[0] != y_true.shape[0]:
        raise ValueError("user_ids must have the same length as labels")

    if method == "auto":
        method = "weighted" if _supports_sample_weight(metric_fn) else "concatenate"
    if method == "weighted" and not _supports_sample_weight(metric_fn):
        raise ValueError("method='weighted' requires metric_fn to accept sample_weight")

    unique_users, inverse = np.unique(uids, return_inverse=True)
    n_users = int(unique_users.size)
    n = int(y_true.shape[0])

    if method == "weighted":
        y_f = y_true.astype(np.float64)
        for _ in range(iterations):
            sampled = rng.randint(0, n_users, size=n_users)
            counts = np.bincount(sampled, minlength=n_users)
            w = counts[inverse]  # weight per row
            pos_weight = float(np.dot(y_f, w))
            total_weight = float(w.sum())
            if pos_weight == 0.0 or pos_weight == total_weight:
                values.append(np.nan)
                continue
            values.append(metric_fn(y_true, y_scores, sample_weight=w))
    else:
        # concatenate route — precompute groups as offsets into flat index array
        order = np.argsort(inverse, kind="stable")
        sorted_inverse = inverse[order]
        boundaries = np.searchsorted(sorted_inverse, np.arange(n_users + 1))
        groups = [order[boundaries[k] : boundaries[k + 1]] for k in range(n_users)]
        for _ in range(iterations):
            sampled = rng.randint(0, n_users, size=n_users)
            idx = np.concatenate([groups[k] for k in sampled])
            y_boot = y_true[idx]
            pos_count = int(y_boot.sum())
            if pos_count == 0 or pos_count == idx.size:
                values.append(np.nan)
                continue
            values.append(metric_fn(y_boot, y_scores[idx]))

    sample = np.asarray(values, dtype=np.float64)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.nanmean(sample)),
        "lower": float(np.nanquantile(sample, alpha)),
        "upper": float(np.nanquantile(sample, 1 - alpha)),
        "std": float(np.nanstd(sample)),
        "n_iterations": int(iterations),
        "cluster_unit": "user",
        "n_clusters_resampled": n_users,
        "method_used": method,
    }


class FraudMetrics:
    """Compatibility wrapper around anomaly-score evaluation."""

    def compute_all_metrics(self, y_true, y_pred=None, y_pred_proba=None) -> Dict[str, float]:
        scores = y_pred_proba if y_pred_proba is not None else y_pred
        if scores is None:
            raise ValueError("Provide anomaly scores via y_pred or y_pred_proba")
        return evaluate_scores(y_true, scores)
