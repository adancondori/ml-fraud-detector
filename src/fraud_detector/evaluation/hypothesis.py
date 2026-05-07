"""Hypothesis testing for HE1-HE4, Holm-Bonferroni, and model comparison.

All functions receive pre-computed scores and proxy labels.
Score convention: higher = more anomalous.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from config.config import settings
from fraud_detector.evaluation.metrics import (
    bootstrap_ci,
    enrichment_factor,
    precision_at_k,
)
from fraud_detector.utils.logger import logger


def _as_arrays(scores, proxy):
    s = np.asarray(scores, dtype=np.float64)
    p = np.asarray(proxy, dtype=np.int8)
    if s.shape[0] != p.shape[0]:
        raise ValueError("scores and proxy must have the same length")
    return s, p


# ── HE1: Statistical separation ────────────────────────────────


def run_mann_whitney(
    scores: np.ndarray,
    proxy: np.ndarray,
    alpha: float = None,
    min_r: float = None,
) -> Dict:
    """HE1: Mann-Whitney U test + rank-biserial effect size.

    Tests whether proxy+ transactions have significantly higher anomaly scores
    than proxy- transactions (one-sided, 'greater').

    Returns dict with U_statistic, p_value, rank_biserial_r, cles, he1_pass.
    """
    alpha = alpha or settings.he1_alpha
    min_r = min_r or settings.he1_min_rank_biserial
    s, p = _as_arrays(scores, proxy)

    scores_pos = s[p == 1]
    scores_neg = s[p == 0]
    n1, n2 = len(scores_pos), len(scores_neg)

    U, p_value = mannwhitneyu(scores_pos, scores_neg, alternative="greater")

    # Rank-biserial r: positive when anomaly > normal
    r = np.clip(2 * U / (n1 * n2) - 1, -1.0, 1.0)
    cles = U / (n1 * n2)

    he1_pass = bool(p_value < alpha and r > min_r)

    logger.info(
        f"HE1: U={U:.0f}, p={p_value:.2e}, r={r:.4f}, CLES={cles:.4f}, pass={he1_pass}"
    )
    return {
        "U_statistic": float(U),
        "p_value": float(p_value),
        "rank_biserial_r": float(r),
        "cles": float(cles),
        "n_anomaly": n1,
        "n_normal": n2,
        "he1_pass": he1_pass,
    }


# ── HE2: Discriminative capacity ───────────────────────────────


def compute_discrimination(
    scores: np.ndarray,
    proxy: np.ndarray,
    min_auc: float = None,
) -> Dict:
    """HE2: AUC-ROC and Average Precision.

    HE2 passes if AUC > threshold AND AP > proxy base rate.
    """
    min_auc = min_auc or settings.he2_min_auc_roc
    s, p = _as_arrays(scores, proxy)

    auc = float(roc_auc_score(p, s))
    ap = float(average_precision_score(p, s))
    base_rate = float(p.mean())
    ap_over_baseline = ap / base_rate if base_rate > 0 else 0.0

    he2_pass = bool(auc > min_auc and ap > base_rate)

    logger.info(
        f"HE2: AUC={auc:.4f}, AP={ap:.4f}, base_rate={base_rate:.4f}, "
        f"AP/base={ap_over_baseline:.2f}x, pass={he2_pass}"
    )
    return {
        "auc_roc": auc,
        "average_precision": ap,
        "base_rate": base_rate,
        "ap_over_baseline": ap_over_baseline,
        "he2_pass": he2_pass,
    }


# ── HE3: Top-K concentration ───────────────────────────────────


def compute_topk(
    scores: np.ndarray,
    proxy: np.ndarray,
    k_values: Optional[List[float]] = None,
) -> Dict:
    """HE3: Enrichment Factor at multiple k values.

    HE3 passes if EF > 1 at top-5%.
    """
    k_values = k_values or settings.top_k_percents_list
    s, p = _as_arrays(scores, proxy)
    base_rate = float(p.mean())

    result = {}
    for k in k_values:
        k_label = int(round(k * 100))
        prec = precision_at_k(p, s, k_pct=k)
        ef = enrichment_factor(p, s, k_pct=k)

        # Recall@k
        n_total_pos = int(p.sum())
        top_k_n = max(1, int(np.ceil(len(p) * k)))
        top_idx = np.argsort(s)[-top_k_n:]
        recall = float(p[top_idx].sum() / n_total_pos) if n_total_pos > 0 else 0.0

        result[f"precision_at_{k_label}pct"] = prec
        result[f"recall_at_{k_label}pct"] = recall
        result[f"ef_at_{k_label}pct"] = ef

    # Primary criterion: EF > 1 at top-5%
    ef_5 = result.get("ef_at_5pct", 0.0)
    result["he3_pass"] = bool(ef_5 > settings.he3_min_enrichment_factor)

    logger.info(f"HE3: EF@5%={ef_5:.4f}, pass={result['he3_pass']}")
    return result


# ── KS test (complementary) ────────────────────────────────────


def ks_test(scores: np.ndarray, proxy: np.ndarray) -> Dict:
    """Kolmogorov-Smirnov two-sample test."""
    s, p = _as_arrays(scores, proxy)
    stat, p_value = ks_2samp(s[p == 1], s[p == 0])
    return {"ks_statistic": float(stat), "p_value": float(p_value)}


# ── Holm-Bonferroni correction ──────────────────────────────────


def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni correction for multiple comparisons.

    Adjusts p-values upward; adjusted >= original always holds.
    """
    n = len(p_values)
    arr = np.asarray(p_values, dtype=np.float64)
    sorted_indices = np.argsort(arr)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = arr[idx] * (n - rank)
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted.tolist()


# ── HE4: Model comparison ──────────────────────────────────────


def compare_models(
    proxy: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    min_wins: int = None,
) -> Dict:
    """HE4: Compare IF vs LOF and OC-SVM on 4 metrics.

    IF must win >= min_wins of 4 metrics (AUC-ROC, AP, Precision@5%, EF@5%).
    """
    min_wins = min_wins or settings.he4_min_metrics_won
    p = np.asarray(proxy, dtype=np.int8)

    metrics_by_model = {}
    for name, scores in model_scores.items():
        s = np.asarray(scores, dtype=np.float64)
        auc = float(roc_auc_score(p, s))
        ap = float(average_precision_score(p, s))
        prec5 = precision_at_k(p, s, k_pct=0.05)
        ef5 = enrichment_factor(p, s, k_pct=0.05)
        metrics_by_model[name] = {
            "auc_roc": auc,
            "ap": ap,
            "precision_at_5pct": prec5,
            "ef_at_5pct": ef5,
        }

    # Count IF wins
    if_metrics = metrics_by_model.get("isolation_forest", {})
    metric_names = ["auc_roc", "ap", "precision_at_5pct", "ef_at_5pct"]
    if_wins = 0
    if_wins_on = []

    for metric in metric_names:
        if_val = if_metrics.get(metric, 0)
        others_best = max(
            metrics_by_model[m].get(metric, 0)
            for m in model_scores if m != "isolation_forest"
        )
        if if_val >= others_best:
            if_wins += 1
            if_wins_on.append(metric)

    he4_pass = bool(if_wins >= min_wins)

    logger.info(
        f"HE4: IF wins {if_wins}/4 metrics {if_wins_on}, pass={he4_pass}"
    )
    return {
        "metrics_comparison": metrics_by_model,
        "if_wins": if_wins,
        "if_wins_on": if_wins_on,
        "he4_pass": he4_pass,
    }


# ── Temporal stability ──────────────────────────────────────────


def temporal_stability(
    model_name: str,
    scores: np.ndarray,
    proxy: np.ndarray,
    dates: np.ndarray,
) -> Dict:
    """Monthly AUC-ROC on test set to detect drift."""
    s, p = _as_arrays(scores, proxy)
    d = np.asarray(dates, dtype="datetime64[M]")

    months = np.unique(d)
    monthly = {}
    for month in months:
        mask = d == month
        if mask.sum() < 10 or len(np.unique(p[mask])) < 2:
            continue
        auc = float(roc_auc_score(p[mask], s[mask]))
        monthly[str(month)] = {
            "auc_roc": auc,
            "n_samples": int(mask.sum()),
            "proxy_rate": float(p[mask].mean()),
        }

    return {"model_name": model_name, "monthly_auc": monthly}


# ── Full evaluation facade ──────────────────────────────────────


def full_evaluation(
    model_name: str,
    scores: np.ndarray,
    proxy: np.ndarray,
    dates: Optional[np.ndarray] = None,
    bootstrap_n: Optional[int] = None,
) -> Dict:
    """Run HE1-HE3 + KS + bootstrap + optional temporal stability."""
    s, p = _as_arrays(scores, proxy)
    n_boot = bootstrap_n or settings.bootstrap_n

    result = {
        "model_name": model_name,
        "he1": run_mann_whitney(s, p),
        "he2": compute_discrimination(s, p),
        "he3": compute_topk(s, p),
        "ks": ks_test(s, p),
        "bootstrap_ci_auc": bootstrap_ci(p, s, roc_auc_score, n_iterations=n_boot),
        "bootstrap_ci_ap": bootstrap_ci(p, s, average_precision_score, n_iterations=n_boot),
    }

    if dates is not None:
        result["temporal_stability"] = temporal_stability(model_name, s, p, dates)

    return result
