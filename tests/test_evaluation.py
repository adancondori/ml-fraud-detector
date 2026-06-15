"""Tests for Fase 7 evaluation — HE1-HE4 contracts.

Uses synthetic data to verify statistical tests, discrimination metrics,
top-k concentration, model comparison, and Holm-Bonferroni correction.
"""
from __future__ import annotations

import numpy as np
import pytest

from fraud_detector.evaluation.metrics import (
    bootstrap_ci,
    enrichment_factor,
    evaluate_scores,
    precision_at_k,
)
from fraud_detector.evaluation.hypothesis import (
    apply_holm_bonferroni,
    compare_models,
    compute_discrimination,
    compute_topk,
    full_evaluation,
    ks_test,
    run_mann_whitney,
)


# --- Fixtures ---


@pytest.fixture
def perfect_separation():
    """Anomalies have score=10, normals have score=0."""
    scores = np.concatenate([np.ones(100) * 10, np.zeros(900)]).astype(np.float32)
    proxy = np.concatenate([np.ones(100), np.zeros(900)]).astype(np.int8)
    return scores, proxy


@pytest.fixture
def random_scores():
    """Random scores uncorrelated with proxy."""
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(1000).astype(np.float32)
    proxy = np.concatenate([np.ones(100), np.zeros(900)]).astype(np.int8)
    return scores, proxy


@pytest.fixture
def three_model_scores():
    """Scores for 3 models where IF is best in 3/4 metrics."""
    rng = np.random.default_rng(42)
    n = 1000
    proxy = np.concatenate([np.ones(100), np.zeros(900)]).astype(np.int8)
    # IF: good separation
    if_scores = np.where(proxy == 1, rng.normal(5, 1, n), rng.normal(0, 1, n)).astype(np.float32)
    # LOF: moderate separation
    lof_scores = np.where(proxy == 1, rng.normal(3, 1.5, n), rng.normal(0, 1, n)).astype(np.float32)
    # OC-SVM: weak separation
    ocsvm_scores = np.where(proxy == 1, rng.normal(2, 2, n), rng.normal(0, 1, n)).astype(np.float32)
    return proxy, if_scores, lof_scores, ocsvm_scores


# --- HE1: Mann-Whitney ---


def test_mann_whitney_perfect_separation_passes_he1(perfect_separation):
    scores, proxy = perfect_separation
    result = run_mann_whitney(scores, proxy)
    assert result["he1_pass"] is True
    assert result["rank_biserial_r"] > 0.10
    assert result["p_value"] < 0.05


def test_mann_whitney_random_scores_fails_he1(random_scores):
    scores, proxy = random_scores
    result = run_mann_whitney(scores, proxy)
    # Random scores should not pass — either p > 0.05 or r < 0.10
    # (with random data, r should be near 0)
    assert abs(result["rank_biserial_r"]) < 0.30


def test_mann_whitney_returns_all_keys(perfect_separation):
    scores, proxy = perfect_separation
    result = run_mann_whitney(scores, proxy)
    expected_keys = {"U_statistic", "p_value", "rank_biserial_r", "cles", "n_anomaly", "n_normal", "he1_pass"}
    assert expected_keys.issubset(result.keys())


# --- HE2: Discrimination ---


def test_auc_roc_perfect_discrimination(perfect_separation):
    scores, proxy = perfect_separation
    result = compute_discrimination(scores, proxy)
    assert result["auc_roc"] == 1.0


def test_auc_roc_random_is_approximately_half(random_scores):
    scores, proxy = random_scores
    result = compute_discrimination(scores, proxy)
    assert abs(result["auc_roc"] - 0.5) < 0.10


def test_discrimination_returns_all_keys(perfect_separation):
    scores, proxy = perfect_separation
    result = compute_discrimination(scores, proxy)
    expected_keys = {"auc_roc", "average_precision", "base_rate", "ap_over_baseline", "he2_pass"}
    assert expected_keys.issubset(result.keys())


# --- HE3: Top-K concentration ---


def test_enrichment_factor_perfect_concentration(perfect_separation):
    scores, proxy = perfect_separation
    result = compute_topk(scores, proxy)
    # With perfect separation, all anomalies in top-k → EF = 1/base_rate
    assert result["ef_at_5pct"] > 1.0
    assert result["he3_pass"] is True


def test_topk_returns_all_k_values(perfect_separation):
    scores, proxy = perfect_separation
    result = compute_topk(scores, proxy)
    for k in [1, 2, 5, 10]:
        assert f"precision_at_{k}pct" in result
        assert f"ef_at_{k}pct" in result


# --- Bootstrap ---


def test_bootstrap_ci_lower_leq_mean_leq_upper(perfect_separation):
    scores, proxy = perfect_separation
    from sklearn.metrics import roc_auc_score
    result = bootstrap_ci(proxy, scores, roc_auc_score, n_iterations=100)
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_bootstrap_ci_backwards_compat(perfect_separation):
    """Sin user_ids el dict NO debe contener `cluster_unit` (back-compat estricta)."""
    scores, proxy = perfect_separation
    from sklearn.metrics import roc_auc_score
    result = bootstrap_ci(proxy, scores, roc_auc_score, n_iterations=100, random_seed=42)
    assert "cluster_unit" not in result
    assert "n_clusters_resampled" not in result
    assert "method_used" not in result
    assert set(result.keys()) == {"mean", "lower", "upper", "std", "n_iterations"}


def test_bootstrap_ci_by_user_bounds():
    """Con user_ids el CI clustered debe respetar lower <= mean <= upper y traer metadatos."""
    rng = np.random.default_rng(0)
    n_users = 200
    txns_per_user = 5
    n = n_users * txns_per_user
    user_ids = np.repeat(np.arange(n_users), txns_per_user)
    proxy = (rng.random(n) < 0.15).astype(np.int8)
    scores = rng.random(n) + 0.2 * proxy  # leve señal
    from sklearn.metrics import roc_auc_score
    res = bootstrap_ci(
        proxy, scores, roc_auc_score,
        n_iterations=200, random_seed=42,
        user_ids=user_ids, method="weighted",
    )
    assert res["lower"] <= res["mean"] <= res["upper"]
    assert res["cluster_unit"] == "user"
    assert res["n_clusters_resampled"] == n_users
    assert res["method_used"] == "weighted"


def test_bootstrap_ci_weighted_vs_concatenate_close():
    """Las rutas weighted y concatenate deben converger para AUC dentro de ±0.005."""
    rng = np.random.default_rng(0)
    n_users = 300
    txns_per_user = 5
    n = n_users * txns_per_user
    user_ids = np.repeat(np.arange(n_users), txns_per_user)
    proxy = (rng.random(n) < 0.1).astype(np.int8)
    scores = rng.random(n) + 0.3 * proxy
    from sklearn.metrics import roc_auc_score
    res_w = bootstrap_ci(
        proxy, scores, roc_auc_score,
        n_iterations=500, random_seed=42,
        user_ids=user_ids, method="weighted",
    )
    res_c = bootstrap_ci(
        proxy, scores, roc_auc_score,
        n_iterations=500, random_seed=42,
        user_ids=user_ids, method="concatenate",
    )
    assert abs(res_w["mean"] - res_c["mean"]) < 0.005
    # ambos identifican el mismo cluster_unit
    assert res_w["cluster_unit"] == res_c["cluster_unit"] == "user"


# --- Holm-Bonferroni ---


def test_holm_bonferroni_increases_p_values():
    original = [0.01, 0.04, 0.03, 0.001]
    adjusted = apply_holm_bonferroni(original)
    for orig, adj in zip(original, adjusted):
        assert adj >= orig


def test_holm_bonferroni_caps_at_one():
    original = [0.5, 0.6, 0.7, 0.8]
    adjusted = apply_holm_bonferroni(original)
    assert all(a <= 1.0 for a in adjusted)


# --- KS test ---


def test_ks_test_perfect_separation(perfect_separation):
    scores, proxy = perfect_separation
    result = ks_test(scores, proxy)
    assert result["ks_statistic"] > 0.5
    assert result["p_value"] < 0.05


# --- HE4: Model comparison ---


def test_compare_models_counts_wins_correctly(three_model_scores):
    proxy, if_scores, lof_scores, ocsvm_scores = three_model_scores
    result = compare_models(
        proxy,
        {"isolation_forest": if_scores, "lof": lof_scores, "ocsvm": ocsvm_scores},
    )
    assert "if_wins" in result
    assert "he4_pass" in result
    assert isinstance(result["if_wins"], int)
    assert 0 <= result["if_wins"] <= 4


def test_compare_models_if_wins_when_best(three_model_scores):
    proxy, if_scores, lof_scores, ocsvm_scores = three_model_scores
    result = compare_models(
        proxy,
        {"isolation_forest": if_scores, "lof": lof_scores, "ocsvm": ocsvm_scores},
    )
    # IF has best separation, should win majority
    assert result["if_wins"] >= 3
    assert result["he4_pass"] is True


# --- Full evaluation ---


def test_full_evaluation_returns_all_keys(perfect_separation):
    scores, proxy = perfect_separation
    result = full_evaluation("test_model", scores, proxy)
    expected_keys = {"model_name", "he1", "he2", "he3", "ks", "bootstrap_ci_auc", "bootstrap_ci_ap"}
    assert expected_keys.issubset(result.keys())


def test_full_evaluation_with_dates():
    """Temporal stability with 2 months of data."""
    rng = np.random.default_rng(42)
    n = 1000
    proxy = np.concatenate([np.ones(100), np.zeros(400), np.ones(100), np.zeros(400)]).astype(np.int8)
    scores = np.where(proxy == 1, rng.normal(5, 1, n), rng.normal(0, 1, n)).astype(np.float32)
    dates = np.array(
        [np.datetime64("2025-09-15")] * 500 + [np.datetime64("2025-10-15")] * 500,
    )
    result = full_evaluation("test_model", scores, proxy, dates=dates, bootstrap_n=50)
    assert "temporal_stability" in result
    assert len(result["temporal_stability"]["monthly_auc"]) == 2
