"""Smoke tests for reporting module — tables and figures."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fraud_detector.reporting import latex_tables as tbl


@pytest.fixture
def sample_results():
    return {
        "isolation_forest": {
            "he1": {"U_statistic": 1e9, "p_value": 0.0, "rank_biserial_r": 0.26, "cles": 0.63, "he1_pass": True},
            "he2": {"auc_roc": 0.63, "average_precision": 0.17, "base_rate": 0.10, "ap_over_baseline": 1.7, "he2_pass": False},
            "he3": {"ef_at_1pct": 3.3, "ef_at_2pct": 2.8, "ef_at_5pct": 2.2, "ef_at_10pct": 1.9, "precision_at_5pct": 0.23, "he3_pass": True},
            "bootstrap_ci_auc": {"mean": 0.63, "lower": 0.628, "upper": 0.632, "std": 0.001},
            "bootstrap_ci_ap": {"mean": 0.17, "lower": 0.168, "upper": 0.172, "std": 0.001},
            "temporal_stability": {"monthly_auc": {"2025-09": {"auc_roc": 0.63}, "2025-10": {"auc_roc": 0.64}}},
        },
        "lof": {
            "he1": {"U_statistic": 8e8, "p_value": 0.0, "rank_biserial_r": 0.09, "cles": 0.55, "he1_pass": False},
            "he2": {"auc_roc": 0.55, "average_precision": 0.12, "base_rate": 0.10, "ap_over_baseline": 1.2, "he2_pass": False},
            "he3": {"ef_at_1pct": 1.5, "ef_at_2pct": 1.3, "ef_at_5pct": 1.2, "ef_at_10pct": 1.1, "precision_at_5pct": 0.13, "he3_pass": True},
            "bootstrap_ci_auc": {"mean": 0.55, "lower": 0.548, "upper": 0.552, "std": 0.001},
            "bootstrap_ci_ap": {"mean": 0.12, "lower": 0.118, "upper": 0.122, "std": 0.001},
            "temporal_stability": {"monthly_auc": {"2025-09": {"auc_roc": 0.55}, "2025-10": {"auc_roc": 0.54}}},
        },
        "ocsvm": {
            "he1": {"U_statistic": 9e8, "p_value": 0.0, "rank_biserial_r": 0.17, "cles": 0.59, "he1_pass": True},
            "he2": {"auc_roc": 0.59, "average_precision": 0.15, "base_rate": 0.10, "ap_over_baseline": 1.5, "he2_pass": False},
            "he3": {"ef_at_1pct": 2.5, "ef_at_2pct": 2.1, "ef_at_5pct": 2.1, "ef_at_10pct": 1.8, "precision_at_5pct": 0.22, "he3_pass": True},
            "bootstrap_ci_auc": {"mean": 0.59, "lower": 0.588, "upper": 0.592, "std": 0.001},
            "bootstrap_ci_ap": {"mean": 0.15, "lower": 0.148, "upper": 0.152, "std": 0.001},
            "temporal_stability": {"monthly_auc": {"2025-09": {"auc_roc": 0.59}, "2025-10": {"auc_roc": 0.58}}},
        },
        "he4": {"metrics_comparison": {
            "isolation_forest": {"auc_roc": 0.63, "ap": 0.17, "precision_at_5pct": 0.23, "ef_at_5pct": 2.2},
            "lof": {"auc_roc": 0.55, "ap": 0.12, "precision_at_5pct": 0.13, "ef_at_5pct": 1.2},
            "ocsvm": {"auc_roc": 0.59, "ap": 0.15, "precision_at_5pct": 0.22, "ef_at_5pct": 2.1},
        }, "if_wins": 4, "if_wins_on": ["auc_roc", "ap", "precision_at_5pct", "ef_at_5pct"], "he4_pass": True},
    }


@pytest.fixture
def sample_sensitivity():
    return {
        "proxy_sensitivity": {
            "unified": {"auc_roc": 0.63, "ap": 0.17, "base_rate": 0.105},
            "tipo_a": {"auc_roc": 0.58, "ap": 0.12, "base_rate": 0.063},
            "wide": {"auc_roc": 0.62, "ap": 0.16, "base_rate": 0.076},
            "delta_auc_tipo_a": 0.054, "delta_ap_tipo_a": 0.05, "delta_auc_wide": 0.01, "robust": False,
        },
        "per_type_metrics": {
            "tipo_a": {"auc_roc": 0.58, "ap": 0.08, "ef_at_5pct": 1.5, "count": 158000, "rate": 0.063},
            "tipo_b": {"auc_roc": None, "ap": None, "ef_at_5pct": None, "count": 0, "rate": 0.0},
            "tipo_c": {"auc_roc": 0.66, "ap": 0.12, "ef_at_5pct": 2.5, "count": 101000, "rate": 0.04},
            "tipo_d": {"auc_roc": 0.99, "ap": 0.80, "ef_at_5pct": 10.0, "count": 10000, "rate": 0.004},
            "tipo_e": {"auc_roc": None, "ap": None, "ef_at_5pct": None, "count": 0, "rate": 0.0},
        },
        "feature18_sensitivity": {
            "auc_31_features": 0.63, "auc_30_features": 0.61, "delta_auc": 0.023,
            "low_sensitivity": False, "jaccard_top5pct": 0.85, "spearman_r": 0.97,
        },
        "ablation_31_vs_21": {
            "model_31": {"auc_roc": 0.63, "ap": 0.17, "precision_at_5pct": 0.23, "enrichment_factor": 2.2},
            "model_21": {"auc_roc": 0.62, "ap": 0.16, "precision_at_5pct": 0.22, "enrichment_factor": 2.1},
            "delta": {"auc_roc": 0.01, "ap": 0.01, "precision_at_5pct": 0.01, "enrichment_factor": 0.1},
            "groups_contribute": False,
        },
        "segment_metrics": {
            "by_role": {"player": {"auc_roc": 0.63, "ap": 0.17, "precision_at_5pct": 0.23, "enrichment_factor": 2.2, "n_transactions": 2000000}},
            "by_category": {"reservation": {"auc_roc": 0.65, "ap": 0.18, "precision_at_5pct": 0.25, "enrichment_factor": 2.5, "n_transactions": 1500000}},
        },
        "anomaly_typology": {"type_distribution": {
            "mixed": {"count": 99000, "pct": 79.0}, "amount": {"count": 13000, "pct": 10.4},
            "velocity": {"count": 2600, "pct": 2.1}, "discount": {"count": 1400, "pct": 1.1},
            "credit_flow": {"count": 6000, "pct": 4.8}, "role_deviation": {"count": 1400, "pct": 1.1},
            "diversity": {"count": 100, "pct": 0.1}, "reversal": {"count": 1200, "pct": 1.0},
            "temporal": {"count": 0, "pct": 0.0},
        }},
        "user_risk_profiles": {"n_users_total": 406000, "n_users_flagged": 15000, "pct_users_flagged": 3.7,
                               "flagged_users_summary": {"mean_concentration": 0.18, "max_concentration": 0.95}},
    }


@pytest.fixture
def sample_posthoc():
    return {"posthoc_analysis": {
        "facility_concentration": {"n_facilities_with_enrichment_gt_2": 118, "top_10_facilities": [
            {"facility_id": 1, "n_transactions": 500, "anomaly_rate": 0.25, "anomaly_enrichment": 5.0},
        ]},
        "manager_concentration": {"mode": "aggregated_manager_intervention", "top_10_managers": [],
                                  "aggregate_only": {"n_transactions_with_manager_intervention": 1400000,
                                                     "n_anomalies_with_manager_intervention": 89000,
                                                     "anomaly_rate_with_manager_intervention": 0.063}},
        "currency_concentration": {"currencies_affected": [
            {"currency": "USD", "n_transactions": 1800000, "anomaly_rate": 0.04, "anomaly_enrichment": 0.8},
            {"currency": "HNL", "n_transactions": 50000, "anomaly_rate": 0.12, "anomaly_enrichment": 2.4},
        ]},
    }}


# --- Table tests ---


def test_table_model_comparison_valid_latex(sample_results):
    tex = tbl.table_model_comparison(sample_results)
    assert r"\begin{table}" in tex
    assert r"\end{table}" in tex
    assert r"\toprule" in tex
    assert "Isolation Forest" in tex


def test_table_he1_valid_latex(sample_results):
    tex = tbl.table_he1_results(sample_results)
    assert r"\begin{table}" in tex
    assert "0.2600" in tex  # rank_biserial_r


def test_table_he4_marks_winner(sample_results):
    tex = tbl.table_he4_comparison(sample_results)
    assert r"\textbf" in tex  # IF wins are bolded


def test_table_hypothesis_summary(sample_results):
    tex = tbl.table_hypothesis_summary(sample_results)
    assert "Respaldada" in tex
    assert "No respaldada" in tex


def test_table_sensitivity_proxy(sample_sensitivity):
    tex = tbl.table_sensitivity_proxy(sample_sensitivity)
    assert "unified" in tex or "Proxy" in tex


def test_table_sensitivity_per_type(sample_sensitivity):
    tex = tbl.table_sensitivity_per_type(sample_sensitivity)
    assert "tipo\\_a" in tex


def test_table_anomaly_types(sample_sensitivity):
    tex = tbl.table_anomaly_types(sample_sensitivity)
    assert "mixed" in tex
    assert "79.00" in tex


def test_table_posthoc_facility(sample_posthoc):
    tex = tbl.table_posthoc_facility(sample_posthoc)
    assert r"\begin{table}" in tex


def test_table_posthoc_currency(sample_posthoc):
    tex = tbl.table_posthoc_currency(sample_posthoc)
    assert "USD" in tex


def test_table_posthoc_manager_aggregate(sample_posthoc):
    tex = tbl.table_posthoc_manager(sample_posthoc)
    assert "manager" in tex.lower()


def test_empty_results_handled():
    tex = tbl.table_hypothesis_summary({})
    assert isinstance(tex, str)
    assert r"\begin{table}" in tex
