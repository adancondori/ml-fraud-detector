#!/usr/bin/env python
"""Fase 8: Sensibilidad y Robustez.

10 steps — all use pre-computed scores from Fase 7 (no retraining).
Outputs: results_sensitivity.json, results_posthoc.json,
         shap_summary.{pdf,png}, user_risk_profiles.parquet.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings
from fraud_detector.data.loader import DataManager
from fraud_detector.evaluation.metrics import enrichment_factor, precision_at_k
from fraud_detector.features.engineering import FEATURE_NAMES, FEATURE_NAMES_21, FEATURE_NAMES_30
from fraud_detector.utils.logger import logger

TOTAL_STEPS = 10


def log_step(step: int, name: str, status: str = "START"):
    logger.info(f"[STEP {step}/{TOTAL_STEPS}] {name} — {status}")


def load_data():
    """Load test features, scores, and proxy labels."""
    df_test = pd.read_parquet(settings.processed_dir / "test_features.parquet")
    X_test = np.load(settings.scores_dir / "X_test.npy")

    scores_df = pd.read_parquet(settings.scores_dir / "test_scores.parquet")
    scores_if = scores_df["score_if"].values

    # Load IF-30 and IF-21 scores
    idx_30 = [FEATURE_NAMES.index(f) for f in FEATURE_NAMES_30]
    idx_21 = [FEATURE_NAMES.index(f) for f in FEATURE_NAMES_21]

    model_if30 = joblib.load(settings.models_output_dir / "isolation_forest_30.joblib")
    scores_if30 = (-model_if30.score_samples(X_test[:, idx_30])).astype(np.float32)

    model_if21 = joblib.load(settings.models_output_dir / "isolation_forest_21.joblib")
    scores_if21 = (-model_if21.score_samples(X_test[:, idx_21])).astype(np.float32)

    # Proxy labels
    y_unified = DataManager.assign_proxy_labels(df_test, "unified", settings).values
    y_tipo_a = DataManager.assign_proxy_labels(df_test, "tipo_a", settings).values
    y_wide = DataManager.assign_proxy_labels(df_test, "wide", settings).values

    logger.info(
        f"Loaded: {len(df_test):,} rows, "
        f"unified={y_unified.sum():,} ({y_unified.mean()*100:.2f}%), "
        f"tipo_a={y_tipo_a.sum():,}, wide={y_wide.sum():,}"
    )
    return df_test, X_test, scores_if, scores_if30, scores_if21, y_unified, y_tipo_a, y_wide


def step1_proxy_sensitivity(scores_if, y_unified, y_tipo_a, y_wide):
    """Proxy robustness: unified vs Tipo A vs wide."""
    log_step(1, "Proxy Sensitivity", "START")
    t0 = time.perf_counter()

    auc_unified = float(roc_auc_score(y_unified, scores_if))
    ap_unified = float(average_precision_score(y_unified, scores_if))

    auc_tipo_a = float(roc_auc_score(y_tipo_a, scores_if))
    ap_tipo_a = float(average_precision_score(y_tipo_a, scores_if))

    auc_wide = float(roc_auc_score(y_wide, scores_if))
    ap_wide = float(average_precision_score(y_wide, scores_if))

    delta_auc_tipo_a = abs(auc_unified - auc_tipo_a)
    delta_auc_wide = abs(auc_unified - auc_wide)

    result = {
        "unified": {"auc_roc": auc_unified, "ap": ap_unified, "base_rate": float(y_unified.mean())},
        "tipo_a": {"auc_roc": auc_tipo_a, "ap": ap_tipo_a, "base_rate": float(y_tipo_a.mean())},
        "wide": {"auc_roc": auc_wide, "ap": ap_wide, "base_rate": float(y_wide.mean())},
        "delta_auc_tipo_a": delta_auc_tipo_a,
        "delta_ap_tipo_a": abs(ap_unified - ap_tipo_a),
        "delta_auc_wide": delta_auc_wide,
        "robust": delta_auc_tipo_a < 0.05,
    }
    log_step(1, "Proxy Sensitivity", f"DONE ({time.perf_counter()-t0:.1f}s) delta_AUC={delta_auc_tipo_a:.4f} robust={result['robust']}")
    return result


def step2_per_type_metrics(df_test, scores_if):
    """AUC per proxy type (A, B, C, D, E)."""
    log_step(2, "Per-Type Metrics", "START")
    t0 = time.perf_counter()
    result = {}
    for ptype in ["tipo_a", "tipo_b", "tipo_c", "tipo_d", "tipo_e"]:
        labels = DataManager.assign_proxy_labels(df_test, ptype, settings).values
        n_pos = int(labels.sum())
        rate = float(labels.mean())
        if n_pos >= 10 and len(np.unique(labels)) == 2:
            auc = float(roc_auc_score(labels, scores_if))
            ap = float(average_precision_score(labels, scores_if))
            ef = enrichment_factor(labels, scores_if, k_pct=0.05)
            result[ptype] = {"auc_roc": auc, "ap": ap, "ef_at_5pct": ef, "count": n_pos, "rate": rate}
            logger.info(f"  {ptype}: AUC={auc:.4f}, AP={ap:.4f}, EF@5%={ef:.4f}, n={n_pos:,}")
        else:
            result[ptype] = {"auc_roc": None, "ap": None, "ef_at_5pct": None, "count": n_pos, "rate": rate}
            logger.info(f"  {ptype}: skipped (n_pos={n_pos})")
    log_step(2, "Per-Type Metrics", f"DONE ({time.perf_counter()-t0:.1f}s)")
    return result


def step3_feature18_sensitivity(scores_if, scores_if30, y_unified):
    """Feature #18 ablation: IF-31 vs IF-30."""
    log_step(3, "Feature #18 Sensitivity", "START")
    t0 = time.perf_counter()

    auc_31 = float(roc_auc_score(y_unified, scores_if))
    auc_30 = float(roc_auc_score(y_unified, scores_if30))
    delta = abs(auc_31 - auc_30)

    # Jaccard similarity at top-5%
    k = int(len(scores_if) * 0.05)
    top5_31 = set(np.argsort(scores_if)[-k:])
    top5_30 = set(np.argsort(scores_if30)[-k:])
    jaccard = len(top5_31 & top5_30) / len(top5_31 | top5_30)

    # Spearman rank correlation
    rho, p_spearman = spearmanr(scores_if, scores_if30)

    result = {
        "auc_31_features": auc_31,
        "auc_30_features": auc_30,
        "delta_auc": delta,
        "low_sensitivity": delta < 0.02,
        "jaccard_top5pct": float(jaccard),
        "spearman_r": float(rho),
        "spearman_p": float(p_spearman),
    }
    log_step(3, "Feature #18 Sensitivity", f"DONE ({time.perf_counter()-t0:.1f}s) delta={delta:.4f} low_sensitivity={result['low_sensitivity']}")
    return result


def step4_ablation_31vs21(scores_if, scores_if21, y_unified):
    """Ablation: IF-31 vs IF-21 (groups F,G,H contribution)."""
    log_step(4, "Ablation IF-31 vs IF-21", "START")
    t0 = time.perf_counter()

    metrics_31 = {
        "auc_roc": float(roc_auc_score(y_unified, scores_if)),
        "ap": float(average_precision_score(y_unified, scores_if)),
        "precision_at_5pct": precision_at_k(y_unified, scores_if, k_pct=0.05),
        "enrichment_factor": enrichment_factor(y_unified, scores_if, k_pct=0.05),
    }
    metrics_21 = {
        "auc_roc": float(roc_auc_score(y_unified, scores_if21)),
        "ap": float(average_precision_score(y_unified, scores_if21)),
        "precision_at_5pct": precision_at_k(y_unified, scores_if21, k_pct=0.05),
        "enrichment_factor": enrichment_factor(y_unified, scores_if21, k_pct=0.05),
    }
    delta = {k: metrics_31[k] - metrics_21[k] for k in metrics_31}

    result = {
        "model_31": metrics_31,
        "model_21": metrics_21,
        "delta": delta,
        "groups_contribute": delta["auc_roc"] > 0.01,
    }
    log_step(4, "Ablation IF-31 vs IF-21", f"DONE ({time.perf_counter()-t0:.1f}s) delta_AUC={delta['auc_roc']:.4f}")
    return result


def step5_segment_metrics(df_test, scores_if, y_unified):
    """Metrics by role and payment category."""
    log_step(5, "Segment Metrics", "START")
    t0 = time.perf_counter()
    result = {"by_role": {}, "by_category": {}}

    # By role
    for role in ["player", "court_manager", "court_operator", "teacher"]:
        mask = df_test["user_role"] == role
        n = int(mask.sum())
        if n < 100:
            continue
        proxy_seg = y_unified[mask.values]
        scores_seg = scores_if[mask.values]
        n_pos = int(proxy_seg.sum())
        if n_pos < 10 or len(np.unique(proxy_seg)) < 2:
            continue
        result["by_role"][role] = {
            "auc_roc": float(roc_auc_score(proxy_seg, scores_seg)),
            "ap": float(average_precision_score(proxy_seg, scores_seg)),
            "precision_at_5pct": precision_at_k(proxy_seg, scores_seg, k_pct=0.05),
            "enrichment_factor": enrichment_factor(proxy_seg, scores_seg, k_pct=0.05),
            "n_transactions": n,
            "n_proxy_positive": n_pos,
            "proxy_rate": float(proxy_seg.mean()),
        }
        logger.info(f"  role={role}: AUC={result['by_role'][role]['auc_roc']:.4f}, n={n:,}")

    # By category
    for cat in df_test["category"].dropna().unique():
        mask = df_test["category"] == cat
        n = int(mask.sum())
        if n < 100:
            continue
        proxy_seg = y_unified[mask.values]
        scores_seg = scores_if[mask.values]
        n_pos = int(proxy_seg.sum())
        if n_pos < 10 or len(np.unique(proxy_seg)) < 2:
            continue
        cat_key = str(cat).replace(" ", "_").lower()
        result["by_category"][cat_key] = {
            "auc_roc": float(roc_auc_score(proxy_seg, scores_seg)),
            "ap": float(average_precision_score(proxy_seg, scores_seg)),
            "precision_at_5pct": precision_at_k(proxy_seg, scores_seg, k_pct=0.05),
            "enrichment_factor": enrichment_factor(proxy_seg, scores_seg, k_pct=0.05),
            "n_transactions": n,
            "n_proxy_positive": n_pos,
            "proxy_rate": float(proxy_seg.mean()),
        }
        logger.info(f"  category={cat_key}: AUC={result['by_category'][cat_key]['auc_roc']:.4f}, n={n:,}")

    log_step(5, "Segment Metrics", f"DONE ({time.perf_counter()-t0:.1f}s)")
    return result


def step6_baselines(scores_if, y_unified, df_test):
    """Sanity baselines: random, amount, z-score."""
    log_step(6, "Sanity Baselines", "START")
    t0 = time.perf_counter()
    result = {}

    # Random
    rng = np.random.default_rng(42)
    random_scores = rng.random(len(y_unified))
    result["random"] = {
        "auc_roc": float(roc_auc_score(y_unified, random_scores)),
        "ap": float(average_precision_score(y_unified, random_scores)),
    }

    # Amount ranking
    amounts = df_test["amount"].values.astype(np.float64)
    result["amount_ranking"] = {
        "auc_roc": float(roc_auc_score(y_unified, amounts)),
        "ap": float(average_precision_score(y_unified, amounts)),
    }

    # Z-score of amount
    mean_a = amounts.mean()
    std_a = amounts.std()
    zscore_amounts = np.abs((amounts - mean_a) / max(std_a, 1e-8))
    result["zscore_amount"] = {
        "auc_roc": float(roc_auc_score(y_unified, zscore_amounts)),
        "ap": float(average_precision_score(y_unified, zscore_amounts)),
    }

    # IF must beat all
    if_auc = float(roc_auc_score(y_unified, scores_if))
    result["if_beats_all"] = all(
        if_auc > result[b]["auc_roc"] for b in ["random", "amount_ranking", "zscore_amount"]
    )
    for b in ["random", "amount_ranking", "zscore_amount"]:
        logger.info(f"  {b}: AUC={result[b]['auc_roc']:.4f}")
    logger.info(f"  IF AUC={if_auc:.4f} beats all={result['if_beats_all']}")

    log_step(6, "Sanity Baselines", f"DONE ({time.perf_counter()-t0:.1f}s)")
    return result


def step7_per_status(df_test, scores_if):
    """Per refund sub-status evaluation."""
    log_step(7, "Per-Status Evaluation", "START")
    t0 = time.perf_counter()
    result = {}
    for status in ["totally_refunded", "refunded_to_credit", "partially_refunded"]:
        proxy_s = (df_test["status"] == status).astype(np.int8).values
        n_pos = int(proxy_s.sum())
        if n_pos >= 10 and len(np.unique(proxy_s)) == 2:
            result[status] = {
                "auc_roc": float(roc_auc_score(proxy_s, scores_if)),
                "count": n_pos,
            }
            logger.info(f"  {status}: AUC={result[status]['auc_roc']:.4f}, n={n_pos:,}")
    log_step(7, "Per-Status Evaluation", f"DONE ({time.perf_counter()-t0:.1f}s)")
    return result


def step8_shap_typology(X_test, scores_if, y_unified):
    """SHAP interpretability + 9-type anomaly typology."""
    log_step(8, "SHAP + Typology", "START")
    t0 = time.perf_counter()

    model_if = joblib.load(settings.models_output_dir / "isolation_forest.joblib")

    # Subsample: top-5% anomalies + 5K normals
    k = int(len(scores_if) * 0.05)
    top5_idx = np.argsort(scores_if)[-k:]
    rng = np.random.default_rng(42)
    normal_mask = np.ones(len(scores_if), dtype=bool)
    normal_mask[top5_idx] = False
    normal_idx = rng.choice(np.where(normal_mask)[0], size=min(5000, normal_mask.sum()), replace=False)
    sample_idx = np.concatenate([top5_idx, normal_idx])
    X_sample = X_test[sample_idx]

    logger.info(f"  SHAP sample: {len(top5_idx):,} anomalies + {len(normal_idx):,} normals = {len(sample_idx):,}")

    # SHAP TreeExplainer
    import shap
    try:
        explainer = shap.TreeExplainer(model_if)
        shap_values = explainer.shap_values(X_sample)
        logger.info(f"  TreeExplainer succeeded: {shap_values.shape}")
    except Exception as e:
        logger.warning(f"  TreeExplainer failed ({e}), falling back to KernelExplainer (500 samples)")
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model_if.decision_function, background)
        shap_values = explainer.shap_values(X_sample[:500])

    # Feature importance ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_ranking = sorted(
        zip(FEATURE_NAMES, mean_abs_shap.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    top10 = [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in feature_ranking[:10]]
    logger.info("  Top 10 features by SHAP:")
    for item in top10:
        logger.info(f"    {item['feature']}: {item['mean_abs_shap']:.6f}")

    # Save SHAP summary plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = settings.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_NAMES, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(fig_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"  Saved shap_summary.pdf + .png")

    # --- Anomaly typology (9 types) ---
    feature_to_type = {}
    type_mapping = {
        "amount": ["amount", "log_amount", "amount_usd_ratio", "amount_facility_ratio", "staff_amount_zscore"],
        "velocity": ["user_txn_count_1h", "user_txn_count_24h", "time_since_last_txn", "user_amount_24h"],
        "discount": ["discount_ratio", "user_discount_ratio_30d"],
        "temporal": ["hour_sin", "hour_cos", "day_of_week", "is_weekend", "is_off_hours"],
        "credit_flow": ["is_club_credit", "user_debit_count_30d", "user_debit_amount_30d", "credit_flow_ratio"],
        "role_deviation": ["is_staff", "paid_by_manager"],
        "diversity": ["user_distinct_facilities_30d", "user_distinct_methods", "category_entropy_30d", "user_merchandise_ratio_30d"],
        "reversal": ["user_reversal_ratio_30d", "user_reversal_count_30d"],
    }
    for typ, features in type_mapping.items():
        for f in features:
            feature_to_type[f] = typ

    # Classify only the top-5% anomalies portion of sample
    n_top5_in_sample = len(top5_idx)
    shap_anomalies = shap_values[:n_top5_in_sample]
    types_list = []
    for row in shap_anomalies:
        abs_row = np.abs(row)
        sorted_idx = np.argsort(abs_row)[::-1]
        top_feat = FEATURE_NAMES[sorted_idx[0]]
        top_val = abs_row[sorted_idx[0]]
        second_val = abs_row[sorted_idx[1]]
        if top_val < 2.0 * second_val:
            types_list.append("mixed")
        else:
            types_list.append(feature_to_type.get(top_feat, "mixed"))

    type_counts = pd.Series(types_list).value_counts()
    all_types = ["amount", "velocity", "discount", "temporal", "credit_flow",
                 "role_deviation", "diversity", "reversal", "mixed"]
    type_distribution = {}
    total = len(types_list)
    for t in all_types:
        c = int(type_counts.get(t, 0))
        type_distribution[t] = {"count": c, "pct": round(c / total * 100, 2) if total > 0 else 0}

    typology_result = {
        "n_anomalies_classified": total,
        "type_distribution": type_distribution,
        "dominance_threshold": 2.0,
        "feature_importance_top10": top10,
    }
    logger.info(f"  Typology: {dict(type_counts)}")

    log_step(8, "SHAP + Typology", f"DONE ({time.perf_counter()-t0:.1f}s)")
    return typology_result, types_list, top5_idx


def step9_user_risk_profiles(df_test, scores_if, types_list, top5_idx):
    """User risk profiles aggregated from test set."""
    log_step(9, "User Risk Profiles", "START")
    t0 = time.perf_counter()

    df = df_test[["user_id"]].copy()
    df["score"] = scores_if
    threshold = np.percentile(scores_if, 95)
    df["is_top5"] = scores_if >= threshold

    # Map anomaly types to full test set
    df["anomaly_type"] = None
    # types_list corresponds to top5_idx from SHAP
    for i, idx in enumerate(top5_idx):
        if i < len(types_list):
            df.iat[idx, df.columns.get_loc("anomaly_type")] = types_list[i]

    profiles = df.groupby("user_id").agg(
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        p95_score=("score", lambda x: float(np.percentile(x, 95))),
        n_total=("score", "count"),
        n_top5pct=("is_top5", "sum"),
    )
    profiles["concentration"] = profiles["n_top5pct"] / profiles["n_total"]

    # Dominant type per user (mode of their top-5% transactions)
    top5_df = df[df["is_top5"] & df["anomaly_type"].notna()]
    if not top5_df.empty:
        dominant = top5_df.groupby("user_id")["anomaly_type"].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        )
        profiles["dominant_type"] = dominant
    else:
        profiles["dominant_type"] = None

    # Save
    profiles_path = settings.project_root / "output" / "user_risk_profiles.parquet"
    profiles.to_parquet(profiles_path)

    n_flagged = int((profiles["concentration"] > 0.10).sum())
    n_total = len(profiles)

    summary = {
        "n_users_total": n_total,
        "n_users_flagged": n_flagged,
        "pct_users_flagged": round(n_flagged / n_total * 100, 2) if n_total > 0 else 0,
    }
    if n_flagged > 0:
        flagged = profiles[profiles["concentration"] > 0.10]
        dominant_dist = flagged["dominant_type"].value_counts().to_dict() if "dominant_type" in flagged.columns else {}
        summary["flagged_users_summary"] = {
            "mean_concentration": round(float(flagged["concentration"].mean()), 4),
            "max_concentration": round(float(flagged["concentration"].max()), 4),
            "dominant_types_distribution": {str(k): int(v) for k, v in dominant_dist.items()},
        }

    logger.info(f"  Users: {n_total:,} total, {n_flagged:,} flagged ({summary['pct_users_flagged']}%)")
    log_step(9, "User Risk Profiles", f"DONE ({time.perf_counter()-t0:.1f}s)")
    return summary


def step10_posthoc(df_test, scores_if):
    """Post-hoc: facility, manager, currency, discount concentration."""
    log_step(10, "Post-Hoc Analysis", "START")
    t0 = time.perf_counter()

    k_pct = 0.05
    threshold = np.percentile(scores_if, 100 * (1 - k_pct))
    df = df_test.copy()
    df["score_if"] = scores_if
    df["is_top_anomaly"] = scores_if >= threshold

    n_anomalies = int(df["is_top_anomaly"].sum())

    # Use discount_ratio as proxy for discount (raw discount not in features parquet)
    discount_col = "discount_ratio" if "discount_ratio" in df.columns else None

    # Facility concentration
    agg_dict = {
        "n_transactions": ("id", "count"),
        "n_anomalies": ("is_top_anomaly", "sum"),
        "anomaly_rate": ("is_top_anomaly", "mean"),
        "mean_score": ("score_if", "mean"),
    }
    if discount_col:
        agg_dict["mean_discount_ratio"] = (discount_col, "mean")

    fac = df.groupby("facility_id").agg(**agg_dict).sort_values("anomaly_rate", ascending=False)
    fac["anomaly_enrichment"] = fac["anomaly_rate"] / k_pct
    fac_enriched = fac[fac["anomaly_enrichment"] > 2.0]

    top_facilities = []
    for fid, row in fac_enriched.head(10).iterrows():
        entry = {
            "facility_id": int(fid),
            "n_transactions": int(row["n_transactions"]),
            "anomaly_rate": round(float(row["anomaly_rate"]), 4),
            "anomaly_enrichment": round(float(row["anomaly_enrichment"]), 2),
        }
        if discount_col:
            entry["mean_discount_ratio"] = round(float(row["mean_discount_ratio"]), 4)
        top_facilities.append(entry)
    logger.info(f"  Facilities with enrichment>2: {len(fac_enriched)}")

    # Manager concentration (aggregated — actor_identity_validated=False)
    df_manager = df[df["paid_by_manager"] == 1] if "paid_by_manager" in df.columns else pd.DataFrame()
    manager_result = {
        "mode": "aggregated_manager_intervention",
        "top_10_managers": [],
        "aggregate_only": {},
    }
    if not df_manager.empty:
        agg_mgr = {
            "n_transactions_with_manager_intervention": int(len(df_manager)),
            "n_anomalies_with_manager_intervention": int(df_manager["is_top_anomaly"].sum()),
            "anomaly_rate_with_manager_intervention": round(float(df_manager["is_top_anomaly"].mean()), 4),
        }
        if discount_col and discount_col in df_manager.columns:
            agg_mgr["mean_discount_ratio_with_manager_intervention"] = round(float(df_manager[discount_col].mean()), 4)
        manager_result["aggregate_only"] = agg_mgr
        logger.info(f"  Manager interventions: {len(df_manager):,}, anomaly_rate={df_manager['is_top_anomaly'].mean():.4f}")

    # Currency concentration
    curr_agg = {
        "n_transactions": ("id", "count"),
        "n_anomalies": ("is_top_anomaly", "sum"),
        "anomaly_rate": ("is_top_anomaly", "mean"),
        "mean_score": ("score_if", "mean"),
    }
    if discount_col:
        curr_agg["mean_discount_ratio"] = (discount_col, "mean")

    curr = df.groupby("currency").agg(**curr_agg).sort_values("anomaly_rate", ascending=False)
    curr["anomaly_enrichment"] = curr["anomaly_rate"] / k_pct

    currencies_affected = []
    for currency, row in curr.iterrows():
        entry = {
            "currency": str(currency),
            "n_transactions": int(row["n_transactions"]),
            "anomaly_rate": round(float(row["anomaly_rate"]), 4),
            "anomaly_enrichment": round(float(row["anomaly_enrichment"]), 2),
        }
        if discount_col:
            entry["mean_discount_ratio"] = round(float(row["mean_discount_ratio"]), 4)
        currencies_affected.append(entry)

    # Discount abuse pattern (using discount_ratio > 0 as proxy for discounted transactions)
    if discount_col:
        df_discount_anom = df[(df["is_top_anomaly"]) & (df[discount_col] > 0)]
        df_discount_mgr = df_discount_anom[df_discount_anom["paid_by_manager"] == 1] if "paid_by_manager" in df.columns else pd.DataFrame()
    else:
        df_discount_anom = pd.DataFrame()
        df_discount_mgr = pd.DataFrame()

    discount_pattern = {
        "n_anomalies_with_discount": int(len(df_discount_anom)),
        "n_anomalies_with_discount_by_manager": int(len(df_discount_mgr)),
    }

    posthoc = {
        "top_k_pct": k_pct,
        "n_total_anomalies": n_anomalies,
        "actor_identity_validated": False,
        "actor_identifier_field": None,
        "facility_concentration": {
            "n_facilities_with_enrichment_gt_2": int(len(fac_enriched)),
            "top_10_facilities": top_facilities,
        },
        "manager_concentration": manager_result,
        "currency_concentration": {"currencies_affected": currencies_affected},
        "discount_abuse_pattern": discount_pattern,
    }

    log_step(10, "Post-Hoc Analysis", f"DONE ({time.perf_counter()-t0:.1f}s)")
    return posthoc


def default_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    t_total = time.perf_counter()

    # Load all data
    df_test, X_test, scores_if, scores_if30, scores_if21, y_unified, y_tipo_a, y_wide = load_data()

    # Steps 1-7: sensitivity (no SHAP needed)
    proxy_sens = step1_proxy_sensitivity(scores_if, y_unified, y_tipo_a, y_wide)
    per_type = step2_per_type_metrics(df_test, scores_if)
    f18_sens = step3_feature18_sensitivity(scores_if, scores_if30, y_unified)
    ablation = step4_ablation_31vs21(scores_if, scores_if21, y_unified)
    segments = step5_segment_metrics(df_test, scores_if, y_unified)
    baselines = step6_baselines(scores_if, y_unified, df_test)
    per_status = step7_per_status(df_test, scores_if)

    # Step 8: SHAP + typology (heaviest step)
    typology, types_list, top5_idx = step8_shap_typology(X_test, scores_if, y_unified)

    # Step 9: user risk profiles
    user_profiles = step9_user_risk_profiles(df_test, scores_if, types_list, top5_idx)

    # Step 10: post-hoc analysis
    posthoc = step10_posthoc(df_test, scores_if)

    # --- Save results_sensitivity.json ---
    sensitivity = {
        "proxy_sensitivity": proxy_sens,
        "per_type_metrics": per_type,
        "feature18_sensitivity": f18_sens,
        "ablation_31_vs_21": ablation,
        "per_status": per_status,
        "segment_metrics": segments,
        "anomaly_typology": typology,
        "user_risk_profiles": user_profiles,
        "baselines": baselines,
    }
    sens_path = settings.project_root / "output" / "results_sensitivity.json"
    sens_path.write_text(json.dumps(sensitivity, indent=2, default=default_serializer))
    logger.info(f"Saved results_sensitivity.json → {sens_path}")

    # --- Save results_posthoc.json ---
    posthoc_path = settings.project_root / "output" / "results_posthoc.json"
    posthoc_path.write_text(json.dumps({"posthoc_analysis": posthoc}, indent=2, default=default_serializer))
    logger.info(f"Saved results_posthoc.json → {posthoc_path}")

    elapsed = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info(f"Fase 8 completada en {elapsed / 60:.1f} min")
    logger.info(f"  Proxy robust: {proxy_sens['robust']} (delta={proxy_sens['delta_auc_tipo_a']:.4f})")
    logger.info(f"  F18 low sensitivity: {f18_sens['low_sensitivity']} (delta={f18_sens['delta_auc']:.4f})")
    logger.info(f"  Groups F,G,H contribute: {ablation['groups_contribute']} (delta={ablation['delta']['auc_roc']:.4f})")
    logger.info(f"  IF beats all baselines: {baselines['if_beats_all']}")
    logger.info(f"  Users flagged: {user_profiles['n_users_flagged']}/{user_profiles['n_users_total']}")
    logger.info(f"  Facilities enriched>2: {posthoc['facility_concentration']['n_facilities_with_enrichment_gt_2']}")


if __name__ == "__main__":
    main()
