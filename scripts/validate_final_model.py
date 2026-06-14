#!/usr/bin/env python
"""Full statistical validation of the winning IF-40 model.

Runs:
  1. Multi-seed analysis (5 seeds) — variance + mean AUC per proxy
  2. Bootstrap 95% CI for AUC (1000 iterations) — winning seed
  3. Temporal stability (monthly AUC for Sep/Oct/Nov/Dec)
  4. Mann-Whitney U test (HE1) on scores positive vs negative

Output:
  output/results_validation_final.json
  output/figures/temporal_stability.csv

Uses pre-computed enriched parquets and the FINAL feature recipe.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings  # noqa: E402
from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.features.engineering import FEATURE_NAMES  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DROP = {"user_reversal_ratio_30d", "user_reversal_count_30d",
        "user_discount_ratio_30d", "user_txn_count_24h",
        "amount", "amount_usd_ratio"}
BASE = [f for f in FEATURE_NAMES if f not in DROP]
INTERACTIONS = ["is_new_user", "is_very_new_user", "new_user_first_facility",
                "rapid_burst", "small_amount_at_facility",
                "very_small_amount_at_facility", "off_hours_high_value"]
RAW_DERIVED = ["is_third_party_payment", "same_amount_count_1h", "same_amount_count_24h",
               "gateway_change_recent", "capture_delay_seconds", "is_main_gateway",
               "is_first_gateway_for_user", "source_change_recent"]
ALL_FEATURES = BASE + INTERACTIONS + RAW_DERIVED

SEEDS = [42, 43, 44, 45, 46]
N_BOOTSTRAP = 1000


def build_extended_proxy(df):
    tipo_a = DataManager.assign_proxy_labels(df, "tipo_a", settings).to_numpy()
    tipo_c = DataManager.assign_proxy_labels(df, "tipo_c", settings).to_numpy()
    tipo_d = DataManager.assign_proxy_labels(df, "tipo_d", settings).to_numpy()
    new_user_burst = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    small_amount = (df["amount_facility_ratio"] < 0.05).to_numpy()
    return (tipo_a | tipo_c | tipo_d | new_user_burst | small_amount).astype(np.int8)


def build_pure_fraud_proxy(df):
    card_test = (df["same_amount_count_1h"] >= 3).to_numpy()
    new_burst = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third_party_burst = ((df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    return (card_test | new_burst | third_party_burst).astype(np.int8)


def get_all_proxies(df):
    return {
        "unified": DataManager.assign_proxy_labels(df, "unified", settings).to_numpy(),
        "tipo_a": DataManager.assign_proxy_labels(df, "tipo_a", settings).to_numpy(),
        "extended": build_extended_proxy(df),
        "pure_fraud": build_pure_fraud_proxy(df),
    }


def fit_score(seed: int, X_train, X_test):
    """Train IF with given seed, return test scores."""
    model = IsolationForest(
        n_estimators=200, max_samples=512, max_features=0.6,
        contamination="auto", random_state=seed, n_jobs=-1,
    )
    model.fit(X_train)
    return -np.asarray(model.decision_function(X_test), dtype=np.float64), model


def main():
    logger.info("=" * 60)
    logger.info("VALIDATION — winning IF-40 model")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    df_train = pd.read_parquet(DATA_DIR / "train_features_enriched.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features_enriched.parquet")
    logger.info(f"  Loaded ({time.perf_counter() - t0:.1f}s)")

    X_train = df_train[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_test = df_test[ALL_FEATURES].to_numpy(dtype=np.float64)
    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
    X_train = np.clip(scaler.transform(X_train).astype(np.float32), -10, 10)
    X_test = np.clip(scaler.transform(X_test).astype(np.float32), -10, 10)
    logger.info(f"  Scaled: train={X_train.shape}, test={X_test.shape}")

    proxies = get_all_proxies(df_test)
    logger.info("  Proxies (test):")
    for k, y in proxies.items():
        logger.info(f"    {k:12s} n={y.sum():>8,}  rate={y.mean():.4f}")

    # ───────── 1. Multi-seed ─────────
    logger.info("")
    logger.info("─── 1. MULTI-SEED (5 seeds) ───")
    multi_seed = {p: [] for p in proxies}
    last_scores = None
    last_model = None
    for seed in SEEDS:
        t = time.perf_counter()
        s_test, model = fit_score(seed, X_train, X_test)
        logger.info(f"  seed={seed} fit+score in {time.perf_counter() - t:.1f}s")
        if seed == 42:
            last_scores = s_test
            last_model = model
        for p, y in proxies.items():
            multi_seed[p].append(float(roc_auc_score(y, s_test)))

    ms_summary = {
        p: {
            "auc_per_seed": multi_seed[p],
            "mean": float(np.mean(multi_seed[p])),
            "std": float(np.std(multi_seed[p], ddof=1)),
            "min": float(np.min(multi_seed[p])),
            "max": float(np.max(multi_seed[p])),
            "range": float(np.max(multi_seed[p]) - np.min(multi_seed[p])),
        }
        for p in proxies
    }
    logger.info("  Summary:")
    for p, s in ms_summary.items():
        logger.info(
            f"    {p:12s} mean={s['mean']:.4f}  std={s['std']:.5f}  "
            f"range={s['range']:.5f}  {'(trivial dispersion)' if s['range'] < 0.005 else '(non-trivial)'}"
        )

    # ───────── 2. Bootstrap CI (seed=42) ─────────
    logger.info("")
    logger.info(f"─── 2. BOOTSTRAP 95% CI ({N_BOOTSTRAP} iter, seed=42) ───")
    rng = np.random.default_rng(42)
    bootstrap_ci = {}
    for pname, y in proxies.items():
        aucs = np.zeros(N_BOOTSTRAP, dtype=np.float64)
        aps = np.zeros(N_BOOTSTRAP, dtype=np.float64)
        n = len(last_scores)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            y_boot = y[idx]
            if y_boot.sum() == 0 or y_boot.sum() == len(y_boot):
                aucs[i] = np.nan
                aps[i] = np.nan
                continue
            aucs[i] = roc_auc_score(y_boot, last_scores[idx])
            aps[i] = average_precision_score(y_boot, last_scores[idx])
        bootstrap_ci[pname] = {
            "auc_mean": float(np.nanmean(aucs)),
            "auc_p2.5": float(np.nanpercentile(aucs, 2.5)),
            "auc_p97.5": float(np.nanpercentile(aucs, 97.5)),
            "ap_mean": float(np.nanmean(aps)),
            "ap_p2.5": float(np.nanpercentile(aps, 2.5)),
            "ap_p97.5": float(np.nanpercentile(aps, 97.5)),
            "n_iterations": N_BOOTSTRAP,
        }
        logger.info(
            f"  {pname:12s} AUC = {bootstrap_ci[pname]['auc_mean']:.4f} "
            f"[{bootstrap_ci[pname]['auc_p2.5']:.4f}, {bootstrap_ci[pname]['auc_p97.5']:.4f}]   "
            f"AP = {bootstrap_ci[pname]['ap_mean']:.4f} "
            f"[{bootstrap_ci[pname]['ap_p2.5']:.4f}, {bootstrap_ci[pname]['ap_p97.5']:.4f}]"
        )

    # ───────── 3. Temporal stability (monthly) ─────────
    logger.info("")
    logger.info("─── 3. TEMPORAL STABILITY (monthly AUC, seed=42) ───")
    df_test_dates = pd.to_datetime(df_test["created_at"])
    months = sorted(df_test_dates.dt.to_period("M").unique())
    temporal = {p: {} for p in proxies}
    for month in months:
        mask = (df_test_dates.dt.to_period("M") == month).to_numpy()
        n = mask.sum()
        if n < 100:
            continue
        for pname, y in proxies.items():
            y_m = y[mask]
            if y_m.sum() == 0 or y_m.sum() == n:
                continue
            auc = float(roc_auc_score(y_m, last_scores[mask]))
            temporal[pname][str(month)] = {"n": int(n), "rate": float(y_m.mean()), "auc": auc}
        logger.info(f"  {month}: n={n:,}")
    for pname in proxies:
        ms = temporal[pname]
        if not ms:
            continue
        aucs = [m["auc"] for m in ms.values()]
        logger.info(
            f"    {pname:12s} per-month AUC: " +
            "  ".join(f"{k}={v['auc']:.4f}" for k, v in ms.items()) +
            f"   range={max(aucs) - min(aucs):.4f}"
        )

    # Save temporal CSV
    rows = []
    for pname, ms in temporal.items():
        for k, v in ms.items():
            rows.append({"proxy": pname, "month": k, "n_samples": v["n"], "proxy_rate": v["rate"], "auc": v["auc"]})
    pd.DataFrame(rows).to_csv(FIG_DIR / "temporal_stability.csv", index=False)
    logger.info(f"  → figures/temporal_stability.csv")

    # ───────── 4. Mann-Whitney U test (HE1) ─────────
    logger.info("")
    logger.info("─── 4. MANN-WHITNEY U (HE1) — scores anomaly vs normal ───")
    he1 = {}
    for pname, y in proxies.items():
        if y.sum() == 0:
            continue
        scores_pos = last_scores[y == 1]
        scores_neg = last_scores[y == 0]
        u, p = mannwhitneyu(scores_pos, scores_neg, alternative="greater")
        n_pos = len(scores_pos)
        n_neg = len(scores_neg)
        cles = float(u / (n_pos * n_neg))  # Common Language Effect Size = AUC
        rank_biserial = 2 * cles - 1
        he1[pname] = {
            "U": float(u), "p_value": float(p),
            "n_anomaly": n_pos, "n_normal": n_neg,
            "rank_biserial_r": rank_biserial,
            "cles": cles,
            "he1_pass": bool(p < 0.001 and rank_biserial > 0.05),
        }
        logger.info(
            f"  {pname:12s} U={u:.2e}  p={p:.2e}  r_b={rank_biserial:.4f}  CLES={cles:.4f}  "
            f"HE1={'PASS' if he1[pname]['he1_pass'] else 'FAIL'}"
        )

    # ───────── Save model (winning seed=42) ─────────
    import joblib
    joblib.dump(last_model, MODELS_DIR / "isolation_forest_final.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_final.joblib")
    (MODELS_DIR / "final_feature_list.json").write_text(json.dumps(ALL_FEATURES, indent=2))
    logger.info(f"\n  Saved: isolation_forest_final.joblib, scaler_final.joblib, final_feature_list.json")

    # ───────── Persist all results ─────────
    out = {
        "model_config": {
            "n_features": len(ALL_FEATURES),
            "features": ALL_FEATURES,
            "model_params": {"n_estimators": 200, "max_samples": 512, "max_features": 0.6, "contamination": "auto"},
            "scaler": "RobustScaler(5-95) + clip(-10, 10)",
        },
        "multi_seed": ms_summary,
        "bootstrap_ci_95pct": bootstrap_ci,
        "temporal_stability": temporal,
        "he1_mann_whitney": he1,
        "seeds_used": SEEDS,
        "n_bootstrap": N_BOOTSTRAP,
    }
    out_path = OUTPUT_DIR / "results_validation_final.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"\n  Written {out_path}")
    logger.info(f"Total: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
