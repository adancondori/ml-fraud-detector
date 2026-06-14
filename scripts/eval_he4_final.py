#!/usr/bin/env python
"""HE4 final: compare IF vs LOF vs OC-SVM on the FINAL enriched feature set.

Reads pre-saved arrays from eval_with_raw_features.py:
  output/scores/X_train_final.npy
  output/scores/X_val_final.npy
  output/scores/X_test_final.npy
  output/scores/if_test_scores_final.npy  (IF already scored)

Trains LOF (novelty=True) and OC-SVM on temporal subsample, scores test,
compares all four metrics (AUC, AP, P@5%, EF@5%) across IF/LOF/OC-SVM
against unified, tipo_a, extended, pure_fraud proxies.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings  # noqa: E402
from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SCORES_DIR = PROJECT_ROOT / "output" / "scores"
OUTPUT_DIR = PROJECT_ROOT / "output"


def temporal_subsample(X, n=100_000):
    if X.shape[0] <= n:
        return X
    idx = np.linspace(0, X.shape[0] - 1, n, dtype=int)
    return X[idx]


def percentile_metrics(scores, y, pcts=(0.01, 0.02, 0.05, 0.10)):
    base = float(y.mean())
    order = np.argsort(-scores)
    out = {}
    for pct in pcts:
        k = max(1, int(np.ceil(len(scores) * pct)))
        prec = float(y[order[:k]].sum() / k)
        out[f"p_{int(pct * 100)}pct"] = prec
        out[f"ef_{int(pct * 100)}pct"] = prec / base if base > 0 else 0.0
    return out


def build_extended_proxy(df):
    tipo_a = DataManager.assign_proxy_labels(df, "tipo_a", settings).to_numpy()
    tipo_c = DataManager.assign_proxy_labels(df, "tipo_c", settings).to_numpy()
    tipo_d = DataManager.assign_proxy_labels(df, "tipo_d", settings).to_numpy()
    new_user_burst = (
        (df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 2)
    ).to_numpy()
    small_amount_extreme = (df["amount_facility_ratio"] < 0.05).to_numpy()
    return (tipo_a | tipo_c | tipo_d | new_user_burst | small_amount_extreme).astype(np.int8)


def build_pure_fraud_proxy(df):
    cols = df.columns
    card_test = (df["same_amount_count_1h"] >= 3).to_numpy() if "same_amount_count_1h" in cols else np.zeros(len(df), dtype=bool)
    new_burst = (
        (df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)
    ).to_numpy()
    third_party_burst = (
        (df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)
    ).to_numpy() if "is_third_party_payment" in cols else np.zeros(len(df), dtype=bool)
    return (card_test | new_burst | third_party_burst).astype(np.int8)


def evaluate(scores, df_meta):
    out = {}
    proxies = {
        "unified": DataManager.assign_proxy_labels(df_meta, "unified", settings).to_numpy(),
        "tipo_a": DataManager.assign_proxy_labels(df_meta, "tipo_a", settings).to_numpy(),
        "extended": build_extended_proxy(df_meta),
        "pure_fraud": build_pure_fraud_proxy(df_meta),
    }
    for label, y in proxies.items():
        if y.sum() == 0:
            continue
        block = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "base_rate": float(y.mean()),
        }
        block.update(percentile_metrics(scores, y))
        out[label] = block
    return out


def main():
    logger.info("=" * 60)
    logger.info("HE4: IF vs LOF vs OC-SVM on FINAL enriched feature set")
    logger.info("=" * 60)

    X_train = np.load(SCORES_DIR / "X_train_final.npy")
    X_val = np.load(SCORES_DIR / "X_val_final.npy")
    X_test = np.load(SCORES_DIR / "X_test_final.npy")
    if_val = np.load(SCORES_DIR / "if_val_scores_final.npy")
    if_test = np.load(SCORES_DIR / "if_test_scores_final.npy")
    df_val = pd.read_parquet(DATA_DIR / "val_features_enriched.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features_enriched.parquet")
    logger.info(f"  Loaded: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    # LOF — full train would be too slow; use subsample (memory bound)
    logger.info("  Training LOF on temporal subsample (200k rows)...")
    t = time.perf_counter()
    X_train_lof = temporal_subsample(X_train, 200_000)
    lof = LocalOutlierFactor(
        n_neighbors=20, novelty=True, metric="minkowski", n_jobs=-1,
    )
    lof.fit(X_train_lof)
    lof_val = -lof.decision_function(X_val)
    lof_test = -lof.decision_function(X_test)
    logger.info(f"    LOF done ({time.perf_counter() - t:.1f}s)")

    # OC-SVM — RBF on small subsample
    logger.info("  Training OC-SVM on temporal subsample (50k rows)...")
    t = time.perf_counter()
    X_train_oc = temporal_subsample(X_train, 50_000)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X_train_oc)
    oc_val = -ocsvm.decision_function(X_val)
    oc_test = -ocsvm.decision_function(X_test)
    logger.info(f"    OC-SVM done ({time.perf_counter() - t:.1f}s)")

    results = {}
    for name, sv, st in [("if", if_val, if_test), ("lof", lof_val, lof_test), ("ocsvm", oc_val, oc_test)]:
        results[name] = {
            "val": evaluate(sv, df_val),
            "test": evaluate(st, df_test),
        }

    logger.info("")
    logger.info("RESULTS — HE4 comparison:")
    for proxy in ("unified", "extended", "pure_fraud", "tipo_a"):
        logger.info(f"\n  {proxy}:")
        logger.info(f"    {'model':10s} {'AUC':>8s} {'AP':>8s} {'P@5%':>8s} {'EF@5%':>8s}")
        if_wins_count = 0
        for name in ("if", "lof", "ocsvm"):
            t = results[name]["test"].get(proxy, {})
            if t:
                logger.info(
                    f"    {name:10s} {t['auc']:>8.4f} {t['ap']:>8.4f} "
                    f"{t.get('p_5pct', 0):>8.3f} {t.get('ef_5pct', 0):>8.2f}"
                )
        # Who wins in AUC
        ifa = results["if"]["test"].get(proxy, {}).get("auc", 0)
        lofa = results["lof"]["test"].get(proxy, {}).get("auc", 0)
        oca = results["ocsvm"]["test"].get(proxy, {}).get("auc", 0)
        if ifa >= lofa and ifa >= oca:
            logger.info(f"    → HE4 holds for {proxy}: IF ≥ {{LOF, OCSVM}}")
        else:
            logger.info(f"    → HE4 FAILS for {proxy}: IF AUC={ifa:.4f}, LOF={lofa:.4f}, OCSVM={oca:.4f}")

    out_path = OUTPUT_DIR / "results_he4_final.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\nWritten {out_path}")


if __name__ == "__main__":
    main()
