#!/usr/bin/env python
"""FS-disjoint ablation: does IF-40's AUC vs proxy_anomalias_operativas survive
when every feature that appears in the proxy definition is removed?

The proxy (see eval_with_raw_features.build_proxy_anomalias_operativas) is built
from: same_amount_count_1h, user_account_age_days, user_txn_count_1h,
is_third_party_payment. Several interaction features are direct recodings of
those variables, so a honest disjunction test must drop them too.

Tiers (cumulative):
  T1 FS-disjoint-36: drop the 4 variables used verbatim in the proxy
  T2 FS-disjoint-31: + recodings (is_new_user, is_very_new_user,
     new_user_first_facility, rapid_burst, same_amount_count_24h)
  T3 FS-disjoint-30: + capture_delay_seconds (post-approval field)

Each tier retrains IF (same recipe as eval_with_raw_features.py) with seeds
42/43/44 and evaluates against pure_fraud, unified and tipo_a on the test set.
For T3 (confirmatory candidate) LOF and OC-SVM are also trained (same recipe
as eval_he4_final.py) for HE4.

Reads:  data/processed/{train,test}_features_enriched.parquet
Writes: output/results_fs_disjoint.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "output" / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

SEEDS = (42, 43, 44)

PROXY_VARS = [
    "same_amount_count_1h",
    "user_account_age_days",
    "user_txn_count_1h",
    "is_third_party_payment",
]
PROXY_RECODINGS = [
    "is_new_user",
    "is_very_new_user",
    "new_user_first_facility",
    "rapid_burst",
    "same_amount_count_24h",
]
POST_APPROVAL = ["capture_delay_seconds"]


def build_proxies(df: pd.DataFrame) -> dict:
    tipo_a = df["status"].isin(["totally_refunded", "refunded_to_credit"]).to_numpy()
    tipo_c = (df["user_discount_ratio_30d"] > 1.0).to_numpy()
    tipo_d = ((df["user_txn_count_24h"] + 1) > 100).to_numpy()
    card_test = (df["same_amount_count_1h"] >= 3).to_numpy()
    new_burst = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third_party_burst = ((df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    return {
        "pure_fraud": (card_test | new_burst | third_party_burst).astype(np.int8),
        "unified": (tipo_a | tipo_c | tipo_d).astype(np.int8),
        "tipo_a": tipo_a.astype(np.int8),
    }


def evaluate(scores: np.ndarray, proxies: dict) -> dict:
    out = {}
    for label, y in proxies.items():
        base = float(y.mean())
        order = np.argsort(-scores)
        block = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "base_rate": base,
        }
        for pct in (0.01, 0.05):
            k = max(1, int(np.ceil(len(scores) * pct)))
            prec = float(y[order[:k]].sum() / k)
            block[f"p_{int(pct * 100)}pct"] = prec
            block[f"ef_{int(pct * 100)}pct"] = prec / base if base > 0 else 0.0
        out[label] = block
    return out


def temporal_subsample(X: np.ndarray, n: int) -> np.ndarray:
    if X.shape[0] <= n:
        return X
    idx = np.linspace(0, X.shape[0] - 1, n, dtype=int)
    return X[idx]


def main() -> None:
    all_features = json.loads((MODELS_DIR / "final_feature_list.json").read_text())
    tiers = {
        "FS-disjoint-36": [f for f in all_features if f not in PROXY_VARS],
        "FS-disjoint-31": [
            f for f in all_features if f not in PROXY_VARS + PROXY_RECODINGS
        ],
        "FS-disjoint-30": [
            f for f in all_features
            if f not in PROXY_VARS + PROXY_RECODINGS + POST_APPROVAL
        ],
    }
    for name, feats in tiers.items():
        expected = int(name.rsplit("-", 1)[1])
        assert len(feats) == expected, f"{name}: {len(feats)} features"

    logger.info("Loading enriched parquets...")
    df_train = pd.read_parquet(DATA_DIR / "train_features_enriched.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features_enriched.parquet")
    proxies = build_proxies(df_test)
    for label, y in proxies.items():
        logger.info(f"  proxy {label}: rate={float(y.mean()):.4f} n={int(y.sum()):,}")

    results = {"seeds": list(SEEDS), "tiers": {}}
    lof_ocsvm_inputs = None

    for tier_name, feats in tiers.items():
        logger.info("=" * 60)
        logger.info(f"{tier_name}: {len(feats)} features")
        t0 = time.perf_counter()
        X_train = df_train[feats].to_numpy(dtype=np.float32)
        X_test = df_test[feats].to_numpy(dtype=np.float32)
        scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
        X_train = np.clip(scaler.transform(X_train).astype(np.float32), -10, 10)
        X_test = np.clip(scaler.transform(X_test).astype(np.float32), -10, 10)
        logger.info(f"  scaled ({time.perf_counter() - t0:.1f}s)")

        per_seed = []
        for seed in SEEDS:
            t = time.perf_counter()
            model = IsolationForest(
                n_estimators=200, max_samples=512, max_features=0.6,
                contamination="auto", random_state=seed, n_jobs=-1,
            )
            model.fit(X_train)
            scores = -np.asarray(model.decision_function(X_test), dtype=np.float64)
            ev = evaluate(scores, proxies)
            per_seed.append(ev)
            logger.info(
                f"  seed {seed}: pure_fraud AUC={ev['pure_fraud']['auc']:.4f} "
                f"EF@1%={ev['pure_fraud']['ef_1pct']:.2f} "
                f"tipo_a AUC={ev['tipo_a']['auc']:.4f} "
                f"({time.perf_counter() - t:.1f}s)"
            )
            if tier_name == "FS-disjoint-30" and seed == SEEDS[0]:
                np.save(OUTPUT_DIR / "scores" / "if_test_scores_fs_disjoint30.npy", scores)

        summary = {}
        for proxy in proxies:
            aucs = [ev[proxy]["auc"] for ev in per_seed]
            summary[proxy] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "auc_min": float(np.min(aucs)),
                "auc_max": float(np.max(aucs)),
                "detail_seed_42": per_seed[0][proxy],
            }
        results["tiers"][tier_name] = {"n_features": len(feats), "features": feats, "if": summary}

        if tier_name == "FS-disjoint-30":
            lof_ocsvm_inputs = (X_train, X_test)
        else:
            del X_train, X_test

    # HE4 on the confirmatory candidate tier (same recipe as eval_he4_final.py)
    X_train, X_test = lof_ocsvm_inputs
    logger.info("=" * 60)
    logger.info("HE4 on FS-disjoint-30: LOF (200k subsample) + OC-SVM (50k subsample)")
    t = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, metric="minkowski", n_jobs=-1)
    lof.fit(temporal_subsample(X_train, 200_000))
    lof_scores = -lof.decision_function(X_test)
    results["tiers"]["FS-disjoint-30"]["lof"] = evaluate(lof_scores, proxies)
    logger.info(f"  LOF done ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(temporal_subsample(X_train, 50_000))
    ocsvm_scores = -ocsvm.decision_function(X_test)
    results["tiers"]["FS-disjoint-30"]["ocsvm"] = evaluate(ocsvm_scores, proxies)
    logger.info(f"  OC-SVM done ({time.perf_counter() - t:.1f}s)")

    out_path = OUTPUT_DIR / "results_fs_disjoint.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Written {out_path}")

    logger.info("")
    logger.info("SUMMARY — AUC vs pure_fraud (mean over seeds):")
    logger.info(f"  {'tier':18s} {'pure_fraud':>11s} {'unified':>9s} {'tipo_a':>8s}")
    logger.info(f"  {'IF-40 (reference)':18s} {'0.8412':>11s} {'0.5891':>9s} {'0.4910':>8s}")
    for tier_name, block in results["tiers"].items():
        s = block["if"]
        logger.info(
            f"  {tier_name:18s} {s['pure_fraud']['auc_mean']:>11.4f} "
            f"{s['unified']['auc_mean']:>9.4f} {s['tipo_a']['auc_mean']:>8.4f}"
        )
    for model_name in ("lof", "ocsvm"):
        b = results["tiers"]["FS-disjoint-30"].get(model_name, {})
        if b:
            logger.info(
                f"  {model_name.upper() + ' (30)':18s} {b['pure_fraud']['auc']:>11.4f} "
                f"{b['unified']['auc']:>9.4f} {b['tipo_a']['auc']:>8.4f}"
            )


if __name__ == "__main__":
    main()
