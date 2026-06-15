#!/usr/bin/env python
"""Add `bootstrap_ci_95pct_by_user` to results_validation_final.json without re-running
the full validate_final_model.py (multi-seed + temporal + MWU).

Loads the saved final IF-40 model, re-scores the test set, computes clustered
bootstrap CI by user for AUC/AP across all 4 proxies, and merges the new key
into the existing JSON. Preserves all other keys bit-for-bit.

Optimized: AUC and AP computed in the same iteration (dual_metric_per_iteration).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

# Import same proxy helpers as validate_final_model
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from validate_final_model import get_all_proxies, ALL_FEATURES  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "output" / "models"
JSON_PATH = PROJECT_ROOT / "output" / "results_validation_final.json"
N_BOOTSTRAP = 1000
SEED = 42


def bootstrap_dual_by_user(y, s, user_ids, n_iter, seed):
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_iter, dtype=np.float64)
    aps = np.empty(n_iter, dtype=np.float64)
    unique_u, inverse = np.unique(user_ids, return_inverse=True)
    n_users = unique_u.size
    y_f = y.astype(np.float64)
    for i in range(n_iter):
        counts = np.bincount(rng.integers(0, n_users, size=n_users), minlength=n_users)
        w = counts[inverse]
        pos_w = float(np.dot(y_f, w))
        tot_w = float(w.sum())
        if pos_w == 0.0 or pos_w == tot_w:
            aucs[i] = np.nan
            aps[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y, s, sample_weight=w)
        aps[i] = average_precision_score(y, s, sample_weight=w)
    return aucs, aps


def main() -> None:
    logger.info("=" * 60)
    logger.info("EVAL V — bootstrap_ci_95pct_by_user (additive update)")
    logger.info("=" * 60)

    # Load existing JSON (must exist)
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Missing: {JSON_PATH}. Run validate_final_model.py first.")
    out = json.loads(JSON_PATH.read_text())
    if "bootstrap_ci_95pct" not in out:
        raise ValueError("Existing JSON has no bootstrap_ci_95pct — refuse to overwrite.")

    # Load test data + final IF model + scaler
    t0 = time.perf_counter()
    df_train = pd.read_parquet(DATA_DIR / "train_features_enriched.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features_enriched.parquet")
    logger.info(f"  Loaded train/test ({time.perf_counter() - t0:.1f}s)")

    X_train = df_train[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_test = df_test[ALL_FEATURES].to_numpy(dtype=np.float64)
    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
    X_test = np.clip(scaler.transform(X_test).astype(np.float32), -10, 10)

    model = joblib.load(MODELS_DIR / "isolation_forest_final.joblib")
    test_scores = -model.decision_function(X_test)
    logger.info(f"  Scored test ({time.perf_counter() - t0:.1f}s total)")

    proxies = get_all_proxies(df_test)
    user_ids_test = df_test["user_id"].to_numpy()
    n_users_test = int(np.unique(user_ids_test).size)

    bootstrap_ci_by_user = {}
    for pname, y in proxies.items():
        logger.info(f"\n  Proxy: {pname}")
        t0 = time.perf_counter()
        auc_arr, ap_arr = bootstrap_dual_by_user(
            y, test_scores, user_ids_test, N_BOOTSTRAP, SEED
        )
        elapsed = time.perf_counter() - t0
        bootstrap_ci_by_user[pname] = {
            "auc_mean":  float(np.nanmean(auc_arr)),
            "auc_p2.5":  float(np.nanquantile(auc_arr, 0.025)),
            "auc_p97.5": float(np.nanquantile(auc_arr, 0.975)),
            "ap_mean":   float(np.nanmean(ap_arr)),
            "ap_p2.5":   float(np.nanquantile(ap_arr, 0.025)),
            "ap_p97.5":  float(np.nanquantile(ap_arr, 0.975)),
            "n_iterations": N_BOOTSTRAP,
            "n_users_resampled": n_users_test,
            "method_used": "weighted",
        }
        v = bootstrap_ci_by_user[pname]
        logger.info(
            f"    AUC={v['auc_mean']:.4f} [{v['auc_p2.5']:.4f}, {v['auc_p97.5']:.4f}]   "
            f"AP={v['ap_mean']:.4f} [{v['ap_p2.5']:.4f}, {v['ap_p97.5']:.4f}]   "
            f"({elapsed:.0f}s)"
        )

    # Merge the new key into the existing JSON, preserving everything else bit-for-bit
    out["bootstrap_ci_95pct_by_user"] = bootstrap_ci_by_user
    JSON_PATH.write_text(json.dumps(out, indent=2))
    logger.info(f"\n  → {JSON_PATH.relative_to(PROJECT_ROOT)} (additive update)")


if __name__ == "__main__":
    main()
