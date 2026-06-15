#!/usr/bin/env python
"""Clustered bootstrap CI for the confirmatory evaluation (§3.4 of the thesis).

OPTIMIZATION: AUC and AP are computed in the SAME iteration (one resample per
iteration, two metrics per resample). Halves the wall-time vs naive impl.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SCORES_DIR = PROJECT_ROOT / "output" / "revision" / "scores"
OUT_PATH = PROJECT_ROOT / "output" / "revision" / "bootstrap_confirmatorio.json"

MODELS = {
    "if":    "if_clean_a_tipo_a_test_scores.parquet",
    "lof":   "lof_clean_a_tipo_a_test_scores.parquet",
    "ocsvm": "ocsvm_clean_a_tipo_a_test_scores.parquet",
}
N_BOOTSTRAP = 1000
SEED = 42


def _ci_summary(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean":  float(np.nanmean(arr)),
        "lower": float(np.nanquantile(arr, 0.025)),
        "upper": float(np.nanquantile(arr, 0.975)),
        "n_iterations": int(arr.size),
    }


def bootstrap_dual_by_txn(
    y: np.ndarray, s: np.ndarray, n_iter: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap por transacción: AUC y AP en la misma iteración."""
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_iter, dtype=np.float64)
    aps = np.empty(n_iter, dtype=np.float64)
    n = len(y)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        s_b = s[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            aucs[i] = np.nan
            aps[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_b, s_b)
        aps[i] = average_precision_score(y_b, s_b)
    return aucs, aps


def bootstrap_dual_by_user(
    y: np.ndarray, s: np.ndarray, user_ids: np.ndarray, n_iter: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap clustered por usuario vía sample_weight. AUC y AP juntos."""
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
    logger.info("CONFIRMATORY BOOTSTRAP CI — dual metric per iteration")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    df_feat = pd.read_parquet(DATA_DIR / "test_features.parquet", columns=["id", "user_id"])
    logger.info(f"  Loaded {len(df_feat):,} rows ({time.perf_counter() - t0:.1f}s)")
    n_users_test = int(df_feat["user_id"].nunique())
    logger.info(f"  n_users_unique = {n_users_test:,}")

    results = {}
    for model, scores_file in MODELS.items():
        scores_path = SCORES_DIR / scores_file
        if not scores_path.exists():
            raise FileNotFoundError(f"Missing scores parquet: {scores_path}")

        t0 = time.perf_counter()
        df_s = pd.read_parquet(scores_path)
        df = df_s.merge(df_feat, on="id", validate="1:1")
        y = df["proxy_label"].to_numpy()
        s = df["anomaly_score"].to_numpy()
        uids = df["user_id"].to_numpy()
        logger.info(
            f"\n── {model.upper()} ──  n={len(df):,}  positives={int(y.sum()):,}  "
            f"loaded+joined in {time.perf_counter() - t0:.1f}s"
        )

        t0 = time.perf_counter()
        auc_txn, ap_txn = bootstrap_dual_by_txn(y, s, N_BOOTSTRAP, SEED)
        logger.info(f"  by_txn  finished in {time.perf_counter() - t0:.1f}s")

        t0 = time.perf_counter()
        auc_usr, ap_usr = bootstrap_dual_by_user(y, s, uids, N_BOOTSTRAP, SEED)
        logger.info(f"  by_user finished in {time.perf_counter() - t0:.1f}s")

        auc_txn_ci = _ci_summary(auc_txn)
        ap_txn_ci = _ci_summary(ap_txn)
        auc_usr_ci = _ci_summary(auc_usr)
        auc_usr_ci.update({"n_users_resampled": n_users_test, "method_used": "weighted"})
        ap_usr_ci = _ci_summary(ap_usr)
        ap_usr_ci.update({"n_users_resampled": n_users_test, "method_used": "weighted"})

        def _ratio(usr: Dict, txn: Dict) -> float:
            w_txn = txn["upper"] - txn["lower"]
            w_usr = usr["upper"] - usr["lower"]
            return float("nan") if w_txn == 0.0 else (w_usr / w_txn)

        results[model] = {
            "auc": {
                "by_transaction": auc_txn_ci,
                "by_user": auc_usr_ci,
                "width_ratio_user_over_txn": _ratio(auc_usr_ci, auc_txn_ci),
            },
            "ap": {
                "by_transaction": ap_txn_ci,
                "by_user": ap_usr_ci,
                "width_ratio_user_over_txn": _ratio(ap_usr_ci, ap_txn_ci),
            },
        }
        logger.info(
            f"  AUC  txn={auc_txn_ci['mean']:.4f} [{auc_txn_ci['lower']:.4f}, {auc_txn_ci['upper']:.4f}]   "
            f"user={auc_usr_ci['mean']:.4f} [{auc_usr_ci['lower']:.4f}, {auc_usr_ci['upper']:.4f}]   "
            f"ratio={results[model]['auc']['width_ratio_user_over_txn']:.2f}x"
        )
        logger.info(
            f"  AP   txn={ap_txn_ci['mean']:.4f} [{ap_txn_ci['lower']:.4f}, {ap_txn_ci['upper']:.4f}]   "
            f"user={ap_usr_ci['mean']:.4f} [{ap_usr_ci['lower']:.4f}, {ap_usr_ci['upper']:.4f}]   "
            f"ratio={results[model]['ap']['width_ratio_user_over_txn']:.2f}x"
        )

    results["_metadata"] = {
        "n_users_unique": n_users_test,
        "n_iterations": N_BOOTSTRAP,
        "seed": SEED,
        "method_clustered": "weighted (sample_weight)",
        "source_scores_dir": str(SCORES_DIR.relative_to(PROJECT_ROOT)),
        "source_features": "data/processed/test_features.parquet",
        "optimization": "dual_metric_per_iteration",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    logger.info(f"\n  → {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
