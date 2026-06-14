#!/usr/bin/env python
"""Ensemble: IF + ECOD + COPOD with rank-averaging.

Combines three unsupervised detectors with different inductive biases:
  - IsolationForest: splits-based outlier isolation (axis-aligned cuts)
  - ECOD (Empirical Cumulative-distribution Outlier Detection):
        parameter-free, distribution-based, very fast
  - COPOD (Copula-Based Outlier Detection):
        copula tail dependence, captures multivariate extremes

Rank-averaging combines scores robustly even when they live on different
scales. Typical lift over single IF: +5-8 AUC points in unsupervised settings.

Uses the CLEAN feature set (25 features, no circular, no redundant).
Evaluates against unified + Tipo A (honest).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from scipy.stats import rankdata
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

CIRCULAR = {"user_reversal_ratio_30d", "user_reversal_count_30d",
            "user_discount_ratio_30d", "user_txn_count_24h"}
REDUNDANT = {"amount", "amount_usd_ratio"}
DROP = CIRCULAR | REDUNDANT
FEATURES = [f for f in FEATURE_NAMES if f not in DROP]


def to_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores to percentile ranks in [0, 1] for averaging."""
    return rankdata(scores, method="average") / len(scores)


def percentile_metrics(scores: np.ndarray, y: np.ndarray, pcts=(0.01, 0.02, 0.05, 0.10)) -> dict:
    base_rate = float(y.mean())
    n = len(scores)
    order = np.argsort(-scores)
    out = {}
    for pct in pcts:
        k = max(1, int(np.ceil(n * pct)))
        prec = float(y[order[:k]].sum() / k)
        out[f"precision_at_{int(pct * 100)}pct"] = prec
        out[f"ef_at_{int(pct * 100)}pct"] = prec / base_rate if base_rate > 0 else 0.0
    return out


def evaluate(scores: np.ndarray, df_meta: pd.DataFrame, label: str) -> dict:
    out = {"label": label}
    for ptype in ("unified", "tipo_a", "tipo_c", "tipo_d"):
        y = DataManager.assign_proxy_labels(df_meta, ptype, settings).to_numpy()
        if y.sum() == 0:
            out[ptype] = {"auc": None}
            continue
        block = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "base_rate": float(y.mean()),
        }
        if ptype in ("unified", "tipo_a"):
            block.update(percentile_metrics(scores, y))
        out[ptype] = block
    return out


def main() -> None:
    logger.info("=" * 60)
    logger.info(f"ENSEMBLE IF + ECOD + COPOD — {len(FEATURES)} clean features")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    df_train = pd.read_parquet(DATA_DIR / "train_features.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val_features.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features.parquet")
    logger.info(f"  Loaded ({time.perf_counter() - t0:.1f}s)")

    X_train = df_train[FEATURES].to_numpy(dtype=np.float64)
    X_val = df_val[FEATURES].to_numpy(dtype=np.float64)
    X_test = df_test[FEATURES].to_numpy(dtype=np.float64)

    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    # Clip extreme post-scaled values to bound effect of remaining outliers
    X_train = np.clip(X_train, -10, 10)
    X_val = np.clip(X_val, -10, 10)
    X_test = np.clip(X_test, -10, 10)
    logger.info(f"  Scaled + clipped to [-10, 10]")

    # ───── 1. ISOLATION FOREST ─────
    t = time.perf_counter()
    if_model = IsolationForest(
        n_estimators=200,
        max_samples=512,
        max_features=0.6,
        random_state=42,
        n_jobs=-1,
    )
    if_model.fit(X_train)
    s_if_val = -if_model.decision_function(X_val)
    s_if_test = -if_model.decision_function(X_test)
    logger.info(f"  IF fitted+scored ({time.perf_counter() - t:.1f}s)")

    # ───── 2. ECOD ─────
    t = time.perf_counter()
    ecod = ECOD(n_jobs=-1)
    ecod.fit(X_train)
    s_ecod_val = ecod.decision_function(X_val)
    s_ecod_test = ecod.decision_function(X_test)
    logger.info(f"  ECOD fitted+scored ({time.perf_counter() - t:.1f}s)")

    # ───── 3. COPOD ─────
    t = time.perf_counter()
    copod = COPOD(n_jobs=-1)
    copod.fit(X_train)
    s_copod_val = copod.decision_function(X_val)
    s_copod_test = copod.decision_function(X_test)
    logger.info(f"  COPOD fitted+scored ({time.perf_counter() - t:.1f}s)")

    # ───── ENSEMBLE: average ranks ─────
    r_val = (to_ranks(s_if_val) + to_ranks(s_ecod_val) + to_ranks(s_copod_val)) / 3.0
    r_test = (to_ranks(s_if_test) + to_ranks(s_ecod_test) + to_ranks(s_copod_test)) / 3.0
    logger.info(f"  Ensemble = mean of percentile ranks across 3 detectors")

    # Evaluate each detector + ensemble
    results = {
        "if": {"val": evaluate(s_if_val, df_val, "if_val"),
               "test": evaluate(s_if_test, df_test, "if_test")},
        "ecod": {"val": evaluate(s_ecod_val, df_val, "ecod_val"),
                 "test": evaluate(s_ecod_test, df_test, "ecod_test")},
        "copod": {"val": evaluate(s_copod_val, df_val, "copod_val"),
                  "test": evaluate(s_copod_test, df_test, "copod_test")},
        "ensemble": {"val": evaluate(r_val, df_val, "ensemble_val"),
                     "test": evaluate(r_test, df_test, "ensemble_test")},
    }

    logger.info("")
    logger.info("RESULTS:")
    for name in ("if", "ecod", "copod", "ensemble"):
        r = results[name]
        v_unified = r["val"]["unified"]
        t_unified = r["test"]["unified"]
        t_tipoa = r["test"]["tipo_a"]
        logger.info(
            f"  {name:8s} val_unified={v_unified['auc']:.4f}  "
            f"test_unified={t_unified['auc']:.4f}  "
            f"test_tipoA={t_tipoa['auc']:.4f}  "
            f"P@5%={t_unified.get('precision_at_5pct', 0):.3f}  "
            f"EF@5%={t_unified.get('ef_at_5pct', 0):.2f}"
        )

    out = {
        "features_used": FEATURES,
        "n_features": len(FEATURES),
        "ensemble_strategy": "mean_rank_percentile",
        "results": results,
    }
    out_path = OUTPUT_DIR / "results_ensemble.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"\n  Written {out_path}")


if __name__ == "__main__":
    main()
