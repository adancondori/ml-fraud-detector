#!/usr/bin/env python
"""Honest AUC: IF without features circularly coupled to Tipo A proxy.

Removes both F18 (user_reversal_ratio_30d) and F33 (user_reversal_count_30d) —
both are rolling functions of status ∈ {totally_refunded, refunded_to_credit},
which is exactly the Tipo A proxy. Evaluating IF on these features against
Tipo A is structurally circular.

This script trains a clean IF on the remaining 29 features and reports
AUC vs Tipo A — the closest available approximation to honest performance.

Requires: val/test_features.parquet already regenerated with warm history
(run scripts/fix_warm_history_revaluate.py first).
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
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings  # noqa: E402
from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.features.engineering import FEATURE_NAMES  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

CIRCULAR_FEATURES = {"user_reversal_ratio_30d", "user_reversal_count_30d"}
FEATURES_29 = [f for f in FEATURE_NAMES if f not in CIRCULAR_FEATURES]
assert len(FEATURES_29) == 29, f"Expected 29 features, got {len(FEATURES_29)}"


def evaluate_all_proxies(scores: np.ndarray, df_meta: pd.DataFrame) -> dict:
    out = {}
    for ptype in ("unified", "tipo_a", "tipo_c", "tipo_d"):
        y = DataManager.assign_proxy_labels(df_meta, ptype, settings).to_numpy()
        if y.sum() == 0:
            out[ptype] = {"auc": None, "ap": None}
            continue
        out[ptype] = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "base_rate": float(y.mean()),
            "n_positives": int(y.sum()),
        }
    return out


def main() -> None:
    logger.info("=" * 60)
    logger.info("HONEST AUC — IF without circular features (F18, F33)")
    logger.info("=" * 60)
    logger.info(f"Using {len(FEATURES_29)} features (removed: {CIRCULAR_FEATURES})")

    t0 = time.perf_counter()
    df_train = pd.read_parquet(DATA_DIR / "train_features.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val_features.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features.parquet")
    logger.info(
        f"Loaded: train={len(df_train):,} val={len(df_val):,} test={len(df_test):,} "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    X_train = df_train[FEATURES_29].to_numpy(dtype=np.float64)
    X_val = df_val[FEATURES_29].to_numpy(dtype=np.float64)
    X_test = df_test[FEATURES_29].to_numpy(dtype=np.float64)

    # Scale (fit-on-train only)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    logger.info(f"Scaled: train {X_train.shape}")

    # Train IF (same params as headline model)
    t_fit = time.perf_counter()
    model = IsolationForest(
        n_estimators=200,
        max_samples=512,
        max_features=1.0,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    logger.info(f"IF-29 fitted ({time.perf_counter() - t_fit:.1f}s)")

    # Score
    t_score = time.perf_counter()
    s_val = -np.asarray(model.decision_function(X_val), dtype=np.float64)
    s_test = -np.asarray(model.decision_function(X_test), dtype=np.float64)
    logger.info(f"Scored val+test ({time.perf_counter() - t_score:.1f}s)")

    # Evaluate
    eval_val = evaluate_all_proxies(s_val, df_val)
    eval_test = evaluate_all_proxies(s_test, df_test)

    logger.info("")
    logger.info("RESULTS (IF-29, train on warm-fix features):")
    logger.info(f"  VAL  unified: AUC={eval_val['unified']['auc']:.4f} AP={eval_val['unified']['ap']:.4f}")
    logger.info(f"  TEST unified: AUC={eval_test['unified']['auc']:.4f} AP={eval_test['unified']['ap']:.4f}")
    logger.info(f"  TEST Tipo A:  AUC={eval_test['tipo_a']['auc']:.4f} AP={eval_test['tipo_a']['ap']:.4f}  ← HONEST")
    if eval_test["tipo_c"]["auc"] is not None:
        logger.info(f"  TEST Tipo C:  AUC={eval_test['tipo_c']['auc']:.4f}")
    if eval_test["tipo_d"]["auc"] is not None:
        logger.info(f"  TEST Tipo D:  AUC={eval_test['tipo_d']['auc']:.4f}")

    out = {
        "features_used": FEATURES_29,
        "features_removed": sorted(CIRCULAR_FEATURES),
        "n_features": len(FEATURES_29),
        "val": eval_val,
        "test": eval_test,
    }
    out_path = OUTPUT_DIR / "results_honest_auc.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Written {out_path}")


if __name__ == "__main__":
    main()
