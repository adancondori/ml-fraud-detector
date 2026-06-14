#!/usr/bin/env python
"""Clean + Honest IF: drop circular features AND redundant features, robust scaling.

What this removes:
  Circular features (defining the proxy):
    - user_reversal_ratio_30d (F18, defines Tipo A)
    - user_reversal_count_30d (F33, defines Tipo A)
    - user_discount_ratio_30d (F19, defines Tipo C)
    - user_txn_count_24h      (F12, defines Tipo D)
  Redundant features (corr=1.0 with each other):
    - amount  (corr=1.00 with amount_usd_ratio; max=$115M dominates scaler)
    - amount_usd_ratio (same as amount up to constant)

Final feature count: 25 features (was 31).
Uses RobustScaler (median/IQR) instead of StandardScaler — robust to long-tail
amounts/ratios.

Evaluates against all 4 proxies (unified, Tipo A, Tipo C, Tipo D). Tipo A is
the only proxy now structurally independent of the model's input features.
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
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings  # noqa: E402
from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.features.engineering import FEATURE_NAMES  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

CIRCULAR_FEATURES = {
    "user_reversal_ratio_30d",
    "user_reversal_count_30d",
    "user_discount_ratio_30d",
    "user_txn_count_24h",
}
REDUNDANT_FEATURES = {
    "amount",
    "amount_usd_ratio",
}
DROP = CIRCULAR_FEATURES | REDUNDANT_FEATURES
FEATURES_CLEAN = [f for f in FEATURE_NAMES if f not in DROP]


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


def evaluate(scores: np.ndarray, df_meta: pd.DataFrame) -> dict:
    out = {}
    for ptype in ("unified", "tipo_a", "tipo_c", "tipo_d"):
        y = DataManager.assign_proxy_labels(df_meta, ptype, settings).to_numpy()
        if y.sum() == 0:
            out[ptype] = {"auc": None, "ap": None}
            continue
        block = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "base_rate": float(y.mean()),
            "n_positives": int(y.sum()),
        }
        if ptype == "unified" or ptype == "tipo_a":
            block.update(percentile_metrics(scores, y))
        out[ptype] = block
    return out


def main() -> None:
    logger.info("=" * 60)
    logger.info(f"CLEAN + HONEST IF — {len(FEATURES_CLEAN)} features, RobustScaler")
    logger.info("=" * 60)
    logger.info(f"  Dropped circular: {sorted(CIRCULAR_FEATURES)}")
    logger.info(f"  Dropped redundant: {sorted(REDUNDANT_FEATURES)}")

    t0 = time.perf_counter()
    df_train = pd.read_parquet(DATA_DIR / "train_features.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val_features.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features.parquet")
    logger.info(f"  Loaded ({time.perf_counter() - t0:.1f}s)")

    X_train = df_train[FEATURES_CLEAN].to_numpy(dtype=np.float64)
    X_val = df_val[FEATURES_CLEAN].to_numpy(dtype=np.float64)
    X_test = df_test[FEATURES_CLEAN].to_numpy(dtype=np.float64)

    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    logger.info(f"  Scaled with RobustScaler (5-95th IQR)")

    t_fit = time.perf_counter()
    model = IsolationForest(
        n_estimators=200,
        max_samples=512,
        max_features=0.6,  # ← from grid search rerun against unified, better than 1.0
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    logger.info(f"  IF fitted ({time.perf_counter() - t_fit:.1f}s)")

    t_score = time.perf_counter()
    s_val = -np.asarray(model.decision_function(X_val), dtype=np.float64)
    s_test = -np.asarray(model.decision_function(X_test), dtype=np.float64)
    logger.info(f"  Scored ({time.perf_counter() - t_score:.1f}s)")

    eval_val = evaluate(s_val, df_val)
    eval_test = evaluate(s_test, df_test)

    logger.info("")
    logger.info("RESULTS (IF-25 clean, RobustScaler):")
    logger.info(f"  VAL  unified: AUC={eval_val['unified']['auc']:.4f} AP={eval_val['unified']['ap']:.4f}")
    logger.info(f"  TEST unified: AUC={eval_test['unified']['auc']:.4f} AP={eval_test['unified']['ap']:.4f}")
    logger.info(f"  TEST Tipo A:  AUC={eval_test['tipo_a']['auc']:.4f} AP={eval_test['tipo_a']['ap']:.4f}  ← HONEST")
    logger.info(f"  TEST Tipo C:  AUC={eval_test['tipo_c']['auc']:.4f}")
    logger.info(f"  TEST Tipo D:  AUC={eval_test['tipo_d']['auc']:.4f}")
    p1 = eval_test['unified'].get('precision_at_1pct', 0)
    e1 = eval_test['unified'].get('ef_at_1pct', 0)
    p5 = eval_test['unified'].get('precision_at_5pct', 0)
    e5 = eval_test['unified'].get('ef_at_5pct', 0)
    logger.info(f"  TEST top 1%:  P={p1:.3f} EF={e1:.2f}x")
    logger.info(f"  TEST top 5%:  P={p5:.3f} EF={e5:.2f}x")

    out = {
        "features_used": FEATURES_CLEAN,
        "features_dropped": sorted(DROP),
        "circular_dropped": sorted(CIRCULAR_FEATURES),
        "redundant_dropped": sorted(REDUNDANT_FEATURES),
        "n_features": len(FEATURES_CLEAN),
        "scaler": "RobustScaler(5-95)",
        "model_params": {"n_estimators": 200, "max_samples": 512, "max_features": 0.6},
        "val": eval_val,
        "test": eval_test,
    }
    out_path = OUTPUT_DIR / "results_clean_honest.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"  Written {out_path}")


if __name__ == "__main__":
    main()
