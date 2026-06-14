#!/usr/bin/env python
"""Grid search IF maximizing AUC vs unified proxy (not Tipo A only).

Fixes the methodological inconsistency where the original pipeline optimized
hyperparameters against Tipo A in validation but reported results vs unified
proxy on test (see run_fase6_modeling.py:46 vs run_fase7_evaluation.py:45).

Also explores a wider range of max_samples since the original best params
(max_samples=512) likely undersample the 3.1M-row train set.

Run after fix_warm_history_revaluate.py so that val/test features have
proper warm history context.

Outputs:
  output/grid_search_if_unified.csv
  output/results_grid_unified.json  (best params + AUC on val/test)
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
from sklearn.model_selection import ParameterGrid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings  # noqa: E402
from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

SCORES_DIR = PROJECT_ROOT / "output" / "scores"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "output" / "models"

# Param grid — explores larger max_samples (was 512 in original)
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_samples": [512, 2048, 4096],
    "max_features": [0.6, 1.0],
    "contamination": ["auto"],
}


def main() -> None:
    t_total = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading scaled train/val/test (post warm-fix)...")
    X_train = np.load(SCORES_DIR / "X_train.npy")
    X_val = np.load(SCORES_DIR / "X_val.npy")
    X_test = np.load(SCORES_DIR / "X_test.npy")
    logger.info(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    df_val = pd.read_parquet(DATA_DIR / "val_features.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features.parquet")
    y_val_unified = DataManager.assign_proxy_labels(df_val, "unified", settings).to_numpy()
    y_test_unified = DataManager.assign_proxy_labels(df_test, "unified", settings).to_numpy()
    y_test_a = DataManager.assign_proxy_labels(df_test, "tipo_a", settings).to_numpy()
    logger.info(
        f"  val proxy_unified rate = {y_val_unified.mean():.4f}, "
        f"test proxy_unified rate = {y_test_unified.mean():.4f}"
    )

    combos = list(ParameterGrid(PARAM_GRID))
    logger.info(f"Grid: {len(combos)} combinations")

    results = []
    for i, params in enumerate(combos, start=1):
        t0 = time.perf_counter()
        model = IsolationForest(random_state=42, n_jobs=-1, **params)
        model.fit(X_train)
        s_val = -model.decision_function(X_val)
        auc_val = float(roc_auc_score(y_val_unified, s_val))
        ap_val = float(average_precision_score(y_val_unified, s_val))
        elapsed = time.perf_counter() - t0
        results.append({**params, "auc_val_unified": auc_val, "ap_val_unified": ap_val,
                        "time_seconds": round(elapsed, 1)})
        logger.info(
            f"  [{i}/{len(combos)}] n_est={params['n_estimators']} "
            f"max_s={params['max_samples']} max_f={params['max_features']} "
            f"AUC_val={auc_val:.4f} AP={ap_val:.4f} ({elapsed:.1f}s)"
        )

    df_grid = pd.DataFrame(results).sort_values("auc_val_unified", ascending=False)
    df_grid.to_csv(OUTPUT_DIR / "grid_search_if_unified.csv", index=False)
    logger.info(f"\nGrid search results -> grid_search_if_unified.csv")

    # Re-train winner and evaluate on test
    best = df_grid.iloc[0].to_dict()
    best_params = {k: best[k] for k in PARAM_GRID if k in best}
    logger.info(f"\nWinning params (max val AUC unified): {best_params}")

    t_fit = time.perf_counter()
    model = IsolationForest(random_state=42, n_jobs=-1, **best_params)
    model.fit(X_train)
    s_test = -model.decision_function(X_test)
    logger.info(f"Final IF re-fit + scored ({time.perf_counter() - t_fit:.1f}s)")

    auc_test_unified = float(roc_auc_score(y_test_unified, s_test))
    ap_test_unified = float(average_precision_score(y_test_unified, s_test))
    auc_test_a = float(roc_auc_score(y_test_a, s_test))
    ap_test_a = float(average_precision_score(y_test_a, s_test))

    out = {
        "param_grid_explored": PARAM_GRID,
        "n_combos": len(combos),
        "best_params": best_params,
        "val": {"auc_unified": best["auc_val_unified"], "ap_unified": best["ap_val_unified"]},
        "test": {
            "auc_unified": auc_test_unified,
            "ap_unified": ap_test_unified,
            "auc_tipo_a": auc_test_a,
            "ap_tipo_a": ap_test_a,
        },
        "baseline_old_pipeline": {
            "auc_test_unified": 0.6299,
            "ap_test_unified": 0.1692,
            "auc_test_tipo_a": 0.5757,
        },
        "delta_vs_baseline": {
            "auc_test_unified": auc_test_unified - 0.6299,
            "ap_test_unified": ap_test_unified - 0.1692,
            "auc_test_tipo_a": auc_test_a - 0.5757,
        },
    }
    out_path = OUTPUT_DIR / "results_grid_unified.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"\nFinal metrics -> {out_path}")
    logger.info(
        f"\nTEST: AUC_unified={auc_test_unified:.4f} (Δ={out['delta_vs_baseline']['auc_test_unified']:+.4f}) "
        f"AUC_TipoA={auc_test_a:.4f} (Δ={out['delta_vs_baseline']['auc_test_tipo_a']:+.4f})"
    )

    logger.info(f"\nTotal elapsed: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
