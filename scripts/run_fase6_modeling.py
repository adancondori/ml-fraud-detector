#!/usr/bin/env python
"""Fase 6: Modelado y Tuning — IF + LOF + OC-SVM.

Grid search on validation set only (test set untouched).
Trains 5 final models: IF-31, IF-30, IF-21, LOF, OC-SVM.
Multi-seed variability check for IF-31.

Usage:
    python scripts/run_fase6_modeling.py              # Full run
    python scripts/run_fase6_modeling.py --step if    # Only IF grid search
    python scripts/run_fase6_modeling.py --step lof   # Only LOF grid search
    python scripts/run_fase6_modeling.py --step ocsvm # Only OC-SVM grid search
    python scripts/run_fase6_modeling.py --step train # Only final training
    python scripts/run_fase6_modeling.py --step seed  # Only multi-seed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings
from fraud_detector.data.loader import DataManager
from fraud_detector.features.engineering import FEATURE_NAMES, FEATURE_NAMES_21, FEATURE_NAMES_30
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
from fraud_detector.models.trainer import ModelTrainer
from fraud_detector.utils.logger import logger


def load_data():
    """Load scaled arrays and proxy labels."""
    scores_dir = PROJECT_ROOT / "output" / "scores"
    data_dir = PROJECT_ROOT / "data" / "processed"

    X_train = np.load(scores_dir / "X_train.npy")
    X_val = np.load(scores_dir / "X_val.npy")

    # Grid search optimizes against the SAME proxy that the final HE2 evaluation
    # uses (unified = OR(A,B,C,D,E)). Using strict_proxy_list (= Tipo A only) here
    # broke train/validate/test consistency: hyperparameters chosen for Tipo A
    # were then reported on unified — see run_fase7_evaluation.py:45.
    df_val = pd.read_parquet(data_dir / "val_features.parquet")
    y_val_proxy = DataManager.assign_proxy_labels(df_val, "unified", settings).to_numpy()

    logger.info(
        f"Loaded: X_train={X_train.shape}, X_val={X_val.shape}, "
        f"proxy_unified_positive={y_val_proxy.sum():,} ({y_val_proxy.mean() * 100:.2f}%)"
    )
    return X_train, X_val, y_val_proxy


def get_feature_indices(feature_list, full_list=FEATURE_NAMES):
    """Get column indices for a feature subset."""
    return [full_list.index(f) for f in feature_list]


def step_grid_search_if(X_train, X_val, y_val_proxy):
    """Grid search IF-31: 240 combinations."""
    output_dir = PROJECT_ROOT / "output"
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    param_grid = {
        "n_estimators": settings.if_n_estimators_list,
        "max_samples": settings.if_max_samples_list,
        "max_features": settings.if_max_features_list,
        "contamination": settings.if_contamination_list,
    }
    total = 1
    for v in param_grid.values():
        total *= len(v)
    logger.info(f"IF grid search: {total} combinations")

    df = ModelTrainer.grid_search_if(
        X_train, X_val, y_val_proxy, param_grid,
        checkpoint_path=str(output_dir / "grid_search_if.csv"),
        checkpoint_every=20,
        random_state=settings.random_seed,
    )

    best_idx = df["auc_roc"].idxmax()
    best = df.iloc[best_idx]
    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_samples": int(best["max_samples"]),
        "max_features": float(best["max_features"]),
        "contamination": float(best["contamination"]),
    }
    logger.info(f"Best IF params: {best_params} → AUC={best['auc_roc']:.6f}")

    with open(models_dir / "best_params_if.json", "w") as f:
        json.dump({"best_params": best_params, "best_auc_roc": float(best["auc_roc"])}, f, indent=2)

    return best_params


def step_grid_search_lof(X_train, X_val, y_val_proxy):
    """Grid search LOF: 3 combinations."""
    output_dir = PROJECT_ROOT / "output"
    models_dir = output_dir / "models"

    df = ModelTrainer.grid_search_lof(
        X_train, X_val, y_val_proxy,
        n_neighbors_list=settings.lof_n_neighbors_list,
    )
    df.to_csv(output_dir / "grid_search_lof.csv", index=False)

    best_idx = df["auc_roc"].idxmax()
    best = df.iloc[best_idx]
    best_params = {"n_neighbors": int(best["n_neighbors"])}
    logger.info(f"Best LOF params: {best_params} → AUC={best['auc_roc']:.6f}")

    with open(models_dir / "best_params_lof.json", "w") as f:
        json.dump({"best_params": best_params, "best_auc_roc": float(best["auc_roc"])}, f, indent=2)

    return best_params


def step_grid_search_ocsvm(X_train, X_val, y_val_proxy):
    """Grid search OC-SVM: 6 combinations with 100K subsample."""
    output_dir = PROJECT_ROOT / "output"
    models_dir = output_dir / "models"

    param_grid = {
        "nu": settings.ocsvm_nu_list,
        "gamma": settings.ocsvm_gamma_list,
    }

    df = ModelTrainer.grid_search_ocsvm(
        X_train, X_val, y_val_proxy, param_grid,
        n_subsample=settings.ocsvm_subsample,
    )
    df.to_csv(output_dir / "grid_search_ocsvm.csv", index=False)

    best_idx = df["auc_roc"].idxmax()
    best = df.iloc[best_idx]
    best_params = {"nu": float(best["nu"]), "gamma": best["gamma"]}
    logger.info(f"Best OC-SVM params: {best_params} → AUC={best['auc_roc']:.6f}")

    with open(models_dir / "best_params_ocsvm.json", "w") as f:
        json.dump({"best_params": best_params, "best_auc_roc": float(best["auc_roc"])}, f, indent=2)

    return best_params


def step_train_final_models(X_train, X_val, y_val_proxy):
    """Train 5 final models with best params."""
    models_dir = PROJECT_ROOT / "output" / "models"
    data_dir = PROJECT_ROOT / "data" / "processed"

    # Load best params
    with open(models_dir / "best_params_if.json") as f:
        if_params = json.load(f)["best_params"]
    with open(models_dir / "best_params_lof.json") as f:
        lof_params = json.load(f)["best_params"]
    with open(models_dir / "best_params_ocsvm.json") as f:
        ocsvm_params = json.load(f)["best_params"]

    # -- IF-31 (full 31 features) --
    logger.info("Training IF-31 (primary)...")
    t0 = time.perf_counter()
    trainer_if = ModelTrainer(model_type="isolation_forest", model_params={
        "n_estimators": if_params["n_estimators"],
        "max_samples": if_params["max_samples"],
        "max_features": if_params["max_features"],
    })
    trainer_if.fit(X_train)
    trainer_if.save_model(str(models_dir / "isolation_forest.joblib"))
    logger.info(f"IF-31 trained in {time.perf_counter() - t0:.1f}s")

    # -- IF-30 (without F18: user_reversal_ratio_30d) --
    logger.info("Training IF-30 (sensitivity, no F18)...")
    idx_30 = get_feature_indices(FEATURE_NAMES_30)
    t0 = time.perf_counter()
    trainer_if30 = ModelTrainer(model_type="isolation_forest", model_params={
        "n_estimators": if_params["n_estimators"],
        "max_samples": if_params["max_samples"],
        "max_features": if_params["max_features"],
    })
    trainer_if30.fit(X_train[:, idx_30])
    trainer_if30.save_model(str(models_dir / "isolation_forest_30.joblib"))
    logger.info(f"IF-30 trained in {time.perf_counter() - t0:.1f}s")

    # -- IF-21 (ablation: only groups A-E) --
    logger.info("Training IF-21 (ablation, 21 base features)...")
    idx_21 = get_feature_indices(FEATURE_NAMES_21)
    t0 = time.perf_counter()
    trainer_if21 = ModelTrainer(model_type="isolation_forest", model_params={
        "n_estimators": if_params["n_estimators"],
        "max_samples": if_params["max_samples"],
        "max_features": if_params["max_features"],
    })
    trainer_if21.fit(X_train[:, idx_21])
    trainer_if21.save_model(str(models_dir / "isolation_forest_21.joblib"))
    logger.info(f"IF-21 trained in {time.perf_counter() - t0:.1f}s")

    # -- LOF --
    logger.info("Training LOF (final)...")
    t0 = time.perf_counter()
    trainer_lof = ModelTrainer(model_type="lof", model_params=lof_params)
    trainer_lof.fit(X_train)
    trainer_lof.save_model(str(models_dir / "lof.joblib"))
    logger.info(f"LOF trained in {time.perf_counter() - t0:.1f}s")

    # -- OC-SVM (on subsample) --
    logger.info("Training OC-SVM (final, 100K subsample)...")
    X_sub = ModelTrainer._subsample_temporal(X_train, n=settings.ocsvm_subsample)
    t0 = time.perf_counter()
    trainer_ocsvm = ModelTrainer(model_type="ocsvm", model_params=ocsvm_params)
    trainer_ocsvm.fit(X_sub)
    trainer_ocsvm.save_model(str(models_dir / "ocsvm.joblib"))
    logger.info(f"OC-SVM trained in {time.perf_counter() - t0:.1f}s")


def step_multi_seed(X_train, X_val, y_val_proxy):
    """Multi-seed variability for IF-31."""
    models_dir = PROJECT_ROOT / "output" / "models"
    output_dir = PROJECT_ROOT / "output"

    with open(models_dir / "best_params_if.json") as f:
        best_params = json.load(f)["best_params"]

    df = ModelTrainer.run_multi_seed(
        X_train, X_val, y_val_proxy, best_params,
        seeds=settings.multi_seeds_list,
    )
    df.to_csv(output_dir / "multi_seed_results.csv", index=False)
    logger.info(f"Multi-seed results saved ({len(df)} seeds)")


def main():
    parser = argparse.ArgumentParser(description="Fase 6: Modelado y Tuning")
    parser.add_argument("--step", choices=["if", "lof", "ocsvm", "train", "seed"],
                        help="Run only a specific step")
    args = parser.parse_args()

    t_total = time.perf_counter()
    X_train, X_val, y_val_proxy = load_data()

    if args.step is None or args.step == "if":
        step_grid_search_if(X_train, X_val, y_val_proxy)

    if args.step is None or args.step == "lof":
        step_grid_search_lof(X_train, X_val, y_val_proxy)

    if args.step is None or args.step == "ocsvm":
        step_grid_search_ocsvm(X_train, X_val, y_val_proxy)

    if args.step is None or args.step == "train":
        step_train_final_models(X_train, X_val, y_val_proxy)

    if args.step is None or args.step == "seed":
        step_multi_seed(X_train, X_val, y_val_proxy)

    elapsed = time.perf_counter() - t_total
    logger.info(f"Fase 6 total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
