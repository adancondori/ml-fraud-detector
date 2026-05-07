#!/usr/bin/env python
"""Fase 5: Preprocesamiento — StandardScaler fit-on-train.

Reads feature parquets from Fase 4, applies StandardScaler fitted on train,
and produces scaled numpy arrays + serialized scaler.

Outputs:
    output/scores/X_train.npy  (float32)
    output/scores/X_val.npy    (float32)
    output/scores/X_test.npy   (float32)
    output/models/scaler.joblib
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detector.features.engineering import FEATURE_NAMES
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
from fraud_detector.utils.logger import logger


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"
    scores_dir = PROJECT_ROOT / "output" / "scores"
    models_dir = PROJECT_ROOT / "output" / "models"
    scores_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Validate prerequisites ---
    splits = {"train": "train_features.parquet", "val": "val_features.parquet", "test": "test_features.parquet"}
    for name, fname in splits.items():
        path = data_dir / fname
        if not path.exists():
            logger.error(f"Missing prerequisite: {path}")
            sys.exit(1)

    # --- Load feature parquets ---
    t0 = time.perf_counter()
    logger.info("Loading feature parquets...")
    df_train = pd.read_parquet(data_dir / "train_features.parquet")
    df_val = pd.read_parquet(data_dir / "val_features.parquet")
    df_test = pd.read_parquet(data_dir / "test_features.parquet")
    logger.info(
        f"Loaded: train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,} "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    # --- Validate features present ---
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        missing = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing:
            logger.error(f"{name} missing features: {missing}")
            sys.exit(1)
        nan_count = df[FEATURE_NAMES].isna().sum().sum()
        if nan_count > 0:
            nan_cols = df[FEATURE_NAMES].columns[df[FEATURE_NAMES].isna().any()].tolist()
            logger.error(f"{name} has {nan_count} NaN values in: {nan_cols}")
            sys.exit(1)

    # --- Fit on train, transform all ---
    logger.info("Fitting StandardScaler on train set...")
    preprocessor = UnsupervisedPreprocessor(variant="full")

    t1 = time.perf_counter()
    X_train = preprocessor.fit_transform(df_train)
    logger.info(f"Train: {X_train.shape}, dtype={X_train.dtype} ({time.perf_counter() - t1:.1f}s)")

    t2 = time.perf_counter()
    X_val = preprocessor.transform(df_val)
    logger.info(f"Val:   {X_val.shape}, dtype={X_val.dtype} ({time.perf_counter() - t2:.1f}s)")

    t3 = time.perf_counter()
    X_test = preprocessor.transform(df_test)
    logger.info(f"Test:  {X_test.shape}, dtype={X_test.dtype} ({time.perf_counter() - t3:.1f}s)")

    # --- Validate outputs ---
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        assert X.dtype == np.float32, f"{name} dtype is {X.dtype}, expected float32"
        assert X.shape[1] == len(FEATURE_NAMES), f"{name} has {X.shape[1]} features, expected {len(FEATURE_NAMES)}"
        assert np.isfinite(X).all(), f"{name} contains NaN or Inf"

    # --- Save ---
    np.save(scores_dir / "X_train.npy", X_train)
    np.save(scores_dir / "X_val.npy", X_val)
    np.save(scores_dir / "X_test.npy", X_test)
    preprocessor.save(str(models_dir / "scaler.joblib"))

    # --- Summary ---
    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Fase 5 completada.")
    logger.info(f"  X_train: {X_train.shape} → {scores_dir / 'X_train.npy'}")
    logger.info(f"  X_val:   {X_val.shape} → {scores_dir / 'X_val.npy'}")
    logger.info(f"  X_test:  {X_test.shape} → {scores_dir / 'X_test.npy'}")
    logger.info(f"  Scaler:  {models_dir / 'scaler.joblib'}")
    logger.info(f"  Tiempo total: {elapsed:.1f}s")

    sizes = {
        "X_train.npy": (scores_dir / "X_train.npy").stat().st_size / 1e6,
        "X_val.npy": (scores_dir / "X_val.npy").stat().st_size / 1e6,
        "X_test.npy": (scores_dir / "X_test.npy").stat().st_size / 1e6,
        "scaler.joblib": (models_dir / "scaler.joblib").stat().st_size / 1e3,
    }
    for name, size in sizes.items():
        unit = "MB" if "npy" in name else "KB"
        logger.info(f"  {name}: {size:.1f} {unit}")


if __name__ == "__main__":
    main()
