"""
Fase 3.5+4 — Normalización monetaria (ya integrada) + Feature Engineering.

Genera 31 features en 8 grupos para train/val/test con warm history
para anti-leakage en bordes de split.

Uso:
    python scripts/run_fase4_features.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config.config import settings
from fraud_detector.data.loader import DataManager
from fraud_detector.features.engineering import (
    FEATURE_NAMES,
    FeatureEngineer,
)
from fraud_detector.utils.logger import logger

settings.ensure_directories()
dm = DataManager(settings)

# ── 1. Load raw splits ──────────────────────────────────────────────
logger.info("Loading raw splits...")
warm = dm.load_split("warm")
train = dm.load_split("train")

# ── 2. Fit on train ─────────────────────────────────────────────────
logger.info("Fitting FeatureEngineer on train...")
engineer = FeatureEngineer()
t0 = time.time()
train_features = engineer.fit_transform(train)
fit_time = time.time() - t0
logger.info(f"fit_transform completed in {fit_time:.1f}s")

# Save feature engineer + state
engineer.save(str(settings.models_output_dir / "feature_engineer.joblib"))
state_after_train = engineer.get_feature_state()

# Save train features
path = settings.processed_dir / "train_features.parquet"
train_features.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
logger.info(f"Saved train_features: {train_features.shape} -> {path}")

# ── 3. Transform val with warm history (train as warm) ──────────────
logger.info("Loading val split...")
val = dm.load_split("val")

logger.info("Transforming val with warm history (train)...")
t0 = time.time()
val_features, state_after_val = engineer.transform_with_warm_history(
    val, train, method_state=state_after_train, return_state=True,
)
val_time = time.time() - t0
logger.info(f"val transform completed in {val_time:.1f}s")

path = settings.processed_dir / "val_features.parquet"
val_features.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
logger.info(f"Saved val_features: {val_features.shape} -> {path}")

# Free memory
del val, train
import gc; gc.collect()

# ── 4. Transform test with warm history (val as warm) ───────────────
logger.info("Loading test split + val raw for warm context...")
val_raw = dm.load_split("val")
test = dm.load_split("test")

logger.info("Transforming test with warm history (val)...")
t0 = time.time()
test_features = engineer.transform_with_warm_history(
    test, val_raw, method_state=state_after_val,
)
test_time = time.time() - t0
logger.info(f"test transform completed in {test_time:.1f}s")

path = settings.processed_dir / "test_features.parquet"
test_features.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
logger.info(f"Saved test_features: {test_features.shape} -> {path}")

del val_raw, test
gc.collect()

# ── 5. Feature statistics ───────────────────────────────────────────
logger.info("Computing feature statistics...")
stats = train_features[FEATURE_NAMES].describe().T
stats["null_count"] = train_features[FEATURE_NAMES].isna().sum()
stats.to_csv(settings.project_root / settings.output_dir / "feature_statistics.csv")

# ── 6. Gate B validation ────────────────────────────────────────────
print("\n" + "=" * 70)
print("GATE B — Validez Metodológica (Anti-Leakage)")
print("=" * 70)

checks = []

for name, df_feat in [
    ("train", train_features),
    ("val", val_features),
    ("test", test_features),
]:
    n_features = len([c for c in df_feat.columns if c in FEATURE_NAMES])
    n_nan = int(df_feat[FEATURE_NAMES].isna().sum().sum())
    n_inf = int(np.isinf(df_feat[FEATURE_NAMES].select_dtypes(include=[np.number]).values).sum())
    n_rows = len(df_feat)

    checks.append(("31 features present", n_features == 31))
    checks.append((f"{name}: 0 NaN", n_nan == 0))
    checks.append((f"{name}: 0 Inf", n_inf == 0))

    print(f"\n{name.upper()}:")
    print(f"  Rows: {n_rows:,} | Features: {n_features} | NaN: {n_nan} | Inf: {n_inf}")

# Check first-txn anti-leakage for velocity features
print("\nAnti-leakage spot check (train, first txn per user):")
first_txn = train_features.groupby("user_id").first() if "user_id" in train_features.columns else None
if first_txn is not None:
    for col in ["user_txn_count_1h", "user_txn_count_24h", "user_amount_24h"]:
        if col in first_txn.columns:
            non_zero = (first_txn[col] != 0).sum()
            pct = non_zero / len(first_txn) * 100
            ok = pct < 5  # allow some from warm history leaking
            checks.append((f"First txn {col} mostly 0", ok))
            print(f"  {col}: {non_zero} non-zero ({pct:.1f}%) [{'PASS' if ok else 'WARN'}]")

print("\nCHECKLIST:")
for desc, ok in checks:
    print(f"  [{'✓' if ok else '✗'}] {desc}")

gate_pass = all(ok for _, ok in checks)
print(f"\n{'=' * 70}")
print(f"GATE B: {'PASS' if gate_pass else 'FAIL'}")
print(f"{'=' * 70}")
print(f"\nTiming: train={fit_time:.1f}s, val={val_time:.1f}s, test={test_time:.1f}s")
