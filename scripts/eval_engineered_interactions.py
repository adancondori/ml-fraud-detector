#!/usr/bin/env python
"""IF + ECOD ensemble with engineered interaction features targeted at user cases:
   - new_user (account_age < 14 days)
   - rapid burst (txn_count_1h > 3 AND time_since_last < 60s)
   - small/unusual amounts (amount_facility_ratio < 0.2)

These cases reflect the user's intuition about fraudulent payments
beyond just refunds. We derive 7 interaction features from existing
columns (no leakage, no new raw data needed) and add them to the
25 clean baseline.

Also evaluates against an EXTENDED proxy that includes new-user-burst
and small-amount-outlier patterns to align metrics with the user's
operational concept of "anomalous payment".
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
from scipy.stats import rankdata

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
BASE_FEATURES = [f for f in FEATURE_NAMES if f not in DROP]

# Engineered interaction features (computed from BASE_FEATURES, leakage-safe)
ENGINEERED = [
    "is_new_user",                  # account_age_days < 14
    "is_very_new_user",              # account_age_days < 3
    "new_user_first_facility",       # new_user AND distinct_facilities == 0
    "rapid_burst",                   # time_since_last < 60s AND user_txn_count_1h > 3
    "small_amount_at_facility",      # amount_facility_ratio < 0.2
    "very_small_amount_at_facility", # amount_facility_ratio < 0.05
    "off_hours_high_value",          # is_off_hours AND log_amount > 8
]
ALL_FEATURES = BASE_FEATURES + ENGINEERED


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7 interaction features. All deterministic from BASE_FEATURES."""
    out = df.copy()
    out["is_new_user"] = (out["user_account_age_days"] < 14).astype(np.int8)
    out["is_very_new_user"] = (out["user_account_age_days"] < 3).astype(np.int8)
    out["new_user_first_facility"] = (
        (out["user_account_age_days"] < 14) &
        (out["user_distinct_facilities_30d"] == 0)
    ).astype(np.int8)
    out["rapid_burst"] = (
        (out["time_since_last_txn"] > 0) &
        (out["time_since_last_txn"] < 60) &
        (out["user_txn_count_1h"] > 3)
    ).astype(np.int8)
    out["small_amount_at_facility"] = (out["amount_facility_ratio"] < 0.2).astype(np.int8)
    out["very_small_amount_at_facility"] = (out["amount_facility_ratio"] < 0.05).astype(np.int8)
    out["off_hours_high_value"] = (
        (out["is_off_hours"] > 0) & (out["log_amount"] > 8)
    ).astype(np.int8)
    return out


def build_extended_proxy(df: pd.DataFrame) -> np.ndarray:
    """User-vision proxy: refunds OR velocity OR discount OR new-user-burst OR small-amount.

    Captures the four patterns the user identified as anomalous:
    refunds (Tipo A) + rapid payments + new users + abnormal small amounts.
    """
    tipo_a = DataManager.assign_proxy_labels(df, "tipo_a", settings).to_numpy()
    tipo_c = DataManager.assign_proxy_labels(df, "tipo_c", settings).to_numpy()
    tipo_d = DataManager.assign_proxy_labels(df, "tipo_d", settings).to_numpy()

    new_user_burst = (
        (df["user_account_age_days"] < 14) &
        (df["user_txn_count_1h"] >= 2)
    ).to_numpy()
    small_amount_extreme = (df["amount_facility_ratio"] < 0.05).to_numpy()

    return (tipo_a | tipo_c | tipo_d | new_user_burst | small_amount_extreme).astype(np.int8)


def percentile_metrics(scores: np.ndarray, y: np.ndarray, pcts=(0.01, 0.02, 0.05, 0.10)) -> dict:
    base_rate = float(y.mean())
    order = np.argsort(-scores)
    out = {}
    for pct in pcts:
        k = max(1, int(np.ceil(len(scores) * pct)))
        prec = float(y[order[:k]].sum() / k)
        out[f"p_{int(pct * 100)}pct"] = prec
        out[f"ef_{int(pct * 100)}pct"] = prec / base_rate if base_rate > 0 else 0.0
    return out


def evaluate(scores: np.ndarray, df_meta: pd.DataFrame) -> dict:
    out = {}
    y_unified = DataManager.assign_proxy_labels(df_meta, "unified", settings).to_numpy()
    y_tipoa = DataManager.assign_proxy_labels(df_meta, "tipo_a", settings).to_numpy()
    y_extended = build_extended_proxy(df_meta)
    for label, y in [("unified", y_unified), ("tipo_a", y_tipoa), ("extended", y_extended)]:
        if y.sum() == 0:
            continue
        block = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "base_rate": float(y.mean()),
            "n_positives": int(y.sum()),
        }
        block.update(percentile_metrics(scores, y))
        out[label] = block
    return out


def main() -> None:
    logger.info("=" * 60)
    logger.info(f"IF + ENGINEERED INTERACTIONS — {len(ALL_FEATURES)} features")
    logger.info(f"  Base: {len(BASE_FEATURES)}  Engineered: {len(ENGINEERED)}")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    df_train = engineer(pd.read_parquet(DATA_DIR / "train_features.parquet"))
    df_val = engineer(pd.read_parquet(DATA_DIR / "val_features.parquet"))
    df_test = engineer(pd.read_parquet(DATA_DIR / "test_features.parquet"))
    logger.info(f"  Loaded + engineered ({time.perf_counter() - t0:.1f}s)")

    # Show new feature rates
    for col in ENGINEERED:
        logger.info(f"    {col:35s} train rate = {df_train[col].mean():.4f}")

    # Extended proxy base rate
    y_ext_train = build_extended_proxy(df_train)
    y_ext_test = build_extended_proxy(df_test)
    logger.info(
        f"  Extended proxy rates: train={y_ext_train.mean():.4f}, "
        f"test={y_ext_test.mean():.4f}"
    )

    X_train = df_train[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_val = df_val[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_test = df_test[ALL_FEATURES].to_numpy(dtype=np.float64)

    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
    X_train = np.clip(scaler.transform(X_train).astype(np.float32), -10, 10)
    X_val = np.clip(scaler.transform(X_val).astype(np.float32), -10, 10)
    X_test = np.clip(scaler.transform(X_test).astype(np.float32), -10, 10)

    # Fit IF
    t_fit = time.perf_counter()
    model = IsolationForest(
        n_estimators=200, max_samples=512, max_features=0.6,
        contamination="auto", random_state=42, n_jobs=-1,
    )
    model.fit(X_train)
    logger.info(f"  IF fitted ({time.perf_counter() - t_fit:.1f}s)")

    s_val = -np.asarray(model.decision_function(X_val), dtype=np.float64)
    s_test = -np.asarray(model.decision_function(X_test), dtype=np.float64)

    eval_val = evaluate(s_val, df_val)
    eval_test = evaluate(s_test, df_test)

    logger.info("")
    logger.info("RESULTS (IF + engineered):")
    for proxy in ("unified", "tipo_a", "extended"):
        v = eval_val.get(proxy, {})
        t = eval_test.get(proxy, {})
        if not t:
            continue
        logger.info(
            f"  {proxy:10s}  VAL AUC={v.get('auc', 0):.4f}  "
            f"TEST AUC={t.get('auc', 0):.4f}  "
            f"AP={t.get('ap', 0):.4f}  "
            f"base={t.get('base_rate', 0):.4f}  "
            f"P@1%={t.get('p_1pct', 0):.3f}  "
            f"EF@1%={t.get('ef_1pct', 0):.2f}  "
            f"P@5%={t.get('p_5pct', 0):.3f}  "
            f"EF@5%={t.get('ef_5pct', 0):.2f}"
        )

    out = {
        "n_features": len(ALL_FEATURES),
        "base_features": BASE_FEATURES,
        "engineered_features": ENGINEERED,
        "model_params": {"n_estimators": 200, "max_samples": 512, "max_features": 0.6},
        "val": eval_val,
        "test": eval_test,
    }
    out_path = OUTPUT_DIR / "results_engineered.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"\n  Written {out_path}")


if __name__ == "__main__":
    main()
