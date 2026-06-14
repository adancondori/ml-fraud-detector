#!/usr/bin/env python
"""IF final: 40 features (clean + interactions + raw-derived) + 5 token features.

Token features from user_tokens MySQL table:
  - has_token (1 if payment has linked token)
  - token_age_days_at_payment
  - is_new_token (< 7d)
  - is_very_new_token (< 1d)
  - is_default_token
  - n_tokens_user_at_payment

Adds new proxy `extended_v2` that includes is_new_token + small_amount pattern
(brand new card used for small amount → classical card testing).
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
from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCORES_DIR = OUTPUT_DIR / "scores"

TOKEN_FEATURES = [
    "has_token",
    "token_age_days_at_payment",
    "is_new_token",
    "is_very_new_token",
    "is_default_token",
    "n_tokens_user_at_payment",
]


def merge_tokens(df_features: pd.DataFrame, df_tokens: pd.DataFrame) -> pd.DataFrame:
    """Add token features by joining on payment id."""
    merged = df_features.merge(
        df_tokens.rename(columns={"payment_id": "id"})[["id"] + TOKEN_FEATURES],
        on="id", how="left",
    )
    for c in TOKEN_FEATURES:
        if c in ("token_age_days_at_payment",):
            merged[c] = merged[c].fillna(-1).astype(np.float32)
        elif c == "n_tokens_user_at_payment":
            merged[c] = merged[c].fillna(0).astype(np.float32)
        else:
            merged[c] = merged[c].fillna(0).astype(np.int8)
    return merged


def build_extended_proxy(df):
    tipo_a = DataManager.assign_proxy_labels(df, "tipo_a", settings).to_numpy()
    tipo_c = DataManager.assign_proxy_labels(df, "tipo_c", settings).to_numpy()
    tipo_d = DataManager.assign_proxy_labels(df, "tipo_d", settings).to_numpy()
    new_user_burst = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    small_amount = (df["amount_facility_ratio"] < 0.05).to_numpy()
    return (tipo_a | tipo_c | tipo_d | new_user_burst | small_amount).astype(np.int8)


def build_proxy_anomalias_operativas(df):
    cols = df.columns
    card_test = (df["same_amount_count_1h"] >= 3).to_numpy() if "same_amount_count_1h" in cols else np.zeros(len(df), dtype=bool)
    new_burst = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third_party_burst = (
        (df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)
    ).to_numpy() if "is_third_party_payment" in cols else np.zeros(len(df), dtype=bool)
    return (card_test | new_burst | third_party_burst).astype(np.int8)


# Alias deprecado (PLAN_REFACTOR_TESIS sec. 9): nombre histórico para compatibilidad
# con scripts existentes. Usar `build_proxy_anomalias_operativas` en código nuevo.
build_pure_fraud_proxy = build_proxy_anomalias_operativas


def build_extended_v2_proxy(df):
    """Adds card-token-testing pattern to extended proxy."""
    base = build_extended_proxy(df)
    if "is_new_token" in df.columns:
        new_token_small = (
            (df["is_new_token"] == 1) & (df["amount_facility_ratio"] < 0.5)
        ).to_numpy()
    else:
        new_token_small = np.zeros(len(df), dtype=bool)
    return (base | new_token_small).astype(np.int8)


def percentile_metrics(scores, y, pcts=(0.01, 0.02, 0.05, 0.10)):
    base = float(y.mean())
    order = np.argsort(-scores)
    out = {}
    for pct in pcts:
        k = max(1, int(np.ceil(len(scores) * pct)))
        prec = float(y[order[:k]].sum() / k)
        out[f"p_{int(pct * 100)}pct"] = prec
        out[f"ef_{int(pct * 100)}pct"] = prec / base if base > 0 else 0.0
    return out


def evaluate(scores, df_meta):
    out = {}
    proxies = {
        "unified": DataManager.assign_proxy_labels(df_meta, "unified", settings).to_numpy(),
        "tipo_a": DataManager.assign_proxy_labels(df_meta, "tipo_a", settings).to_numpy(),
        "extended": build_extended_proxy(df_meta),
        "extended_v2": build_extended_v2_proxy(df_meta),
        "pure_fraud": build_pure_fraud_proxy(df_meta),
    }
    for label, y in proxies.items():
        if y.sum() == 0:
            continue
        out[label] = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "base_rate": float(y.mean()),
            "n_positives": int(y.sum()),
            **percentile_metrics(scores, y),
        }
    return out


def main():
    logger.info("=" * 60)
    logger.info("IF final + token features")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    df_train = pd.read_parquet(DATA_DIR / "train_features_enriched.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val_features_enriched.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_features_enriched.parquet")
    df_tok = pd.read_parquet(DATA_DIR / "payment_token_features.parquet")
    logger.info(f"  Loaded ({time.perf_counter() - t0:.1f}s)")
    logger.info(f"  Token features parquet: {df_tok.shape}")

    df_train = merge_tokens(df_train, df_tok)
    df_val = merge_tokens(df_val, df_tok)
    df_test = merge_tokens(df_test, df_tok)
    logger.info(
        f"  Merged. Token coverage — train: {df_train['has_token'].mean():.4f}, "
        f"val: {df_val['has_token'].mean():.4f}, test: {df_test['has_token'].mean():.4f}"
    )
    logger.info(f"  is_new_token rates — train: {df_train['is_new_token'].mean():.4f}, "
                f"test: {df_test['is_new_token'].mean():.4f}")

    # All features
    from fraud_detector.features.engineering import FEATURE_NAMES
    DROP = {"user_reversal_ratio_30d", "user_reversal_count_30d",
            "user_discount_ratio_30d", "user_txn_count_24h",
            "amount", "amount_usd_ratio"}
    BASE = [f for f in FEATURE_NAMES if f not in DROP]
    INTERACTIONS = ["is_new_user", "is_very_new_user", "new_user_first_facility",
                    "rapid_burst", "small_amount_at_facility",
                    "very_small_amount_at_facility", "off_hours_high_value"]
    RAW_DERIVED = ["is_third_party_payment", "same_amount_count_1h", "same_amount_count_24h",
                   "gateway_change_recent", "capture_delay_seconds", "is_main_gateway",
                   "is_first_gateway_for_user", "source_change_recent"]
    ALL_FEATURES = BASE + INTERACTIONS + RAW_DERIVED + TOKEN_FEATURES

    logger.info(f"  Total features: {len(ALL_FEATURES)} "
                f"(base={len(BASE)} interactions={len(INTERACTIONS)} raw={len(RAW_DERIVED)} tokens={len(TOKEN_FEATURES)})")

    X_train = df_train[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_val = df_val[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_test = df_test[ALL_FEATURES].to_numpy(dtype=np.float64)

    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
    X_train = np.clip(scaler.transform(X_train).astype(np.float32), -10, 10)
    X_val = np.clip(scaler.transform(X_val).astype(np.float32), -10, 10)
    X_test = np.clip(scaler.transform(X_test).astype(np.float32), -10, 10)

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
    logger.info(f"RESULTS (IF on {len(ALL_FEATURES)} features, with tokens):")
    for proxy in ("unified", "tipo_a", "extended", "extended_v2", "pure_fraud"):
        v = eval_val.get(proxy, {})
        t = eval_test.get(proxy, {})
        if not t:
            continue
        logger.info(
            f"  {proxy:13s} VAL AUC={v.get('auc', 0):.4f}  TEST AUC={t['auc']:.4f}  "
            f"AP={t['ap']:.4f}  base={t['base_rate']:.4f}  "
            f"P@1%={t['p_1pct']:.3f}  EF@1%={t['ef_1pct']:.2f}  "
            f"P@5%={t['p_5pct']:.3f}  EF@5%={t['ef_5pct']:.2f}"
        )

    out = {
        "n_features": len(ALL_FEATURES),
        "features": ALL_FEATURES,
        "token_features": TOKEN_FEATURES,
        "model_params": {"n_estimators": 200, "max_samples": 512, "max_features": 0.6},
        "val": eval_val,
        "test": eval_test,
    }
    out_path = OUTPUT_DIR / "results_with_tokens.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"\n  Written {out_path}")


if __name__ == "__main__":
    main()
