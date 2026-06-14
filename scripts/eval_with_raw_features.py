#!/usr/bin/env python
"""IF on enriched feature set: 25 clean + 7 interactions + raw-derived.

Adds features from raw columns that the original pipeline did NOT use:
  - is_third_party_payment          (effective_user_id != user_id)
  - same_amount_count_1h            (card-testing burst: same exact $ repeated)
  - same_amount_count_24h
  - gateway_change_recent           (gateway differs from user's last txn)
  - capture_delay_seconds           (created_at → captured_at delay)
  - is_main_gateway                 (transaction on user's most common gateway)
  - is_first_gateway_for_user       (first time user uses this gateway in data)
  - source_change_recent            (source_enum differs from user's last txn)

Removes all circular features (F12, F18, F19, F33) and redundant (amount, amount_usd_ratio).
"""
from __future__ import annotations

import gc
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
SCORES_DIR = OUTPUT_DIR / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

DROP = {
    "user_reversal_ratio_30d", "user_reversal_count_30d",
    "user_discount_ratio_30d", "user_txn_count_24h",
    "amount", "amount_usd_ratio",
}
BASE = [f for f in FEATURE_NAMES if f not in DROP]

INTERACTIONS = [
    "is_new_user", "is_very_new_user", "new_user_first_facility",
    "rapid_burst", "small_amount_at_facility",
    "very_small_amount_at_facility", "off_hours_high_value",
]

RAW_DERIVED = [
    "is_third_party_payment",
    "same_amount_count_1h",
    "same_amount_count_24h",
    "gateway_change_recent",
    "capture_delay_seconds",
    "is_main_gateway",
    "is_first_gateway_for_user",
    "source_change_recent",
]

ALL_FEATURES = BASE + INTERACTIONS + RAW_DERIVED


def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_new_user"] = (df["user_account_age_days"] < 14).astype(np.int8)
    df["is_very_new_user"] = (df["user_account_age_days"] < 3).astype(np.int8)
    df["new_user_first_facility"] = (
        (df["user_account_age_days"] < 14) &
        (df["user_distinct_facilities_30d"] == 0)
    ).astype(np.int8)
    df["rapid_burst"] = (
        (df["time_since_last_txn"] > 0) &
        (df["time_since_last_txn"] < 60) &
        (df["user_txn_count_1h"] > 3)
    ).astype(np.int8)
    df["small_amount_at_facility"] = (df["amount_facility_ratio"] < 0.2).astype(np.int8)
    df["very_small_amount_at_facility"] = (df["amount_facility_ratio"] < 0.05).astype(np.int8)
    df["off_hours_high_value"] = (
        (df["is_off_hours"] > 0) & (df["log_amount"] > 8)
    ).astype(np.int8)
    return df


def per_user_same_amount_count(df: pd.DataFrame, hours: int) -> np.ndarray:
    """For each row, count of prior rows from same user with same exact amount
    within `hours`. Vectorized per user×amount group."""
    out = np.zeros(len(df), dtype=np.int32)
    sec = hours * 3600
    times = df["created_at"].view("int64").to_numpy() // 1_000_000_000  # to epoch seconds
    for (uid, amt), group in df.groupby(["user_id", "amount_rounded"], sort=False):
        if len(group) <= 1:
            continue
        idx = group.index.to_numpy()
        t = times[idx]
        # for each i, count j < i with t[j] >= t[i] - sec
        j = 0
        for i in range(len(idx)):
            cutoff = t[i] - sec
            while j < i and t[j] < cutoff:
                j += 1
            out[idx[i]] = i - j
        # reset j tracking — actually since group is sorted, j tracking accumulates correctly
        # but we need to reset between groups: handled by loop scope
    return out


def build_raw_derived(
    df_features: pd.DataFrame,
    df_raw: pd.DataFrame,
    df_warm_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Compute raw-derived features. Uses df_warm_raw for history context.
    Joins on `id` to preserve features-frame row order.
    """
    df_features = df_features.copy()
    raw = pd.concat([df_warm_raw, df_raw], ignore_index=True)
    raw["created_at"] = pd.to_datetime(raw["created_at"])
    raw["captured_at"] = pd.to_datetime(raw["captured_at"], errors="coerce")
    raw["gateway"] = raw["gateway"].fillna("unknown").astype(str)
    raw["source_enum"] = raw["source_enum"].fillna("unknown").astype(str)
    raw["amount_rounded"] = raw["amount"].round(2)

    raw = raw.sort_values(["user_id", "created_at", "id"]).reset_index(drop=True)

    # 1: third party payment
    raw["is_third_party_payment"] = (
        raw["effective_user_id"] != raw["user_id"]
    ).astype(np.int8)

    # 2, 3: same amount burst
    logger.info("    Computing same-amount bursts (1h, 24h)...")
    t0 = time.perf_counter()
    raw["same_amount_count_1h"] = per_user_same_amount_count(raw, 1).astype(np.float32)
    raw["same_amount_count_24h"] = per_user_same_amount_count(raw, 24).astype(np.float32)
    logger.info(f"      done ({time.perf_counter() - t0:.1f}s)")

    # 4: gateway change vs previous txn (per user)
    prev_gw = raw.groupby("user_id")["gateway"].shift(1)
    raw["gateway_change_recent"] = (
        prev_gw.notna() & (prev_gw != raw["gateway"])
    ).astype(np.int8)

    # 5: capture delay (seconds)
    raw["capture_delay_seconds"] = (
        (raw["captured_at"] - raw["created_at"]).dt.total_seconds()
        .fillna(0).clip(lower=-86400, upper=86400).astype(np.float32)
    )

    # 6: is on user's most common gateway (computed across combined frame)
    main_gw = raw.groupby("user_id")["gateway"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"
    )
    raw["_main_gw"] = raw["user_id"].map(main_gw)
    raw["is_main_gateway"] = (raw["gateway"] == raw["_main_gw"]).astype(np.int8)

    # 7: is first gateway use for user (cumulative distinct check)
    raw["_gateway_seen_count"] = raw.groupby(["user_id", "gateway"]).cumcount()
    raw["is_first_gateway_for_user"] = (raw["_gateway_seen_count"] == 0).astype(np.int8)

    # 8: source change vs previous txn
    prev_src = raw.groupby("user_id")["source_enum"].shift(1)
    raw["source_change_recent"] = (
        prev_src.notna() & (prev_src != raw["source_enum"])
    ).astype(np.int8)

    warm_ids = set(df_warm_raw["id"].tolist())
    keep = raw[~raw["id"].isin(warm_ids)].copy()

    new_cols = ["id"] + RAW_DERIVED
    merged = df_features.merge(keep[new_cols], on="id", how="left")
    for c in RAW_DERIVED:
        merged[c] = merged[c].fillna(0).astype(np.float32)
    return merged


def build_extended_proxy(df: pd.DataFrame) -> np.ndarray:
    tipo_a = DataManager.assign_proxy_labels(df, "tipo_a", settings).to_numpy()
    tipo_c = DataManager.assign_proxy_labels(df, "tipo_c", settings).to_numpy()
    tipo_d = DataManager.assign_proxy_labels(df, "tipo_d", settings).to_numpy()
    new_user_burst = (
        (df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 2)
    ).to_numpy()
    small_amount_extreme = (df["amount_facility_ratio"] < 0.05).to_numpy()
    return (tipo_a | tipo_c | tipo_d | new_user_burst | small_amount_extreme).astype(np.int8)


def build_proxy_anomalias_operativas(df: pd.DataFrame) -> np.ndarray:
    """Proxy operativo estricto de anomalía: card-testing + new-user-burst + third-party-burst."""
    cols = df.columns
    card_test = (df["same_amount_count_1h"] >= 3).to_numpy() if "same_amount_count_1h" in cols else np.zeros(len(df), dtype=bool)
    new_burst = (
        (df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)
    ).to_numpy()
    third_party_burst = (
        (df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)
    ).to_numpy() if "is_third_party_payment" in cols else np.zeros(len(df), dtype=bool)
    return (card_test | new_burst | third_party_burst).astype(np.int8)


# Alias deprecado (PLAN_REFACTOR_TESIS sec. 9): nombre histórico para compatibilidad
# con scripts existentes. Usar `build_proxy_anomalias_operativas` en código nuevo.
build_pure_fraud_proxy = build_proxy_anomalias_operativas


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
        "pure_fraud": build_pure_fraud_proxy(df_meta),
    }
    for label, y in proxies.items():
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


def main():
    logger.info("=" * 60)
    logger.info(f"IF + clean + interactions + raw-derived — {len(ALL_FEATURES)} features")
    logger.info(f"  base={len(BASE)} interactions={len(INTERACTIONS)} raw={len(RAW_DERIVED)}")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    df_train_feat = build_interactions(pd.read_parquet(DATA_DIR / "train_features.parquet"))
    df_val_feat = build_interactions(pd.read_parquet(DATA_DIR / "val_features.parquet"))
    df_test_feat = build_interactions(pd.read_parquet(DATA_DIR / "test_features.parquet"))

    df_warm_raw = pd.read_parquet(DATA_DIR / "warm_raw.parquet")
    df_train_raw = pd.read_parquet(DATA_DIR / "train_raw.parquet")
    df_val_raw = pd.read_parquet(DATA_DIR / "val_raw.parquet")
    df_test_raw = pd.read_parquet(DATA_DIR / "test_raw.parquet")
    logger.info(f"  Loaded features + raw ({time.perf_counter() - t0:.1f}s)")

    logger.info("  Computing raw-derived for TRAIN...")
    t = time.perf_counter()
    df_train_feat = build_raw_derived(df_train_feat, df_train_raw, df_warm_raw)
    logger.info(f"    train done ({time.perf_counter() - t:.1f}s)")

    cutoff = df_train_raw["created_at"].max() - pd.Timedelta(days=35)
    warm_for_val = df_train_raw[df_train_raw["created_at"] >= cutoff]
    logger.info("  Computing raw-derived for VAL...")
    t = time.perf_counter()
    df_val_feat = build_raw_derived(df_val_feat, df_val_raw, warm_for_val)
    logger.info(f"    val done ({time.perf_counter() - t:.1f}s)")

    cutoff = df_val_raw["created_at"].max() - pd.Timedelta(days=35)
    warm_for_test = df_val_raw[df_val_raw["created_at"] >= cutoff]
    logger.info("  Computing raw-derived for TEST...")
    t = time.perf_counter()
    df_test_feat = build_raw_derived(df_test_feat, df_test_raw, warm_for_test)
    logger.info(f"    test done ({time.perf_counter() - t:.1f}s)")

    del df_train_raw, df_val_raw, df_test_raw, df_warm_raw, warm_for_val, warm_for_test
    gc.collect()

    logger.info("")
    logger.info("  Train rates for new features:")
    for c in RAW_DERIVED:
        s = df_train_feat[c]
        if set(s.unique()).issubset({0, 1}):
            logger.info(f"    {c:35s} rate = {float(s.mean()):.4f}")
        else:
            logger.info(f"    {c:35s} mean={float(s.mean()):.2f} p99={float(s.quantile(0.99)):.2f}")

    y_pure_test = build_pure_fraud_proxy(df_test_feat)
    y_ext_test = build_extended_proxy(df_test_feat)
    logger.info(
        f"  Proxy rates (test): "
        f"unified={DataManager.assign_proxy_labels(df_test_feat, 'unified', settings).mean():.4f}  "
        f"tipo_a={DataManager.assign_proxy_labels(df_test_feat, 'tipo_a', settings).mean():.4f}  "
        f"extended={y_ext_test.mean():.4f}  anomalias_operativas={y_pure_test.mean():.4f}"
    )

    X_train = df_train_feat[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_val = df_val_feat[ALL_FEATURES].to_numpy(dtype=np.float64)
    X_test = df_test_feat[ALL_FEATURES].to_numpy(dtype=np.float64)

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

    np.save(SCORES_DIR / "X_train_final.npy", X_train)
    np.save(SCORES_DIR / "X_val_final.npy", X_val)
    np.save(SCORES_DIR / "X_test_final.npy", X_test)
    np.save(SCORES_DIR / "if_test_scores_final.npy", s_test)
    np.save(SCORES_DIR / "if_val_scores_final.npy", s_val)
    df_val_feat.to_parquet(DATA_DIR / "val_features_enriched.parquet", index=False)
    df_test_feat.to_parquet(DATA_DIR / "test_features_enriched.parquet", index=False)
    df_train_feat.to_parquet(DATA_DIR / "train_features_enriched.parquet", index=False)
    logger.info("  Saved enriched feature parquets + arrays for HE4")

    eval_val = evaluate(s_val, df_val_feat)
    eval_test = evaluate(s_test, df_test_feat)

    logger.info("")
    logger.info(f"RESULTS (IF on {len(ALL_FEATURES)} features):")
    for proxy in ("unified", "tipo_a", "extended", "pure_fraud"):
        v = eval_val.get(proxy, {})
        t = eval_test.get(proxy, {})
        if not t:
            continue
        logger.info(
            f"  {proxy:12s} VAL AUC={v.get('auc', 0):.4f}  TEST AUC={t['auc']:.4f}  AP={t['ap']:.4f}  "
            f"base={t['base_rate']:.4f}  P@1%={t['p_1pct']:.3f}  EF@1%={t['ef_1pct']:.2f}  "
            f"P@5%={t['p_5pct']:.3f}  EF@5%={t['ef_5pct']:.2f}"
        )

    out = {
        "n_features": len(ALL_FEATURES),
        "base_features": BASE,
        "interaction_features": INTERACTIONS,
        "raw_derived_features": RAW_DERIVED,
        "circular_features_dropped": ["F12", "F18", "F19", "F33"],
        "redundant_features_dropped": ["amount", "amount_usd_ratio"],
        "model_params": {"n_estimators": 200, "max_samples": 512, "max_features": 0.6},
        "val": eval_val,
        "test": eval_test,
    }
    out_path = OUTPUT_DIR / "results_final.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"\n  Written {out_path}")


if __name__ == "__main__":
    main()
