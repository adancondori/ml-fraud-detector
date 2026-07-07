"""Diagnostico H-A: misma construccion de features, medida en VAL vs TEST.

Reproduce disjoint30_frames de exp_frames_improvement.py (fuente del 2.14x)
PERO mide el top-5% amount ratio en:
  - VAL   completo (Jul-Ago)   -> lo que reporta produccion
  - TEST  completo (Sep-Dic)   -> lo que reporto el experimento
  - VAL   muestra 1/20
  - TEST  muestra 1/20         -> configuracion EXACTA del experimento

Un solo modelo (train completo, features de marco, IF seed 42) evaluado en ambos.
Aisla si la ventana explica 7x (val) vs 2x (test).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

ROOT = Path("/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector")
DATA = ROOT / "data" / "processed"
SCRATCH = ROOT / "output" / "revision"
sys.path.insert(0, str(ROOT / "src"))
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402

OFF_HOURS = list({23, 0, 1, 2, 3, 4, 5, 6})
NEEDED = [
    "facility_id", "amount", "created_at", "currency",
    "discount_ratio", "has_tip", "time_since_last_txn",
    "user_amount_24h", "user_distinct_facilities_30d", "user_distinct_methods",
    "user_debit_count_30d", "user_debit_amount_30d", "credit_flow_ratio",
    "is_club_credit", "paid_by_manager", "user_role",
    "category_entropy_30d", "user_merchandise_ratio_30d",
    "gateway_change_recent", "is_main_gateway",
    "is_first_gateway_for_user", "source_change_recent", "is_off_hours", "id",
]


def own_stats():
    tr = pd.read_parquet(DATA / "train_features_enriched.parquet", columns=["amount", "facility_id"])
    g = tr.groupby("facility_id")["amount"]
    mean = g.mean().to_dict()
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    iqr_g = iqr.fillna(float(iqr.median())).to_dict()
    med = g.median().to_dict()
    gmean = float(tr["amount"].mean())
    return mean, med, iqr_g, gmean


def tz_lookup():
    stats = json.load(open(ROOT / "output" / "models" / "facility_stats_v1.json"))
    return {int(k): (e.get("iana_tz") or "Etc/UTC") for k, e in stats["facilities"].items()}


def local_hours(df, tz_by_fid):
    fid = df["facility_id"].to_numpy(np.int64)
    n = len(df)
    created = pd.to_datetime(df["created_at"])
    if created.dt.tz is not None:
        created = created.dt.tz_convert("UTC").dt.tz_localize(None)
    iana = np.array([tz_by_fid.get(int(f), "Etc/UTC") for f in fid])
    hour = np.zeros(n, np.int64); dow = np.zeros(n, np.int64)
    for tzn in np.unique(iana):
        mask = iana == tzn
        loc = created[mask].dt.tz_localize("UTC").dt.tz_convert(tzn)
        hour[mask] = loc.dt.hour.to_numpy(); dow[mask] = loc.dt.dayofweek.to_numpy()
    return hour, dow


def build(df, mean_by, gmean, tz_by_fid):
    df = df.copy()
    fid = df["facility_id"].to_numpy(np.int64)
    amt = df["amount"].to_numpy(np.float64)
    fmean = np.array([mean_by.get(int(f), gmean) for f in fid], dtype=np.float64)
    afr = amt / (fmean + 0.01)
    df["log_amount_fac"] = np.log1p(amt / (fmean + 0.01)).astype(np.float32)
    df["amount_facility_ratio"] = afr.astype(np.float32)
    df["user_amount_24h_fac"] = (df["user_amount_24h"].to_numpy(np.float64) / (fmean + 0.01)).astype(np.float32)
    df["user_debit_amount_30d_fac"] = (df["user_debit_amount_30d"].to_numpy(np.float64) / (fmean + 0.01)).astype(np.float32)
    df["small_amount_at_facility"] = (afr < 0.2).astype(np.float32)
    df["very_small_amount_at_facility"] = (afr < 0.05).astype(np.float32)
    hour, dow = local_hours(df, tz_by_fid)
    df["hour_sin_loc"] = np.sin(2*np.pi*hour/24).astype(np.float32)
    df["hour_cos_loc"] = np.cos(2*np.pi*hour/24).astype(np.float32)
    df["dow_sin_loc"] = np.sin(2*np.pi*dow/7).astype(np.float32)
    df["dow_cos_loc"] = np.cos(2*np.pi*dow/7).astype(np.float32)
    df["is_weekend_loc"] = (dow >= 5).astype(np.float32)
    isoff = np.isin(hour, OFF_HOURS).astype(np.float32)
    df["is_off_hours_loc"] = isoff
    df["off_hours_high_value_loc"] = ((isoff > 0) & (afr > 3)).astype(np.float32)
    roles = df["user_role"].fillna("player").astype(str)
    df["is_staff"] = roles.isin(["court_manager", "court_operator", "teacher"]).astype(np.float32).to_numpy()
    df["staff_amount_zscore"] = np.zeros(len(df), np.float32)
    return df


def top5(scores, amount):
    k5 = int(len(scores) * 0.05)
    top = np.argsort(scores)[-k5:]
    return float(amount[top].mean() / amount.mean())


def main():
    t0 = time.perf_counter()
    mean_by, med_by, iqr_by, gmean = own_stats()
    tz_by_fid = tz_lookup()
    train_df = pd.read_parquet(DATA / "train_features_enriched.parquet", columns=NEEDED)
    val_df = pd.read_parquet(DATA / "val_features_enriched.parquet", columns=NEEDED)
    test_df = pd.read_parquet(DATA / "test_features_enriched.parquet", columns=NEEDED)
    print(f"loaded train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} ({time.perf_counter()-t0:.1f}s)")

    tr = build(train_df, mean_by, gmean, tz_by_fid)
    va = build(val_df, mean_by, gmean, tz_by_fid)
    te = build(test_df, mean_by, gmean, tz_by_fid)

    Xtr = np.nan_to_num(tr[FRAME_V1_FEATURE_NAMES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(Xtr)
    Xtr = np.clip(scaler.transform(Xtr), -10, 10).astype(np.float32)
    m = IsolationForest(n_estimators=200, max_samples=512, max_features=0.6,
                        contamination="auto", random_state=42, n_jobs=-1).fit(Xtr)

    out = {}
    for tag, dfb, raw in [("VAL_full", va, val_df), ("TEST_full", te, test_df)]:
        X = np.nan_to_num(dfb[FRAME_V1_FEATURE_NAMES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
        sc = -m.decision_function(X)
        r = top5(sc, raw["amount"].to_numpy(np.float64))
        out[tag] = r
        print(f"[{tag}] top5 = {r:.3f}x  (n={len(sc):,})")
        # muestra 1/20 (config exacta del experimento)
        mask = (raw["id"].to_numpy() % 20) == 0
        r20 = top5(sc[mask], raw["amount"].to_numpy(np.float64)[mask])
        out[tag + "_mod20"] = r20
        print(f"[{tag} mod20] top5 = {r20:.3f}x  (n={int(mask.sum()):,})")

    json.dump(out, open(SCRATCH / "diag_window_ab.json", "w"), indent=2)
    print("\n=== RESUMEN ===")
    for k, v in out.items():
        print(f"  {k:16s} {v:.3f}x")


if __name__ == "__main__":
    main()
