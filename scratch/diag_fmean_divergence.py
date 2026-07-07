"""Diagnostico H-B: fmean del artefacto (currency-fallback) vs stats propias.

Objetivo: cuantificar cuanto del sesgo top-5% amount viene del denominador fmean.
NO toca produccion. Entrena en VAL/TRAIN local, mide en VAL completo.

Compara dos construcciones de las MISMAS 30 features (FRAME_V1_FEATURE_NAMES),
cambiando UNA sola variable: como se calcula fmean/fmedian/fiqr por facility.

  config A "artifact"   : usa facility_stats_v1.json (currency-fallback para n<30)
  config B "own_stats"  : media/mediana/IQR propias por facility (como el experimento),
                          sin fallback de currency, gmean/gmed/giqr solo si facility ausente.

Salida: top-5% amount ratio en VAL para A y B, con el MISMO modelo IF (seed 42),
misma receta (RobustScaler(5,95)+clip[-10,10], IF 200/512/0.6).
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
MODELS = ROOT / "output" / "models"
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
    "is_first_gateway_for_user", "source_change_recent", "is_off_hours",
]

# Features de marco que dependen del denominador fmean/fmedian/fiqr:
#   log_amount_fac, amount_facility_ratio, user_amount_24h_fac, user_debit_amount_30d_fac,
#   small_amount_at_facility, very_small_amount_at_facility, off_hours_high_value_loc
# Las temporales locales y staff_zscore NO cambian entre A y B -> las tomamos del artefacto
# para aislar SOLO el efecto del denominador facility.


def load_artifact_stats() -> dict:
    with open(MODELS / "facility_stats_v1.json") as f:
        return json.load(f)


def artifact_lookup(stats: dict):
    gfb = stats["global_fallback"]
    g_mean, g_median, g_iqrg = float(gfb["mean"]), float(gfb["median"]), float(gfb["iqr_guarded"])
    m, med, iqrg, tz = {}, {}, {}, {}
    for fid_str, e in stats["facilities"].items():
        fid = int(fid_str)
        tz[fid] = e.get("iana_tz", "Etc/UTC") or "Etc/UTC"
        fmean = float(e.get("mean") or 0)
        if fmean > 0:
            m[fid] = fmean
            med[fid] = float(e.get("median") or 0)
            iqrg[fid] = float(e.get("iqr_guarded") or 1.0)
        else:
            m[fid], med[fid], iqrg[fid] = g_mean, g_median, g_iqrg
    return m, med, iqrg, tz, (g_mean, g_median, g_iqrg)


def own_stats_from_train():
    """Media/mediana/IQR PROPIAS por facility desde el train COMPLETO (sin fallback n<30)."""
    tr = pd.read_parquet(DATA / "train_features_enriched.parquet",
                         columns=["amount", "facility_id"])
    g = tr.groupby("facility_id")["amount"]
    mean = g.mean().to_dict()
    median = g.median().to_dict()
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    iqr = (q3 - q1)
    iqr_guarded = iqr.clip(lower=1.0).to_dict()  # guarded como produccion max(iqr,1)
    g_mean = float(tr["amount"].mean())
    g_median = float(tr["amount"].median())
    g_iqrg = max(float(tr["amount"].quantile(0.75) - tr["amount"].quantile(0.25)), 1.0)
    return mean, median, iqr_guarded, (g_mean, g_median, g_iqrg)


def build_local_hours(df: pd.DataFrame, tz_by_fid: dict):
    fid = df["facility_id"].to_numpy(dtype=np.int64)
    n = len(df)
    created = pd.to_datetime(df["created_at"])
    if created.dt.tz is not None:
        created = created.dt.tz_convert("UTC").dt.tz_localize(None)
    iana = np.array([tz_by_fid.get(int(f), "Etc/UTC") for f in fid])
    hour = np.zeros(n, dtype=np.int64)
    dow = np.zeros(n, dtype=np.int64)
    for tzn in np.unique(iana):
        mask = iana == tzn
        loc = created[mask].dt.tz_localize("UTC").dt.tz_convert(tzn)
        hour[mask] = loc.dt.hour.to_numpy()
        dow[mask] = loc.dt.dayofweek.to_numpy()
    return hour, dow


def build_features(df, fmean_by, fmed_by, fiqr_by, gtuple, tz_by_fid, staff_source_df):
    """Construye las 30 features. fmean/fmed/fiqr vienen del config (A o B).
    Temporales y staff se toman de staff_source_df para que sean identicas entre configs."""
    df = df.copy()
    fid = df["facility_id"].to_numpy(dtype=np.int64)
    amt = df["amount"].to_numpy(dtype=np.float64)
    g_mean, g_median, g_iqrg = gtuple
    fmean = np.array([fmean_by.get(int(f), g_mean) for f in fid], dtype=np.float64)
    # (fmed/fiqr no se usan en FRAME_V1 salvo via amount_fac_z, ausente; se dejan por si acaso)

    amount_facility_ratio = amt / (fmean + 0.01)
    df["log_amount_fac"] = np.log1p(amt / (fmean + 0.01)).astype(np.float32)
    df["amount_facility_ratio"] = amount_facility_ratio.astype(np.float32)
    df["user_amount_24h_fac"] = (df["user_amount_24h"].to_numpy(np.float64) / (fmean + 0.01)).astype(np.float32)
    df["user_debit_amount_30d_fac"] = (df["user_debit_amount_30d"].to_numpy(np.float64) / (fmean + 0.01)).astype(np.float32)
    df["small_amount_at_facility"] = (amount_facility_ratio < 0.2).astype(np.float32)
    df["very_small_amount_at_facility"] = (amount_facility_ratio < 0.05).astype(np.float32)

    # temporales locales (identicas entre configs)
    hour, dow = build_local_hours(df, tz_by_fid)
    df["hour_sin_loc"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos_loc"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["dow_sin_loc"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["dow_cos_loc"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    df["is_weekend_loc"] = (dow >= 5).astype(np.float32)
    is_off = np.isin(hour, OFF_HOURS).astype(np.float32)
    df["is_off_hours_loc"] = is_off
    df["off_hours_high_value_loc"] = ((is_off > 0) & (amount_facility_ratio > 3)).astype(np.float32)

    # staff / is_staff desde source_df (mismo para A y B): lo copiamos por indice
    df["is_staff"] = staff_source_df["is_staff"].to_numpy()
    df["staff_amount_zscore"] = staff_source_df["staff_amount_zscore"].to_numpy()
    return df


def top5_ratio(scores, amount):
    n = len(scores)
    k5 = int(n * 0.05)
    top = np.argsort(scores)[-k5:]
    return float(amount[top].mean() / amount.mean())


def run(tag, train_df, val_df, fmean_by, fmed_by, fiqr_by, gtuple, tz_by_fid, staff_tr, staff_val):
    tr = build_features(train_df, fmean_by, fmed_by, fiqr_by, gtuple, tz_by_fid, staff_tr)
    va = build_features(val_df, fmean_by, fmed_by, fiqr_by, gtuple, tz_by_fid, staff_val)
    Xtr = np.nan_to_num(tr[FRAME_V1_FEATURE_NAMES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(va[FRAME_V1_FEATURE_NAMES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sc = RobustScaler(quantile_range=(5.0, 95.0)).fit(Xtr)
    Xtr = np.clip(sc.transform(Xtr), -10, 10).astype(np.float32)
    Xva = np.clip(sc.transform(Xva), -10, 10).astype(np.float32)
    m = IsolationForest(n_estimators=200, max_samples=512, max_features=0.6,
                        contamination="auto", random_state=42, n_jobs=-1).fit(Xtr)
    scores = -m.decision_function(Xva)
    amount = val_df["amount"].to_numpy(np.float64)
    r = top5_ratio(scores, amount)
    print(f"[{tag}] top5 amount ratio (VAL) = {r:.3f}x")
    return r


def main():
    t0 = time.perf_counter()
    stats = load_artifact_stats()
    a_mean, a_med, a_iqr, a_tz, a_g = artifact_lookup(stats)
    print(f"artifact loaded ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    train_df = pd.read_parquet(DATA / "train_features_enriched.parquet", columns=NEEDED)
    val_df = pd.read_parquet(DATA / "val_features_enriched.parquet", columns=NEEDED)
    print(f"train={train_df.shape} val={val_df.shape} ({time.perf_counter()-t0:.1f}s)")

    # staff features via artifact calculator seria caro; usamos aproximacion neutra:
    # is_staff desde user_role, staff_zscore=0 (constante entre A y B -> no afecta comparacion)
    def staff_cols(df):
        roles = df["user_role"].fillna("player").astype(str)
        return pd.DataFrame({
            "is_staff": roles.isin(["court_manager", "court_operator", "teacher"]).astype(np.float32).to_numpy(),
            "staff_amount_zscore": np.zeros(len(df), dtype=np.float32),
        })
    staff_tr, staff_val = staff_cols(train_df), staff_cols(val_df)

    # --- Config A: artifact stats (currency-fallback) ---
    rA = run("A_artifact", train_df, val_df, a_mean, a_med, a_iqr, a_g, a_tz, staff_tr, staff_val)

    # --- Config B: own per-facility stats (como el experimento) ---
    t0 = time.perf_counter()
    b_mean, b_med, b_iqr, b_g = own_stats_from_train()
    print(f"own stats built ({time.perf_counter()-t0:.1f}s)")
    rB = run("B_own_stats", train_df, val_df, b_mean, b_med, b_iqr, b_g, a_tz, staff_tr, staff_val)

    print("\n=== RESULTADO ===")
    print(f"A (artifact, currency-fallback n<30): {rA:.3f}x")
    print(f"B (own per-facility stats):           {rB:.3f}x")
    print(f"delta = {rA - rB:.3f}x   (si B<<A -> H-B confirmada: el fallback de currency causa el sesgo)")
    json.dump({"A_artifact": rA, "B_own_stats": rB},
              open(ROOT / "output" / "revision" / "diag_fmean_divergence.json", "w"), indent=2)


if __name__ == "__main__":
    main()
