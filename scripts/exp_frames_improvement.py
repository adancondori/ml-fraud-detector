"""Prueba con datos pequenos: ¿la normalizacion de marcos mejora el ML?

4 configs x 3 seeds, muestra 1/20 del train/test enriched:
  clean29_v0        FS-clean-A-29 tal cual          -> ¿reproduce ~0.50 vs tipo_a?
  clean29_frames    FS-clean-A-29 marco-normalizado -> ¿mejora vs tipo_a?
  disjoint30_v0     FS-disjoint-30 tal cual         -> ¿reproduce ~0.636 vs pure_fraud?
  disjoint30_frames FS-disjoint-30 marco-normalizado-> ¿mejora vs pure_fraud?

Receta identica al pipeline: RobustScaler(5,95) + clip(+-10), IF(200, 512, 0.6).
Proxies solo para evaluacion. Stats de facility ajustadas SOLO en train completo.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

ROOT = Path("/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector")
DATA = ROOT / "data" / "processed"
SCRATCH = ROOT / "output" / "revision"
OUT = SCRATCH / "frames_improvement_results.json"

SAMPLE_MOD = 20
SEEDS = (42, 43, 44)
OFF_HOURS = [23, 0, 1, 2, 3, 4, 5, 6]

FEATURES_31 = [
    "amount", "log_amount", "amount_usd_ratio", "discount_ratio", "has_tip",
    "hour_sin", "hour_cos", "day_of_week", "is_weekend", "is_off_hours",
    "user_txn_count_1h", "user_txn_count_24h", "time_since_last_txn",
    "user_amount_24h", "user_distinct_facilities_30d", "user_distinct_methods",
    "user_reversal_ratio_30d", "user_account_age_days", "user_discount_ratio_30d",
    "facility_avg_amount", "amount_facility_ratio", "is_club_credit",
    "user_debit_count_30d", "user_debit_amount_30d", "credit_flow_ratio",
    "is_staff", "paid_by_manager", "staff_amount_zscore", "category_entropy_30d",
    "user_reversal_count_30d", "user_merchandise_ratio_30d",
]
CLEAN29 = [f for f in FEATURES_31
           if f not in ("user_reversal_ratio_30d", "user_reversal_count_30d")]

ALL40 = json.loads((ROOT / "output" / "models" / "final_feature_list.json").read_text())
PROXY_VARS = ["same_amount_count_1h", "user_account_age_days",
              "user_txn_count_1h", "is_third_party_payment"]
PROXY_RECODINGS = ["is_new_user", "is_very_new_user", "new_user_first_facility",
                   "rapid_burst", "same_amount_count_24h"]
DISJOINT30 = [f for f in ALL40
              if f not in PROXY_VARS + PROXY_RECODINGS + ["capture_delay_seconds"]]
assert len(CLEAN29) == 29 and len(DISJOINT30) == 30

# Reemplazos de marco: feature contaminada -> lista de reemplazos normalizados
FRAME_MAP = {
    "amount": ["amount_fac_z"],
    "log_amount": ["log_amount_fac"],
    "amount_usd_ratio": [],            # redundante con amount_facility_ratio
    "facility_avg_amount": [],         # proxy puro de tamano de facility
    "user_amount_24h": ["user_amount_24h_fac"],
    "user_debit_amount_30d": ["user_debit_amount_30d_fac"],
    "hour_sin": ["hour_sin_loc"],
    "hour_cos": ["hour_cos_loc"],
    "day_of_week": ["dow_sin_loc", "dow_cos_loc"],
    "is_weekend": ["is_weekend_loc"],
    "is_off_hours": ["is_off_hours_loc"],
    "off_hours_high_value": ["off_hours_high_value_loc"],
}

RAILS_TZ_TO_IANA = {
    "Eastern Time (US & Canada)": "America/New_York",
    "Central Time (US & Canada)": "America/Chicago",
    "Mountain Time (US & Canada)": "America/Denver",
    "Pacific Time (US & Canada)": "America/Los_Angeles",
    "Arizona": "America/Phoenix", "Hawaii": "Pacific/Honolulu",
    "Atlantic Time (Canada)": "America/Halifax",
    "Central America": "America/Guatemala", "Caracas": "America/Caracas",
    "Puerto Rico": "America/Puerto_Rico", "Kuala Lumpur": "Asia/Kuala_Lumpur",
    "Melbourne": "Australia/Melbourne", "Sydney": "Australia/Sydney",
    "Hong Kong": "Asia/Hong_Kong", "Singapore": "Asia/Singapore",
    "Monterrey": "America/Monterrey", "Quito": "America/Guayaquil",
    "Abu Dhabi": "Asia/Dubai", "Bogota": "America/Bogota",
    "Karachi": "Asia/Karachi", "Islamabad": "Asia/Karachi",
    "Brisbane": "Australia/Brisbane", "Mexico City": "America/Mexico_City",
    "Guadalajara": "America/Mexico_City", "Tijuana": "America/Tijuana",
    "Cairo": "Africa/Cairo", "Jerusalem": "Asia/Jerusalem",
    "Tokyo": "Asia/Tokyo", "Sapporo": "Asia/Tokyo", "London": "Europe/London",
    "Auckland": "Pacific/Auckland", "Wellington": "Pacific/Auckland",
    "Mumbai": "Asia/Kolkata", "New Delhi": "Asia/Kolkata",
    "Berlin": "Europe/Berlin", "Istanbul": "Europe/Istanbul",
    "Dublin": "Europe/Dublin", "La Paz": "America/La_Paz",
    "Athens": "Europe/Athens", "Perth": "Australia/Perth",
    "Stockholm": "Europe/Stockholm", "Bangkok": "Asia/Bangkok",
    "Hanoi": "Asia/Bangkok", "Pretoria": "Africa/Johannesburg",
    "Zurich": "Europe/Zurich", "Kyiv": "Europe/Kyiv",
    "Brasilia": "America/Sao_Paulo", "Sri Jayawardenepura": "Asia/Colombo",
    "Buenos Aires": "America/Argentina/Buenos_Aires",
    "New Caledonia": "Pacific/Noumea", "Paris": "Europe/Paris",
}


def load_split(name: str) -> pd.DataFrame:
    cols = sorted(set(ALL40) | set(FEATURES_31)
                  | {"id", "facility_id", "created_at", "status", "currency"})
    df = pd.read_parquet(DATA / f"{name}_features_enriched.parquet", columns=cols)
    df = df[(df["id"] % SAMPLE_MOD) == 0].reset_index(drop=True)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def facility_stats() -> dict:
    tr = pd.read_parquet(DATA / "train_features_enriched.parquet",
                         columns=["amount", "facility_id"])
    g = tr.groupby("facility_id")["amount"]
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    return {
        "mean": g.mean().to_dict(), "median": g.median().to_dict(),
        "iqr": iqr.fillna(float(iqr.median())).to_dict(),
        "gmean": float(tr["amount"].mean()), "gmed": float(tr["amount"].median()),
        "giqr": float(tr["amount"].quantile(0.75) - tr["amount"].quantile(0.25)) or 1.0,
    }


def local_hour_dow(df: pd.DataFrame, tz_map: dict) -> tuple:
    hour = np.zeros(len(df), dtype=np.int64)
    dow = np.zeros(len(df), dtype=np.int64)
    utc = pd.to_datetime(df["created_at"], utc=True)
    tz_name = df["facility_id"].map(tz_map)
    for name, grp in df.groupby(tz_name, dropna=False).groups.items():
        idx = df.index.get_indexer(grp)
        iana = RAILS_TZ_TO_IANA.get(str(name))
        loc = utc.iloc[idx] if iana is None else utc.iloc[idx].dt.tz_convert(iana)
        hour[idx] = loc.dt.hour.to_numpy()
        dow[idx] = loc.dt.dayofweek.to_numpy()  # 0=lunes
    return hour, dow


def add_frame_features(df: pd.DataFrame, st: dict, tz_map: dict) -> pd.DataFrame:
    fid = df["facility_id"].to_numpy()
    amt = df["amount"].to_numpy(dtype=np.float64)
    fmean = np.array([st["mean"].get(f, st["gmean"]) for f in fid])
    fmed = np.array([st["median"].get(f, st["gmed"]) for f in fid])
    fiqr = np.array([st["iqr"].get(f, st["giqr"]) for f in fid])
    df = df.copy()
    df["amount_fac_z"] = (amt - fmed) / (fiqr + 1e-6)
    df["log_amount_fac"] = np.log1p(amt / (fmean + 0.01))
    df["user_amount_24h_fac"] = df["user_amount_24h"].to_numpy() / (fmean + 0.01)
    df["user_debit_amount_30d_fac"] = df["user_debit_amount_30d"].to_numpy() / (fmean + 0.01)
    hour, dow = local_hour_dow(df, tz_map)
    df["hour_sin_loc"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos_loc"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin_loc"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos_loc"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend_loc"] = (dow >= 5).astype(np.int8)
    df["is_off_hours_loc"] = np.isin(hour, OFF_HOURS).astype(np.int8)
    df["off_hours_high_value_loc"] = (
        (df["is_off_hours_loc"] > 0) & (df["amount_facility_ratio"] > 3)
    ).astype(np.int8)
    df["_hour_utc"] = df["created_at"].dt.hour
    df["_hour_loc"] = hour
    return df


def frame_version(features: list) -> list:
    out = []
    for f in features:
        out.extend(FRAME_MAP.get(f, [f]) if f in FRAME_MAP else [f])
    return out


def build_proxies(df: pd.DataFrame) -> dict:
    tipo_a = df["status"].isin(["totally_refunded", "refunded_to_credit"]).to_numpy()
    card = (df["same_amount_count_1h"] >= 3).to_numpy()
    newb = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third = ((df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    return {"tipo_a": tipo_a.astype(np.int8),
            "pure_fraud": (card | newb | third).astype(np.int8)}


def evaluate(scores, proxies, amount) -> dict:
    out = {}
    order = np.argsort(-scores)
    k5 = max(1, int(np.ceil(len(scores) * 0.05)))
    k1 = max(1, int(np.ceil(len(scores) * 0.01)))
    out["top5_amount_x_avg"] = float(amount[order[:k5]].mean() / amount.mean())
    for label, y in proxies.items():
        base = float(y.mean())
        out[label] = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "p_1pct": float(y[order[:k1]].mean()),
            "ef_1pct": float(y[order[:k1]].mean() / base),
            "p_5pct": float(y[order[:k5]].mean()),
            "ef_5pct": float(y[order[:k5]].mean() / base),
            "base_rate": base,
        }
    return out


def run_config(name, feats, df_tr, df_te, proxies, amount_te, results):
    Xtr = np.nan_to_num(df_tr[feats].to_numpy(dtype=np.float32),
                        nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(df_te[feats].to_numpy(dtype=np.float32),
                        nan=0.0, posinf=0.0, neginf=0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(Xtr)
    Xtr = np.clip(scaler.transform(Xtr), -10, 10).astype(np.float32)
    Xte = np.clip(scaler.transform(Xte), -10, 10).astype(np.float32)
    per_seed = []
    for seed in SEEDS:
        t = time.perf_counter()
        m = IsolationForest(n_estimators=200, max_samples=512, max_features=0.6,
                            contamination="auto", random_state=seed, n_jobs=-1).fit(Xtr)
        sc = -np.asarray(m.decision_function(Xte), dtype=np.float64)
        ev = evaluate(sc, proxies, amount_te)
        per_seed.append(ev)
        print(f"  {name} seed={seed}: tipo_a AUC={ev['tipo_a']['auc']:.4f} "
              f"pure_fraud AUC={ev['pure_fraud']['auc']:.4f} "
              f"EF1%={ev['pure_fraud']['ef_1pct']:.2f} "
              f"top5amt={ev['top5_amount_x_avg']:.1f}x ({time.perf_counter()-t:.0f}s)")
    summary = {"n_features": len(feats), "features": feats, "per_seed": per_seed}
    for label in proxies:
        aucs = [e[label]["auc"] for e in per_seed]
        ef1 = [e[label]["ef_1pct"] for e in per_seed]
        ef5 = [e[label]["ef_5pct"] for e in per_seed]
        summary[f"{label}_auc_mean"] = float(np.mean(aucs))
        summary[f"{label}_auc_std"] = float(np.std(aucs))
        summary[f"{label}_ef1_mean"] = float(np.mean(ef1))
        summary[f"{label}_ef5_mean"] = float(np.mean(ef5))
    summary["top5_amount_x_avg_mean"] = float(
        np.mean([e["top5_amount_x_avg"] for e in per_seed]))
    results[name] = summary


def main():
    t0 = time.perf_counter()
    tz_map = pd.read_parquet(SCRATCH / "facility_tz.parquet") \
        .set_index("facility_id")["time_zone"].to_dict()
    print("Cargando splits (muestra 1/%d)..." % SAMPLE_MOD)
    df_tr, df_te = load_split("train"), load_split("test")
    print(f"  train={len(df_tr):,} test={len(df_te):,} ({time.perf_counter()-t0:.0f}s)")
    st = facility_stats()
    df_tr = add_frame_features(df_tr, st, tz_map)
    df_te = add_frame_features(df_te, st, tz_map)
    proxies = build_proxies(df_te)
    for k, y in proxies.items():
        print(f"  proxy {k}: rate={y.mean():.4f} n={int(y.sum()):,}")
    print(f"  off-hours test: UTC={np.isin(df_te['_hour_utc'], OFF_HOURS).mean():.3f} "
          f"local={np.isin(df_te['_hour_loc'], OFF_HOURS).mean():.3f}")
    amount_te = df_te["amount"].to_numpy(dtype=np.float64)

    results = {"sample_mod": SAMPLE_MOD, "seeds": list(SEEDS),
               "n_train": len(df_tr), "n_test": len(df_te),
               "offhours_utc": float(np.isin(df_te["_hour_utc"], OFF_HOURS).mean()),
               "offhours_local": float(np.isin(df_te["_hour_loc"], OFF_HOURS).mean())}
    configs = [
        ("clean29_v0", CLEAN29),
        ("clean29_frames", frame_version(CLEAN29)),
        ("disjoint30_v0", DISJOINT30),
        ("disjoint30_frames", frame_version(DISJOINT30)),
    ]
    for name, feats in configs:
        print(f"\n== {name} ({len(feats)} features) ==")
        run_config(name, feats, df_tr, df_te, proxies, amount_te, results)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nGuardado en {OUT}")
    print(f"\n{'config':22s}{'AUC tipo_a':>12s}{'AUC pure_fr':>12s}"
          f"{'EF1 pure':>10s}{'top5 amt':>10s}")
    for name, _ in configs:
        s = results[name]
        print(f"{name:22s}{s['tipo_a_auc_mean']:>12.4f}{s['pure_fraud_auc_mean']:>12.4f}"
              f"{s['pure_fraud_ef1_mean']:>10.2f}{s['top5_amount_x_avg_mean']:>9.1f}x")


if __name__ == "__main__":
    main()
