"""
Experimento controlado: efecto de los MARCOS DE REFERENCIA en la deteccion.

Compara el MISMO Isolation Forest con dos conjuntos de features que difieren
UNICAMENTE en el marco en que se definen:

  V0 (contaminado)  : magnitud absoluta (amount, log_amount, amount/media_global)
                      + tiempo en UTC (hora, off-hours, dia de semana)
  V1 (marco-normal.): magnitud relativa a la facility (ratio y z-score robusto)
                      + tiempo en hora LOCAL de la facility

Cohorte: currency='USD' (controla el confound de moneda por construccion).
Split temporal: train Ene-Ago 2025 / test Sep-Dic 2025 (anti-leakage).
Proxy (solo evaluacion): status IN ('totally_refunded','refunded_to_credit').
NINGUN feature derivado de `status` en las features (sin leakage de etiqueta).

Uso:  ./venv/bin/python scripts/exp_reference_frames.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import clickhouse_connect  # noqa: E402
from config.config import settings  # noqa: E402

SEED = 42
SAMPLE_MOD = 25  # ~1/25 de la cohorte USD para correr rapido
OUT = Path("output/revision/frames_experiment.json")

# Rails ActiveSupport::TimeZone  ->  IANA (las presentes en la cohorte USD)
RAILS_TZ_TO_IANA = {
    "Eastern Time (US & Canada)": "America/New_York",
    "Central Time (US & Canada)": "America/Chicago",
    "Mountain Time (US & Canada)": "America/Denver",
    "Pacific Time (US & Canada)": "America/Los_Angeles",
    "Arizona": "America/Phoenix",
    "Hawaii": "Pacific/Honolulu",
    "Atlantic Time (Canada)": "America/Halifax",
    "Caracas": "America/Caracas",
    "Puerto Rico": "America/Puerto_Rico",
    "Central America": "America/Guatemala",
    "Quito": "America/Guayaquil",
    "Bogota": "America/Bogota",
    "La Paz": "America/La_Paz",
    "Mexico City": "America/Mexico_City",
    "Tijuana": "America/Tijuana",
    "Buenos Aires": "America/Argentina/Buenos_Aires",
}

OFF_HOURS = {23, 0, 1, 2, 3, 4, 5, 6}


def get_client():
    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
        secure=settings.clickhouse_secure,
    )


def fetch(client, start: str, end: str) -> pd.DataFrame:
    q = f"""
    SELECT p.id           AS id,
           p.facility_id  AS facility_id,
           p.created_at   AS created_at,
           p.reservation_paid_out AS amount,
           p.discount     AS discount,
           p.tip          AS tip,
           p.status       AS status,
           f.time_zone    AS time_zone
    FROM {settings.clickhouse_database}.payments AS p FINAL
    LEFT JOIN {settings.clickhouse_database}.facilities AS f ON f.id = p.facility_id
    WHERE p.currency = 'USD'
      AND p.payment_method != 'reversal'
      AND p.payment_method != 'free'
      AND p.user_id != 0
      AND p.created_at >= '{start}' AND p.created_at < '{end}'
      AND cityHash64(p.id) % {SAMPLE_MOD} = 0
    """
    df = client.query_df(q)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["amount"] = df["amount"].astype(float).clip(lower=0)
    df["discount"] = df["discount"].astype(float).fillna(0)
    df["tip"] = df["tip"].astype(float).fillna(0)
    df["y"] = df["status"].isin(["totally_refunded", "refunded_to_credit"]).astype(int)
    return df


def local_hour_dow(df: pd.DataFrame):
    """Convierte created_at (UTC) a hora local de la facility segun su time_zone."""
    hour = np.zeros(len(df), dtype=np.int64)
    dow = np.zeros(len(df), dtype=np.int64)
    utc = pd.to_datetime(df["created_at"], utc=True)
    for tz_name, group_idx in df.groupby("time_zone").groups.items():
        iana = RAILS_TZ_TO_IANA.get(str(tz_name))
        idx = df.index.get_indexer(group_idx)
        local = utc.iloc[idx] if iana is None else utc.iloc[idx].dt.tz_convert(iana)
        hour[idx] = local.dt.hour.to_numpy()
        dow[idx] = local.dt.dayofweek.to_numpy()  # 0=lunes
    return hour, dow


def temporal_block(hour, dow):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    is_off = np.isin(hour, list(OFF_HOURS)).astype(float)
    is_weekend = (dow >= 5).astype(float)
    return np.column_stack([hour_sin, hour_cos, is_off, is_weekend])


def build_v0(df, stats):
    """Marco contaminado: magnitud absoluta + hora UTC."""
    amount = df["amount"].to_numpy()
    log_amount = np.log1p(amount)
    amount_global = amount / max(stats["global_mean"], 1e-8)
    discount_ratio = df["discount"].to_numpy() / (amount + 0.01)
    has_tip = (df["tip"].to_numpy() > 0).astype(float)
    hour = df["created_at"].dt.hour.to_numpy()          # UTC
    dow = df["created_at"].dt.dayofweek.to_numpy()      # UTC
    temporal = temporal_block(hour, dow)
    mag = np.column_stack([amount, log_amount, amount_global, discount_ratio, has_tip])
    return np.column_stack([mag, temporal])


def build_v1(df, stats):
    """Marco normalizado: magnitud relativa a la facility + hora local."""
    amount = df["amount"].to_numpy()
    fid = df["facility_id"].to_numpy()
    fmean = np.array([stats["fac_mean"].get(f, stats["global_mean"]) for f in fid])
    fmed = np.array([stats["fac_median"].get(f, stats["global_median"]) for f in fid])
    fiqr = np.array([stats["fac_iqr"].get(f, stats["global_iqr"]) for f in fid])
    amount_fac_ratio = amount / (fmean + 0.01)
    amount_fac_z = (amount - fmed) / (fiqr + 1e-6)
    discount_ratio = df["discount"].to_numpy() / (amount + 0.01)
    has_tip = (df["tip"].to_numpy() > 0).astype(float)
    hour, dow = local_hour_dow(df)                      # LOCAL
    temporal = temporal_block(hour, dow)
    mag = np.column_stack([amount_fac_ratio, amount_fac_z, discount_ratio, has_tip])
    return np.column_stack([mag, temporal])


def fit_stats(train: pd.DataFrame) -> dict:
    g = train.groupby("facility_id")["amount"]
    q1 = g.quantile(0.25)
    q3 = g.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    return {
        "global_mean": float(train["amount"].mean()),
        "global_median": float(train["amount"].median()),
        "global_iqr": float((train["amount"].quantile(0.75) - train["amount"].quantile(0.25)) or 1.0),
        "fac_mean": g.mean().to_dict(),
        "fac_median": g.median().to_dict(),
        "fac_iqr": iqr.fillna(iqr.median()).to_dict(),
    }


def train_score(Xtr, Xte, yte, test_df):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    model = IsolationForest(
        n_estimators=200, max_samples=512, max_features=1.0,
        contamination="auto", random_state=SEED, n_jobs=-1,
    ).fit(Xtr_s)
    scores = -model.score_samples(Xte_s)
    auc = float(roc_auc_score(yte, scores))
    ap = float(average_precision_score(yte, scores))
    # enriquecimiento en el top-5%
    k = max(1, int(len(scores) * 0.05))
    top = np.argsort(scores)[-k:]
    prec5 = float(yte.to_numpy()[top].mean())
    ef5 = prec5 / max(yte.mean(), 1e-9)
    # diagnostico: que mide el score? (correlacion con magnitud cruda)
    amount = test_df["amount"].to_numpy()
    corr_amount = float(pd.Series(scores).corr(pd.Series(amount), method="spearman"))
    top_amount_mult = float(amount[top].mean() / max(amount.mean(), 1e-9))
    return {
        "auc_roc": auc, "auc_pr": ap,
        "precision_at_5pct": prec5, "ef_at_5pct": ef5,
        "spearman_score_vs_amount": corr_amount,
        "top5_amount_x_avg": top_amount_mult,
    }


def main():
    client = get_client()
    print("Descargando cohorte USD (muestreada 1/%d)..." % SAMPLE_MOD)
    train = fetch(client, "2025-01-01", "2025-09-01")
    test = fetch(client, "2025-09-01", "2026-01-01")
    print(f"  train={len(train):,}  test={len(test):,}")
    print(f"  base rate train={train['y'].mean():.4f}  test={test['y'].mean():.4f}")

    # sanity: off-hours UTC vs local (evidencia del confound temporal)
    h_utc = test["created_at"].dt.hour.to_numpy()
    hl, _ = local_hour_dow(test)
    print(f"  off-hours UTC={np.isin(h_utc, list(OFF_HOURS)).mean():.4f}  "
          f"local={np.isin(hl, list(OFF_HOURS)).mean():.4f}")

    stats = fit_stats(train)

    print("\nEntrenando V0 (marco contaminado)...")
    v0 = train_score(build_v0(train, stats), build_v0(test, stats), test["y"], test)
    print("Entrenando V1 (marco normalizado)...")
    v1 = train_score(build_v1(train, stats), build_v1(test, stats), test["y"], test)

    result = {
        "cohort": "USD",
        "sample_mod": SAMPLE_MOD,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "base_rate_test": float(test["y"].mean()),
        "offhours_utc": float(np.isin(h_utc, list(OFF_HOURS)).mean()),
        "offhours_local": float(np.isin(hl, list(OFF_HOURS)).mean()),
        "V0_contaminated": v0,
        "V1_frame_normalized": v1,
        "delta_auc": v1["auc_roc"] - v0["auc_roc"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 56)
    print(f"{'metric':<22}{'V0 contam.':>14}{'V1 normaliz.':>16}")
    print("-" * 56)
    for m in ("auc_roc", "auc_pr", "precision_at_5pct", "ef_at_5pct",
              "spearman_score_vs_amount", "top5_amount_x_avg"):
        print(f"{m:<22}{v0[m]:>14.4f}{v1[m]:>16.4f}")
    print("=" * 56)
    print(f"Delta AUC-ROC (V1 - V0): {result['delta_auc']:+.4f}")
    print(f"Resultado guardado en {OUT}")


if __name__ == "__main__":
    main()
