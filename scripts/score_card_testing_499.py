#!/usr/bin/env python3
"""PRUEBA DE FUEGO: ¿ve el IF frame-v1 el ataque de card testing distribuido de
facility 499 (2026-05-13) aunque la regla per-usuario NO lo etiquete?

Vía de extracción: (a) camino real-time de producción.
  UserContextProvider.get_context(...)  -> agregados rolling desde HISTORIAL
                                            COMPLETO de ClickHouse (SQL de context.py)
  FrameV1FeatureCalculator.calculate(payment, context) -> vector de 30 features
  IF frame-v1 (isolation_forest_frame_v1.joblib) + scaler_frame_v1 + clip[-10,10]
  score = -decision_function   (idéntico a build_v2_scores.py que produjo la
                                 distribución de referencia test_scores_v2.parquet)

Compara el percentil de cada txn del ataque contra la distribución poblacional
(test set Sep-Dic 2025, 2.5M) y contra una BASELINE de facility 499 en un día
normal (2025-09-15) puntuada con el MISMO camino.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detector.data.clickhouse_connector import ClickHouseConnector  # noqa: E402
from fraud_detector.scoring.context import UserContextProvider  # noqa: E402
from fraud_detector.scoring.features_frame_v1 import (  # noqa: E402
    FRAME_V1_FEATURE_NAMES,
    FrameV1FeatureCalculator,
)
from config.config import Settings  # noqa: E402

MODELS = ROOT / "output" / "models"
REVISION = ROOT / "output" / "revision"

# Raw columns needed by FrameV1FeatureCalculator.calculate(payment, context).
# Depuración idéntica al pipeline: payment_method NOT IN (reversal, free), user_id != 0.
EXTRACT_SQL = """
SELECT
    p.id AS id,
    p.user_id AS user_id,
    p.effective_user_id AS effective_user_id,
    p.facility_id AS facility_id,
    p.created_at AS created_at,
    p.status AS status,
    p.reservation_paid_out AS reservation_paid_out,
    p.discount AS discount,
    p.tip AS tip,
    p.currency AS currency,
    p.gateway AS gateway,
    toString(p.source_enum) AS source_enum,
    p.payment_method AS payment_method,
    p.category AS category,
    p.club_credit_flag AS club_credit_flag,
    p.paid_by_manager AS paid_by_manager
FROM pbp_productionDB_optimized.payments AS p FINAL
WHERE p.facility_id = {fid:Int64}
  AND p.created_at >= {start:DateTime}
  AND p.created_at <  {end:DateTime}
  AND p.payment_method NOT IN ('reversal', 'free')
  AND p.user_id != 0
  AND p._peerdb_is_deleted = 0
ORDER BY p.created_at, p.id
"""

# Facility 499 IANA tz para las features temporales locales (matches artefacto)
FACILITY_499_IANA = "America/New_York"


def get_client():
    s = Settings()
    conn = ClickHouseConnector(
        host=s.clickhouse_host, port=s.clickhouse_port, user=s.clickhouse_user,
        password=s.clickhouse_password, database=s.clickhouse_database,
        secure=s.clickhouse_secure,
    )
    return conn.client, conn


def extract(client, fid, start, end):
    return client.query_df(
        EXTRACT_SQL,
        parameters={"fid": fid, "start": start, "end": end},
    )


def score_txns(df, provider, calc, model, scaler, label):
    """Devuelve np.ndarray de scores (score = -decision_function, clip[-10,10])."""
    vectors = []
    n = len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        payment = {
            "id": int(r["id"]),
            "user_id": int(r["user_id"]),
            "effective_user_id": int(r["effective_user_id"]) if r["effective_user_id"] else None,
            "facility_id": int(r["facility_id"]),
            "created_at": pd.Timestamp(r["created_at"]),
            "reservation_paid_out": float(r["reservation_paid_out"]),
            "discount": float(r["discount"]),
            "tip": float(r["tip"]),
            "currency": str(r["currency"] or "USD"),
            "gateway": str(r["gateway"] or "unknown"),
            "source_enum": str(r["source_enum"] or "unknown"),
            "club_credit_flag": bool(r["club_credit_flag"]),
            "paid_by_manager": bool(r["paid_by_manager"]),
            "facility_time_zone_iana": FACILITY_499_IANA,
        }
        ctx = provider.get_context(
            user_id=payment["user_id"],
            facility_id=payment["facility_id"],
            timestamp=payment["created_at"].to_pydatetime(),
            payment=payment,
        )
        vec = calc.calculate(payment, ctx)
        vectors.append(vec)
        if (i + 1) % 50 == 0:
            print(f"    [{label}] {i+1}/{n} contextos computados", flush=True)

    X = np.vstack(vectors).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
    scores = (-model.decision_function(Xs)).astype(np.float64)
    return scores, X


def main():
    print("[1] Conectando a ClickHouse y cargando artefactos frame-v1...", flush=True)
    client, conn = get_client()
    import json
    stats = json.load(open(MODELS / "facility_stats_v1.json"))
    calc = FrameV1FeatureCalculator(
        facility_stats=stats,
        feature_engineer_path=str(MODELS / "feature_engineer.joblib"),
    )
    provider = UserContextProvider(conn)
    model = joblib.load(MODELS / "isolation_forest_frame_v1.joblib")
    scaler = joblib.load(MODELS / "scaler_frame_v1.joblib")

    print("[2] Extrayendo txns del ATAQUE (facility 499, 2026-05-13)...", flush=True)
    df_attack = extract(client, 499, "2026-05-13 00:00:00", "2026-05-14 00:00:00")
    print(f"    ataque: {len(df_attack)} txns depuradas, {df_attack['user_id'].nunique()} usuarios", flush=True)

    print("[3] Extrayendo BASELINE facility 499 día normal (2025-09-15)...", flush=True)
    df_base = extract(client, 499, "2025-09-15 00:00:00", "2025-09-16 00:00:00")
    print(f"    baseline: {len(df_base)} txns depuradas, {df_base['user_id'].nunique()} usuarios", flush=True)

    print("[4] Puntuando ATAQUE (camino real-time con contexto de ClickHouse)...", flush=True)
    s_attack, X_attack = score_txns(df_attack, provider, calc, model, scaler, "attack")

    print("[5] Puntuando BASELINE...", flush=True)
    s_base, _ = score_txns(df_base, provider, calc, model, scaler, "baseline")

    print("[6] Cargando distribución poblacional (test_scores_v2.parquet)...", flush=True)
    ref = pd.read_parquet(REVISION / "test_scores_v2.parquet", columns=["score"])["score"].to_numpy()
    ref_sorted = np.sort(ref)
    n_ref = len(ref_sorted)

    def pctile(vals):
        # percentil poblacional: % de la población con score <= val (higher score = más anómalo)
        idx = np.searchsorted(ref_sorted, vals, side="right")
        return 100.0 * idx / n_ref

    p_attack = pctile(s_attack)
    p_base = pctile(s_base)

    # Umbrales top-k de la población (higher score = más anómalo)
    thr_top10 = np.percentile(ref, 90)
    thr_top5 = np.percentile(ref, 95)
    thr_top1 = np.percentile(ref, 99)

    def topk(scores):
        return {
            "top10": float((scores >= thr_top10).mean() * 100),
            "top5": float((scores >= thr_top5).mean() * 100),
            "top1": float((scores >= thr_top1).mean() * 100),
        }

    tk_attack = topk(s_attack)
    tk_base = topk(s_base)

    # Refund breakdown dentro del ataque
    refund_mask = df_attack["status"].isin(["totally_refunded", "refunded_to_credit"]).to_numpy()

    print("\n" + "=" * 72)
    print("RESULTADOS — PRUEBA DE FUEGO card testing facility 499 (2026-05-13)")
    print("=" * 72)
    print(f"Vía de extracción: (a) camino real-time producción "
          f"(UserContextProvider + FrameV1FeatureCalculator.calculate)")
    print(f"Modelo: IF frame-v1 (30 feats) | score = -decision_function, clip[-10,10]")
    print(f"Distribución poblacional: test_scores_v2.parquet (n={n_ref:,}, Sep-Dic 2025)")
    print(f"Umbrales pob.: top10 score>={thr_top10:.5f} | top5>={thr_top5:.5f} | top1>={thr_top1:.5f}")
    print()
    print(f"{'':<28}{'ATAQUE':>16}{'BASELINE 499':>16}")
    print(f"{'n txns':<28}{len(s_attack):>16}{len(s_base):>16}")
    print(f"{'score medio':<28}{s_attack.mean():>16.5f}{s_base.mean():>16.5f}")
    print(f"{'score mediano':<28}{np.median(s_attack):>16.5f}{np.median(s_base):>16.5f}")
    print(f"{'percentil pob. MEDIO':<28}{p_attack.mean():>15.2f}%{p_base.mean():>15.2f}%")
    print(f"{'percentil pob. MEDIANO':<28}{np.median(p_attack):>15.2f}%{np.median(p_base):>15.2f}%")
    print(f"{'% en top-1% pob (azar 1%)':<28}{tk_attack['top1']:>15.2f}%{tk_base['top1']:>15.2f}%")
    print(f"{'% en top-5% pob (azar 5%)':<28}{tk_attack['top5']:>15.2f}%{tk_base['top5']:>15.2f}%")
    print(f"{'% en top-10% pob (azar 10%)':<28}{tk_attack['top10']:>15.2f}%{tk_base['top10']:>15.2f}%")
    print()
    print("EF implícito (ataque, vs azar):")
    print(f"    top-1%:  {tk_attack['top1']/1.0:.2f}x")
    print(f"    top-5%:  {tk_attack['top5']/5.0:.2f}x")
    print(f"    top-10%: {tk_attack['top10']/10.0:.2f}x")
    print()
    print(f"Dentro del ataque: refunded={int(refund_mask.sum())} vs no-refunded={int((~refund_mask).sum())}")
    print(f"    percentil medio refunded:     {p_attack[refund_mask].mean():.2f}%")
    print(f"    percentil medio NO-refunded:  {p_attack[~refund_mask].mean():.2f}%")
    print()
    # También: reportar percentil del ataque relativo a la baseline (control facility 499)
    base_sorted = np.sort(s_base)
    pos_vs_base = 100.0 * np.searchsorted(base_sorted, s_attack, side="right") / len(base_sorted)
    print(f"Percentil MEDIO del ataque relativo a la baseline facility-499: {pos_vs_base.mean():.2f}%")
    print("=" * 72)

    # Guardar CSV de scores para inspección
    out = pd.DataFrame({
        "id": df_attack["id"].values,
        "user_id": df_attack["user_id"].values,
        "created_at": df_attack["created_at"].values,
        "status": df_attack["status"].values,
        "reservation_paid_out": df_attack["reservation_paid_out"].values,
        "score": s_attack,
        "pop_percentile": p_attack,
    })
    outp = ROOT / "output" / "revision" / "card_testing_499_scores.csv"
    out.to_csv(outp, index=False)
    print(f"\nGuardado detalle: {outp}")


if __name__ == "__main__":
    main()
