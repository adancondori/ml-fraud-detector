#!/usr/bin/env python3
"""PRUEBA DE FUEGO (versión batcheada): IF frame-v1 vs ataque card testing 499.

Vía de extracción: (b) agregados computados vía SQL batcheado (agg_card_testing_sql.py),
replicando 1:1 las definiciones de context.py, y enriquecidos con
add_frame_features_from_artifact (MISMO camino batch que produjo test_scores_v2.parquet).

Incluye VALIDACIÓN: compara el vector de features de una muestra contra el camino
real-time (UserContextProvider + FrameV1FeatureCalculator.calculate) — deben coincidir
salvo <1e-4 (paridad batch↔calculator garantizada por retrain_frame_v1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fraud_detector.data.clickhouse_connector import ClickHouseConnector  # noqa: E402
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES, FrameV1FeatureCalculator  # noqa: E402
from fraud_detector.scoring.context import UserContextProvider  # noqa: E402
from config.config import Settings  # noqa: E402
from retrain_frame_v1 import add_frame_features_from_artifact  # noqa: E402
from agg_card_testing_sql import build_target_sql, compute_aggregates  # noqa: E402

MODELS = ROOT / "output" / "models"
REVISION = ROOT / "output" / "revision"
FACILITY_499_IANA = "America/New_York"


def get_conn():
    s = Settings()
    return ClickHouseConnector(
        host=s.clickhouse_host, port=s.clickhouse_port, user=s.clickhouse_user,
        password=s.clickhouse_password, database=s.clickhouse_database, secure=s.clickhouse_secure,
    )


def enrich_and_vectorize(agg_df, stats):
    """agg_df tiene las columnas raw -> añade facility_time_zone_iana y enriquece."""
    df = agg_df.copy()
    df["facility_time_zone_iana"] = FACILITY_499_IANA
    # add_frame_features_from_artifact usa 'currency' y 'user_role' del row; ya están.
    enr = add_frame_features_from_artifact(df, stats)
    X = enr[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, enr


def score(X, model, scaler):
    Xs = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
    return (-model.decision_function(Xs)).astype(np.float64)


def validate_against_realtime(agg_df, stats, conn, calc, n=12):
    """Compara vectores batch vs real-time sobre n txns; imprime max diff."""
    prov = UserContextProvider(conn)
    sub = agg_df.head(n)
    X_batch, _ = enrich_and_vectorize(sub, stats)
    max_diffs = []
    for i, (_, r) in enumerate(sub.iterrows()):
        payment = {
            "id": int(r["id"]), "user_id": int(r["user_id"]),
            "effective_user_id": int(r["effective_user_id"]) if r.get("effective_user_id") else None,
            "facility_id": int(r["facility_id"]), "created_at": pd.Timestamp(r["created_at"]),
            "reservation_paid_out": float(r["amount"]),  # ya en USD; RT re-normaliza USD->USD (rate 1)
            "discount": float(r["discount_usd"]), "tip": float(r["tip"]),
            "currency": str(r["currency"] or "USD"), "gateway": str(r["gateway"] or "unknown"),
            "source_enum": str(r["source_enum"] or "unknown"),
            "club_credit_flag": bool(r["is_club_credit"]), "paid_by_manager": bool(r["paid_by_manager"]),
            "facility_time_zone_iana": FACILITY_499_IANA,
        }
        ctx = prov.get_context(payment["user_id"], payment["facility_id"],
                               payment["created_at"].to_pydatetime(), payment)
        vec_rt = calc.calculate(payment, ctx).astype(np.float64)
        d = np.abs(vec_rt - X_batch[i].astype(np.float64))
        max_diffs.append((int(r["id"]), float(d.max()), int(np.argmax(d))))
    print("  Validación batch↔real-time (primeras %d txns):" % n, flush=True)
    for tid, md, ai in max_diffs:
        flag = "" if md < 1e-3 else "  <-- REVISAR"
        print(f"    id={tid} max_diff={md:.2e} feat={FRAME_V1_FEATURE_NAMES[ai]}{flag}", flush=True)
    overall = max(m[1] for m in max_diffs)
    print(f"  max_diff global: {overall:.2e}", flush=True)
    return overall


def pct_report(scores, ref_sorted):
    n_ref = len(ref_sorted)
    idx = np.searchsorted(ref_sorted, scores, side="right")
    return 100.0 * idx / n_ref


def main():
    print("[1] Conexión + artefactos...", flush=True)
    conn = get_conn()
    client = conn.client
    stats = json.load(open(MODELS / "facility_stats_v1.json"))
    calc = FrameV1FeatureCalculator(stats, str(MODELS / "feature_engineer.joblib"))
    model = joblib.load(MODELS / "isolation_forest_frame_v1.joblib")
    scaler = joblib.load(MODELS / "scaler_frame_v1.joblib")

    print("[2] Extrayendo + agregando ATAQUE (499, 2026-05-13)...", flush=True)
    tgt_attack = client.query_df(build_target_sql(499, "2026-05-13 00:00:00", "2026-05-14 00:00:00"),
                                 parameters={"fid": 499, "start": "2026-05-13 00:00:00", "end": "2026-05-14 00:00:00"})
    print(f"    {len(tgt_attack)} txns objetivo", flush=True)
    agg_attack = compute_aggregates(client, tgt_attack, 499)

    print("[3] Extrayendo + agregando BASELINE (499, 2025-09-15)...", flush=True)
    tgt_base = client.query_df(build_target_sql(499, "2025-09-15 00:00:00", "2025-09-16 00:00:00"),
                               parameters={"fid": 499, "start": "2025-09-15 00:00:00", "end": "2025-09-16 00:00:00"})
    print(f"    {len(tgt_base)} txns objetivo", flush=True)
    agg_base = compute_aggregates(client, tgt_base, 499)

    print("[4] VALIDACIÓN batch vs real-time (muestra del ataque)...", flush=True)
    vdiff = validate_against_realtime(agg_attack, stats, conn, calc, n=12)

    print("[5] Enriqueciendo + puntuando...", flush=True)
    X_attack, enr_attack = enrich_and_vectorize(agg_attack, stats)
    X_base, _ = enrich_and_vectorize(agg_base, stats)
    s_attack = score(X_attack, model, scaler)
    s_base = score(X_base, model, scaler)

    print("[6] Distribución poblacional...", flush=True)
    ref = pd.read_parquet(REVISION / "test_scores_v2.parquet", columns=["score"])["score"].to_numpy()
    ref_sorted = np.sort(ref); n_ref = len(ref_sorted)
    p_attack = pct_report(s_attack, ref_sorted)
    p_base = pct_report(s_base, ref_sorted)
    thr10, thr5, thr1 = np.percentile(ref, [90, 95, 99])

    def topk(s):
        return (float((s >= thr1).mean()*100), float((s >= thr5).mean()*100), float((s >= thr10).mean()*100))
    a1, a5, a10 = topk(s_attack)
    b1, b5, b10 = topk(s_base)

    refund_mask = agg_attack["status"].isin(["totally_refunded", "refunded_to_credit"]).to_numpy()
    base_sorted = np.sort(s_base)
    pos_vs_base = 100.0 * np.searchsorted(base_sorted, s_attack, side="right") / len(base_sorted)

    print("\n" + "="*74)
    print("RESULTADOS — PRUEBA DE FUEGO card testing facility 499 (2026-05-13)")
    print("="*74)
    print(f"Vía: (b) agregados SQL batcheado (context.py 1:1) + add_frame_features_from_artifact")
    print(f"Validación batch↔real-time: max_diff={vdiff:.2e} ({'OK <1e-3' if vdiff<1e-3 else 'REVISAR'})")
    print(f"Modelo: IF frame-v1 (30 feats), score=-decision_function, clip[-10,10]")
    print(f"Poblacional: test_scores_v2.parquet n={n_ref:,} | thr top10>={thr10:.5f} top5>={thr5:.5f} top1>={thr1:.5f}")
    print()
    print(f"{'':<30}{'ATAQUE':>14}{'BASELINE 499':>16}")
    print(f"{'n txns':<30}{len(s_attack):>14}{len(s_base):>16}")
    print(f"{'score medio':<30}{s_attack.mean():>14.5f}{s_base.mean():>16.5f}")
    print(f"{'score mediano':<30}{np.median(s_attack):>14.5f}{np.median(s_base):>16.5f}")
    print(f"{'percentil pob MEDIO':<30}{p_attack.mean():>13.2f}%{p_base.mean():>15.2f}%")
    print(f"{'percentil pob MEDIANO':<30}{np.median(p_attack):>13.2f}%{np.median(p_base):>15.2f}%")
    print(f"{'% top-1% pob (azar 1%)':<30}{a1:>13.2f}%{b1:>15.2f}%")
    print(f"{'% top-5% pob (azar 5%)':<30}{a5:>13.2f}%{b5:>15.2f}%")
    print(f"{'% top-10% pob (azar 10%)':<30}{a10:>13.2f}%{b10:>15.2f}%")
    print()
    print(f"EF implícito ataque: top-1%={a1/1.0:.2f}x  top-5%={a5/5.0:.2f}x  top-10%={a10/10.0:.2f}x")
    print(f"Dentro del ataque: refunded={int(refund_mask.sum())} no-refunded={int((~refund_mask).sum())}")
    print(f"    percentil medio refunded:    {p_attack[refund_mask].mean():.2f}%")
    if (~refund_mask).sum() > 0:
        print(f"    percentil medio NO-refunded: {p_attack[~refund_mask].mean():.2f}%")
    print(f"Percentil medio ataque vs baseline-499: {pos_vs_base.mean():.2f}%")
    print("="*74)

    out = pd.DataFrame({
        "id": agg_attack["id"].values, "user_id": agg_attack["user_id"].values,
        "created_at": agg_attack["created_at"].values, "status": agg_attack["status"].values,
        "amount_usd": agg_attack["amount"].values, "account_age_days": agg_attack["user_account_age_days"].values,
        "score": s_attack, "pop_percentile": p_attack,
    })
    outp = REVISION / "card_testing_499_scores.csv"
    out.sort_values("pop_percentile", ascending=False).to_csv(outp, index=False)
    print(f"Guardado: {outp}")


if __name__ == "__main__":
    main()
