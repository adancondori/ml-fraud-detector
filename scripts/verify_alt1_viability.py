"""Alternativa 1 · Pruebas de viabilidad — detección por capas (reglas + IF frame-v1).

Verifica, ANTES del refactor, que el cambio de paradigma es viable:

  T1. Artefactos frame-v1 cargan y puntúan con el recipe del scorer en vivo.
  T2. Métricas titulares nuevas (EF@k / P@k por proxy tipificado) sostienen el
      abandono de AUC-vs-reembolso como métrica de cabecera.
  T3. Capa de reglas: volúmenes de alerta operables por regla y por umbral.
  T4. Complementariedad reglas <-> IF: las capas cubren poblaciones distintas.

Proxies SOLO para evaluación (nunca entrenamiento). El proxy de reembolso
(tipo_a) se reporta únicamente como referencia histórica, NO como gate.

Salida: output/revision/alt1_viability.json
Uso:  ./venv/bin/python scripts/verify_alt1_viability.py [--sample-mod N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "output" / "models"
OUT = ROOT / "output" / "revision" / "alt1_viability.json"

sys.path.insert(0, str(ROOT / "src"))
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from retrain_frame_v1 import NEEDED_COLS, add_frame_features_from_artifact  # noqa: E402

TEST_DAYS = 122  # Sep-Dic 2025

RULE_COLS = [
    "same_amount_count_1h",
    "same_amount_count_24h",
    "user_account_age_days",
    "user_txn_count_1h",
    "user_txn_count_24h",
    "is_third_party_payment",
    "user_discount_ratio_30d",
]


def score_frame_v1(df: pd.DataFrame) -> np.ndarray:
    """T1: replica scoring/scorer.py: transform -> clip[-10,10] -> -decision_function."""
    stats = json.load(open(MODELS / "facility_stats_v1.json"))
    df = add_frame_features_from_artifact(df, stats)
    model = joblib.load(MODELS / "isolation_forest_frame_v1.joblib")
    scaler = joblib.load(MODELS / "scaler_frame_v1.joblib")
    X = df[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float64)
    Xs = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
    return -np.asarray(model.decision_function(Xs), dtype=np.float64), df


def build_typed_proxies(df: pd.DataFrame) -> dict:
    """Proxies tipificados (evaluación). Coinciden con backtest_shadow.build_proxies."""
    card = (df["same_amount_count_1h"] >= 3).to_numpy()
    newb = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third = ((df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    return {
        "card_testing": card.astype(np.int8),
        "new_user_burst": newb.astype(np.int8),
        "third_party_multi": third.astype(np.int8),
        "pure_fraud_union": (card | newb | third).astype(np.int8),
        # circular parcial: frame-v1 contiene discount_ratio -> solo descriptivo
        "discount_extreme_CIRC": (df["user_discount_ratio_30d"] > 1.0).to_numpy().astype(np.int8),
        "velocity_extreme": (df["user_txn_count_24h"] > 100).to_numpy().astype(np.int8),
        # referencia histórica, NO gate
        "tipo_a_refund_REF": df["status"]
        .isin(["totally_refunded", "refunded_to_credit"])
        .to_numpy()
        .astype(np.int8),
    }


def eval_typed(scores: np.ndarray, proxies: dict) -> dict:
    """T2: métricas titulares candidatas por proxy tipificado."""
    n = len(scores)
    order = np.argsort(-scores)
    k1, k5 = max(1, int(np.ceil(n * 0.01))), max(1, int(np.ceil(n * 0.05)))
    out = {}
    for label, y in proxies.items():
        base = float(y.mean())
        if base == 0:
            out[label] = {"base_rate": 0.0, "note": "sin positivos en muestra"}
            continue
        out[label] = {
            "base_rate": base,
            "p_1pct": float(y[order[:k1]].mean()),
            "ef_1pct": float(y[order[:k1]].mean() / base),
            "p_5pct": float(y[order[:k5]].mean()),
            "ef_5pct": float(y[order[:k5]].mean() / base),
            "ap": float(average_precision_score(y, scores)),
            "auc_ref_only": float(roc_auc_score(y, scores)),
        }
    return out


def eval_rule_volumes(df: pd.DataFrame, scale: int) -> dict:
    """T3: volumen de alertas por regla y umbral (estimado a población completa)."""
    n = len(df)
    variants = {
        "card_testing_ge3": df["same_amount_count_1h"] >= 3,
        "card_testing_ge5": df["same_amount_count_1h"] >= 5,
        "card_testing_ge10": df["same_amount_count_1h"] >= 10,
        "new_user_burst_3in1h": (df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3),
        "new_user_burst_5in1h": (df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 5),
        "third_party_multi": (df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2),
        "velocity_gt100_24h": df["user_txn_count_24h"] > 100,
        "discount_ratio_gt100pct": df["user_discount_ratio_30d"] > 1.0,
    }
    out = {}
    for name, mask in variants.items():
        rate = float(mask.mean())
        out[name] = {
            "rate": rate,
            "est_alerts_per_day_full_population": round(rate * n * scale / TEST_DAYS, 1),
        }
    return out


def eval_complementarity(scores: np.ndarray, df: pd.DataFrame, proxies: dict) -> dict:
    """T4: ¿las capas cubren poblaciones distintas?"""
    n = len(scores)
    k5 = max(1, int(np.ceil(n * 0.05)))
    if_top5 = np.zeros(n, dtype=bool)
    if_top5[np.argsort(-scores)[:k5]] = True
    rules_any = proxies["pure_fraud_union"].astype(bool) | proxies["velocity_extreme"].astype(bool)
    overlap = float((if_top5 & rules_any).sum() / if_top5.sum())
    union = if_top5 | rules_any
    return {
        "if_top5_flagged_by_rules_pct": overlap,
        "if_top5_net_new_behavioral_pct": 1.0 - overlap,
        "union_alert_rate": float(union.mean()),
        "recall_union_vs_tipo_a_ref": float(
            proxies["tipo_a_refund_REF"][union].sum() / max(1, proxies["tipo_a_refund_REF"].sum())
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mod", type=int, default=10)
    args = ap.parse_args()
    t0 = time.perf_counter()

    load_cols = sorted(
        set(NEEDED_COLS) | set(RULE_COLS) | {"id", "status", "amount", "facility_id", "currency"}
    )
    print(f"Cargando test set (muestra 1/{args.sample_mod})...")
    df = pd.read_parquet(DATA / "test_features_enriched.parquet", columns=load_cols)
    df = df[(df["id"] % args.sample_mod) == 0].reset_index(drop=True)
    print(f"  n={len(df):,} ({time.perf_counter()-t0:.0f}s)")

    print("T1: cargando artefactos frame-v1 y puntuando...")
    scores, df_feat = score_frame_v1(df)
    t1 = {
        "artifacts_loaded": True,
        "n_features": len(FRAME_V1_FEATURE_NAMES),
        "n_scored": int(len(scores)),
        "score_finite_pct": float(np.isfinite(scores).mean()),
    }
    print(f"  OK: {t1}")

    proxies = build_typed_proxies(df)
    print("T2: métricas tipificadas...")
    t2 = eval_typed(scores, proxies)
    print("T3: volúmenes de reglas...")
    t3 = eval_rule_volumes(df, args.sample_mod)
    print("T4: complementariedad...")
    t4 = eval_complementarity(scores, df, proxies)

    # Sesgos estructurales del top-5% (gates ya aprobados en shadow; re-chequeo barato)
    order = np.argsort(-scores)
    k5 = max(1, int(np.ceil(len(scores) * 0.05)))
    amt = df["amount"].to_numpy(dtype=np.float64)
    amt_w = np.clip(amt, None, np.percentile(amt, 99.9))
    bias = {
        "top5_amount_x_avg_winsorized_p999": float(amt_w[order[:k5]].mean() / amt_w.mean()),
        "top5_off_hours_local_pct": float(
            df_feat["is_off_hours_loc"].to_numpy()[order[:k5]].mean()
        ),
    }

    result = {
        "note": "Viabilidad Alternativa 1 — capas (reglas + IF frame-v1). "
        "Proxies solo evaluación; tipo_a solo referencia.",
        "sample_mod": args.sample_mod,
        "n_test_sample": int(len(df)),
        "t1_artifacts": t1,
        "t2_typed_metrics": t2,
        "t3_rule_volumes": t3,
        "t4_complementarity": t4,
        "top5_structural_bias": bias,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nEscrito: {OUT} ({result['elapsed_s']}s)")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
