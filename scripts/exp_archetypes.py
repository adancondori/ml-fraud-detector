"""Plan B · Paso 1-2 — Tipificación SHAP de anomalías sobre frame-v1 (offline).

Sobre el artefacto frame-v1 (marco limpio, NO circular):
  1. Puntúa el test set, toma el top-5% de anomalías.
  2. TreeSHAP atribuye la contribución de cada feature al score de anomalía.
  3. Mapea features -> arquetipo (9 grupos), asigna arquetipo DOMINANTE por transacción
     (>=50% de la contribución positiva) o `mixed`.
  4. Reporta distribución de arquetipos + tabla cruzada descriptiva vs proxies
     (descriptivo, NUNCA objetivo de entrenamiento).

Salida: output/revision/archetype_report.json
Uso:  ./venv/bin/python scripts/exp_archetypes.py [--sample-mod N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "output" / "models"
OUT = ROOT / "output" / "revision" / "archetype_report.json"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402
from fraud_detector.scoring.archetypes import (  # noqa: E402
    ARCHETYPE_GROUPS as GROUPS,
    DEFAULT_DOMINANCE,
    assign_archetype,
)
from retrain_frame_v1 import NEEDED_COLS, add_frame_features_from_artifact  # noqa: E402

DOMINANCE = DEFAULT_DOMINANCE  # 0.35 (ajustado tras el prototipo con 0.50 -> 65.7% mixed)


def build_proxies(df: pd.DataFrame) -> dict:
    tipo_a = df["status"].isin(["totally_refunded", "refunded_to_credit"]).to_numpy()
    card = (df["same_amount_count_1h"] >= 3).to_numpy()
    newb = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third = ((df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    return {"tipo_a": tipo_a.astype(np.int8), "pure_fraud": (card | newb | third).astype(np.int8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mod", type=int, default=10)
    args = ap.parse_args()
    import shap
    t0 = time.perf_counter()

    stats = json.load(open(MODELS / "facility_stats_v1.json"))
    load_cols = sorted(set(NEEDED_COLS) | {"id", "facility_id", "created_at", "status",
                        "currency", "amount", "same_amount_count_1h", "user_account_age_days",
                        "user_txn_count_1h", "is_third_party_payment"})
    df = pd.read_parquet(DATA / "test_features_enriched.parquet", columns=load_cols)
    df = df[(df["id"] % args.sample_mod) == 0].reset_index(drop=True)
    df = add_frame_features_from_artifact(df, stats)
    print(f"test n={len(df):,} ({time.perf_counter()-t0:.0f}s)")

    model = joblib.load(MODELS / "isolation_forest_frame_v1.joblib")
    scaler = joblib.load(MODELS / "scaler_frame_v1.joblib")
    X = np.nan_to_num(df[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
    scores = -np.asarray(model.decision_function(Xs), dtype=np.float64)

    # Top-5% anomalías
    k5 = max(1, int(len(scores) * 0.05))
    top_idx = np.argsort(scores)[-k5:]
    print(f"top-5% = {len(top_idx):,} anomalías")

    # SHAP sobre el top-5% (TreeSHAP explica decision_function; anomalía = -decision_function,
    # así que negamos para que POSITIVO = hacia anomalía).
    t1 = time.perf_counter()
    explainer = shap.TreeExplainer(model)
    shap_top = explainer.shap_values(Xs[top_idx], check_additivity=False)
    shap_top = -np.asarray(shap_top)  # signo: positivo = empuja hacia anomalía
    print(f"SHAP top-5% listo ({time.perf_counter()-t1:.0f}s)")

    proxies = build_proxies(df)
    dominants = []
    group_totals = {g: 0.0 for g in GROUPS}
    for r in range(len(top_idx)):
        dom, gc, _top = assign_archetype(shap_top[r], dominance=DOMINANCE, top_k=2)
        dominants.append(dom)
        for g, v in gc.items():
            group_totals[g] += v

    dist = Counter(dominants)
    n = len(dominants)
    # Cruzada descriptiva: tasa de proxy dentro de cada arquetipo (NO es objetivo)
    cross = {}
    dom_arr = np.array(dominants)
    for arch in sorted(dist):
        mask = dom_arr == arch
        idx = top_idx[mask]
        cross[arch] = {
            "n": int(mask.sum()),
            "pct_of_top5": round(float(mask.mean()), 4),
            "tipo_a_rate": round(float(proxies["tipo_a"][idx].mean()), 4),
            "pure_fraud_rate": round(float(proxies["pure_fraud"][idx].mean()), 4),
        }

    report = {
        "note": "SHAP sobre frame-v1 (NO circular); proxies descriptivos, no objetivo de entrenamiento",
        "sample_mod": args.sample_mod, "n_test": int(len(df)),
        "n_top5": int(n), "dominance_threshold": DOMINANCE,
        "archetype_distribution": {a: round(dist[a] / n, 4) for a in sorted(dist)},
        "archetype_cross_proxy": cross,
        "mean_group_contribution": {g: round(group_totals[g] / n, 6) for g in GROUPS},
        "base_rates_top5": {
            "tipo_a": round(float(proxies["tipo_a"][top_idx].mean()), 4),
            "pure_fraud": round(float(proxies["pure_fraud"][top_idx].mean()), 4),
        },
    }
    OUT.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(f"{'arquetipo dominante':<20}{'% top-5%':>10}{'tipo_a':>9}{'pure_fr':>9}")
    print("-" * 60)
    for a in sorted(dist, key=lambda x: -dist[x]):
        c = cross[a]
        print(f"{a:<20}{c['pct_of_top5']*100:>9.1f}%{c['tipo_a_rate']*100:>8.1f}%"
              f"{c['pure_fraud_rate']*100:>8.1f}%")
    print("=" * 60)
    print(f"Guardado en {OUT}")


if __name__ == "__main__":
    main()
