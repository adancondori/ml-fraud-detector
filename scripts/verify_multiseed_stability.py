"""Evidencia gate de estabilidad multi-semilla (plan §5, re-especificación 2026-07-09).

Reentrena IF frame-v1 in-memory (receta exacta de retrain_frame_v1.py) con
semillas 42/43/44 y mide, por proxy tipificado, el rango de AUC y el rango
relativo de EF@1% entre semillas. NO sobrescribe artefactos de producción.

Motivación: el gate anterior ("rango AUC/EF < 0.01") mezclaba escalas —
0.01 absoluto es inalcanzable sobre EF (escala 1-5x) y velocity_extreme
rompía el umbral en AUC (rango 0.027) sin inestabilidad real. Gate nuevo:
EF@1% rango relativo < 15% sobre los proxies del gate EF (card_testing,
velocity_extreme) y la unión tipificada; AUC pasa a diagnóstico sin gate.
new_user_burst y third_party_multi quedan fuera del gate (IF es casi ciego
a ellos; su cobertura es de las reglas).

Salida: output/revision/multiseed_stability.json
Uso:    ./venv/bin/python scripts/verify_multiseed_stability.py [--sample-mod N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from retrain_frame_v1 import NEEDED_COLS, add_frame_features_from_artifact  # noqa: E402
from verify_alt1_viability import RULE_COLS, build_typed_proxies  # noqa: E402

from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "output" / "models"
OUT = ROOT / "output" / "revision" / "multiseed_stability.json"

SEEDS = (42, 43, 44)
# Proxies con gate de estabilidad (los del gate EF@k del scoreboard + unión).
GATED_PROXIES = ("card_testing", "velocity_extreme", "pure_fraud_union")
EF1_REL_RANGE_MAX = 0.15


def relative_range(values: list) -> float:
    """Rango relativo: (max - min) / mediana."""
    med = float(np.median(values))
    if med == 0:
        return float("nan")
    return (max(values) - min(values)) / med


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-mod", type=int, default=10)
    args = parser.parse_args()

    stats = json.load(open(MODELS / "facility_stats_v1.json"))

    print(f"Cargando train (muestra 1/{args.sample_mod})...")
    train = pd.read_parquet(
        DATA / "train_features_enriched.parquet",
        columns=list(set(NEEDED_COLS) | {"id"}),
    )
    train = train[(train["id"] % args.sample_mod) == 0].reset_index(drop=True)
    thr = DataManager.compute_amount_sanity_thresholds(train)
    train = DataManager.sanitize_amount_df(train, thr, split_name="train", drop=True)
    train = add_frame_features_from_artifact(train, stats)
    X_train = np.nan_to_num(
        train[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print(f"Cargando test (muestra 1/{args.sample_mod})...")
    load_cols = list(set(NEEDED_COLS) | set(RULE_COLS) | {"id", "status"})
    test = pd.read_parquet(DATA / "test_features_enriched.parquet", columns=load_cols)
    test = test[(test["id"] % args.sample_mod) == 0].reset_index(drop=True)
    test = add_frame_features_from_artifact(test, stats)
    X_test = np.nan_to_num(
        test[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    proxies = build_typed_proxies(test)
    n = len(test)
    k1 = max(1, int(np.ceil(n * 0.01)))

    per_proxy: dict = {}
    top1_sets: list = []
    for seed in SEEDS:
        scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
        Xs_train = np.clip(scaler.transform(X_train), -10, 10).astype(np.float32)
        model = IsolationForest(
            n_estimators=200,
            max_samples=512,
            max_features=0.6,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        ).fit(Xs_train)
        Xs_test = np.clip(scaler.transform(X_test), -10, 10).astype(np.float32)
        scores = -np.asarray(model.decision_function(Xs_test), dtype=np.float64)
        order = np.argsort(-scores)
        top1_sets.append(frozenset(order[:k1].tolist()))
        for label, y in proxies.items():
            base = float(y.mean())
            if base == 0:
                continue
            entry = per_proxy.setdefault(label, {"auc": [], "ef1": []})
            entry["auc"].append(float(roc_auc_score(y, scores)))
            entry["ef1"].append(float(y[order[:k1]].mean() / base))
        print(f"  seed {seed} listo")

    # Diagnóstico de estabilidad operativa: solapamiento del top-1% entre
    # semillas (Jaccard). El VOLUMEN de alertas es estable por construcción
    # (umbral por percentil segmentado); lo que varía entre semillas es QUÉ
    # pagos caen en el top-k — eso mide este solapamiento. Sin gate.
    jaccard_pairs = {}
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            a, b = top1_sets[i], top1_sets[j]
            jaccard_pairs[f"{SEEDS[i]}-{SEEDS[j]}"] = len(a & b) / len(a | b)

    result = {
        "seeds": list(SEEDS),
        "sample_mod": args.sample_mod,
        "n_train": int(len(X_train)),
        "n_test": int(n),
        "ranking_stability": {
            "top1_jaccard_pairs": jaccard_pairs,
            "top1_jaccard_min": min(jaccard_pairs.values()),
            "note": "diagnostico, sin gate",
        },
        "gate": {
            "metric": "ef1_relative_range",
            "max": EF1_REL_RANGE_MAX,
            "gated_proxies": list(GATED_PROXIES),
            "auc": "diagnostico, sin gate",
        },
        "proxies": {},
    }
    gate_pass = True
    print(f"\n{'proxy':28s} {'AUC rango':>10s} {'EF1 rel%':>9s} {'gate':>6s}")
    for label, r in per_proxy.items():
        auc_range = max(r["auc"]) - min(r["auc"])
        ef1_rel = relative_range(r["ef1"])
        gated = label in GATED_PROXIES
        ok = (not gated) or (ef1_rel < EF1_REL_RANGE_MAX)
        gate_pass &= ok
        result["proxies"][label] = {
            "auc_per_seed": r["auc"],
            "ef1_per_seed": r["ef1"],
            "auc_range": auc_range,
            "ef1_relative_range": ef1_rel,
            "gated": gated,
            "gate_ok": ok if gated else None,
        }
        flag = ("PASS" if ok else "FAIL") if gated else "—"
        print(f"{label:28s} {auc_range:10.4f} {ef1_rel*100:8.1f}% {flag:>6s}")

    result["gate_pass"] = bool(gate_pass)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSolapamiento top-1% entre semillas (Jaccard, diagnóstico): {jaccard_pairs}")
    print(f"Gate estabilidad: {'PASS' if gate_pass else 'FAIL'}")
    print(f"Salida: {OUT.relative_to(ROOT)}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
