"""Plan A · Paso 1 — Decisión shadow: campeón IF-40-v1 vs retador frame-v1.

Compara los ARTEFACTOS DESPLEGADOS (no feature-sets re-entrenados inline):
  - Campeón:  isolation_forest_final.joblib + scaler_final.joblib + final_feature_list.json (40)
  - Retador:  isolation_forest_frame_v1.joblib + scaler_frame_v1.joblib + FRAME_V1 (30)

Recipe de scoring idéntico al scorer en vivo (scoring/scorer.py:183-190):
  X_scaled = scaler.transform(X); X_scaled = clip(X_scaled, -10, 10)
  score = -model.decision_function(X_scaled)   # higher = more anomalous

Evalúa los gates de decisión de docs/plan-normalizacion-marcos.md §A.1 sobre el
test set (Sep-Dic 2025), proxies SOLO para evaluación (nunca entrenamiento).

Salida: output/revision/shadow_decision_frame_v1.json
Uso:  ./venv/bin/python scripts/backtest_shadow.py [--sample-mod N]
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
OUT = ROOT / "output" / "revision" / "shadow_decision_frame_v1.json"

sys.path.insert(0, str(ROOT / "src"))
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402

# Reutilizar el build vectorizado con paridad verificada (retrain_frame_v1.py)
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_frame_v1 import NEEDED_COLS, add_frame_features_from_artifact  # noqa: E402

OFF_HOURS = [23, 0, 1, 2, 3, 4, 5, 6]

CHAMPION_FEATURES = json.loads((MODELS / "final_feature_list.json").read_text())
PROXY_COLS = ["same_amount_count_1h", "user_account_age_days",
              "user_txn_count_1h", "is_third_party_payment"]


def score_model(model, scaler, X: np.ndarray) -> np.ndarray:
    """Replica scoring/scorer.py: transform -> clip[-10,10] -> -decision_function."""
    Xs = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
    return -np.asarray(model.decision_function(Xs), dtype=np.float64)


def build_proxies(df: pd.DataFrame) -> dict:
    tipo_a = df["status"].isin(["totally_refunded", "refunded_to_credit"]).to_numpy()
    card = (df["same_amount_count_1h"] >= 3).to_numpy()
    newb = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third = ((df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    return {"tipo_a": tipo_a.astype(np.int8),
            "pure_fraud": (card | newb | third).astype(np.int8)}


def eval_scores(scores, proxies, amount) -> dict:
    order = np.argsort(-scores)
    k5 = max(1, int(np.ceil(len(scores) * 0.05)))
    k1 = max(1, int(np.ceil(len(scores) * 0.01)))
    amt_p999 = float(np.percentile(amount, 99.9))
    amt_wins = np.clip(amount, None, amt_p999)
    out = {
        "top5_amount_x_avg_raw": float(amount[order[:k5]].mean() / amount.mean()),
        "top5_amount_x_avg_winsorized_p999": float(
            amt_wins[order[:k5]].mean() / amt_wins.mean()),
    }
    for label, y in proxies.items():
        base = float(y.mean())
        out[label] = {
            "auc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "ef_1pct": float(y[order[:k1]].mean() / base) if base else 0.0,
            "ef_5pct": float(y[order[:k5]].mean() / base) if base else 0.0,
            "base_rate": base,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mod", type=int, default=10,
                    help="1/N del test set (default 10 ~250K filas)")
    args = ap.parse_args()
    t0 = time.perf_counter()

    stats = json.load(open(MODELS / "facility_stats_v1.json"))

    # Columnas: unión de features campeón, NEEDED_COLS frame, proxies y metadata
    load_cols = sorted(set(CHAMPION_FEATURES) | set(NEEDED_COLS) | set(PROXY_COLS)
                       | {"id", "facility_id", "created_at", "status", "currency", "amount",
                          "is_off_hours"})
    print(f"Cargando test set (muestra 1/{args.sample_mod})...")
    df = pd.read_parquet(DATA / "test_features_enriched.parquet", columns=load_cols)
    df = df[(df["id"] % args.sample_mod) == 0].reset_index(drop=True)
    print(f"  test n={len(df):,} ({time.perf_counter()-t0:.0f}s)")

    proxies = build_proxies(df)
    amount = df["amount"].to_numpy(dtype=np.float64)
    for k, y in proxies.items():
        print(f"  proxy {k}: rate={y.mean():.4f} n={int(y.sum()):,}")

    # --- Cobertura de datos (FrameFlags-like) ---
    known_fac = set(int(x) for x in stats["facilities"].keys())
    tz_missing = np.array([int(f) not in known_fac for f in df["facility_id"]]).mean()
    cur_missing = df["currency"].isna().mean() + (df["currency"].fillna("").str.strip() == "").mean()
    coverage = 1.0 - max(float(tz_missing), float(cur_missing))
    print(f"  cobertura: tz_missing={tz_missing:.4f} currency_missing={float(cur_missing):.4f} "
          f"-> coverage={coverage:.4f}")

    # --- Campeón IF-40-v1 ---
    print("Puntuando campeón IF-40-v1...")
    champ_model = joblib.load(MODELS / "isolation_forest_final.joblib")
    champ_scaler = joblib.load(MODELS / "scaler_final.joblib")
    Xc = np.nan_to_num(df[CHAMPION_FEATURES].to_numpy(dtype=np.float32),
                       nan=0.0, posinf=0.0, neginf=0.0)
    champ_scores = score_model(champ_model, champ_scaler, Xc)
    champ_eval = eval_scores(champ_scores, proxies, amount)
    champ_eval["off_hours_utc_pct"] = float(np.isin(df["is_off_hours"].to_numpy().astype(bool), [True]).mean()) \
        if df["is_off_hours"].dtype == bool else float(df["is_off_hours"].mean())

    # --- Retador frame-v1 ---
    print("Puntuando retador frame-v1 (build de features + score)...")
    t1 = time.perf_counter()
    df = add_frame_features_from_artifact(df, stats)
    print(f"  frame features listas ({time.perf_counter()-t1:.0f}s)")
    chal_model = joblib.load(MODELS / "isolation_forest_frame_v1.joblib")
    chal_scaler = joblib.load(MODELS / "scaler_frame_v1.joblib")
    Xf = np.nan_to_num(df[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32),
                       nan=0.0, posinf=0.0, neginf=0.0)
    chal_scores = score_model(chal_model, chal_scaler, Xf)
    chal_eval = eval_scores(chal_scores, proxies, amount)
    chal_eval["off_hours_local_pct"] = float(df["is_off_hours_loc"].mean())

    # --- Gates de decisión (§A.1) ---
    champ_auc_pf = champ_eval["pure_fraud"]["auc"]
    chal_auc_pf = chal_eval["pure_fraud"]["auc"]
    gates = {
        "size_bias_frame_lt_3x": {
            "value": chal_eval["top5_amount_x_avg_winsorized_p999"],
            "champion": champ_eval["top5_amount_x_avg_winsorized_p999"],
            "threshold": 3.0, "pass": chal_eval["top5_amount_x_avg_winsorized_p999"] < 3.0,
        },
        "offhours_local_band": {
            "value": chal_eval["off_hours_local_pct"],
            "champion_utc": champ_eval["off_hours_utc_pct"],
            "band": [0.03, 0.07],
            "pass": 0.03 <= chal_eval["off_hours_local_pct"] <= 0.07,
        },
        "auc_pure_fraud_not_inferior": {
            "value": chal_auc_pf, "champion": champ_auc_pf,
            "margin": 0.02, "pass": chal_auc_pf >= champ_auc_pf - 0.02,
        },
        "ef5_pure_fraud_ge_1_2_and_champion": {
            "value": chal_eval["pure_fraud"]["ef_5pct"],
            "champion": champ_eval["pure_fraud"]["ef_5pct"],
            "pass": (chal_eval["pure_fraud"]["ef_5pct"] >= 1.2
                     and chal_eval["pure_fraud"]["ef_5pct"] >= champ_eval["pure_fraud"]["ef_5pct"]),
        },
        "data_coverage_gt_98pct": {
            "value": coverage, "threshold": 0.98, "pass": coverage > 0.98,
        },
    }
    all_pass = all(g["pass"] for g in gates.values())

    result = {
        "generated_at_note": "sample-based backtest; test set Sep-Dic 2025",
        "sample_mod": args.sample_mod,
        "n_test": int(len(df)),
        "coverage": {"tz_missing": float(tz_missing), "currency_missing": float(cur_missing),
                     "coverage": coverage},
        "champion_IF40": champ_eval,
        "challenger_frame_v1": chal_eval,
        "gates": gates,
        "decision": "PROMOTE" if all_pass else "HOLD",
    }
    OUT.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 68)
    print(f"{'gate':<38}{'valor':>10}{'ref':>10}{'':>6}")
    print("-" * 68)
    for name, g in gates.items():
        ref = g.get("champion", g.get("champion_utc", g.get("threshold", "")))
        refs = f"{ref:.3f}" if isinstance(ref, float) else str(ref)
        print(f"{name:<38}{g['value']:>10.3f}{refs:>10}{'  PASS' if g['pass'] else '  FAIL'}")
    print("=" * 68)
    print(f"AUC pure_fraud:  campeón={champ_auc_pf:.4f}  frame-v1={chal_auc_pf:.4f}")
    print(f"AUC tipo_a:      campeón={champ_eval['tipo_a']['auc']:.4f}  "
          f"frame-v1={chal_eval['tipo_a']['auc']:.4f}")
    print(f"top5 amt (wins): campeón={champ_eval['top5_amount_x_avg_winsorized_p999']:.2f}x  "
          f"frame-v1={chal_eval['top5_amount_x_avg_winsorized_p999']:.2f}x")
    print(f"\nDECISIÓN: {result['decision']}  -> {OUT}")


if __name__ == "__main__":
    main()
