#!/usr/bin/env python3
"""Construye el parquet de entrada para eval_scoreboard.py (confirmatorio V2).

Puntúa el test set con IF frame-v1 y adjunta los campos externos que consumen
las reglas tipificadas (rules.py). Parametrizado para smoke (muestra 1/N) y
corrida completa. Opcionalmente adjunta failed_count_1h (de failed_payment_logs),
scores LOF/OC-SVM frame-v1 y scores multi-semilla si los artefactos existen.

Uso:
  python scripts/build_v2_scores.py --sample-mod 10 --out output/revision/test_scores_v2_sample.parquet
  python scripts/build_v2_scores.py --sample-mod 1  --with-failed --out output/revision/test_scores_v2.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from retrain_frame_v1 import NEEDED_COLS, add_frame_features_from_artifact  # noqa: E402
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "output" / "models"
AUX = ROOT / "output" / "extended" / "auxiliary"

# Campos externos que consumen las reglas (rules.py) — NO son features del modelo
RULE_FIELDS = [
    "same_amount_count_1h",
    "user_txn_count_24h",
    "user_account_age_days",
    "user_txn_count_1h",
    "status",
]


def score_with(model_path, scaler_path, X):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    Xs = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
    return (-model.decision_function(Xs)).astype(np.float64)


def compute_failed_count_1h(df):
    """failed_count_1h = # de fallos del mismo usuario en la ventana [t-1h, t].

    Se aproxima uniendo por user_id (el test no trae user_token_id): para cada
    pago, cuenta fallos del mismo usuario en la ventana [t-1h, t]. Conservador.

    Implementación vectorizada: itera por USUARIO (no por fila). Para cada
    usuario ordena sus timestamps de fallo una vez y usa dos np.searchsorted
    por lote de pagos del usuario:
      - idx_hi = # fallos con ts <= t          (searchsorted 'right' en t)
      - idx_lo = # fallos con ts <  t - 1h     (searchsorted 'left'  en t-1h)
      - count  = idx_hi - idx_lo
    Sólo se recorren usuarios presentes en el test que además tienen fallos;
    los demás quedan en 0. Escala a millones de filas en pocos minutos.
    """
    fp = pd.read_parquet(AUX / "failed_payment_logs_with_user.parquet")
    # columnas esperadas: user_id, created_at (fallos)
    tcol = "created_at" if "created_at" in fp.columns else next(
        c for c in fp.columns if "creat" in c.lower()
    )
    fp = fp[["user_id", tcol]].copy()
    fp[tcol] = pd.to_datetime(fp[tcol], utc=True, errors="coerce")
    fp = fp.dropna(subset=["user_id", tcol])

    out = np.zeros(len(df), dtype=np.float32)

    # Timestamps del test (int64 ns; NaT -> se ignoran vía máscara)
    df_t = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    t_ns = df_t.to_numpy(dtype="datetime64[ns]").view("int64")  # NaT == NaT sentinel
    valid_t = ~df_t.isna().to_numpy()
    uids = df["user_id"].to_numpy()
    win_ns = np.int64(3600) * 1_000_000_000  # 1 hora en ns

    # Fallos agrupados por usuario, ya como int64 ns ordenados
    fp_uid = fp["user_id"].to_numpy()
    fp_ns = fp[tcol].to_numpy(dtype="datetime64[ns]").view("int64")
    order = np.argsort(fp_uid, kind="stable")
    fp_uid = fp_uid[order]
    fp_ns = fp_ns[order]
    # Fronteras de cada bloque de usuario en el array de fallos
    uniq_fail_uids, starts = np.unique(fp_uid, return_index=True)
    ends = np.append(starts[1:], len(fp_uid))
    fail_ranges = {u: (s, e) for u, s, e in zip(uniq_fail_uids, starts, ends)}

    # Índices de las filas del test agrupados por usuario (una pasada)
    test_order = np.argsort(uids, kind="stable")
    uids_sorted = uids[test_order]
    uniq_test_uids, t_starts = np.unique(uids_sorted, return_index=True)
    t_ends = np.append(t_starts[1:], len(uids_sorted))

    for u, ts, te in zip(uniq_test_uids, t_starts, t_ends):
        rng = fail_ranges.get(u)
        if rng is None:
            continue  # usuario sin fallos -> count 0
        fs, fe = rng
        fails = fp_ns[fs:fe]  # ya ordenados asc (dentro del bloque de usuario)
        # Nota: el ordenamiento por (user_id) es estable pero NO ordena los
        # timestamps dentro del usuario; ordenarlos explícitamente.
        fails = np.sort(fails)

        rows = test_order[ts:te]
        row_t = t_ns[rows]
        row_valid = valid_t[rows]
        idx_hi = np.searchsorted(fails, row_t, side="right")
        idx_lo = np.searchsorted(fails, row_t - win_ns, side="left")
        counts = (idx_hi - idx_lo).astype(np.float32)
        counts[~row_valid] = 0.0
        out[rows] = counts

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mod", type=int, default=10, help="1=completo, 10=muestra 1/10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-failed", action="store_true", help="computar failed_count_1h (lento)")
    ap.add_argument("--with-comparators", action="store_true", help="adjuntar LOF/OC-SVM frame-v1 si existen")
    ap.add_argument("--with-seeds", action="store_true", help="adjuntar scores seed 43/44 si existen")
    args = ap.parse_args()

    print(f"[1/5] Cargando test set (sample-mod={args.sample_mod})...", flush=True)
    df = pd.read_parquet(DATA / "test_features_enriched.parquet")
    if args.sample_mod > 1:
        df = df[(df["id"] % args.sample_mod) == 0].reset_index(drop=True)
    print(f"      {len(df):,} filas", flush=True)

    print("[2/5] Construyendo features frame-v1...", flush=True)
    stats = json.load(open(MODELS / "facility_stats_v1.json"))
    keep = list(dict.fromkeys(NEEDED_COLS + RULE_FIELDS + ["id", "user_id"]))
    keep = [c for c in keep if c in df.columns]
    sub = df[keep].copy()
    sub = add_frame_features_from_artifact(sub, stats)

    print("[3/5] Puntuando IF frame-v1...", flush=True)
    X = sub[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    score = score_with(
        MODELS / "isolation_forest_frame_v1.joblib",
        MODELS / "scaler_frame_v1.joblib", X,
    )
    print(f"      score mean={score.mean():.4f} std={score.std():.4f}", flush=True)

    out = pd.DataFrame({
        "id": sub["id"].values,
        "user_id": sub["user_id"].values,
        "status": sub["status"].astype(str).values,
        "score": score,
        "same_amount_count_1h": sub["same_amount_count_1h"].fillna(0).to_numpy(np.float32),
        "user_txn_count_24h": sub["user_txn_count_24h"].fillna(0).to_numpy(np.float32),
        "user_account_age_days": sub["user_account_age_days"].fillna(0).to_numpy(np.int32),
        "user_txn_count_1h": sub["user_txn_count_1h"].fillna(0).to_numpy(np.float32),
    })

    if args.with_failed:
        print("[3b] Computando failed_count_1h...", flush=True)
        out["failed_count_1h"] = compute_failed_count_1h(
            df[["user_id", "created_at"]].assign(created_at=df["created_at"]))
    else:
        out["failed_count_1h"] = np.zeros(len(out), dtype=np.float32)

    if args.with_comparators:
        for name in ["lof", "ocsvm"]:
            mp = MODELS / f"{name}_frame_v1.joblib"
            sp = MODELS / f"scaler_frame_v1.joblib"
            if mp.exists():
                print(f"[3c] Puntuando {name} frame-v1...", flush=True)
                out[f"score_{name}"] = score_with(mp, sp, X)

    if args.with_seeds:
        # El scaler es determinista (RobustScaler); los seeds solo cambian el
        # random_state del IsolationForest. Reusar el scaler principal si no hay
        # uno específico por semilla.
        for seed in [43, 44]:
            mp = MODELS / f"isolation_forest_frame_v1_seed{seed}.joblib"
            sp = MODELS / f"scaler_frame_v1_seed{seed}.joblib"
            if not sp.exists():
                sp = MODELS / "scaler_frame_v1.joblib"
            if mp.exists():
                print(f"[3d] Puntuando seed {seed}...", flush=True)
                out[f"score_seed{seed}"] = score_with(mp, sp, X)
        out["score_seed42"] = score  # el score principal es seed 42

    print("[4/5] Escribiendo parquet...", flush=True)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outp, engine="pyarrow", compression="snappy")
    print(f"[5/5] Listo: {outp}  ({len(out):,} filas, cols={list(out.columns)})", flush=True)


if __name__ == "__main__":
    main()
