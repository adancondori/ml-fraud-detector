"""Entrena los modelos faltantes del confirmatorio V2 sobre FS-frame-v1 (30 features).

Reutiliza la MISMA receta y el MISMO scaler que el frame-v1 principal
(retrain_frame_v1.py). NO entrena un scaler nuevo: carga scaler_frame_v1.joblib.

Artefactos producidos:
  - output/models/isolation_forest_frame_v1_seed43.joblib  (IF, receta principal, random_state=43)
  - output/models/isolation_forest_frame_v1_seed44.joblib  (IF, receta principal, random_state=44)
  - output/models/lof_frame_v1.joblib                       (LOF novelty, fit sobre submuestra 100K)
  - output/models/ocsvm_frame_v1.joblib                     (OC-SVM rbf, fit sobre submuestra 100K)

Recetas:
  IF (seeds 43/44): IsolationForest(n_estimators=200, max_samples=512, max_features=0.6,
                    contamination="auto", random_state=SEED, n_jobs=-1)
  LOF (HE4):        LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
                    fit sobre submuestra aleatoria de 100.000 filas (random_state=42)
  OC-SVM (HE4):     OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
                    fit sobre la MISMA submuestra 100K (complejidad O(n^2)-O(n^3))

Pipeline de features: idéntico a retrain_frame_v1.py / backtest_shadow.py:
  - add_frame_features_from_artifact(df, stats)
  - saneo de montos aguas arriba (compute_amount_sanity_thresholds + sanitize_amount_df)
  - scaler.transform(X) -> clip[-10, 10]

Validación final: cada artefacto carga y puntúa (decision_function) una muestra
del test set sin error, con shape correcta.

Uso:  ./venv/bin/python scripts/train_frame_v1_extra.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "output" / "models"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402
from retrain_frame_v1 import NEEDED_COLS, add_frame_features_from_artifact  # noqa: E402

STATS_PATH = MODELS / "facility_stats_v1.json"
SCALER_PATH = MODELS / "scaler_frame_v1.joblib"
TRAIN_PARQUET = DATA / "train_features_enriched.parquet"
TEST_PARQUET = DATA / "test_features_enriched.parquet"

SUBSAMPLE_N = 100_000
SUBSAMPLE_SEED = 42


def build_frame_matrix(df: pd.DataFrame, stats: dict, scaler) -> np.ndarray:
    """add_frame_features -> matriz float32 -> scaler.transform -> clip[-10,10]."""
    df = add_frame_features_from_artifact(df, stats)
    X = df[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
    return Xs


def main():
    t_total = time.perf_counter()
    print("=" * 66)
    print("train_frame_v1_extra.py — modelos faltantes confirmatorio V2 (FS-frame-v1)")
    print("=" * 66)

    # ------------------------------------------------------------------
    # [1] Artefactos base (scaler REUTILIZADO — no re-entrenar)
    # ------------------------------------------------------------------
    print("\n[1] Cargando artefactos base...")
    t0 = time.perf_counter()
    stats = json.load(open(STATS_PATH))
    scaler = joblib.load(SCALER_PATH)
    print(f"  facility_stats_v1.json: {len(stats['facilities'])} facilities")
    print(f"  scaler_frame_v1.joblib: {type(scaler).__name__} "
          f"(n_features_in={getattr(scaler, 'n_features_in_', '?')}) — REUTILIZADO")
    print(f"  ({time.perf_counter()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # [2] Train set + frame features + scaler (mismo saneo que retrain_frame_v1)
    # ------------------------------------------------------------------
    print("\n[2] Cargando train set y construyendo matriz frame-v1...")
    t0 = time.perf_counter()
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=NEEDED_COLS)
    print(f"  train shape (raw): {train_df.shape} ({time.perf_counter()-t0:.1f}s)")

    # Saneo de montos aguas arriba (idéntico a retrain_frame_v1.py)
    amount_thresholds = DataManager.compute_amount_sanity_thresholds(train_df)
    train_df = DataManager.sanitize_amount_df(
        train_df, amount_thresholds, split_name="train", drop=True
    )
    print(f"  train shape (saneado): {train_df.shape} "
          f"({len(amount_thresholds)} monedas)")

    t0 = time.perf_counter()
    X_train = build_frame_matrix(train_df, stats, scaler)
    print(f"  X_train scaled shape: {X_train.shape} ({time.perf_counter()-t0:.1f}s)")
    del train_df

    # Submuestra fija (100K) para LOF y OC-SVM — declarada en la tesis
    n_train = X_train.shape[0]
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    sub_n = min(SUBSAMPLE_N, n_train)
    sub_idx = rng.choice(n_train, size=sub_n, replace=False)
    X_sub = X_train[sub_idx]
    print(f"  submuestra LOF/OC-SVM: {X_sub.shape} (seed={SUBSAMPLE_SEED})")

    saved = {}  # name -> path

    # ------------------------------------------------------------------
    # [3] IsolationForest seeds 43 y 44 (MISMA receta, cambia random_state)
    # ------------------------------------------------------------------
    for seed in (43, 44):
        print(f"\n[3] Entrenando IsolationForest random_state={seed}...")
        t0 = time.perf_counter()
        model = IsolationForest(
            n_estimators=200,
            max_samples=512,
            max_features=0.6,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        ).fit(X_train)
        dt = time.perf_counter() - t0
        path = MODELS / f"isolation_forest_frame_v1_seed{seed}.joblib"
        joblib.dump(model, path)
        saved[f"isolation_forest_frame_v1_seed{seed}"] = path
        print(f"  IF seed{seed} entrenado en {dt:.1f}s -> {path.name}")

    del X_train

    # ------------------------------------------------------------------
    # [4] LOF frame-v1 (HE4) — novelty=True, fit sobre submuestra 100K
    # ------------------------------------------------------------------
    print("\n[4] Entrenando LOF frame-v1 (novelty=True, submuestra 100K)...")
    t0 = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(X_sub)
    dt = time.perf_counter() - t0
    lof_path = MODELS / "lof_frame_v1.joblib"
    joblib.dump(lof, lof_path)
    saved["lof_frame_v1"] = lof_path
    print(f"  LOF entrenado en {dt:.1f}s -> {lof_path.name}")

    # ------------------------------------------------------------------
    # [5] OC-SVM frame-v1 (HE4) — misma submuestra 100K
    # ------------------------------------------------------------------
    print("\n[5] Entrenando OC-SVM frame-v1 (rbf, nu=0.05, gamma=scale, submuestra 100K)...")
    t0 = time.perf_counter()
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(X_sub)
    dt = time.perf_counter() - t0
    ocsvm_path = MODELS / "ocsvm_frame_v1.joblib"
    joblib.dump(ocsvm, ocsvm_path)
    saved["ocsvm_frame_v1"] = ocsvm_path
    print(f"  OC-SVM entrenado en {dt:.1f}s -> {ocsvm_path.name}")

    del X_sub

    # ------------------------------------------------------------------
    # [6] Validación: cada artefacto carga y puntúa muestra del test
    # ------------------------------------------------------------------
    print("\n[6] Validación: carga + decision_function sobre 1000 filas del test...")
    t0 = time.perf_counter()
    test_df = pd.read_parquet(TEST_PARQUET, columns=NEEDED_COLS).head(1000)
    X_test = build_frame_matrix(test_df, stats, scaler)
    print(f"  X_test (muestra) shape: {X_test.shape} ({time.perf_counter()-t0:.1f}s)")

    all_ok = True
    for name, path in saved.items():
        mdl = joblib.load(path)
        assert hasattr(mdl, "decision_function"), f"{name} SIN decision_function"
        scores = np.asarray(mdl.decision_function(X_test), dtype=np.float64)
        ok = scores.shape == (X_test.shape[0],) and np.isfinite(scores).all()
        all_ok = all_ok and ok
        size_mb = path.stat().st_size / 1e6
        print(f"  {name:38s} | decision_function OK shape={scores.shape} "
              f"| finite={np.isfinite(scores).all()} | {size_mb:.2f} MB | {'PASS' if ok else 'FAIL'}")

    print(f"\n{'='*66}")
    print(f"COMPLETADO en {time.perf_counter()-t_total:.0f}s — "
          f"{'TODOS PASAN' if all_ok else 'HAY FALLOS'}")
    for name, path in saved.items():
        print(f"  {path} ({path.stat().st_size/1e6:.2f} MB)")
    print(f"{'='*66}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
