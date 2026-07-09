"""Puntúa el set dorado (output/golden_set_v0.parquet) con el path frame-v1.

frame-normalization-v1 (task 3.6 — scenario rollback/restaura-scoring):
permite verificar que restaurar artefactos previos reproduce los scores
previos del set dorado. El scoring usa exactamente el mismo path offline que
la calibración (add_frame_features_from_artifact + scaler + IF frame-v1).

Uso:
  ./venv/bin/python scripts/score_golden_frame_v1.py \
      --stats output/models/facility_stats_v1.json \
      --out /tmp/golden_scores_baseline.npy

Imprime el SHA-256 del array de scores (float64, orden del parquet) para
comparación exacta entre corridas con distintos artefactos de stats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

GOLDEN_PARQUET = ROOT / "output" / "golden_set_v0.parquet"
MODEL_PATH = ROOT / "output" / "models" / "isolation_forest_frame_v1.joblib"
SCALER_PATH = ROOT / "output" / "models" / "scaler_frame_v1.joblib"


def main() -> None:
    from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES
    from retrain_frame_v1 import NEEDED_COLS, add_frame_features_from_artifact

    parser = argparse.ArgumentParser(description="Score golden set con frame-v1")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with open(args.stats) as f:
        stats = json.load(f)

    df = pd.read_parquet(GOLDEN_PARQUET, columns=NEEDED_COLS)
    df = add_frame_features_from_artifact(df, stats)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = df[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = np.clip(scaler.transform(X), -10, 10).astype(np.float32)
    scores = (-model.decision_function(X_scaled)).astype(np.float64)

    digest = hashlib.sha256(scores.tobytes()).hexdigest()
    print(f"rows: {len(scores)}")
    print(f"stats: {args.stats}")
    print(f"scores_sha256: {digest}")
    print(f"p50={np.percentile(scores, 50):.8f} p95={np.percentile(scores, 95):.8f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, scores)
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
