"""Offline script to calibrate segmented thresholds from frame-v1 model scores.

Computes scores using the trained isolation_forest_frame_v1.joblib model applied
to the validation set (val_features_enriched.parquet), then fits
SegmentedThresholdCalibrator to produce thresholds_segmented_v1.json.

CRITICAL: scores come from -model.decision_function (frame-v1 p95 ≈ 0.0436).
DO NOT use output/scores/ files — those contain IF-40 scores (p95 ≈ 0.024).

GUARDRAIL: global binary_threshold must fall in [0.040, 0.048].
If it falls near 0.024 the wrong scores were used — the script will abort.

Artifacts produced:
    output/models/thresholds_segmented_v1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fraud_detector.calibration.segmented import SegmentedThresholdCalibrator
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES

# Import the vectorized feature-adder from retrain_frame_v1 without executing main()
# retrain_frame_v1.py protects its main() with "if __name__ == '__main__'", so
# importing it here only loads function/constant definitions.
from retrain_frame_v1 import add_frame_features_from_artifact, NEEDED_COLS  # type: ignore[import]

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

MODELS = ROOT / "output" / "models"
DATA = ROOT / "data" / "processed"

MODEL_PATH = MODELS / "isolation_forest_frame_v1.joblib"
SCALER_PATH = MODELS / "scaler_frame_v1.joblib"
STATS_PATH = MODELS / "facility_stats_v1.json"
VAL_PARQUET = DATA / "val_features_enriched.parquet"
OUTPUT_PATH = MODELS / "thresholds_segmented_v1.json"

# Guardrail: global p95 of frame-v1 scores must be in this range.
# If it falls near 0.024, IF-40 scores were used by mistake — abort.
GUARDRAIL_LOW = 0.040
GUARDRAIL_HIGH = 0.048


def main() -> None:
    print("=" * 60)
    print("calibrate_segmented_thresholds.py — Offline calibration")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    print("\n[1/4] Loading artifacts...")
    with open(STATS_PATH) as f:
        stats = json.load(f)
    print(f"  facility_stats_v1.json: {len(stats['facilities'])} entries")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  Scaler: {SCALER_PATH.name}")

    # ------------------------------------------------------------------
    # Load val set and compute frame-v1 features
    # ------------------------------------------------------------------
    print("\n[2/4] Loading val set and computing frame features...")
    val_df = pd.read_parquet(VAL_PARQUET, columns=NEEDED_COLS)
    print(f"  val shape (raw): {val_df.shape}")

    val_df = add_frame_features_from_artifact(val_df, stats)
    print(f"  frame features added: {val_df.shape}")

    X = val_df[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = np.clip(scaler.transform(X), -10, 10).astype(np.float32)

    # ------------------------------------------------------------------
    # Compute frame-v1 scores (NOT IF-40 scores from output/scores/)
    # decision_function returns negative anomaly scores in sklearn:
    # more anomalous = more negative, so we negate to get positive scores
    # where higher = more anomalous. Identical to retrain_frame_v1.py:508.
    # ------------------------------------------------------------------
    print("\n[3/4] Computing frame-v1 scores via decision_function...")
    scores = -model.decision_function(X_scaled)
    p95 = float(np.percentile(scores, 95.0))
    print(f"  scores shape: {scores.shape}")
    print(f"  scores p95: {p95:.6f}")
    print(f"  scores p50: {float(np.percentile(scores, 50.0)):.6f}")
    print(f"  scores p99: {float(np.percentile(scores, 99.0)):.6f}")

    # GUARDRAIL: abort if scores look like IF-40 (p95 ≈ 0.024)
    if not (GUARDRAIL_LOW <= p95 <= GUARDRAIL_HIGH):
        raise RuntimeError(
            f"GUARDRAIL FAILED: global p95 = {p95:.6f} is not in "
            f"[{GUARDRAIL_LOW}, {GUARDRAIL_HIGH}]. "
            f"Expected frame-v1 scores (p95 ≈ 0.0436). "
            f"If p95 ≈ 0.024, IF-40 scores were loaded by mistake. "
            f"Check that isolation_forest_frame_v1.joblib is the correct model."
        )
    print(f"  GUARDRAIL PASSED: p95 = {p95:.6f} in [{GUARDRAIL_LOW}, {GUARDRAIL_HIGH}]")

    # ------------------------------------------------------------------
    # Fit segmented calibrator
    # ------------------------------------------------------------------
    print("\n[4/4] Fitting SegmentedThresholdCalibrator (MIN_N=200)...")
    facility_ids = val_df["facility_id"].to_numpy()
    currencies = val_df["currency"].to_numpy(dtype=str)

    calibrator = SegmentedThresholdCalibrator()
    thresholds = calibrator.fit(
        scores=scores,
        facility_ids=facility_ids,
        currencies=currencies,
        percentile=95.0,
    )

    # ------------------------------------------------------------------
    # Post-fit assertions
    # ------------------------------------------------------------------
    print("\n  Assertions...")

    n_facilities = len(thresholds["by_facility"])
    n_currencies = len(thresholds["by_currency"])

    assert GUARDRAIL_LOW <= thresholds["binary_threshold"] <= GUARDRAIL_HIGH, (
        f"binary_threshold={thresholds['binary_threshold']} not in guardrail range"
    )
    assert {"binary_threshold", "score_percentiles"} <= set(thresholds), (
        "Root-level backward-compat keys missing"
    )
    assert n_currencies == 17, (
        f"Expected 17 currency entries (MXN n=88 and INR n=2 fall to global), "
        f"got {n_currencies}"
    )
    assert all(v["n"] >= 200 for v in thresholds["by_facility"].values()), (
        "Some facility entries have n < 200"
    )
    assert all(v["n"] >= 200 for v in thresholds["by_currency"].values()), (
        "Some currency entries have n < 200"
    )

    # Warn if facility count deviates significantly from expected ~452
    if not (400 <= n_facilities <= 500):
        print(
            f"  WARNING: by_facility has {n_facilities} entries "
            f"(expected ~452; check val set composition)"
        )

    # ------------------------------------------------------------------
    # Write artifact
    # ------------------------------------------------------------------
    with open(OUTPUT_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n  Written: {OUTPUT_PATH}")

    print("\n  Summary:")
    print(f"    global p95:  {thresholds['binary_threshold']:.6f}")
    print(f"    by_facility: {n_facilities}")
    print(f"    by_currency: {n_currencies}")
    print(f"    val rows:    {thresholds['calibration_rows']}")
    print("\n  OK all assertions passed.")

    # Final output lines for verification
    print(f"global p95: {thresholds['binary_threshold']}")
    print(f"by_facility: {n_facilities} by_currency: {n_currencies}")


if __name__ == "__main__":
    main()
