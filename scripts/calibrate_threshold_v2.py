"""Calibrate IF-40 thresholds on the validation split.

This intentionally uses validation, not test. The test split remains reserved for
final reporting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_payment import PaymentScorer


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "output" / "models"
VAL_PATH = ROOT / "data" / "processed" / "val_features_enriched.parquet"


def main() -> None:
    scorer = PaymentScorer()
    df_val = pd.read_parquet(VAL_PATH)
    out = scorer.score_frame(df_val, with_percentile=False)
    scores = out["score"].to_numpy(dtype=np.float64)

    threshold = float(np.percentile(scores, 95))
    percentiles = [float(np.percentile(scores, i / 10.0)) for i in range(1001)]

    thresholds = {
        "binary_threshold": threshold,
        "threshold_source": "percentile_95_validation_set",
        "model_version": "IF-40-v1",
        "feature_version": "enriched-40",
        "score_function": "decision_function",
        "threshold_version": "v2",
        "calibration_date": datetime.now(timezone.utc).isoformat(),
        "calibration_rows": int(len(scores)),
        "expected_anomaly_rate": float((scores > threshold).mean()),
        "score_percentiles": percentiles,
    }

    out_path = MODELS_DIR / "thresholds_v2.json"
    out_path.write_text(json.dumps(thresholds, indent=2))
    print(
        f"Wrote {out_path} threshold={threshold:.6f} "
        f"rows={len(scores):,} anomaly_rate={(scores > threshold).mean():.4f}"
    )


if __name__ == "__main__":
    main()
