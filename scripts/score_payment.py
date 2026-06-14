#!/usr/bin/env python
"""Score a payment (single or batch) with the final IF-40 model.

Usage (CLI):
    python scripts/score_payment.py --payment-id 1234567
    python scripts/score_payment.py --batch data/processed/some_payments.parquet
    python scripts/score_payment.py --batch some.parquet --top 1000

Usage (programmatic):
    from scripts.score_payment import score_payments, PaymentScorer
    scorer = PaymentScorer()
    df_scored = scorer.score_frame(df_payments_with_features)

Expected input columns: the 40 features used by the winning model
(see output/models/final_feature_list.json).

Output:
    a DataFrame with columns: id, score, decile (0=safest, 9=most anomalous),
    is_top_1pct, is_top_5pct.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detector.utils.logger import logger  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "output" / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class PaymentScorer:
    """Wraps the trained IF + scaler + feature list. Scores higher = more anomalous."""

    def __init__(
        self,
        model_path: Path | str = MODELS_DIR / "isolation_forest_final.joblib",
        scaler_path: Path | str = MODELS_DIR / "scaler_final.joblib",
        feature_list_path: Path | str = MODELS_DIR / "final_feature_list.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.feature_list_path = Path(feature_list_path)
        for p in (self.model_path, self.scaler_path, self.feature_list_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing artifact: {p}. "
                    "Run scripts/validate_final_model.py first to train the model."
                )
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.features = json.loads(self.feature_list_path.read_text())
        logger.info(f"Loaded model with {len(self.features)} features")

    def score_frame(self, df: pd.DataFrame, *, with_percentile: bool = True) -> pd.DataFrame:
        """Score every row. Returns a NEW DataFrame with id, score, decile, top flags."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        X = df[self.features].to_numpy(dtype=np.float64)
        X = self.scaler.transform(X).astype(np.float32)
        X = np.clip(X, -10, 10)
        scores = -np.asarray(self.model.decision_function(X), dtype=np.float64)

        out = pd.DataFrame({"id": df["id"].to_numpy() if "id" in df.columns else np.arange(len(df))})
        out["score"] = scores
        if with_percentile:
            order = np.argsort(-scores)
            rank = np.empty_like(order)
            rank[order] = np.arange(len(scores))
            pct_rank = (rank + 1) / len(scores)
            out["percentile_rank"] = pct_rank.astype(np.float32)
            out["decile"] = (pct_rank * 10).astype(np.int8).clip(0, 9)
            out["is_top_1pct"] = (pct_rank <= 0.01).astype(np.int8)
            out["is_top_5pct"] = (pct_rank <= 0.05).astype(np.int8)
        return out

    def score_payment_id(self, payment_id: int, df_enriched: pd.DataFrame) -> dict:
        """Score a single payment looked up by id from an enriched DataFrame."""
        row = df_enriched[df_enriched["id"] == payment_id]
        if row.empty:
            raise ValueError(f"payment_id {payment_id} not found")
        result = self.score_frame(row, with_percentile=False)
        return result.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score payments with the final IF model")
    parser.add_argument("--payment-id", type=int, help="Single payment id to look up in test enriched parquet")
    parser.add_argument("--batch", type=str, help="Path to a parquet/csv with the 40 features + id")
    parser.add_argument("--top", type=int, default=None, help="Output only top N most-anomalous")
    parser.add_argument("--out", type=str, default=None, help="Output file (csv/parquet); else stdout")
    args = parser.parse_args()

    scorer = PaymentScorer()

    if args.payment_id is not None:
        df_enriched = pd.read_parquet(DATA_DIR / "test_features_enriched.parquet")
        result = scorer.score_payment_id(args.payment_id, df_enriched)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.batch:
        path = Path(args.batch)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        logger.info(f"Loaded {len(df):,} rows from {path}")
        out = scorer.score_frame(df)
        if args.top:
            out = out.nlargest(args.top, "score")
        if args.out:
            outp = Path(args.out)
            if outp.suffix == ".parquet":
                out.to_parquet(outp, index=False)
            else:
                out.to_csv(outp, index=False)
            logger.info(f"Wrote {len(out):,} rows to {outp}")
        else:
            print(out.to_string(index=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
