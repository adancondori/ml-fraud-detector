from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detector.scoring.scorer import SingleTransactionScorer
from scorer.artifact_loader import load_artifacts
from scripts.score_payment import PaymentScorer


def test_if40_artifacts_match_feature_contract():
    artifacts = load_artifacts(Path("output/models"))

    assert artifacts.metadata["model_version"] == "IF-40-v1"
    assert artifacts.metadata["score_function"] == "decision_function"
    assert len(artifacts.feature_list) == 40
    assert artifacts.model.n_features_in_ == 40
    assert artifacts.scaler.n_features_in_ == 40
    assert artifacts.thresholds["threshold_version"] == "v2"


def test_if40_scores_match_offline_payment_scorer():
    df = pd.read_parquet("data/processed/val_features_enriched.parquet").head(10)
    artifacts = load_artifacts(Path("output/models"))
    scorer = SingleTransactionScorer(
        feature_engineer_path="output/models/feature_engineer.joblib",
        artifacts=artifacts,
    )
    offline_scores = PaymentScorer().score_frame(df, with_percentile=False)

    for _, row in df.iterrows():
        features = scorer._feature_calc.calculate_from_feature_row(row)
        online_score, _ = scorer.score_features(features)
        offline_score = float(
            offline_scores.loc[offline_scores["id"] == row["id"], "score"].iloc[0]
        )
        assert abs(online_score - offline_score) < 1e-7
