"""Versioned artifact loading for the FastAPI scorer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class Artifacts:
    model: Any
    scaler: Any
    feature_list: list[str]
    thresholds: dict
    metadata: dict


def load_artifacts(model_dir: Path) -> Artifacts:
    """Load model artifacts atomically and validate their feature contract."""
    model_dir = Path(model_dir)
    metadata = _load_metadata(model_dir)
    files = metadata["artifact_files"]

    model = joblib.load(model_dir / files["model"])
    scaler = joblib.load(model_dir / files["scaler"])
    feature_list_path = model_dir / files["feature_list"]
    if feature_list_path.exists():
        feature_list = json.loads(feature_list_path.read_text())
    else:
        from fraud_detector.features.engineering import FEATURE_NAMES

        feature_list = list(FEATURE_NAMES)
    thresholds = json.loads((model_dir / files["thresholds"]).read_text())

    _validate_artifacts(model, scaler, feature_list, thresholds, metadata)
    return Artifacts(
        model=model,
        scaler=scaler,
        feature_list=feature_list,
        thresholds=thresholds,
        metadata=metadata,
    )


def _load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "model_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())

    legacy_features = model_dir / "final_feature_list.json"
    if legacy_features.exists():
        return {
            "model_version": "IF-40-v1",
            "feature_version": "enriched-40",
            "score_function": "decision_function",
            "threshold_version": "v2",
            "artifact_files": {
                "model": "isolation_forest_final.joblib",
                "scaler": "scaler_final.joblib",
                "feature_list": "final_feature_list.json",
                "thresholds": "thresholds_v2.json",
            },
        }

    return {
        "model_version": "IF-31-v1",
        "feature_version": "base-31",
        "score_function": "score_samples",
        "threshold_version": "v1",
        "artifact_files": {
            "model": "isolation_forest.joblib",
            "scaler": "scaler.joblib",
            "feature_list": "feature_list_legacy.json",
            "thresholds": "thresholds.json",
        },
    }


def _validate_artifacts(
    model: Any,
    scaler: Any,
    feature_list: list[str],
    thresholds: dict,
    metadata: dict,
) -> None:
    if not feature_list:
        raise ValueError("Artifact feature list is empty")

    model_features = getattr(model, "n_features_in_", None)
    if model_features is not None and int(model_features) != len(feature_list):
        raise ValueError(
            f"Model expects {model_features} features, feature list has {len(feature_list)}"
        )

    scaler_obj = getattr(scaler, "scaler", scaler)
    scaler_features = getattr(scaler_obj, "n_features_in_", None)
    if scaler_features is not None and int(scaler_features) != len(feature_list):
        raise ValueError(
            f"Scaler expects {scaler_features} features, feature list has {len(feature_list)}"
        )

    required_threshold_keys = {"binary_threshold", "score_percentiles"}
    missing_threshold_keys = required_threshold_keys - set(thresholds)
    if missing_threshold_keys:
        raise ValueError(f"Thresholds missing keys: {sorted(missing_threshold_keys)}")

    score_function = metadata.get("score_function")
    if score_function not in {"decision_function", "score_samples"}:
        raise ValueError(f"Unsupported score_function={score_function!r}")
