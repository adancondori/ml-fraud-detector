"""Versioned artifact loading for the FastAPI scorer."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib


@dataclass(frozen=True)
class Artifacts:
    model: Any
    scaler: Any
    feature_list: list[str]
    thresholds: dict
    metadata: dict
    facility_stats: Optional[dict] = field(default=None)
    thresholds_segmented: Optional[dict] = field(default=None)


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

    # Optional: load facility_stats and thresholds_segmented if referenced in metadata.
    # For the IF-40 legacy path these keys are absent → both remain None (backward compat).
    facility_stats: Optional[dict] = None
    stats_artifact = metadata.get("stats_artifact")
    if stats_artifact:
        stats_path = model_dir / stats_artifact
        if stats_path.exists():
            facility_stats = json.loads(stats_path.read_text())

    thresholds_segmented: Optional[dict] = None
    seg_artifact = metadata.get("thresholds_segmented_artifact")
    if seg_artifact:
        seg_path = model_dir / seg_artifact
        if seg_path.exists():
            thresholds_segmented = json.loads(seg_path.read_text())

    if facility_stats is not None and thresholds_segmented is None:
        warnings.warn(
            "facility_stats loaded but thresholds_segmented is None — "
            "check that thresholds_segmented_artifact is set in metadata.",
            stacklevel=2,
        )

    _validate_artifacts(model, scaler, feature_list, thresholds, metadata)
    return Artifacts(
        model=model,
        scaler=scaler,
        feature_list=feature_list,
        thresholds=thresholds,
        metadata=metadata,
        facility_stats=facility_stats,
        thresholds_segmented=thresholds_segmented,
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
