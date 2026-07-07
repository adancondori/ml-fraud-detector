"""Tests for Artifacts dataclass and load_artifacts — retrocompat + frame-v1 extension."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scorer.artifact_loader import Artifacts, load_artifacts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_DIR = Path("output/models")

FRAME_V1_METADATA = MODEL_DIR / "model_metadata_frame_v1.json"
FRAME_V1_FILES = [
    "isolation_forest_frame_v1.joblib",
    "scaler_frame_v1.joblib",
    "feature_list_frame_v1.json",
    "thresholds_segmented_v1.json",
    "facility_stats_v1.json",
]


def _make_frame_v1_dir(tmp_path: Path) -> Path:
    """
    Create a temp directory that mimics a frame-v1 model directory:
    - model_metadata.json  ← the frame-v1 metadata (renamed so _load_metadata picks it up)
    - all artifact files symlinked from output/models/
    """
    meta = json.loads(FRAME_V1_METADATA.read_text())
    (tmp_path / "model_metadata.json").write_text(json.dumps(meta, indent=2))

    for fname in FRAME_V1_FILES:
        src = MODEL_DIR / fname
        if src.exists():
            dst = tmp_path / fname
            shutil.copy2(src, dst)

    return tmp_path


# ---------------------------------------------------------------------------
# Legacy backward-compat tests
# ---------------------------------------------------------------------------


class TestArtifactsLegacyDefaults:
    """Artifacts without the new optional fields behaves identically to pre-02-03."""

    def test_legacy_fields_default_to_none(self):
        """Constructing Artifacts without facility_stats/thresholds_segmented → None."""
        # We use IF-40 artifacts since that's what output/models currently resolves to
        artifacts = load_artifacts(MODEL_DIR)
        # IF-40 metadata has no stats_artifact / thresholds_segmented_artifact
        # → both new fields must be None
        assert artifacts.facility_stats is None
        assert artifacts.thresholds_segmented is None

    def test_legacy_if40_still_loads(self):
        """IF-40 path: model_version, feature count, score_function intact."""
        artifacts = load_artifacts(MODEL_DIR)
        assert artifacts.metadata["model_version"] == "IF-40-v1"
        assert artifacts.metadata["score_function"] == "decision_function"
        assert len(artifacts.feature_list) == 40
        assert artifacts.model.n_features_in_ == 40

    def test_artifacts_dataclass_accepts_none_new_fields(self):
        """Direct construction: new optional fields accept None explicitly."""
        import joblib

        artifacts = load_artifacts(MODEL_DIR)
        rebuilt = Artifacts(
            model=artifacts.model,
            scaler=artifacts.scaler,
            feature_list=artifacts.feature_list,
            thresholds=artifacts.thresholds,
            metadata=artifacts.metadata,
            facility_stats=None,
            thresholds_segmented=None,
        )
        assert rebuilt.facility_stats is None
        assert rebuilt.thresholds_segmented is None


# ---------------------------------------------------------------------------
# Frame-v1 extension tests
# ---------------------------------------------------------------------------


class TestArtifactsFrameV1:
    """load_artifacts from a frame-v1 model dir populates both new optional fields."""

    def test_frame_v1_loads_facility_stats(self, tmp_path):
        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)
        assert artifacts.facility_stats is not None, "facility_stats should be loaded for frame-v1"

    def test_frame_v1_loads_thresholds_segmented(self, tmp_path):
        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)
        assert (
            artifacts.thresholds_segmented is not None
        ), "thresholds_segmented should be loaded for frame-v1"

    def test_frame_v1_thresholds_segmented_has_by_facility(self, tmp_path):
        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)
        assert "by_facility" in artifacts.thresholds_segmented

    def test_frame_v1_model_version(self, tmp_path):
        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)
        assert artifacts.metadata["model_version"] == "frame-v1"

    def test_frame_v1_feature_count(self, tmp_path):
        frame_dir = _make_frame_v1_dir(tmp_path)
        artifacts = load_artifacts(frame_dir)
        assert len(artifacts.feature_list) == 30
        assert artifacts.model.n_features_in_ == 30

    def test_frame_v1_validate_artifacts_passes(self, tmp_path):
        """_validate_artifacts must not raise for frame-v1 (thresholds has binary_threshold etc.)."""
        frame_dir = _make_frame_v1_dir(tmp_path)
        # load_artifacts internally calls _validate_artifacts; if it raises the test fails
        artifacts = load_artifacts(frame_dir)
        assert artifacts is not None


# ---------------------------------------------------------------------------
# Warning behavior when stats present but segmented absent
# ---------------------------------------------------------------------------


class TestArtifactsWarnings:
    def test_warns_when_stats_present_but_segmented_missing(self, tmp_path):
        """Metadata with stats_artifact but no thresholds_segmented_artifact → UserWarning."""
        # Build a minimal frame-v1 dir but strip thresholds_segmented_artifact from metadata
        frame_dir = _make_frame_v1_dir(tmp_path)
        meta_path = frame_dir / "model_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta.pop("thresholds_segmented_artifact", None)
        # Also update artifact_files.thresholds to point to thresholds_v2.json equivalent
        # We need a valid thresholds file with the required keys.
        # Use thresholds_segmented_v1.json (it has binary_threshold + score_percentiles).
        meta_path.write_text(json.dumps(meta, indent=2))

        with pytest.warns(UserWarning, match="thresholds_segmented is None"):
            load_artifacts(frame_dir)


# ---------------------------------------------------------------------------
# metadata_filename override tests (SHAD-01 — dual-run support)
# ---------------------------------------------------------------------------


class TestMetadataFilenameOverride:
    """load_artifacts(metadata_filename=...) loads the correct model without renaming files."""

    def test_default_filename_loads_if40(self):
        """Default metadata_filename='model_metadata.json' → IF-40-v1 (regression guard)."""
        artifacts = load_artifacts(MODEL_DIR)
        assert artifacts.metadata["model_version"] == "IF-40-v1"
        assert len(artifacts.feature_list) == 40
        assert artifacts.facility_stats is None
        assert artifacts.thresholds_segmented is None

    def test_explicit_default_filename_same_as_default(self):
        """Passing metadata_filename='model_metadata.json' explicitly behaves identically."""
        artifacts_default = load_artifacts(MODEL_DIR)
        artifacts_explicit = load_artifacts(MODEL_DIR, metadata_filename="model_metadata.json")
        assert (
            artifacts_default.metadata["model_version"]
            == artifacts_explicit.metadata["model_version"]
        )
        assert len(artifacts_default.feature_list) == len(artifacts_explicit.feature_list)

    def test_frame_v1_filename_loads_frame_v1(self):
        """metadata_filename='model_metadata_frame_v1.json' → frame-v1 with 30 features."""
        artifacts = load_artifacts(MODEL_DIR, metadata_filename="model_metadata_frame_v1.json")
        assert artifacts.metadata["model_version"] == "frame-v1"
        assert len(artifacts.feature_list) == 30
        assert artifacts.facility_stats is not None, "facility_stats must be loaded for frame-v1"
        assert (
            artifacts.thresholds_segmented is not None
        ), "thresholds_segmented must be loaded for frame-v1"

    def test_missing_explicit_filename_raises(self, tmp_path):
        """An explicit metadata_filename that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="nonexistent.json"):
            load_artifacts(tmp_path, metadata_filename="nonexistent.json")
