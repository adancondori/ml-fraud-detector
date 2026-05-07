"""Tests for run_pipeline.py orchestrator."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pytest

# Import orchestrator functions directly
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_pipeline import should_run, validate_prerequisites, STEP_INPUTS


def test_should_run_all_by_default():
    args = argparse.Namespace(step=None, from_step=None)
    for s in range(1, 9):
        assert should_run(s, args) is True


def test_should_run_single_step():
    args = argparse.Namespace(step=3, from_step=None)
    assert should_run(1, args) is False
    assert should_run(2, args) is False
    assert should_run(3, args) is True
    assert should_run(4, args) is False


def test_should_run_from_step():
    args = argparse.Namespace(step=None, from_step=5)
    assert should_run(1, args) is False
    assert should_run(4, args) is False
    assert should_run(5, args) is True
    assert should_run(8, args) is True


def test_prerequisite_validation_passes_with_files(tmp_path):
    # Create prerequisite files for step 3
    for p in STEP_INPUTS[3]:
        f = tmp_path / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("test")
    validate_prerequisites(3, tmp_path)  # Should not raise


def test_prerequisite_validation_fails_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing prerequisite"):
        validate_prerequisites(3, tmp_path)


def test_step1_has_no_prerequisites():
    assert STEP_INPUTS[1] == []


def test_all_steps_defined():
    for s in range(1, 9):
        assert s in STEP_INPUTS
