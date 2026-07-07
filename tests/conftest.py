"""Shared fixtures for all test modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_scores():
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=200).astype(np.float32)


@pytest.fixture
def sample_proxy_labels():
    rng = np.random.default_rng(42)
    return rng.choice([0, 1], size=200, p=[0.94, 0.06]).astype(np.int8)
