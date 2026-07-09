"""Tests for archetype assignment (Plan B — typed anomalies).

TDD: deterministic synthetic SHAP vectors, no model loading.
Convention under test: positive contribution = pushes toward anomaly.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detector.scoring.archetypes import (
    ARCHETYPE_GROUPS,
    ARCHETYPE_OF,
    DEFAULT_DOMINANCE,
    MIXED,
    assign_archetype,
    group_contributions,
)
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES

IDX = {name: i for i, name in enumerate(FRAME_V1_FEATURE_NAMES)}


def _zeros() -> np.ndarray:
    return np.zeros(len(FRAME_V1_FEATURE_NAMES), dtype=np.float64)


def _vec(**feature_values) -> np.ndarray:
    v = _zeros()
    for name, val in feature_values.items():
        v[IDX[name]] = val
    return v


def test_mapping_covers_contract_exactly():
    assert set(ARCHETYPE_OF) == set(FRAME_V1_FEATURE_NAMES)


def test_single_feature_dominates():
    # All positive mass on discount_ratio -> archetype 'discount'.
    dom, contrib, top = assign_archetype(_vec(discount_ratio=1.0))
    assert dom == "discount"
    assert contrib["discount"] == pytest.approx(1.0)
    assert top[0] == "discount"


def test_magnitude_group_aggregates_features():
    # Two magnitude features -> magnitude dominates.
    dom, contrib, _ = assign_archetype(
        _vec(log_amount_fac=0.6, amount_facility_ratio=0.4)
    )
    assert dom == "magnitude"
    assert contrib["magnitude"] == pytest.approx(1.0)


def test_evenly_split_is_mixed():
    # Equal mass across three distinct groups: top share = 1/3 < 0.35 -> mixed.
    dom, _, top = assign_archetype(
        _vec(discount_ratio=1.0, is_club_credit=1.0, is_staff=1.0),
        dominance=DEFAULT_DOMINANCE,
    )
    assert dom == MIXED
    assert len(top) == 2  # top_k default


def test_negative_only_is_mixed_with_no_top():
    # Everything pushes toward normal -> no positive mass -> mixed, empty top.
    dom, contrib, top = assign_archetype(_vec(discount_ratio=-2.0, is_staff=-1.0))
    assert dom == MIXED
    assert top == []
    assert all(v == 0.0 for v in contrib.values())


def test_dominance_threshold_boundary():
    # magnitude=0.5 (unique top), rest split -> top share exactly 0.5, no tie.
    v = _vec(log_amount_fac=0.5, discount_ratio=0.25, is_club_credit=0.25)
    assert assign_archetype(v, dominance=0.5)[0] == "magnitude"  # >= passes
    assert assign_archetype(v, dominance=0.51)[0] == MIXED


def test_top_k_returns_ranked_positive_groups():
    v = _vec(discount_ratio=0.5, is_club_credit=0.3, is_staff=0.2)
    _, _, top = assign_archetype(v, top_k=2)
    assert top == ["discount", "credit_flow"]


def test_lower_dominance_reduces_mixed():
    # A 40% top group is 'mixed' at 0.5 but dominant at 0.35.
    v = _vec(discount_ratio=0.4, is_club_credit=0.35, is_staff=0.25)
    assert assign_archetype(v, dominance=0.50)[0] == MIXED
    assert assign_archetype(v, dominance=0.35)[0] == "discount"


def test_determinism():
    v = _vec(credit_flow_ratio=0.7, discount_ratio=0.3)
    assert assign_archetype(v) == assign_archetype(v)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        group_contributions(np.zeros(5))


def test_invalid_dominance_raises():
    with pytest.raises(ValueError):
        assign_archetype(_zeros(), dominance=1.5)
