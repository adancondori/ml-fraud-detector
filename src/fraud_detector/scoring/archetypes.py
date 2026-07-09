"""Archetype assignment for typed anomaly alerts (Plan B).

Maps a per-transaction SHAP contribution vector (convention: POSITIVE value =
pushes toward anomaly) onto behavioral archetype groups, and assigns a dominant
archetype (or ``mixed`` when no group concentrates enough of the signal).

The mapping is keyed to FS-frame-v1 (30 features). SHAP explains the frame-v1
model, which is frame-normalized and non-circular, so archetypes reflect
behavior — not currency/scale/timezone artifacts.

Design notes:
  - Only positive contributions count: a feature that pushes a transaction toward
    "normal" does not explain why it is anomalous.
  - ``dominance`` is the share of total positive contribution the top group must
    reach to be declared dominant; otherwise the transaction is ``mixed``.
  - ``top_k`` returns the k highest-contributing groups (with positive mass) for
    operator context, independent of the dominant/``mixed`` decision.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES

MIXED = "mixed"
DEFAULT_DOMINANCE = 0.35

# Feature -> archetype group (docs/plan-normalizacion-marcos.md §B.2.2)
ARCHETYPE_OF: Dict[str, str] = {
    "log_amount_fac": "magnitude",
    "amount_facility_ratio": "magnitude",
    "user_amount_24h_fac": "magnitude",
    "small_amount_at_facility": "magnitude",
    "very_small_amount_at_facility": "magnitude",
    "hour_sin_loc": "temporal",
    "hour_cos_loc": "temporal",
    "dow_sin_loc": "temporal",
    "dow_cos_loc": "temporal",
    "is_weekend_loc": "temporal",
    "is_off_hours_loc": "temporal",
    "off_hours_high_value_loc": "temporal",
    "discount_ratio": "discount",
    "is_club_credit": "credit_flow",
    "user_debit_count_30d": "credit_flow",
    "user_debit_amount_30d_fac": "credit_flow",
    "credit_flow_ratio": "credit_flow",
    "user_distinct_facilities_30d": "diversity",
    "user_distinct_methods": "diversity",
    "category_entropy_30d": "diversity",
    "user_merchandise_ratio_30d": "diversity",
    "is_staff": "staff_role",
    "paid_by_manager": "staff_role",
    "staff_amount_zscore": "staff_role",
    "gateway_change_recent": "gateway_channel",
    "is_main_gateway": "gateway_channel",
    "is_first_gateway_for_user": "gateway_channel",
    "source_change_recent": "gateway_channel",
    "has_tip": "tip",
    "time_since_last_txn": "velocity",
}

ARCHETYPE_GROUPS: List[str] = sorted(set(ARCHETYPE_OF.values()))

# Fail fast if the frame-v1 contract and the mapping drift apart.
_missing = set(FRAME_V1_FEATURE_NAMES) - set(ARCHETYPE_OF)
_extra = set(ARCHETYPE_OF) - set(FRAME_V1_FEATURE_NAMES)
if _missing or _extra:
    raise ValueError(
        f"ARCHETYPE_OF must cover FRAME_V1_FEATURE_NAMES exactly. "
        f"missing={sorted(_missing)} extra={sorted(_extra)}"
    )


def group_contributions(
    shap_row: Sequence[float],
    feature_names: Sequence[str] = FRAME_V1_FEATURE_NAMES,
) -> Dict[str, float]:
    """Sum the POSITIVE (toward-anomaly) SHAP contributions per archetype group.

    Args:
        shap_row: Per-feature contributions; positive = pushes toward anomaly.
        feature_names: Feature order matching ``shap_row``.

    Returns:
        Dict archetype -> summed positive contribution (0.0 for groups with none).
    """
    row = np.asarray(shap_row, dtype=np.float64)
    if row.shape[0] != len(feature_names):
        raise ValueError(
            f"shap_row length {row.shape[0]} != feature_names length {len(feature_names)}"
        )
    pos = np.clip(row, 0.0, None)
    contrib = {g: 0.0 for g in ARCHETYPE_GROUPS}
    for i, name in enumerate(feature_names):
        contrib[ARCHETYPE_OF[name]] += float(pos[i])
    return contrib


def assign_archetype(
    shap_row: Sequence[float],
    feature_names: Sequence[str] = FRAME_V1_FEATURE_NAMES,
    dominance: float = DEFAULT_DOMINANCE,
    top_k: int = 2,
) -> Tuple[str, Dict[str, float], List[str]]:
    """Assign a dominant archetype to a transaction from its SHAP contributions.

    Args:
        shap_row: Per-feature contributions; positive = toward anomaly.
        feature_names: Feature order matching ``shap_row``.
        dominance: Share of total positive contribution the top group must reach
            to be declared dominant (else ``mixed``). In [0, 1].
        top_k: Number of top positive groups to return for context.

    Returns:
        (dominant, group_contrib, top_archetypes) where ``dominant`` is a group
        name or ``mixed``, and ``top_archetypes`` are the up-to-``top_k`` groups
        with positive contribution, highest first.
    """
    if not 0.0 <= dominance <= 1.0:
        raise ValueError(f"dominance must be in [0, 1], got {dominance}")
    contrib = group_contributions(shap_row, feature_names)
    total = sum(contrib.values())
    ranked = sorted(contrib.items(), key=lambda kv: (-kv[1], kv[0]))
    top_archetypes = [g for g, v in ranked[:top_k] if v > 0.0]

    if total <= 0.0:
        return MIXED, contrib, top_archetypes

    top_group, top_val = ranked[0]
    dominant = top_group if (top_val / total) >= dominance else MIXED
    return dominant, contrib, top_archetypes
