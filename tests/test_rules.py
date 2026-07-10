"""Tests para la Capa 1 — reglas deterministas del confirmatorio V2.

Variable criterio tipificada NO circular (card_testing, velocity_extreme,
new_user_burst, typed_union) + control negativo de reembolso. Cubre umbrales
exactos, campos nulos, ambas superficies (DataFrame y fila), y la verificación
de disyunción feature↔proxy (invariante de no-circularidad, plan §7.8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detector.scoring import rules
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES

# ---------------------------------------------------------------------------
# card_testing — umbral exacto en same_amount_count_1h y failed_count_1h
# ---------------------------------------------------------------------------


def test_card_testing_same_amount_boundary():
    # 5 dispara, 4 no (umbral >= 5).
    assert rules.card_testing({"same_amount_count_1h": 5}) is True
    assert rules.card_testing({"same_amount_count_1h": 4}) is False


def test_card_testing_failed_boundary():
    assert rules.card_testing({"failed_count_1h": 5}) is True
    assert rules.card_testing({"failed_count_1h": 4}) is False


def test_card_testing_is_or_of_both_signals():
    # Ninguna señal por su cuenta -> no dispara; una sí -> dispara.
    assert rules.card_testing({"same_amount_count_1h": 0, "failed_count_1h": 0}) is False
    assert rules.card_testing({"same_amount_count_1h": 6, "failed_count_1h": 0}) is True
    assert rules.card_testing({"same_amount_count_1h": 0, "failed_count_1h": 7}) is True


def test_card_testing_missing_fields_default_to_false():
    assert rules.card_testing({}) is False


def test_card_testing_dataframe_surface():
    df = pd.DataFrame(
        {
            "same_amount_count_1h": [5, 4, 0, 0],
            "failed_count_1h": [0, 0, 5, 4],
        }
    )
    out = rules.card_testing(df)
    assert isinstance(out, np.ndarray)
    assert out.dtype == bool
    assert out.tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# velocity_extreme — umbral estricto > 100
# ---------------------------------------------------------------------------


def test_velocity_extreme_boundary_strict():
    # 100 NO dispara, 101 sí (umbral estricto).
    assert rules.velocity_extreme({"user_txn_count_24h": 100}) is False
    assert rules.velocity_extreme({"user_txn_count_24h": 101}) is True


def test_velocity_extreme_dataframe():
    df = pd.DataFrame({"user_txn_count_24h": [99, 100, 101, 500]})
    assert rules.velocity_extreme(df).tolist() == [False, False, True, True]


# ---------------------------------------------------------------------------
# new_user_burst — edad < 14 estricta AND txn_1h >= 3
# ---------------------------------------------------------------------------


def test_new_user_burst_age_boundary():
    # age=14 no dispara (estricto <14); age=13 con 3 txns sí.
    assert rules.new_user_burst({"user_account_age_days": 14, "user_txn_count_1h": 3}) is False
    assert rules.new_user_burst({"user_account_age_days": 13, "user_txn_count_1h": 3}) is True


def test_new_user_burst_txn_boundary():
    # txn_1h >= 3: 2 no dispara, 3 sí (con usuario nuevo).
    assert rules.new_user_burst({"user_account_age_days": 5, "user_txn_count_1h": 2}) is False
    assert rules.new_user_burst({"user_account_age_days": 5, "user_txn_count_1h": 3}) is True


def test_new_user_burst_requires_both_conditions():
    # Usuario viejo con muchas txns no dispara.
    assert rules.new_user_burst({"user_account_age_days": 400, "user_txn_count_1h": 10}) is False


def test_new_user_burst_dataframe():
    df = pd.DataFrame(
        {
            "user_account_age_days": [13, 14, 5, 400],
            "user_txn_count_1h": [3, 3, 2, 10],
        }
    )
    assert rules.new_user_burst(df).tolist() == [True, False, False, False]


# ---------------------------------------------------------------------------
# typed_union — OR de las tres
# ---------------------------------------------------------------------------


def test_typed_union_fires_on_any_rule():
    assert rules.typed_union({"same_amount_count_1h": 5}) is True
    assert rules.typed_union({"user_txn_count_24h": 101}) is True
    assert rules.typed_union({"user_account_age_days": 3, "user_txn_count_1h": 3}) is True


def test_typed_union_false_when_none_fires():
    row = {
        "same_amount_count_1h": 1,
        "failed_count_1h": 1,
        "user_txn_count_24h": 10,
        "user_account_age_days": 400,
        "user_txn_count_1h": 1,
    }
    assert rules.typed_union(row) is False


def test_typed_union_dataframe_matches_or():
    df = pd.DataFrame(
        {
            "same_amount_count_1h": [5, 0, 0, 0],
            "failed_count_1h": [0, 0, 0, 0],
            "user_txn_count_24h": [0, 101, 0, 0],
            "user_account_age_days": [400, 400, 3, 400],
            "user_txn_count_1h": [0, 0, 3, 0],
        }
    )
    expected = rules.card_testing(df) | rules.velocity_extreme(df) | rules.new_user_burst(df)
    assert rules.typed_union(df).tolist() == expected.tolist()
    assert rules.typed_union(df).tolist() == [True, True, True, False]


# ---------------------------------------------------------------------------
# refund_negative_control
# ---------------------------------------------------------------------------


def test_refund_negative_control_positive_statuses():
    assert rules.refund_negative_control({"status": "totally_refunded"}) is True
    assert rules.refund_negative_control({"status": "refunded_to_credit"}) is True


def test_refund_negative_control_negative_statuses():
    assert rules.refund_negative_control({"status": "captured"}) is False
    assert rules.refund_negative_control({"status": "partially_refunded"}) is False
    assert rules.refund_negative_control({}) is False


def test_refund_negative_control_dataframe():
    df = pd.DataFrame({"status": ["totally_refunded", "captured", "refunded_to_credit", "empty"]})
    assert rules.refund_negative_control(df).tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# Disyunción feature↔proxy (no-circularidad)
# ---------------------------------------------------------------------------


def test_assert_disjoint_passes_against_real_contract():
    report = rules.assert_disjoint_from_features(FRAME_V1_FEATURE_NAMES)
    assert report["disjoint"] is True
    assert report["overlap"] == []
    # Se valida contra el contrato real (30 features), no una copia.
    assert report["n_feature_names"] == 30


def test_assert_disjoint_default_uses_real_contract():
    # Sin argumentos debe usar FRAME_V1_FEATURE_NAMES.
    report = rules.assert_disjoint_from_features()
    assert report["disjoint"] is True


def test_assert_disjoint_raises_when_rule_field_is_a_model_feature():
    # Inyectar un campo de regla como si fuera feature del modelo -> debe fallar.
    poisoned = list(FRAME_V1_FEATURE_NAMES) + ["user_txn_count_24h"]
    with pytest.raises(ValueError, match="Disyunción"):
        rules.assert_disjoint_from_features(poisoned)


def test_assert_disjoint_reports_all_overlaps():
    poisoned = list(FRAME_V1_FEATURE_NAMES) + ["status", "same_amount_count_1h"]
    with pytest.raises(ValueError) as exc:
        rules.assert_disjoint_from_features(poisoned)
    msg = str(exc.value)
    assert "same_amount_count_1h" in msg
    assert "status" in msg


def test_rule_fields_cover_all_rules():
    # Cada regla del scoreboard documenta sus campos externos.
    assert set(rules.RULE_FIELDS) == {
        "card_testing",
        "velocity_extreme",
        "new_user_burst",
        "refund_negative_control",
    }


# ---------------------------------------------------------------------------
# Stub diferido
# ---------------------------------------------------------------------------


def test_multi_account_token_is_deferred_stub():
    with pytest.raises(NotImplementedError):
        rules.multi_account_token({"token": "abc"})
