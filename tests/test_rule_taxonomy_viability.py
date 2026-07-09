"""Tests para verify_rule_taxonomy_viability.py (taxonomía extendida, plan §4.1b).

Sin conexión a ClickHouse: valida la lógica pura de veredictos, la disyunción
programática señal↔feature contra el contrato REAL de frame-v1 (no una copia),
la política de refund y la clasificación de volumen. Cubre los casos 8–12 del
plan §7 (TDD).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from verify_rule_taxonomy_viability import (  # noqa: E402
    MEASUREMENT_QUERIES,
    OPERABLE_MAX_PER_DAY,
    SHADOW_MAX_PER_DAY,
    build_taxonomy,
    classify_volume,
    is_circular,
    scoreboard_eligibility,
)

from fraud_detector.scoring.features_frame_v1 import (  # noqa: E402
    FRAME_V1_FEATURE_NAMES,
)


@pytest.fixture(scope="module")
def taxonomy():
    return build_taxonomy()


@pytest.fixture(scope="module")
def by_rule(taxonomy):
    return {t["regla"]: t for t in taxonomy}


# ---------------------------------------------------------------------------
# Caso 12 §7 — clasificación de volumen: bordes exactos
# ---------------------------------------------------------------------------


class TestClassifyVolume:
    def test_zero_is_operable(self):
        assert classify_volume(0.0) == "operable"

    def test_exact_operable_boundary(self):
        assert classify_volume(OPERABLE_MAX_PER_DAY) == "operable"

    def test_just_above_operable_is_shadow(self):
        assert classify_volume(OPERABLE_MAX_PER_DAY + 0.1) == "shadow"

    def test_exact_shadow_boundary(self):
        assert classify_volume(SHADOW_MAX_PER_DAY) == "shadow"

    def test_above_shadow_is_aggregate(self):
        assert classify_volume(SHADOW_MAX_PER_DAY + 0.1) == "senal_agregada"


# ---------------------------------------------------------------------------
# Caso 8 §7 — disyunción programática contra FRAME_V1_FEATURE_NAMES real
# ---------------------------------------------------------------------------


class TestDisjunction:
    CIRCULAR_RULES = ("payment_method_switch", "high_amount", "odd_hours", "discount_extreme")

    def test_is_circular_detects_frame_feature(self):
        assert is_circular(frozenset({"user_distinct_methods"}))
        assert is_circular(frozenset({"amount_facility_ratio"}))
        assert is_circular(frozenset({"is_off_hours_loc"}))
        assert is_circular(frozenset({"discount_ratio"}))

    def test_is_circular_rejects_disjoint_signal(self):
        assert not is_circular(frozenset({"token_shared_accounts"}))
        assert not is_circular(frozenset({"quick_cancel_count_7d"}))
        assert not is_circular(frozenset())

    @pytest.mark.parametrize("rule", CIRCULAR_RULES)
    def test_circular_categories_never_scoreboard(self, by_rule, rule):
        entry = by_rule[rule]
        assert entry["scoreboard_eligible"] is False
        assert "circular" in entry["scoreboard_reason"]

    def test_circularity_checked_against_real_contract(self, by_rule):
        """La señal circular citada debe existir en el contrato real de 30 features."""
        for rule in self.CIRCULAR_RULES:
            inter = set(by_rule[rule]["signals"]) & set(FRAME_V1_FEATURE_NAMES)
            assert inter, f"{rule} marcada circular sin señal en frame-v1"


# ---------------------------------------------------------------------------
# Caso 9 §7 — política refund: excluido del scoreboard aunque sea disjunto
# ---------------------------------------------------------------------------


class TestRefundPolicy:
    @pytest.mark.parametrize("rule", ["refund_extreme", "merchant_outlier"])
    def test_refund_based_never_scoreboard(self, by_rule, rule):
        entry = by_rule[rule]
        assert entry["scoreboard_eligible"] is False
        assert "reembolso" in entry["scoreboard_reason"]

    def test_refund_policy_precedes_disjunction(self):
        """refund_count_30d NO es feature frame-v1: la exclusión es por política."""
        entry = dict(
            regla="x",
            veredicto="factible_regla",
            signals=frozenset({"refund_count_30d"}),
            uses_refund_status=True,
        )
        assert not is_circular(entry["signals"])
        eligible, reason = scoreboard_eligibility(entry)
        assert eligible is False
        assert "política" in reason


# ---------------------------------------------------------------------------
# Caso 10 §7 — multi_account_token: identificador correcto
# ---------------------------------------------------------------------------


class TestMultiAccountToken:
    def test_uses_gateway_token_not_last4(self):
        sql = MEASUREMENT_QUERIES["multi_account_token"]
        assert "user_tokens" in sql
        assert "token" in sql
        assert "last_four" not in sql and "card_brand" not in sql

    def test_threshold_is_three_accounts(self):
        assert "n_users >= 3" in MEASUREMENT_QUERIES["multi_account_token"]

    def test_empty_token_excluded(self):
        assert "token != ''" in MEASUREMENT_QUERIES["multi_account_token"]

    def test_verdict_and_eligibility(self, by_rule):
        entry = by_rule["multi_account_token"]
        assert entry["veredicto"] == "factible_regla"
        assert entry["scoreboard_eligible"] is True  # candidato (disjunto)
        assert "candidato" in entry["scoreboard_reason"]

    def test_flow_query_alerts_on_new_account_joining_shared_token(self):
        """La regla alerta por flujo (rn>=3 por first_seen), no por stock."""
        sql = MEASUREMENT_QUERIES["multi_account_token_flow"]
        assert "user_tokens" in sql
        assert "min(created_at)" in sql
        assert "rn >= 3" in sql
        assert "first_seen >=" in sql


# ---------------------------------------------------------------------------
# Caso 11 §7 — reglas de reservas: exclusiones y shadow
# ---------------------------------------------------------------------------


class TestReservationRules:
    def test_booking_burst_excludes_admin_surfaces(self):
        sql = MEASUREMENT_QUERIES["booking_burst"]
        assert "admin_booked = 0" in sql
        assert "generated_by_court = 0" in sql
        assert "recurring_event_id = 0" in sql

    def test_cancel_after_booking_is_quick_cancel(self):
        sql = MEASUREMENT_QUERIES["cancel_after_booking"]
        assert "deleted_at > created_at" in sql
        assert "INTERVAL 1 HOUR" in sql
        assert "qc >= 3" in sql

    @pytest.mark.parametrize("rule", ["booking_burst", "cancel_after_booking"])
    def test_born_in_shadow(self, by_rule, rule):
        assert by_rule[rule]["born_in_shadow"] is True


# ---------------------------------------------------------------------------
# Integridad de la taxonomía (las 16 categorías, veredictos válidos)
# ---------------------------------------------------------------------------


class TestTaxonomyIntegrity:
    VALID_VERDICTS = {
        "existente",
        "factible_regla",
        "factible_senal_agregada",
        "cubierto_por_if",
        "diferido",
        "descartado",
    }
    EXPECTED_RULES = {
        "velocity_extreme",
        "discount_extreme",
        "card_testing",
        "refund_extreme",
        "high_amount",
        "odd_hours",
        "geo_anomaly",
        "device_change",
        "new_customer_risk",
        "payment_method_switch",
        "failed_payment_burst",
        "membership_abuse",
        "booking_burst",
        "cancel_after_booking",
        "multi_account_token",
        "merchant_outlier",
    }

    def test_sixteen_categories_exactly_once(self, taxonomy):
        rules = [t["regla"] for t in taxonomy]
        assert len(rules) == 16
        assert set(rules) == self.EXPECTED_RULES
        assert len(set(rules)) == len(rules)

    def test_all_verdicts_valid(self, taxonomy):
        for t in taxonomy:
            assert t["veredicto"] in self.VALID_VERDICTS, t["regla"]

    def test_discarded_have_no_signals_and_evidence(self, by_rule):
        for rule in ("geo_anomaly", "device_change"):
            entry = by_rule[rule]
            assert entry["signals"] == []
            assert entry["scoreboard_eligible"] is False
            assert "Sin datos" in entry["motivo"] or "Sin device" in entry["motivo"]

    def test_geo_evidence_query_exists(self):
        """El descarte de geo se re-verifica en cada corrida en vivo (plan §11.5)."""
        assert "billing_address_id > 0" in MEASUREMENT_QUERIES["geo_anomaly_evidence"]

    def test_every_entry_has_reason(self, taxonomy):
        for t in taxonomy:
            assert t["motivo"].strip()
            assert t["scoreboard_reason"].strip()

    def test_headline_proxies_remain_eligible(self, by_rule):
        for rule in ("velocity_extreme", "card_testing", "new_customer_risk"):
            assert by_rule[rule]["scoreboard_eligible"] is True

    def test_all_final_and_peerdb_filters(self):
        """Toda query en vivo respeta FINAL + _peerdb_is_deleted (regla ClickHouse)."""
        for name, sql in MEASUREMENT_QUERIES.items():
            assert "FINAL" in sql, name
            assert "_peerdb_is_deleted = 0" in sql, name
