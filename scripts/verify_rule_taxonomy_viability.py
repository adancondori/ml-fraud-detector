"""Taxonomía extendida · Viabilidad de las 16 categorías de reglas/señales (§4.1b del plan).

Evalúa contra ClickHouse (solo lectura) cada categoría de la taxonomía extendida
propuesta 2026-07-09 y emite un veredicto por categoría:

  existente              — ya implementada/planificada en §4.1
  factible_regla         — regla nueva con volumen operable o en shadow
  factible_senal_agregada— señal por usuario/facility (no alerta por transacción)
  cubierto_por_if        — la señal es feature central de frame-v1 (regla redundante,
                           proxy PROHIBIDO por circularidad)
  diferido               — datos existen pero falta definición de negocio
  descartado             — sin datos (evidencia por query, no supuesto)

Reglas de política (no negociables, ver plan §1.1 y §4.1b):
  * Ninguna señal basada en estados de reembolso entra al scoreboard.
  * Ninguna señal que intersecte FRAME_V1_FEATURE_NAMES entra al scoreboard.

Salida: output/revision/rule_taxonomy_viability.json
Uso:    ./venv/bin/python scripts/verify_rule_taxonomy_viability.py [--skip-live]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402

OUT = ROOT / "output" / "revision" / "rule_taxonomy_viability.json"

# Ventana de medición estándar (coincide con las pruebas de viabilidad del plan)
WINDOW_START = "2025-09-01"
WINDOW_END = "2025-12-01"
WINDOW_DAYS = 91

# Umbrales de clasificación de volumen (alertas/día)
OPERABLE_MAX_PER_DAY = 100.0
SHADOW_MAX_PER_DAY = 500.0

DB = "pbp_productionDB_optimized"

# Universo estándar de pagos (§3 del plan)
_PAY_UNIVERSE = (
    "_peerdb_is_deleted = 0 AND user_id != 0 " "AND payment_method NOT IN ('reversal','free')"
)


def classify_volume(per_day: float) -> str:
    """Clasifica el volumen diario de alertas de una regla candidata."""
    if per_day <= OPERABLE_MAX_PER_DAY:
        return "operable"
    if per_day <= SHADOW_MAX_PER_DAY:
        return "shadow"
    return "senal_agregada"


def is_circular(signals: frozenset, frame_features=FRAME_V1_FEATURE_NAMES) -> bool:
    """True si alguna señal de la categoría es feature de frame-v1 (proxy circular)."""
    return bool(signals & set(frame_features))


def scoreboard_eligibility(entry: dict, frame_features=FRAME_V1_FEATURE_NAMES) -> tuple:
    """(elegible, motivo) para el scoreboard §5. Política antes que disyunción."""
    if entry.get("uses_refund_status"):
        return False, "usa estados de reembolso — excluido por política (plan §1.1)"
    if is_circular(entry["signals"], frame_features):
        inter = sorted(entry["signals"] & set(frame_features))
        return False, f"circular: señal(es) {inter} son features de frame-v1"
    if entry["veredicto"] in ("descartado", "diferido"):
        return False, f"veredicto {entry['veredicto']} — sin señal utilizable"
    if entry.get("granularity") == "facility":
        return False, "granularidad facility, no transacción"
    if entry["veredicto"] == "existente" and entry.get("already_headline"):
        return True, "proxy tipificado ya titular (§5)"
    return True, (
        "candidato: señal disjunta de frame-v1 — requiere etiqueta a nivel pago, "
        "disyunción en CI y face validity HITL antes de entrar al scoreboard"
    )


# ---------------------------------------------------------------------------
# Taxonomía — las 16 categorías propuestas (2026-07-09)
# ---------------------------------------------------------------------------


def build_taxonomy() -> list:
    """Las 16 categorías con veredicto y elegibilidad de scoreboard calculados."""
    raw = [
        dict(
            categoria="Velocidad",
            regla="velocity_extreme",
            veredicto="existente",
            signals=frozenset({"user_txn_count_24h"}),
            already_headline=True,
            motivo="Regla activa §4.1 (~86/día); proxy tipificado titular.",
        ),
        dict(
            categoria="Descuentos",
            regla="discount_extreme",
            veredicto="existente",
            signals=frozenset({"discount_ratio"}),
            motivo="Solo señal agregada por facility (~830/día por txn es inviable).",
        ),
        dict(
            categoria="Card Testing",
            regla="card_testing",
            veredicto="existente",
            signals=frozenset({"same_amount_count_1h", "failed_count_1h"}),
            already_headline=True,
            motivo="card_testing_burst (~103/día) + card_testing_failed activas §4.1.",
        ),
        dict(
            categoria="Refunds",
            regla="refund_extreme",
            veredicto="factible_senal_agregada",
            signals=frozenset({"refund_count_30d"}),
            uses_refund_status=True,
            motivo="≥5 refunds/30d ≈ 34/día como señal por usuario; post-hoc por "
            "naturaleza (el refund ocurre después del pago).",
        ),
        dict(
            categoria="Montos",
            regla="high_amount",
            veredicto="cubierto_por_if",
            signals=frozenset({"amount_facility_ratio"}),
            motivo="Feature central de frame-v1; regla >10x promedio facility ≈ 181/día.",
        ),
        dict(
            categoria="Horarios",
            regla="odd_hours",
            veredicto="cubierto_por_if",
            signals=frozenset({"is_off_hours_loc"}),
            motivo="4.43% de la población es off-hours local; feature + gate de sesgo §5.",
        ),
        dict(
            categoria="Ubicación",
            regla="geo_anomaly",
            veredicto="descartado",
            signals=frozenset(),
            motivo="Sin datos: billing_address_id=0 en 100% del universo 2025; "
            "audits sin registros de Payment; no hay IP/país por pago.",
        ),
        dict(
            categoria="Dispositivo",
            regla="device_change",
            veredicto="descartado",
            signals=frozenset(),
            motivo="Sin device fingerprint en ninguna tabla replicada (jwt_tokens "
            "solo trae refresh_hash/signer/session_reference).",
        ),
        dict(
            categoria="Cliente",
            regla="new_customer_risk",
            veredicto="existente",
            signals=frozenset({"user_account_age_days", "user_txn_count_1h"}),
            already_headline=True,
            motivo="Equivale a new_user_burst (~24/día), regla activa §4.1.",
        ),
        dict(
            categoria="Método de pago",
            regla="payment_method_switch",
            veredicto="factible_regla",
            signals=frozenset({"user_distinct_methods"}),
            motivo="≥4 métodos/30d ≈ 21/día (≥3 da ~196/día). Regla operativa sí; "
            "proxy no (user_distinct_methods es feature frame-v1).",
        ),
        dict(
            categoria="Fallos",
            regla="failed_payment_burst",
            veredicto="existente",
            signals=frozenset({"failed_count_1h"}),
            already_headline=True,
            motivo="Equivale a card_testing_failed sobre failed_payment_logs (§4.1).",
        ),
        dict(
            categoria="Membresías",
            regla="membership_abuse",
            veredicto="diferido",
            signals=frozenset({"guest_passes_30d"}),
            motivo="Datos existen (4,658 membership_payments con ≥10 guest passes) "
            "pero 'abuso' exige join con límites del plan — definición de "
            "negocio pendiente (Fase 3).",
        ),
        dict(
            categoria="Reservas",
            regla="booking_burst",
            veredicto="factible_regla",
            signals=frozenset({"booking_count_1h"}),
            born_in_shadow=True,
            motivo="≥10 reservas/h no-admin ≈ 136/día → shadow y calibrar umbral (~20).",
        ),
        dict(
            categoria="Cancelaciones",
            regla="cancel_after_booking",
            veredicto="factible_regla",
            signals=frozenset({"quick_cancel_count_7d"}),
            born_in_shadow=True,
            motivo="≥3 cancelaciones <1h/semana ≈ 49 usuarios/día; 30% de reservas "
            "tienen deleted_at benigno → shadow obligatorio.",
        ),
        dict(
            categoria="Cuenta",
            regla="multi_account_token",
            veredicto="factible_regla",
            signals=frozenset({"token_shared_accounts"}),
            motivo="user_tokens.token: 6,423 tokens en ≥3 cuentas (máx 133) como "
            "stock; como flujo (cuenta nueva se une a token ya compartido por "
            "≥2) ≈ 3.1 alertas/día. last4+card_brand descartado como "
            "identificador (3,569 firmas para 6.7M pagos — espacio de "
            "colisión). Guard de familias (users_relations) antes de activar.",
        ),
        dict(
            categoria="Comercio",
            regla="merchant_outlier",
            veredicto="factible_senal_agregada",
            signals=frozenset({"facility_refund_rate_zscore"}),
            uses_refund_status=True,
            granularity="facility",
            motivo="Refund rate > mu+3sigma en cohorte >=200 txns: 10 de 549 "
            "facilities (Sep-Nov) → señal mensual operable.",
        ),
    ]
    for entry in raw:
        eligible, reason = scoreboard_eligibility(entry)
        entry["scoreboard_eligible"] = eligible
        entry["scoreboard_reason"] = reason
        entry["signals"] = sorted(entry["signals"])
    return raw


# ---------------------------------------------------------------------------
# Mediciones en vivo (ClickHouse, solo lectura)
# ---------------------------------------------------------------------------

MEASUREMENT_QUERIES = {
    "geo_anomaly_evidence": f"""
        SELECT countIf(billing_address_id > 0) AS with_billing, count() AS total
        FROM {DB}.payments FINAL
        WHERE {_PAY_UNIVERSE}
          AND created_at >= '2025-01-01' AND created_at < '2026-01-01'
    """,
    "payment_method_switch": f"""
        SELECT countIf(n_methods >= 4) AS user_months
        FROM (
            SELECT user_id, toStartOfMonth(created_at) AS m,
                   uniqExact(payment_method) AS n_methods
            FROM {DB}.payments FINAL
            WHERE {_PAY_UNIVERSE}
              AND created_at >= '{WINDOW_START}' AND created_at < '{WINDOW_END}'
            GROUP BY user_id, m
        )
    """,
    "refund_extreme": f"""
        SELECT countIf(refunds >= 5) AS user_months
        FROM (
            SELECT user_id, toStartOfMonth(created_at) AS m,
                   countIf(status IN ('totally_refunded','refunded_to_credit')) AS refunds
            FROM {DB}.payments FINAL
            WHERE {_PAY_UNIVERSE}
              AND created_at >= '{WINDOW_START}' AND created_at < '{WINDOW_END}'
            GROUP BY user_id, m
        )
    """,
    "multi_account_token": f"""
        SELECT countIf(n_users >= 3) AS shared_3plus, max(n_users) AS max_users
        FROM (
            SELECT token, uniqExact(user_id) AS n_users
            FROM {DB}.user_tokens FINAL
            WHERE _peerdb_is_deleted = 0 AND token != ''
            GROUP BY token
        )
    """,
    # Flujo de alertas (no stock): cuenta nueva que se une a un token que ya
    # tenía >=2 cuentas — es lo que la regla alertaría por día.
    "multi_account_token_flow": f"""
        SELECT count() AS events
        FROM (
            SELECT token, user_id, first_seen,
                   row_number() OVER (PARTITION BY token ORDER BY first_seen) AS rn
            FROM (
                SELECT token, user_id, min(created_at) AS first_seen
                FROM {DB}.user_tokens FINAL
                WHERE _peerdb_is_deleted = 0 AND token != ''
                GROUP BY token, user_id
            )
        )
        WHERE rn >= 3
          AND first_seen >= '{WINDOW_START}' AND first_seen < '{WINDOW_END}'
    """,
    "booking_burst": f"""
        SELECT count() AS user_hours
        FROM (
            SELECT user_id, toStartOfHour(created_at) AS h, count() AS c
            FROM {DB}.reservations FINAL
            WHERE _peerdb_is_deleted = 0 AND user_id != 0
              AND admin_booked = 0 AND generated_by_court = 0
              AND recurring_event_id = 0
              AND created_at >= '{WINDOW_START}' AND created_at < '{WINDOW_END}'
            GROUP BY user_id, h HAVING c >= 10
        )
    """,
    "cancel_after_booking": f"""
        SELECT count() AS user_weeks
        FROM (
            SELECT user_id, toStartOfWeek(created_at) AS w,
                   countIf(deleted_at > created_at
                           AND deleted_at < created_at + INTERVAL 1 HOUR) AS qc
            FROM {DB}.reservations FINAL
            WHERE _peerdb_is_deleted = 0 AND user_id != 0 AND admin_booked = 0
              AND created_at >= '{WINDOW_START}' AND created_at < '{WINDOW_END}'
            GROUP BY user_id, w HAVING qc >= 3
        )
    """,
    "merchant_outlier": f"""
        WITH fr AS (
            SELECT facility_id, count() AS n,
                   countIf(status IN ('totally_refunded','refunded_to_credit'))/count()
                       AS refund_rate
            FROM {DB}.payments FINAL
            WHERE {_PAY_UNIVERSE}
              AND created_at >= '{WINDOW_START}' AND created_at < '{WINDOW_END}'
            GROUP BY facility_id HAVING n >= 200
        ),
        stats AS (SELECT avg(refund_rate) AS mu, stddevPop(refund_rate) AS sigma FROM fr)
        SELECT count() AS facilities, countIf(refund_rate > mu + 3*sigma) AS outliers
        FROM fr, stats
    """,
}


def run_live_measurements(client) -> dict:
    """Ejecuta las queries de medición y deriva alertas/día por regla."""
    measures = {}
    for name, sql in MEASUREMENT_QUERIES.items():
        t0 = time.time()
        row = client.query(sql).first_row
        measures[name] = {
            "raw": [int(v) if v is not None else None for v in row],
            "elapsed_s": round(time.time() - t0, 1),
        }
    # Derivar tasas diarias comparables
    measures["payment_method_switch"]["per_day"] = round(
        measures["payment_method_switch"]["raw"][0] / WINDOW_DAYS, 1
    )
    measures["refund_extreme"]["per_day"] = round(
        measures["refund_extreme"]["raw"][0] / WINDOW_DAYS, 1
    )
    measures["booking_burst"]["per_day"] = round(
        measures["booking_burst"]["raw"][0] / WINDOW_DAYS, 1
    )
    measures["cancel_after_booking"]["per_day"] = round(
        measures["cancel_after_booking"]["raw"][0] / WINDOW_DAYS, 1
    )
    measures["multi_account_token_flow"]["per_day"] = round(
        measures["multi_account_token_flow"]["raw"][0] / WINDOW_DAYS, 1
    )
    return measures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="No consultar ClickHouse; emitir solo la taxonomía con veredictos.",
    )
    args = parser.parse_args()

    taxonomy = build_taxonomy()
    result = {
        "generated_for": "plan §4.1b — taxonomía extendida (16 categorías)",
        "window": {"start": WINDOW_START, "end": WINDOW_END, "days": WINDOW_DAYS},
        "thresholds": {
            "operable_max_per_day": OPERABLE_MAX_PER_DAY,
            "shadow_max_per_day": SHADOW_MAX_PER_DAY,
        },
        "taxonomy": taxonomy,
        "summary": {
            "existente": sum(1 for t in taxonomy if t["veredicto"] == "existente"),
            "factible_regla": sum(1 for t in taxonomy if t["veredicto"] == "factible_regla"),
            "factible_senal_agregada": sum(
                1 for t in taxonomy if t["veredicto"] == "factible_senal_agregada"
            ),
            "cubierto_por_if": sum(1 for t in taxonomy if t["veredicto"] == "cubierto_por_if"),
            "diferido": sum(1 for t in taxonomy if t["veredicto"] == "diferido"),
            "descartado": sum(1 for t in taxonomy if t["veredicto"] == "descartado"),
        },
    }

    if not args.skip_live:
        import clickhouse_connect

        from config.config import settings

        client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            secure=settings.clickhouse_secure,
        )
        try:
            measures = run_live_measurements(client)
        finally:
            client.close()
        result["live_measurements"] = measures
        # Clasificación de volumen para las reglas nuevas medibles por día
        result["volume_classification"] = {
            name: classify_volume(measures[name]["per_day"])
            for name in (
                "payment_method_switch",
                "refund_extreme",
                "booking_burst",
                "cancel_after_booking",
                "multi_account_token_flow",
            )
        }
        # Gate de evidencia: geo sigue sin datos
        with_billing, total = measures["geo_anomaly_evidence"]["raw"]
        result["geo_anomaly_still_discarded"] = with_billing == 0
        if with_billing > 0:
            print(
                f"AVISO: {with_billing}/{total} pagos con billing_address_id > 0 — "
                "reevaluar geo_anomaly (plan §11.5)."
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"OK — veredictos: {result['summary']}")
    print(f"Salida: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
