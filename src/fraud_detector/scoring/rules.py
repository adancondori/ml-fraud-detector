"""Capa 1 — reglas deterministas del confirmatorio V2 (variable criterio + operativas).

Este módulo define, como funciones puras sobre un ``DataFrame`` o ``dict`` de
campos, la **variable criterio tipificada NO circular** del confirmatorio V2 y
el **control negativo de reembolso**:

  Variable criterio (unión tipificada, evaluación titular):
    - card_testing     : ráfaga de mismo monto en 1h o ráfaga de fallos por token.
    - velocity_extreme : conteo de transacciones en 24h por encima de umbral.
    - new_user_burst   : usuario recién creado con varias transacciones en 1h.
    - typed_union      : OR de las tres reglas anteriores.

  Control negativo:
    - refund_negative_control : status de reembolso (NO circular con features, pero
      es un proxy administrativo — su rol es de banda de equivalencia HE3, jamás
      variable criterio; ver plan §1.1).

**Disyunción feature↔proxy (crítica, plan §7.8):** cada regla se basa
exclusivamente en CAMPOS EXTERNOS a ``FRAME_V1_FEATURE_NAMES``. Esto es lo que
garantiza que la capacidad discriminativa medida por el scoreboard no sea
circular. :func:`assert_disjoint_from_features` verifica programáticamente esa
disyunción (mismo patrón que ``scripts/validate_if40_pivot_disjoint.py`` y
``scripts/verify_rule_taxonomy_viability.py``).

Convención de score de anomalía: no aplica aquí; estas reglas producen
etiquetas binarias (variable criterio), nunca un score continuo.

Superficies aceptadas por cada regla:
  - ``pd.DataFrame`` -> devuelve ``np.ndarray`` booleano (una entrada por fila).
  - ``dict`` / ``pd.Series`` (una fila) -> devuelve ``bool`` escalar.

Los umbrales son constantes nombradas al tope del módulo (auditables y
reutilizables por config, tests y la corrida confirmatoria).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Mapping, Union

import numpy as np
import pandas as pd

from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Umbrales nombrados (auditables — coinciden con plan §4.1 y la tabla de gates V2)
# ---------------------------------------------------------------------------

#: card_testing — mínimo de pagos del mismo monto en 1h para disparar la ráfaga.
CARD_TESTING_SAME_AMOUNT_MIN: int = 5

#: card_testing — mínimo de fallos por ``user_token_id`` en 1h (failed_payment_logs).
CARD_TESTING_FAILED_MIN: int = 5

#: velocity_extreme — umbral (estricto) de transacciones en 24h por usuario.
VELOCITY_EXTREME_TXN_24H_MIN: int = 100

#: new_user_burst — edad máxima (estricta) de la cuenta en días.
NEW_USER_MAX_AGE_DAYS: int = 14

#: new_user_burst — mínimo de transacciones en 1h para el usuario nuevo.
NEW_USER_BURST_TXN_1H_MIN: int = 3

#: Estados de reembolso que definen el control negativo (Tipo A).
REFUND_STATUSES: FrozenSet[str] = frozenset({"totally_refunded", "refunded_to_credit"})

# ---------------------------------------------------------------------------
# Campos EXTERNOS que consume cada regla (mapa señal↔campo, para disyunción)
# ---------------------------------------------------------------------------
# NINGUNO de estos campos puede pertenecer a FRAME_V1_FEATURE_NAMES. Ese es el
# invariante que hace no-circular a la variable criterio.

#: Campos externos por regla. ``card_testing`` usa el conteo de mismo monto en 1h
#: y el conteo de fallos por token en 1h (derivado de failed_payment_logs, campo
#: externo al vector de features del modelo).
RULE_FIELDS: Dict[str, FrozenSet[str]] = {
    "card_testing": frozenset({"same_amount_count_1h", "failed_count_1h"}),
    "velocity_extreme": frozenset({"user_txn_count_24h"}),
    "new_user_burst": frozenset({"user_account_age_days", "user_txn_count_1h"}),
    "refund_negative_control": frozenset({"status"}),
}

#: Unión de todos los campos externos usados por las reglas del scoreboard.
ALL_RULE_FIELDS: FrozenSet[str] = frozenset().union(*RULE_FIELDS.values())


# ---------------------------------------------------------------------------
# Acceso uniforme a campos (DataFrame | dict | Series)
# ---------------------------------------------------------------------------

_RowLike = Union[Mapping, pd.Series]
_Input = Union[pd.DataFrame, _RowLike]


def _is_frame(obj: _Input) -> bool:
    return isinstance(obj, pd.DataFrame)


def _get(obj: _Input, field: str, default: float = 0):
    """Extrae ``field`` como escalar (fila) o np.ndarray (DataFrame).

    Campos ausentes degradan al ``default`` (usuario sin historial), sin
    lanzar excepción — coherente con el manejo de nulos del feature calculator.
    """
    if _is_frame(obj):
        if field in obj.columns:
            return obj[field].fillna(default).to_numpy()
        return np.full(len(obj), default)
    # dict / Series
    val = obj.get(field, default) if hasattr(obj, "get") else default
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def _to_bool(mask, is_frame: bool):
    """Normaliza el resultado: ndarray[bool] para frame, bool escalar para fila."""
    if is_frame:
        return np.asarray(mask, dtype=bool)
    return bool(mask)


# ---------------------------------------------------------------------------
# Reglas — variable criterio (unión tipificada NO circular)
# ---------------------------------------------------------------------------


def card_testing(obj: _Input):
    """Card testing: ráfaga de mismo monto en 1h O ráfaga de fallos por token.

    Campos externos: ``same_amount_count_1h`` (conteo de pagos del mismo monto
    del usuario en la última hora, excluyendo la fila actual) y ``failed_count_1h``
    (conteo de fallos por ``user_token_id`` en 1h, derivado de
    ``failed_payment_logs``). Ambos son externos a FRAME_V1_FEATURE_NAMES.

    Dispara si ``same_amount_count_1h >= CARD_TESTING_SAME_AMOUNT_MIN`` (5) o
    ``failed_count_1h >= CARD_TESTING_FAILED_MIN`` (5).
    """
    is_frame = _is_frame(obj)
    same = _get(obj, "same_amount_count_1h", 0)
    failed = _get(obj, "failed_count_1h", 0)
    mask = (np.asarray(same) >= CARD_TESTING_SAME_AMOUNT_MIN) | (
        np.asarray(failed) >= CARD_TESTING_FAILED_MIN
    )
    return _to_bool(mask, is_frame)


def velocity_extreme(obj: _Input):
    """Velocidad extrema: ``user_txn_count_24h > VELOCITY_EXTREME_TXN_24H_MIN`` (100).

    Umbral ESTRICTO: 100 no dispara, 101 sí. Campo externo:
    ``user_txn_count_24h`` (conteo de transacciones del usuario en 24h, no está
    en el vector de features del modelo).
    """
    is_frame = _is_frame(obj)
    txn_24h = np.asarray(_get(obj, "user_txn_count_24h", 0))
    mask = txn_24h > VELOCITY_EXTREME_TXN_24H_MIN
    return _to_bool(mask, is_frame)


def new_user_burst(obj: _Input):
    """Usuario nuevo con ráfaga: edad < 14d Y ``user_txn_count_1h >= 3``.

    Edad ESTRICTA (< 14): 14 no dispara, 13 sí. Conteo con ``>=``: 3 dispara.
    Campos externos: ``user_account_age_days`` y ``user_txn_count_1h``.
    """
    is_frame = _is_frame(obj)
    age = np.asarray(_get(obj, "user_account_age_days", 0))
    txn_1h = np.asarray(_get(obj, "user_txn_count_1h", 0))
    mask = (age < NEW_USER_MAX_AGE_DAYS) & (txn_1h >= NEW_USER_BURST_TXN_1H_MIN)
    return _to_bool(mask, is_frame)


def typed_union(obj: _Input):
    """Unión tipificada: OR de card_testing, velocity_extreme y new_user_burst.

    Es la **variable criterio** del confirmatorio V2 (evaluación titular HE1/HE2).
    """
    is_frame = _is_frame(obj)
    if is_frame:
        mask = card_testing(obj) | velocity_extreme(obj) | new_user_burst(obj)
        return np.asarray(mask, dtype=bool)
    return bool(card_testing(obj) or velocity_extreme(obj) or new_user_burst(obj))


# ---------------------------------------------------------------------------
# Control negativo (refund) — banda de equivalencia HE3, NUNCA variable criterio
# ---------------------------------------------------------------------------


def refund_negative_control(obj: _Input):
    """Control negativo: ``status in {totally_refunded, refunded_to_credit}``.

    Rol EXCLUSIVO de control negativo (HE3): se espera que el score NO lo
    discrimine (bandas de equivalencia). El reembolso es un proxy administrativo
    post-hoc, jamás variable criterio (plan §1.1).
    """
    is_frame = _is_frame(obj)
    if is_frame:
        status = obj["status"] if "status" in obj.columns else pd.Series([""] * len(obj))
        return status.isin(REFUND_STATUSES).to_numpy(dtype=bool)
    status = _get(obj, "status", "")
    return bool(status in REFUND_STATUSES)


# ---------------------------------------------------------------------------
# Stub diferido — documentado (plan §4.1c / §4.1b #15)
# ---------------------------------------------------------------------------


def multi_account_token(obj: _Input):  # noqa: D401
    """STUB (diferido). Regla de flujo: cuenta nueva que se une a un token de
    gateway (``user_tokens.token``) ya compartido por >= 3 cuentas.

    Diferida en Fase 5A: requiere la superficie ``user_tokens`` y un guard de
    familias (``users_relations`` / ``user_children``) para evitar falsos
    positivos por tarjetas familiares legítimas (plan §4.1c, riesgos §10). No
    entra al scoreboard V2. El identificador es ``user_tokens.token`` — NUNCA
    ``last4 + card_brand`` (espacio de colisión, plan §4.1b #15).

    Raises:
        NotImplementedError: siempre; implementación diferida a la fase de reglas.
    """
    raise NotImplementedError(
        "multi_account_token está diferida (Fase de reglas): requiere la superficie "
        "user_tokens + guard de familias. No participa del confirmatorio V2."
    )


# ---------------------------------------------------------------------------
# Verificación de disyunción feature↔proxy (patrón validate_if40_pivot_disjoint)
# ---------------------------------------------------------------------------


def assert_disjoint_from_features(
    feature_names=FRAME_V1_FEATURE_NAMES,
) -> Dict[str, object]:
    """Verifica que NINGÚN campo usado por las reglas ∈ ``feature_names``.

    Este es el invariante de no-circularidad del confirmatorio V2: la variable
    criterio se construye sobre campos externos al vector del modelo. Reutiliza
    el patrón de ``scripts/validate_if40_pivot_disjoint.py`` (intersección
    señal↔feature).

    Args:
        feature_names: nombres de las features del modelo. Por defecto el
            contrato real ``FRAME_V1_FEATURE_NAMES`` (no una copia — se valida
            contra la fuente de verdad, plan §7.8).

    Returns:
        Dict con el reporte de disyunción: ``disjoint`` (bool), ``overlap``
        (lista ordenada de campos infractores) y ``rule_fields`` (mapa
        regla -> campos). Apto para registrarse en el JSON del scoreboard.

    Raises:
        ValueError: si algún campo de regla pertenece a ``feature_names``.
    """
    feature_set = set(feature_names)
    overlap = sorted(ALL_RULE_FIELDS & feature_set)
    report: Dict[str, object] = {
        "disjoint": len(overlap) == 0,
        "overlap": overlap,
        "rule_fields": {rule: sorted(fields) for rule, fields in RULE_FIELDS.items()},
        "n_feature_names": len(feature_set),
    }
    if overlap:
        raise ValueError(
            "Disyunción feature↔proxy violada: los campos de regla "
            f"{overlap} pertenecen a FRAME_V1_FEATURE_NAMES (proxy circular). "
            "La variable criterio dejaría de ser no-circular."
        )
    return report
