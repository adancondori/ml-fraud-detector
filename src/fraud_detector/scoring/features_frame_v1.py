"""FrameV1FeatureCalculator — feature calculator for FS-frame-v1 (30 features).

Frame-normalized: magnitude features are relative to per-facility distribution
(not USD absolute), temporal features use local IANA time with DST (not UTC).

Provides two surfaces sharing identical arithmetic via _compute_frame_features:
  - calculate(payment, context): real-time path (payment dict + UserContext)
  - calculate_from_row(row): batch/parity path (enriched parquet row)

Paridad garantizada: ambas superficies producen vectores idénticos (<1e-8)
para la misma transacción.

FS-frame-v1 satisface FRAME-01, FRAME-02, FRAME-03.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from fraud_detector.scoring.context import UserContext
from fraud_detector.utils.currency import normalize_amount_value

# ---------------------------------------------------------------------------
# Feature contract — FS-frame-v1 (30 features, fuente canónica: frame_version(DISJOINT30))
# ---------------------------------------------------------------------------

FRAME_V1_FEATURE_NAMES = [
    "log_amount_fac",  # log1p(amount / (fmean + 0.01))
    "discount_ratio",  # discount / max(amount, 0.01)
    "has_tip",  # 1 si tip > 0
    "hour_sin_loc",  # sin(2π * local_hour / 24)
    "hour_cos_loc",  # cos(2π * local_hour / 24)
    "dow_sin_loc",  # sin(2π * local_dow / 7)
    "dow_cos_loc",  # cos(2π * local_dow / 7)
    "is_weekend_loc",  # local_dow >= 5
    "is_off_hours_loc",  # local_hour in OFF_HOURS
    "time_since_last_txn",  # segundos desde última transacción
    "user_amount_24h_fac",  # user_amount_24h / (fmean + 0.01)
    "user_distinct_facilities_30d",
    "user_distinct_methods",
    "amount_facility_ratio",  # amount / (fmean + 0.01)
    "is_club_credit",
    "user_debit_count_30d",
    "user_debit_amount_30d_fac",  # user_debit_amount_30d / (fmean + 0.01)
    "credit_flow_ratio",  # debit_amount / max(prepaid, 0.01)
    "is_staff",
    "paid_by_manager",
    "staff_amount_zscore",  # (amount - staff_mean) / staff_std
    "category_entropy_30d",
    "user_merchandise_ratio_30d",
    "small_amount_at_facility",  # amount_facility_ratio < 0.2
    "very_small_amount_at_facility",  # amount_facility_ratio < 0.05
    "off_hours_high_value_loc",  # is_off_hours_loc AND amount_facility_ratio > 3
    "gateway_change_recent",
    "is_main_gateway",
    "is_first_gateway_for_user",
    "source_change_recent",
]

assert (
    len(FRAME_V1_FEATURE_NAMES) == 30
), f"FS-frame-v1 contract violated: expected 30 features, got {len(FRAME_V1_FEATURE_NAMES)}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OFF_HOURS: frozenset = frozenset({23, 0, 1, 2, 3, 4, 5, 6})

_FEATURE_IDX: Dict[str, int] = {name: i for i, name in enumerate(FRAME_V1_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# FrameV1FeatureCalculator
# ---------------------------------------------------------------------------


class FrameV1FeatureCalculator:
    """Calcula el vector FS-frame-v1 (30 features) para una transacción.

    Carga:
      - facility_stats: dict del artefacto facility_stats_v1.json (ya deserializado)
      - feature_engineer_path: ruta a feature_engineer.joblib (para staff z-score stats)

    Las dos superficies públicas delegan a _compute_frame_features — única fuente
    de aritmética — para garantizar paridad batch↔real-time.
    """

    def __init__(self, facility_stats: dict, feature_engineer_path: str):
        self._stats = facility_stats

        # Cargar solo las stats de staff desde feature_engineer.joblib
        fe = joblib.load(feature_engineer_path)
        self._staff_role_currency = fe._groups[6]._role_currency_stats
        self._staff_currency = fe._groups[6]._currency_stats
        self._staff_global_mean: float = float(fe._groups[6]._global_mean)
        self._staff_global_std: float = float(fe._groups[6]._global_std or 1.0)

    # ------------------------------------------------------------------
    # Public: static utility
    # ------------------------------------------------------------------

    @staticmethod
    def _local_hour_dow(ts_utc_naive: pd.Timestamp, iana_tz: str) -> Tuple[int, int]:
        """Convierte un timestamp UTC naive a hora y día-de-semana locales.

        Args:
            ts_utc_naive: Timestamp naive (sin tzinfo) interpretado como UTC.
            iana_tz: Nombre de zona IANA (e.g. "America/New_York").

        Returns:
            (hour_local, dow_local) donde dow 0=lunes, 6=domingo.
        """
        utc_aware = ts_utc_naive.tz_localize("UTC")
        local = utc_aware.astimezone(ZoneInfo(iana_tz))
        return local.hour, local.dayofweek

    # ------------------------------------------------------------------
    # Private: facility lookup with fallback chain
    # ------------------------------------------------------------------

    def _lookup_facility(self, fid: int) -> Tuple[float, float, float, str]:
        """Retorna (fmean, fmedian, iqr_guarded, iana_tz) con fallback chain.

        Orden: facility (n>=30) → currency-fallback stats → global stats.
        iana_tz siempre disponible: todas las 1876 facilities tienen entrada en el
        artefacto con iana_tz. Para facilities desconocidas se usa "Etc/UTC".
        """
        fid_str = str(fid)
        entry = self._stats["facilities"].get(fid_str)
        iana_tz = entry.get("iana_tz", "Etc/UTC") if entry is not None else "Etc/UTC"

        if entry is not None:
            fmean = float(entry.get("mean") or 0)
            fmedian = float(entry.get("median") or 0)
            iqr_guarded = float(entry.get("iqr_guarded") or 1.0)
            if fmean > 0:
                # Cubre tanto fallback_level=facility como fallback_level=currency
                # (ambos tienen mean positivo en el artefacto)
                return fmean, fmedian, iqr_guarded, iana_tz

        # Fallback global: facility desconocida o mean==0 (raro)
        g = self._stats["global_fallback"]
        return (
            float(g["mean"]),
            float(g["median"]),
            float(g["iqr_guarded"]),
            iana_tz,
        )

    def _lookup_staff_zscore(self, amount: float, currency: str, role: str) -> float:
        """Calcula staff_amount_zscore con fallback chain (role,currency)→currency→global.

        Mismo orden de lookup que SingleFeatureCalculator (decisión 00-01:
        usar actual_role sin forzar 'player').
        """
        actual_role = role or "player"
        currency_key = (actual_role, currency)
        if currency_key in self._staff_role_currency:
            s = self._staff_role_currency[currency_key]
        elif currency in self._staff_currency:
            s = self._staff_currency[currency]
        else:
            s = {"mean": self._staff_global_mean, "std": self._staff_global_std}
        staff_mean = float(s["mean"])
        staff_std = float(s.get("std") or 1.0) or 1.0
        return (amount - staff_mean) / staff_std

    # ------------------------------------------------------------------
    # Private: core arithmetic (única fuente de verdad — garantiza paridad)
    # ------------------------------------------------------------------

    def _compute_frame_features(
        self,
        *,
        amount_usd: float,
        facility_id: int,
        created_at: pd.Timestamp,
        currency: str,
        currency_original: str,
        user_role: str,
        discount_ratio: float,
        has_tip: float,
        time_since_last_txn: float,
        user_amount_24h: float,
        user_distinct_facilities_30d: float,
        user_distinct_methods: float,
        user_debit_count_30d: float,
        user_debit_amount_30d: float,
        credit_flow_ratio: float,
        is_club_credit: float,
        paid_by_manager: float,
        category_entropy_30d: float,
        user_merchandise_ratio_30d: float,
        gateway_change_recent: float,
        is_main_gateway: float,
        is_first_gateway_for_user: float,
        source_change_recent: float,
    ) -> np.ndarray:
        """Única implementación de la aritmética de marco.

        Recibe primitivos; retorna np.ndarray de shape (30,) en orden FRAME_V1_FEATURE_NAMES.
        """
        # 1. Lookup facility stats (mean, median, iqr_guarded, iana_tz)
        fmean, fmedian, iqr_guarded, iana_tz = self._lookup_facility(facility_id)

        # 2. Magnitud relativa a facility
        log_amount_fac = math.log1p(amount_usd / (fmean + 0.01))
        amount_facility_ratio = amount_usd / (fmean + 0.01)
        user_amount_24h_fac = user_amount_24h / (fmean + 0.01)
        user_debit_amount_30d_fac = user_debit_amount_30d / (fmean + 0.01)

        # 3. Temporales en hora local con DST
        hour_loc, dow_loc = self._local_hour_dow(created_at, iana_tz)
        hour_sin_loc = math.sin(2 * math.pi * hour_loc / 24)
        hour_cos_loc = math.cos(2 * math.pi * hour_loc / 24)
        dow_sin_loc = math.sin(2 * math.pi * dow_loc / 7)
        dow_cos_loc = math.cos(2 * math.pi * dow_loc / 7)
        is_weekend_loc = float(dow_loc >= 5)
        is_off_hours_loc = float(hour_loc in OFF_HOURS)

        # 4. Interacciones derivadas (usan amount_facility_ratio calculado arriba)
        small_amount_at_facility = float(amount_facility_ratio < 0.2)
        very_small_amount_at_facility = float(amount_facility_ratio < 0.05)
        off_hours_high_value_loc = float(is_off_hours_loc > 0 and amount_facility_ratio > 3)

        # 5. Staff z-score — usa moneda original para lookup en stats aprendidos
        staff_amount_zscore = self._lookup_staff_zscore(amount_usd, currency_original, user_role)

        # 6. is_staff — roles que son staff
        is_staff = float(user_role in ("court_manager", "court_operator", "teacher"))

        # 7. Ensamblar vector en orden FRAME_V1_FEATURE_NAMES
        return np.array(
            [
                log_amount_fac,
                discount_ratio,
                has_tip,
                hour_sin_loc,
                hour_cos_loc,
                dow_sin_loc,
                dow_cos_loc,
                is_weekend_loc,
                is_off_hours_loc,
                time_since_last_txn,
                user_amount_24h_fac,
                float(user_distinct_facilities_30d),
                float(user_distinct_methods),
                amount_facility_ratio,
                float(is_club_credit),
                float(user_debit_count_30d),
                user_debit_amount_30d_fac,
                credit_flow_ratio,
                is_staff,
                float(paid_by_manager),
                staff_amount_zscore,
                category_entropy_30d,
                user_merchandise_ratio_30d,
                small_amount_at_facility,
                very_small_amount_at_facility,
                off_hours_high_value_loc,
                float(gateway_change_recent),
                float(is_main_gateway),
                float(is_first_gateway_for_user),
                float(source_change_recent),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Public: surface 1 — real-time path
    # ------------------------------------------------------------------

    def calculate(self, payment: Dict, context: UserContext) -> np.ndarray:
        """Calcula el vector FS-frame-v1 desde un pago real-time + UserContext.

        amount se normaliza a USD usando el patrón _payment_with_usd_amounts.
        La zona IANA se resuelve desde el artefacto por facility_id.

        Args:
            payment: Dict con campos del pago (reservation_paid_out, currency,
                     created_at, facility_id, discount, tip, club_credit_flag,
                     paid_by_manager).
            context: UserContext con agregados rolling del usuario.

        Returns:
            np.ndarray de shape (30,) en orden FRAME_V1_FEATURE_NAMES.
        """
        # Normalizar amount a USD
        payment_usd = _payment_with_usd_amounts(payment)
        amount_usd = float(payment_usd.get("reservation_paid_out", 0) or 0)
        discount = float(payment_usd.get("discount", 0) or 0)
        tip = float(payment_usd.get("tip", 0) or 0)

        fid = int(payment.get("facility_id", 0))
        # currency es la moneda de la transacción (para normalización USD).
        # original_currency es para el lookup de staff stats (aprendidos por moneda original).
        # Si se pasa 'original_currency', se usa para staff zscore; si no, se usa 'currency'.
        currency = (payment.get("currency") or "USD").upper()
        currency_original = (payment.get("original_currency") or currency).upper()
        created_at = pd.Timestamp(payment["created_at"])
        if created_at.tzinfo is not None:
            created_at = created_at.tz_convert("UTC").tz_localize(None)

        discount_ratio = discount / max(amount_usd, 0.01)
        has_tip = float(tip > 0)

        # time_since_last_txn: usar campo pre-computado si está disponible, sino derivar
        if context.time_since_last_txn >= 0:
            time_since = float(context.time_since_last_txn)
        elif context.last_txn_at is not None:
            time_since = max((created_at - pd.Timestamp(context.last_txn_at)).total_seconds(), 0.0)
        else:
            time_since = 0.0

        # credit_flow_ratio: usar campo pre-computado si está disponible, sino derivar
        if context.credit_flow_ratio >= 0:
            credit_flow = float(context.credit_flow_ratio)
        else:
            prepaid = max(float(context.prepaid_spend_30d or 0), 0.01)
            credit_flow = float(context.debit_amount_30d or 0) / prepaid

        # category_entropy_30d
        if context.category_entropy_30d >= 0:
            cat_entropy = float(context.category_entropy_30d)
        else:
            cat_entropy = _shannon_entropy(context.categories_30d)

        is_club_credit = float(bool(payment.get("club_credit_flag")))
        paid_by_mgr = float(bool(payment.get("paid_by_manager")))

        return self._compute_frame_features(
            amount_usd=amount_usd,
            facility_id=fid,
            created_at=created_at,
            currency=currency,
            currency_original=currency_original,
            user_role=context.user_role,
            discount_ratio=discount_ratio,
            has_tip=has_tip,
            time_since_last_txn=time_since,
            user_amount_24h=float(context.amount_24h or 0),
            user_distinct_facilities_30d=float(context.distinct_facilities_30d or 0),
            user_distinct_methods=float(context.distinct_methods or 0),
            user_debit_count_30d=float(context.debit_count_30d or 0),
            user_debit_amount_30d=float(context.debit_amount_30d or 0),
            credit_flow_ratio=credit_flow,
            is_club_credit=is_club_credit,
            paid_by_manager=paid_by_mgr,
            category_entropy_30d=cat_entropy,
            user_merchandise_ratio_30d=float(context.merchandise_ratio_30d or 0),
            gateway_change_recent=float(context.gateway_change_recent or 0),
            is_main_gateway=float(context.is_main_gateway or 0),
            is_first_gateway_for_user=float(context.is_first_gateway_for_user or 0),
            source_change_recent=float(context.source_change_recent or 0),
        )

    # ------------------------------------------------------------------
    # Public: surface 2 — batch/parity path
    # ------------------------------------------------------------------

    def calculate_from_row(self, row) -> np.ndarray:
        """Calcula el vector FS-frame-v1 desde una fila del parquet enriquecido.

        La zona IANA se resuelve SIEMPRE desde self._stats por facility_id.
        No requiere columna time_zone en el row (Pitfall 1).

        amount en el row ya está en USD (normalizado por FeatureEngineer).

        Args:
            row: pd.Series o dict con campos del parquet enriquecido.

        Returns:
            np.ndarray de shape (30,) en orden FRAME_V1_FEATURE_NAMES.
        """
        amount_usd = float(row.get("amount", 0) or 0)
        fid = int(row.get("facility_id", 0))
        currency = (str(row.get("currency", "USD") or "USD")).upper()
        created_at = pd.Timestamp(row["created_at"])
        if created_at.tzinfo is not None:
            created_at = created_at.tz_convert("UTC").tz_localize(None)

        discount_ratio = float(row.get("discount_ratio", 0) or 0)
        has_tip = float(int(row.get("has_tip", 0) or 0))

        # Campos rolling que el parquet ya tiene pre-computados
        time_since = float(row.get("time_since_last_txn", 0) or 0)
        user_amount_24h = float(row.get("user_amount_24h", 0) or 0)
        user_debit_amount_30d = float(row.get("user_debit_amount_30d", 0) or 0)
        credit_flow = float(row.get("credit_flow_ratio", 0) or 0)
        user_role = str(row.get("user_role", "player") or "player")
        cat_entropy = float(row.get("category_entropy_30d", 0) or 0)

        return self._compute_frame_features(
            amount_usd=amount_usd,
            facility_id=fid,
            created_at=created_at,
            currency=currency,
            currency_original=currency,  # parquet currency IS the original
            user_role=user_role,
            discount_ratio=discount_ratio,
            has_tip=has_tip,
            time_since_last_txn=time_since,
            user_amount_24h=user_amount_24h,
            user_distinct_facilities_30d=float(row.get("user_distinct_facilities_30d", 0) or 0),
            user_distinct_methods=float(row.get("user_distinct_methods", 0) or 0),
            user_debit_count_30d=float(row.get("user_debit_count_30d", 0) or 0),
            user_debit_amount_30d=user_debit_amount_30d,
            credit_flow_ratio=credit_flow,
            is_club_credit=float(int(row.get("is_club_credit", 0) or 0)),
            paid_by_manager=float(int(row.get("paid_by_manager", 0) or 0)),
            category_entropy_30d=cat_entropy,
            user_merchandise_ratio_30d=float(row.get("user_merchandise_ratio_30d", 0) or 0),
            gateway_change_recent=float(row.get("gateway_change_recent", 0) or 0),
            is_main_gateway=float(row.get("is_main_gateway", 0) or 0),
            is_first_gateway_for_user=float(row.get("is_first_gateway_for_user", 0) or 0),
            source_change_recent=float(row.get("source_change_recent", 0) or 0),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _payment_with_usd_amounts(payment: Dict) -> Dict:
    """Normaliza reservation_paid_out, discount y tip a USD.

    Patrón reutilizado de EnrichedFeatureCalculator._payment_with_usd_amounts.
    """
    currency = payment.get("currency")
    out = dict(payment)
    out["reservation_paid_out"] = normalize_amount_value(
        payment.get("reservation_paid_out"), currency
    )
    out["discount"] = normalize_amount_value(payment.get("discount"), currency)
    out["tip"] = normalize_amount_value(payment.get("tip"), currency)
    return out


def _shannon_entropy(categories: list) -> float:
    """Entropía de Shannon para una lista de categorías."""
    from collections import Counter

    if not categories:
        return 0.0
    counts = Counter(categories)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
