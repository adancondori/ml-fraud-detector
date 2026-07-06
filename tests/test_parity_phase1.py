"""Fase 1: paridad batch↔real-time del FrameV1FeatureCalculator.

Verifica que calculate(payment, context) y calculate_from_row(row) producen
vectores idénticos (diff <1e-8) para >=100 pagos estratificados de >=20 facilities.

Cubre también:
- test_calculate_from_row_no_time_zone_column: el calculator resuelve iana_tz
  vía el artefacto por facility_id sin necesitar columna time_zone en el row.
- test_magnitude_relative_to_facility: para amount > fmean, log_amount_fac > 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detector.scoring.context import UserContext
from fraud_detector.scoring.features_frame_v1 import (
    FRAME_V1_FEATURE_NAMES,
    FrameV1FeatureCalculator,
)

FE_PATH = "output/models/feature_engineer.joblib"
STATS_PATH = "output/models/facility_stats_v1.json"
GOLDEN_PARQUET = "data/processed/val_features_enriched.parquet"

N_ROWS = 100
MIN_FACILITIES = 20
TOL = 1e-8

pytestmark = pytest.mark.skipif(
    not Path(FE_PATH).exists()
    or not Path(STATS_PATH).exists()
    or not Path(GOLDEN_PARQUET).exists(),
    reason="requiere artefactos entrenados, facility_stats_v1.json y val parquet",
)


@pytest.fixture(scope="module")
def facility_stats() -> dict:
    with open(STATS_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def frame_calc(facility_stats) -> FrameV1FeatureCalculator:
    return FrameV1FeatureCalculator(
        facility_stats=facility_stats,
        feature_engineer_path=FE_PATH,
    )


@pytest.fixture(scope="module")
def golden_rows() -> pd.DataFrame:
    """Set dorado estratificado por facility_id: >=20 facilities, >=100 filas."""
    df = pd.read_parquet(GOLDEN_PARQUET)
    rows_per_fac = max(1, N_ROWS // MIN_FACILITIES)
    sampled = (
        df.groupby("facility_id", group_keys=False)
        .apply(lambda g: g.head(rows_per_fac), include_groups=False)
    )
    if "facility_id" not in sampled.columns:
        sampled = sampled.join(df[["facility_id"]])
    if len(sampled) < N_ROWS:
        sampled = df.head(N_ROWS)
    sampled = sampled.head(max(N_ROWS, len(sampled))).reset_index(drop=True)
    assert sampled["facility_id"].nunique() >= MIN_FACILITIES, (
        f"Set estratificado cubre solo {sampled['facility_id'].nunique()} facilities "
        f"(mínimo: {MIN_FACILITIES})"
    )
    assert len(sampled) >= N_ROWS, (
        f"Set tiene solo {len(sampled)} filas (mínimo: {N_ROWS})"
    )
    return sampled


def _row_to_payment(row) -> dict:
    """Construir dict de pago desde una fila del parquet enriquecido.

    El parquet almacena amount ya en USD (normalizado por FeatureEngineer).
    Para que calculate() reproduzca exactamente el mismo vector que calculate_from_row():
    - Pasar el amount ya en USD con currency='USD' → evita doble conversión
    - Para staff_amount_zscore: el artefacto (feature_engineer.joblib) almacena
      stats por moneda original. Pasamos original_currency como campo adicional
      para que _compute_frame_features haga el lookup correcto.
    - discount: reconstituido como discount_ratio * max(amount, 0.01) para
      reproducir el mismo discount_ratio que el parquet pre-computó.
    """
    amount = float(row.get("amount", 0) or 0)
    dr = float(row.get("discount_ratio", 0) or 0)
    discount_reconstituted = dr * max(amount, 0.01)
    original_currency = str(row.get("currency", "USD") or "USD").upper()
    return {
        "reservation_paid_out": amount,
        "created_at": row["created_at"],
        "facility_id": int(row["facility_id"]),
        "currency": "USD",           # amount ya en USD, sin doble conversión
        "original_currency": original_currency,  # para staff z-score lookup
        "discount": discount_reconstituted,
        "tip": 1.0 if float(row.get("has_tip", 0) or 0) > 0 else 0.0,
        "club_credit_flag": bool(int(row.get("is_club_credit", 0) or 0)),
        "paid_by_manager": bool(int(row.get("paid_by_manager", 0) or 0)),
    }


def _row_to_context(row) -> UserContext:
    """Construir UserContext desde una fila del parquet.

    Los campos rolling (amount_24h, debit_count_30d, etc.) se pasan directamente
    para que _compute_frame_features use los mismos valores que el parquet.
    """
    return UserContext(
        txn_count_24h=int(float(row.get("user_txn_count_24h", 0) or 0)),
        amount_24h=float(row.get("user_amount_24h", 0) or 0),
        time_since_last_txn=float(row.get("time_since_last_txn", 0) or 0),
        distinct_facilities_30d=int(float(row.get("user_distinct_facilities_30d", 0) or 0)),
        distinct_methods=int(float(row.get("user_distinct_methods", 0) or 0)),
        debit_count_30d=int(float(row.get("user_debit_count_30d", 0) or 0)),
        debit_amount_30d=float(row.get("user_debit_amount_30d", 0) or 0),
        credit_flow_ratio=float(row.get("credit_flow_ratio", 0) or 0),
        category_entropy_30d=float(row.get("category_entropy_30d", 0) or 0),
        merchandise_ratio_30d=float(row.get("user_merchandise_ratio_30d", 0) or 0),
        user_role=str(row.get("user_role", "player") or "player"),
        gateway_change_recent=float(row.get("gateway_change_recent", 0) or 0),
        is_main_gateway=float(row.get("is_main_gateway", 0) or 0),
        is_first_gateway_for_user=float(row.get("is_first_gateway_for_user", 0) or 0),
        source_change_recent=float(row.get("source_change_recent", 0) or 0),
    )


class TestFrameV1FeatureNames:
    """Verificar el contrato del feature set FS-frame-v1."""

    def test_feature_names_count(self):
        assert len(FRAME_V1_FEATURE_NAMES) == 31, (
            f"FRAME_V1_FEATURE_NAMES tiene {len(FRAME_V1_FEATURE_NAMES)} features, esperado 31"
        )

    def test_feature_names_unique(self):
        assert len(FRAME_V1_FEATURE_NAMES) == len(set(FRAME_V1_FEATURE_NAMES)), (
            "FRAME_V1_FEATURE_NAMES contiene duplicados"
        )

    def test_canonical_features_present(self):
        """Las features canónicas del FS-frame-v1 deben estar presentes."""
        required = [
            "log_amount_fac", "hour_sin_loc", "hour_cos_loc",
            "dow_sin_loc", "dow_cos_loc", "is_weekend_loc", "is_off_hours_loc",
            "amount_facility_ratio", "amount_fac_z", "user_amount_24h_fac",
            "user_debit_amount_30d_fac", "off_hours_high_value_loc", "staff_amount_zscore",
        ]
        for feat in required:
            assert feat in FRAME_V1_FEATURE_NAMES, f"Feature canónica faltante: {feat}"

    def test_eliminated_features_absent(self):
        """Features eliminadas del FS-frame-v1 no deben estar presentes."""
        eliminated = [
            "facility_avg_amount", "amount_usd_ratio", "log_amount",
            "hour_sin", "hour_cos", "day_of_week", "is_weekend", "is_off_hours",
            "user_txn_count_1h", "same_amount_count_1h", "same_amount_count_24h",
            "is_third_party_payment", "user_account_age_days",
        ]
        for feat in eliminated:
            assert feat not in FRAME_V1_FEATURE_NAMES, (
                f"Feature eliminada no debe estar en FRAME_V1_FEATURE_NAMES: {feat}"
            )


class TestFrameV1Calculator:
    """Paridad batch↔real-time del FrameV1FeatureCalculator."""

    def test_calculate_returns_shape_30(self, golden_rows, frame_calc):
        """calculate() produce vector de shape (30,)."""
        row = golden_rows.iloc[0]
        payment = _row_to_payment(row)
        context = _row_to_context(row)
        vec = frame_calc.calculate(payment, context)
        assert vec.shape == (31,), f"Esperado (31,), obtenido {vec.shape}"

    def test_calculate_from_row_returns_shape_30(self, golden_rows, frame_calc):
        """calculate_from_row() produce vector de shape (30,)."""
        row = golden_rows.iloc[0]
        vec = frame_calc.calculate_from_row(row)
        assert vec.shape == (31,), f"Esperado (31,), obtenido {vec.shape}"

    def test_frame_features_parity(self, golden_rows, frame_calc):
        """calculate() == calculate_from_row() para >=100 pagos, diff <1e-8."""
        max_diffs = []
        for _, row in golden_rows.iterrows():
            payment = _row_to_payment(row)
            context = _row_to_context(row)
            rt_vec = frame_calc.calculate(payment, context).astype(np.float64)
            batch_vec = frame_calc.calculate_from_row(row).astype(np.float64)
            diff = np.max(np.abs(rt_vec - batch_vec))
            max_diffs.append(diff)
            assert diff < TOL, (
                f"Paridad fallida: diff={diff:.2e} (tol={TOL:.0e}) "
                f"facility={row['facility_id']}\n"
                f"  rt  = {rt_vec}\n"
                f"  bat = {batch_vec}"
            )
        overall_max = max(max_diffs)
        assert overall_max < TOL, (
            f"Max diff global={overall_max:.2e} sobre {len(golden_rows)} filas, "
            f"{golden_rows['facility_id'].nunique()} facilities"
        )

    def test_calculate_from_row_no_time_zone_column(self, golden_rows, frame_calc):
        """calculate_from_row() funciona sin columna time_zone — zona resuelta del artefacto.

        Elimina la clave time_zone del row (si existe) y verifica que el calculator
        produce el vector (30,) sin KeyError, resolviendo iana_tz por facility_id.
        """
        row = golden_rows.iloc[0].copy()
        # Eliminar time_zone si existe (pitfall 1 del research)
        if hasattr(row, "drop"):
            row = row.drop(labels=["time_zone"], errors="ignore")
        assert "time_zone" not in row.index, "time_zone debe estar ausente del row"
        # No debe levantar KeyError ni excepción
        vec = frame_calc.calculate_from_row(row)
        assert vec.shape == (31,), f"Esperado (31,), obtenido {vec.shape}"
        assert not np.any(np.isnan(vec)), "El vector no debe contener NaN"

    def test_magnitude_relative_to_facility(self, golden_rows, frame_calc, facility_stats):
        """Para amount > fmean, log_amount_fac > 0 y amount_facility_ratio > 1."""
        fac_idx = FRAME_V1_FEATURE_NAMES.index("log_amount_fac")
        ratio_idx = FRAME_V1_FEATURE_NAMES.index("amount_facility_ratio")
        tested = 0
        for _, row in golden_rows.iterrows():
            fid = str(int(row["facility_id"]))
            entry = facility_stats["facilities"].get(fid, {})
            fmean = entry.get("mean") or facility_stats["global_fallback"]["mean"]
            if fmean is None:
                continue
            amount = float(row["amount"])
            if amount <= fmean:
                continue
            payment = _row_to_payment(row)
            context = _row_to_context(row)
            vec = frame_calc.calculate(payment, context)
            assert vec[fac_idx] > 0, (
                f"log_amount_fac debe ser >0 cuando amount ({amount:.2f}) > fmean ({fmean:.2f}), "
                f"obtenido {vec[fac_idx]:.4f}"
            )
            assert vec[ratio_idx] > 1, (
                f"amount_facility_ratio debe ser >1 cuando amount ({amount:.2f}) > fmean ({fmean:.2f}), "
                f"obtenido {vec[ratio_idx]:.4f}"
            )
            tested += 1
        assert tested >= 5, f"Solo {tested} filas con amount > fmean — fixture insuficiente"

    def test_facility_coverage_multiple_facilities(self, golden_rows, frame_calc):
        """El set estratificado cubre >=20 facilities diferentes."""
        assert golden_rows["facility_id"].nunique() >= MIN_FACILITIES

    def test_no_nan_in_output(self, golden_rows, frame_calc):
        """El vector de features no debe contener NaN para ningún pago del golden set."""
        for _, row in golden_rows.iterrows():
            vec = frame_calc.calculate_from_row(row)
            assert not np.any(np.isnan(vec)), (
                f"NaN en el vector para facility_id={row['facility_id']}"
            )
