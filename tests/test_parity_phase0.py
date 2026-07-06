"""Fase 0: paridad batch <-> real-time del SingleFeatureCalculator.

Verifica que los valores calculados por SingleFeatureCalculator.calculate()
en el path real-time son idénticos (<1e-6) a los valores precalculados por
FeatureEngineer.transform() almacenados en val_features_enriched.parquet.

Nota: solo se verifican facility_avg_amount y staff_amount_zscore porque son
las features cuyos valores dependían de los stats aprendidos en fit() y eran
los afectados por el bug de getattr. Las demás features (velocidad, comportamiento
rolling) dependen de context rolling que el parquet enriquecido no reconstruye 1:1
desde un único payment dict, por lo que no se verifican aquí.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detector.features.engineering import FEATURE_NAMES
from fraud_detector.scoring.context import UserContext
from fraud_detector.scoring.features import SingleFeatureCalculator

FE_PATH = "output/models/feature_engineer.joblib"
GOLDEN_PARQUET = "data/processed/val_features_enriched.parquet"

FACILITY_AVG_IDX = FEATURE_NAMES.index("facility_avg_amount")   # 19
STAFF_ZSCORE_IDX = FEATURE_NAMES.index("staff_amount_zscore")   # 27
TOL = 1e-6
N_ROWS = 500
MIN_FACILITIES = 20

pytestmark = pytest.mark.skipif(
    not Path(FE_PATH).exists() or not Path(GOLDEN_PARQUET).exists(),
    reason="requiere artefactos entrenados y val_features_enriched.parquet",
)


@pytest.fixture(scope="module")
def golden_rows():
    """Set dorado estratificado por facility_id: >=20 facilities, >=500 filas."""
    df = pd.read_parquet(GOLDEN_PARQUET)
    # Estratificar: tomar ~N_ROWS/MIN_FACILITIES filas por facility para cubrir diversidad.
    sampled = (
        df.groupby("facility_id", group_keys=False)
        .apply(lambda g: g.head(max(1, N_ROWS // MIN_FACILITIES)), include_groups=False)
    )
    # Restore facility_id after include_groups=False drops grouping column
    if "facility_id" not in sampled.columns:
        sampled = sampled.join(df[["facility_id"]])
    if len(sampled) < N_ROWS:
        sampled = df.head(N_ROWS)
    sampled = sampled.head(max(N_ROWS, len(sampled))).reset_index(drop=True)
    assert sampled["facility_id"].nunique() >= MIN_FACILITIES, (
        f"Set estratificado cubre solo {sampled['facility_id'].nunique()} facilities "
        f"(mínimo requerido: {MIN_FACILITIES})"
    )
    assert len(sampled) >= N_ROWS, (
        f"Set estratificado tiene solo {len(sampled)} filas (mínimo: {N_ROWS})"
    )
    return sampled


@pytest.fixture(scope="module")
def calculator():
    """SingleFeatureCalculator cargado desde el artefacto de producción."""
    return SingleFeatureCalculator(FE_PATH)


def _row_to_payment(row) -> dict:
    """Construir dict de pago desde una fila del parquet enriquecido.

    Solo se pasan los campos que afectan facility_avg_amount y staff_amount_zscore:
    amount (reservation_paid_out), facility_id, currency, created_at.
    discount y tip en 0.0 — no afectan las features de paridad verificadas.
    """
    return {
        "reservation_paid_out": float(row.get("amount", 0) or 0),
        "created_at": row["created_at"],
        "facility_id": int(row["facility_id"]),
        "currency": str(row.get("currency", "USD") or "USD"),
        "discount": 0.0,
        "tip": 0.0,
    }


def _row_to_context(row) -> UserContext:
    """Construir UserContext desde una fila del parquet.

    Solo user_role es load-bearing para staff_amount_zscore; el resto en defaults.
    """
    return UserContext(user_role=str(row.get("user_role", "player") or "player"))


class TestParityPhase0:
    """Guardrail de regresión: paridad batch↔real-time sobre >=500 pagos estratificados."""

    def test_facility_avg_parity(self, golden_rows, calculator):
        """facility_avg_amount debe ser idéntico en batch y real-time para el mismo pago."""
        deltas = []
        for _, row in golden_rows.iterrows():
            payment = _row_to_payment(row)
            context = _row_to_context(row)
            rt_features = calculator.calculate(payment, context)

            batch_val = float(row["facility_avg_amount"])
            rt_val = float(rt_features[FACILITY_AVG_IDX])
            deltas.append(abs(batch_val - rt_val))

        max_delta = max(deltas)
        assert max_delta < TOL, (
            f"Paridad fallida en facility_avg_amount: "
            f"max_delta={max_delta:.2e} (tolerancia={TOL:.0e}) "
            f"sobre {len(golden_rows)} filas, {golden_rows['facility_id'].nunique()} facilities"
        )

    def test_staff_zscore_parity(self, golden_rows, calculator):
        """staff_amount_zscore debe ser idéntico en batch y real-time.

        Depende de amount, user_role y currency — los tres campos se pasan
        desde el parquet para garantizar el mismo cómputo de z-score.
        """
        deltas = []
        for _, row in golden_rows.iterrows():
            payment = _row_to_payment(row)
            context = _row_to_context(row)
            rt_features = calculator.calculate(payment, context)

            batch_val = float(row["staff_amount_zscore"])
            rt_val = float(rt_features[STAFF_ZSCORE_IDX])
            deltas.append(abs(batch_val - rt_val))

        max_delta = max(deltas)
        assert max_delta < TOL, (
            f"Paridad fallida en staff_amount_zscore: "
            f"max_delta={max_delta:.2e} (tolerancia={TOL:.0e}) "
            f"sobre {len(golden_rows)} filas"
        )

    def test_facility_avgs_not_global_fallback(self, calculator):
        """Regresión directa del bug getattr: _facility_avgs no puede ser vacío ni uniforme.

        Con el bug activo, getattr(…, '_facility_avg', {}) devolvía {} y todas
        las lookups caían al global mean (368.61). Post-fix, debe haber >1 facility
        y los valores deben variar entre facilities (no todos iguales al global).
        """
        assert len(calculator._facility_avgs) > 1, (
            f"_facility_avgs tiene solo {len(calculator._facility_avgs)} entry — "
            "regresión del bug getattr detectada"
        )
        vals = list(calculator._facility_avgs.values())
        unique_vals = set(round(v, 4) for v in vals)
        assert len(unique_vals) > 1, (
            f"Todas las {len(vals)} facilities mapean al mismo valor — "
            "posible regresión del bug getattr (fallback al global)"
        )
