"""
Tests de Gate B — Validación del catálogo de 20 features.

Ejecutar antes de continuar a Fase 4 (Preprocesamiento).
Todos deben pasar: pytest tests/test_features.py -v
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from fraud_detector.features.engineering import (
    FEATURE_NAMES,
    FEATURE_NAMES_19,
    FeatureEngineer,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_txn(
    user_id: int,
    created_at: str,
    facility_id: int = 1,
    amount: float = 100.0,
    discount: float = 0.0,
    tip: float = 0.0,
    payment_method: str = "card",
    status: str = "paid",
    txn_id: int = None,
) -> dict:
    return {
        "id": txn_id,
        "user_id": user_id,
        "facility_id": facility_id,
        "created_at": pd.Timestamp(created_at),
        "amount": amount,
        "discount": discount,
        "tip": tip,
        "payment_method": payment_method,
        "status": status,
    }


@pytest.fixture(scope="module")
def sample_data():
    """Dataset sintético con múltiples usuarios y patrones variados."""
    rows = []
    txn_id = 1

    # Usuario 1: 5 transacciones en distintos horarios, 2 facilities, sin reversiones
    for i, (ts, fac, amt) in enumerate([
        ("2025-01-01 08:00:00", 1, 200.0),
        ("2025-01-01 09:30:00", 2, 150.0),
        ("2025-01-02 23:00:00", 1, 300.0),
        ("2025-01-03 02:00:00", 1, 50.0),
        ("2025-01-10 10:00:00", 3, 400.0),
    ]):
        rows.append(_make_txn(1, ts, fac, amt, txn_id=txn_id))
        txn_id += 1

    # Usuario 2: 3 transacciones con una reversión
    for ts, fac, amt, status in [
        ("2025-01-01 10:00:00", 1, 100.0, "paid"),
        ("2025-01-02 11:00:00", 1, 200.0, "totally_refunded"),
        ("2025-01-03 12:00:00", 2, 120.0, "paid"),
    ]:
        rows.append(_make_txn(2, ts, fac, amt, status=status, txn_id=txn_id))
        txn_id += 1

    # Usuario 3: 2 transacciones con propina y descuento
    for ts, amt, disc, tip in [
        ("2025-01-05 15:00:00", 500.0, 50.0, 20.0),
        ("2025-01-06 16:00:00", 300.0, 0.0, 0.0),
    ]:
        rows.append(_make_txn(3, ts, 2, amt, discount=disc, tip=tip, txn_id=txn_id))
        txn_id += 1

    # Usuario 4: transacción con amount = 0 (edge case)
    rows.append(_make_txn(4, "2025-01-07 08:00:00", 1, 0.0, txn_id=txn_id))
    txn_id += 1

    # Usuario 5: cold-start (solo aparece en val, no en train)
    rows.append(_make_txn(5, "2025-02-01 10:00:00", 1, 100.0, txn_id=txn_id))

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df.sort_values(["user_id", "created_at"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def train_val_split(sample_data):
    """Divide el dataset en train (Ene) y val (Feb)."""
    df_train = sample_data[sample_data["created_at"] < "2025-02-01"].copy()
    df_val = sample_data[sample_data["created_at"] >= "2025-02-01"].copy()
    return df_train, df_val


@pytest.fixture(scope="module")
def fitted_engineer_and_features(train_val_split):
    """FeatureEngineer fit en train + features de train y val."""
    df_train, df_val = train_val_split
    fe = FeatureEngineer()
    train_feat = fe.fit_transform(df_train)
    val_feat = fe.transform(df_val)
    return fe, train_feat, val_feat


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFeatureNamesCatalog:
    """Gate B — Tests #1 y #4: integridad del catálogo."""

    def test_feature_names_count(self):
        """FEATURE_NAMES debe tener exactamente 20 entradas sin duplicados."""
        assert len(FEATURE_NAMES) == 20
        assert len(set(FEATURE_NAMES)) == 20

    def test_feature_names_19_count(self):
        """FEATURE_NAMES_19 debe tener 19 entradas (sin user_reversal_ratio_30d)."""
        assert len(FEATURE_NAMES_19) == 19
        assert "user_reversal_ratio_30d" not in FEATURE_NAMES_19

    def test_feature_names_order(self):
        """Las 5 primeras features deben ser las transaccionales."""
        assert FEATURE_NAMES[:5] == [
            "amount", "log_amount", "amount_usd_ratio", "discount_ratio", "has_tip"
        ]


class TestFirstTransactionZero:
    """Gate B — Test #1: primera transacción tiene counts de velocidad == 0."""

    def test_first_txn_velocity_counts_zero(self, fitted_engineer_and_features):
        _, train_feat, _ = fitted_engineer_and_features
        first_txns = train_feat.sort_values("created_at").groupby("user_id").first()
        assert (first_txns["user_txn_count_1h"] == 0).all(), \
            "Primera txn de algún usuario tiene user_txn_count_1h != 0"
        assert (first_txns["user_txn_count_24h"] == 0).all(), \
            "Primera txn de algún usuario tiene user_txn_count_24h != 0"

    def test_first_txn_distinct_facilities_zero(self, fitted_engineer_and_features):
        """Antes de la primera transacción, user_distinct_facilities_cumul == 0."""
        _, train_feat, _ = fitted_engineer_and_features
        first_txns = train_feat.sort_values("created_at").groupby("user_id").first()
        assert (first_txns["user_distinct_facilities_cumul"] == 0).all(), \
            "Primera txn de algún usuario tiene distinct_facilities != 0"


class TestFacilityAvgMatchesTrain:
    """Gate B — Test #3: facility_avg_amount coincide con media del training."""

    def test_facility_avg_matches_train(self, train_val_split, fitted_engineer_and_features):
        df_train, _ = train_val_split
        fe, train_feat, _ = fitted_engineer_and_features

        # Obtener el grupo ContextualFeatures
        contextual = next(
            g for g in fe._groups
            if g.__class__.__name__ == "ContextualFeatures"
        )
        train_avgs = df_train.groupby("facility_id")["amount"].mean()
        for fid, avg in train_avgs.items():
            assert np.isclose(contextual._facility_avg_amount[fid], avg), \
                f"facility_id={fid}: esperado {avg:.4f}, obtenido {contextual._facility_avg_amount[fid]:.4f}"


class TestNoNans:
    """Gate B — Test #5: no hay NaN en ninguna feature después de transform."""

    def test_no_nans_in_train_features(self, fitted_engineer_and_features):
        _, train_feat, _ = fitted_engineer_and_features
        for col in FEATURE_NAMES:
            nan_count = train_feat[col].isna().sum()
            assert nan_count == 0, f"NaN encontrados en {col} (train): {nan_count}"

    def test_no_nans_in_val_features(self, fitted_engineer_and_features):
        _, _, val_feat = fitted_engineer_and_features
        for col in FEATURE_NAMES:
            if col in val_feat.columns:
                nan_count = val_feat[col].isna().sum()
                assert nan_count == 0, f"NaN encontrados en {col} (val): {nan_count}"


class TestCorrelationMatrix:
    """Gate B — Test #6: advertencia si |r| > 0.95 entre algún par de features."""

    def test_no_perfect_correlation(self, fitted_engineer_and_features):
        _, train_feat, _ = fitted_engineer_and_features
        corr = train_feat[FEATURE_NAMES].corr()
        high_corr = []
        for i in range(len(FEATURE_NAMES)):
            for j in range(i + 1, len(FEATURE_NAMES)):
                r = abs(corr.iloc[i, j])
                if r > 0.95:
                    high_corr.append((FEATURE_NAMES[i], FEATURE_NAMES[j], round(r, 4)))
        if high_corr:
            warnings.warn(
                f"Pares con correlación alta (|r|>0.95): {high_corr}. "
                "No es bloqueante pero documentar en tesis."
            )


class TestFeatureRanges:
    """Gate B — Test #7: rangos esperados para features clave."""

    def test_amount_non_negative(self, fitted_engineer_and_features):
        _, train_feat, _ = fitted_engineer_and_features
        assert (train_feat["amount"] >= 0).all()

    def test_binary_features_in_range(self, fitted_engineer_and_features):
        _, train_feat, _ = fitted_engineer_and_features
        for col in ["has_tip", "is_weekend", "is_off_hours"]:
            assert train_feat[col].isin([0, 1]).all(), f"{col} tiene valores fuera de [0,1]"

    def test_day_of_week_in_range(self, fitted_engineer_and_features):
        _, train_feat, _ = fitted_engineer_and_features
        assert train_feat["day_of_week"].between(1, 7).all()

    def test_hour_sin_cos_in_range(self, fitted_engineer_and_features):
        _, train_feat, _ = fitted_engineer_and_features
        assert train_feat["hour_sin"].between(-1.0, 1.0).all()
        assert train_feat["hour_cos"].between(-1.0, 1.0).all()


class TestColdStartUsers:
    """Gate B — Test #9: usuarios nuevos en val/test manejados correctamente."""

    def test_cold_start_counts_zero(self, train_val_split, fitted_engineer_and_features):
        df_train, _ = train_val_split
        _, _, val_feat = fitted_engineer_and_features

        train_users = set(df_train["user_id"].unique())
        new_val_users = set(val_feat["user_id"].unique()) - train_users

        if not new_val_users:
            pytest.skip("No hay usuarios cold-start en val")

        new_first = (
            val_feat[val_feat["user_id"].isin(new_val_users)]
            .sort_values("created_at")
            .groupby("user_id")
            .first()
        )
        assert (new_first["user_txn_count_1h"] == 0).all(), \
            "Cold-start users no tienen count 1h == 0"
        assert (new_first["user_account_age_days"] == 0).all(), \
            "Cold-start users no tienen account_age == 0"


class TestEdgeCases:
    """Gate B — Test #10: robustez con amount=0 y discount > amount."""

    def test_amount_zero_finite_features(self, fitted_engineer_and_features):
        """Features no deben explotar con amount=0."""
        _, train_feat, _ = fitted_engineer_and_features
        zero_amt = train_feat[train_feat["amount"] == 0]
        if zero_amt.empty:
            pytest.skip("No hay filas con amount=0 en el dataset de prueba")

        assert np.isfinite(zero_amt["log_amount"]).all(), "log_amount infinito con amount=0"
        assert np.isfinite(zero_amt["discount_ratio"]).all(), "discount_ratio infinito con amount=0"
        assert np.isfinite(zero_amt["amount_facility_ratio"]).all(), \
            "amount_facility_ratio infinito con amount=0"

    def test_high_discount_finite(self, fitted_engineer_and_features):
        """discount_ratio puede ser > 1 pero debe ser finito."""
        _, train_feat, _ = fitted_engineer_and_features
        high_disc = train_feat[train_feat["discount_ratio"] > 1.0]
        if high_disc.empty:
            pytest.skip("No hay filas con discount > amount en el dataset de prueba")
        assert np.isfinite(high_disc["discount_ratio"]).all()


class TestFitTransformContract:
    """Tests de contrato del FeatureEngineer."""

    def test_transform_raises_without_fit(self, sample_data):
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fit()"):
            fe.transform(sample_data)

    def test_output_columns_match_catalog(self, fitted_engineer_and_features):
        _, train_feat, _ = fitted_engineer_and_features
        for col in FEATURE_NAMES:
            assert col in train_feat.columns, f"Feature {col!r} falta en el output"

    def test_missing_required_column_raises(self, sample_data):
        fe = FeatureEngineer()
        broken = sample_data.drop(columns=["tip"])
        with pytest.raises(ValueError, match="Columnas faltantes"):
            fe.fit(broken)

    def test_save_load_roundtrip(self, fitted_engineer_and_features, tmp_path, sample_data):
        fe, train_feat, _ = fitted_engineer_and_features
        path = str(tmp_path / "fe.joblib")
        fe.save(path)
        fe2 = FeatureEngineer.load(path)
        df_train = sample_data[sample_data["created_at"] < "2025-02-01"].copy()
        train_feat2 = fe2.transform(df_train)
        pd.testing.assert_frame_equal(
            train_feat[FEATURE_NAMES].reset_index(drop=True),
            train_feat2[FEATURE_NAMES].reset_index(drop=True),
        )
