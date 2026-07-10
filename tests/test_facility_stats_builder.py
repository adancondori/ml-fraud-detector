"""Tests for the facility stats builder and validator (frame-normalization-v1).

Covers (specs/artefactos-stats, change frame-normalization-v1):
- Universo canónico declarado (4 predicados) y rechazo del string legado.
- Ventana temporal anti-fuga (stats_window_start/end vs shadow_period_start).
- Mapping de monto fuente declarado (reservation_paid_out -> amount).
- Umbral exacto MIN_N=30.
- iana_tz desde la columna replicada facilities.tzinfo_identifier
  (sin diccionario Rails->IANA; tz_mapping.py retirado).
- Integration tests contra el artefacto materializado (skip si legado/ausente).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
IANA_PARQUET = ROOT / "output" / "revision" / "facility_iana.parquet"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train_features_enriched.parquet"
STATS_JSON = ROOT / "output" / "models" / "facility_stats_v1.json"

# Ventana canónica de train (config.py) usada por los tests sintéticos.
WINDOW_KWARGS = {
    "stats_window_start": "2025-01-01",
    "stats_window_end": "2025-06-30",
    "shadow_period_start": "2025-07-01",
}


def _build(train_df, iana_map, fid_currency, **overrides):
    from fraud_detector.stats.builder import FacilityStatsBuilder

    kwargs = {**WINDOW_KWARGS, **overrides}
    return FacilityStatsBuilder().build(train_df, iana_map, fid_currency, **kwargs)


# ---------------------------------------------------------------------------
# tz_mapping retirado (design D6)
# ---------------------------------------------------------------------------


def test_tz_mapping_module_removed():
    """El diccionario Rails->IANA queda retirado: la fuente es la columna replicada."""
    with pytest.raises(ModuleNotFoundError):
        import fraud_detector.stats.tz_mapping  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures sintéticos
# ---------------------------------------------------------------------------


def _created_at_within_window(n: int) -> pd.Series:
    """n timestamps distribuidos dentro de la ventana canónica de test
    (2025-01-01 .. 2025-06-30, shadow 2025-07-01), naive.

    Cubre el borde inclusivo (max = 2025-06-30 23:59:54) sin tocar shadow.
    """
    if n <= 1:
        return pd.to_datetime(["2025-01-01 00:00:00"][:n])
    return pd.date_range(
        start="2025-01-01 00:00:00",
        end="2025-06-30 23:59:54",
        periods=n,
    )


@pytest.fixture
def synthetic_train_df():
    """Synthetic DataFrame with four facilities.

    A: n=100, normal spread  -> fallback_level='facility'
    B: n=5,   n < MIN_N      -> fallback_level in {'currency','global'}
    C: n=50,  all same amount (IQR=0) -> iqr_guarded=1.0

    Cada fila lleva ``created_at`` dentro de la ventana canónica de test
    (2025-01-01 .. 2025-06-30, shadow 2025-07-01) para pasar el gate anti-fuga.
    """
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(100):
        rows.append({"facility_id": 1001, "amount": float(rng.normal(50, 15)), "currency": "USD"})
    for _ in range(5):
        rows.append({"facility_id": 1002, "amount": float(rng.normal(30, 5)), "currency": "USD"})
    for _ in range(50):
        rows.append({"facility_id": 1003, "amount": 25.0, "currency": "CAD"})
    df = pd.DataFrame(rows)
    df["created_at"] = _created_at_within_window(len(df))
    return df


@pytest.fixture
def synthetic_iana_map():
    """{facility_id -> tzinfo_identifier} tal como viene de la columna replicada.

    Facility D (1004) no tiene filas en train. Facility E (1005) tiene
    tzinfo_identifier vacío en ClickHouse.
    """
    return {
        1001: "America/New_York",
        1002: "America/Los_Angeles",
        1003: "America/Denver",
        1004: "America/Chicago",  # tz-only, no train rows
        1005: "",  # tzinfo_identifier vacío -> iana_tz nulo
    }


@pytest.fixture
def synthetic_fid_currency(synthetic_train_df):
    return (
        synthetic_train_df.groupby("facility_id")["currency"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )


@pytest.fixture
def built_stats(synthetic_train_df, synthetic_iana_map, synthetic_fid_currency):
    return _build(synthetic_train_df, synthetic_iana_map, synthetic_fid_currency)


# ---------------------------------------------------------------------------
# Universo canónico (scenarios: universo/artefacto-declarado)
# ---------------------------------------------------------------------------


def test_universe_filter_declares_four_predicates(built_stats):
    """universe_filter debe declarar los 4 predicados literales del universo canónico."""
    uf = built_stats["universe_filter"]
    assert "FINAL" in uf
    assert "_peerdb_is_deleted=0" in uf
    assert "payment_method NOT IN ('reversal','free')" in uf
    assert "user_id != 0" in uf


def test_builder_min_n():
    from fraud_detector.stats.builder import FacilityStatsBuilder

    assert FacilityStatsBuilder.MIN_N == 30


# ---------------------------------------------------------------------------
# Ventana temporal anti-fuga (scenario: universo/sin-fuga-temporal)
# ---------------------------------------------------------------------------


def test_artifact_declares_stats_window(built_stats):
    assert built_stats["stats_window_start"] == "2025-01-01"
    assert built_stats["stats_window_end"] == "2025-06-30"


def test_build_fails_if_window_overlaps_shadow(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """La generación FALLA si la ventana de stats solapa el período de shadow."""
    with pytest.raises(ValueError, match="solapa"):
        _build(
            synthetic_train_df,
            synthetic_iana_map,
            synthetic_fid_currency,
            stats_window_end="2025-07-15",
            shadow_period_start="2025-07-01",
        )


def test_build_fails_if_window_end_equals_shadow_start(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """stats_window_end == shadow_period_start también es solape (se exige end < T)."""
    with pytest.raises(ValueError, match="solapa"):
        _build(
            synthetic_train_df,
            synthetic_iana_map,
            synthetic_fid_currency,
            stats_window_end="2025-07-01",
            shadow_period_start="2025-07-01",
        )


def test_build_fails_if_window_inverted(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    with pytest.raises(ValueError):
        _build(
            synthetic_train_df,
            synthetic_iana_map,
            synthetic_fid_currency,
            stats_window_start="2025-06-30",
            stats_window_end="2025-01-01",
        )


# ---------------------------------------------------------------------------
# Anti-fuga temporal EJECUTADO sobre created_at (change anti-fuga-builder-enforce)
# specs/anti-fuga-stats/spec.md — cada test cita su scenario.
# ---------------------------------------------------------------------------


def test_build_fails_if_created_at_missing(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """Scenario: falta la columna created_at.

    train_df sin created_at -> ValueError accionable que nombra la columna,
    sin producir artefacto y antes de agregar estadísticas.
    """
    df = synthetic_train_df.drop(columns=["created_at"])
    with pytest.raises(ValueError, match="created_at"):
        _build(df, synthetic_iana_map, synthetic_fid_currency)


def test_build_fails_if_created_at_has_nulls(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """Scenario: created_at con nulos -> ValueError que indica nulos/NaT."""
    df = synthetic_train_df.copy()
    df.loc[df.index[0], "created_at"] = pd.NaT
    with pytest.raises(ValueError, match="(?i)nulos|nat"):
        _build(df, synthetic_iana_map, synthetic_fid_currency)


def test_build_fails_if_created_at_unparseable(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """Scenario: created_at no parsea como datetime -> ValueError."""
    df = synthetic_train_df.copy()
    df["created_at"] = df["created_at"].astype(object)
    df.loc[df.index[0], "created_at"] = "no-es-fecha"
    with pytest.raises(ValueError, match="(?i)created_at"):
        _build(df, synthetic_iana_map, synthetic_fid_currency)


def test_build_fails_if_observed_min_before_window_start(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """Scenario: observed_min anterior al inicio de ventana.

    Ventana 2025-01-01..2025-06-30, shadow 2025-07-01; una fila el 2024-12-31.
    """
    df = synthetic_train_df.copy()
    df["created_at"] = df["created_at"].astype("datetime64[ns]")
    df.loc[df.index[0], "created_at"] = pd.Timestamp("2024-12-31 23:00:00")
    with pytest.raises(ValueError, match="(?i)fuera de ventana|observed_min"):
        _build(df, synthetic_iana_map, synthetic_fid_currency)


def test_build_fails_if_observed_max_after_window_end_before_shadow(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """Scenario: observed_max posterior al fin de ventana pero anterior al shadow.

    stats_window_end=2025-06-30, shadow=2025-08-01, fila el 2025-07-15.
    """
    df = synthetic_train_df.copy()
    df["created_at"] = df["created_at"].astype("datetime64[ns]")
    df.loc[df.index[0], "created_at"] = pd.Timestamp("2025-07-15 10:00:00")
    with pytest.raises(ValueError, match="(?i)fuera de ventana|observed_max"):
        _build(
            df,
            synthetic_iana_map,
            synthetic_fid_currency,
            stats_window_end="2025-06-30",
            shadow_period_start="2025-08-01",
        )


def test_build_passes_on_inclusive_window_end_boundary(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """Scenario: borde inclusivo — máximo en el último día de la ventana PASA.

    max created_at = 2025-06-30 23:59:54 -> build COMPLETA.
    """
    df = synthetic_train_df.copy()
    df["created_at"] = df["created_at"].astype("datetime64[ns]")
    df.loc[df.index[-1], "created_at"] = pd.Timestamp("2025-06-30 23:59:54")
    stats = _build(df, synthetic_iana_map, synthetic_fid_currency)
    assert stats["observed_max_created_at"] == "2025-06-30"


def test_build_fails_on_shadow_boundary_instant(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """Scenario: borde estricto — instante dentro del shadow FALLA.

    max created_at = 2025-07-01 00:00:00 (>= shadow_period_start) -> FUGA.
    """
    df = synthetic_train_df.copy()
    df["created_at"] = df["created_at"].astype("datetime64[ns]")
    df.loc[df.index[-1], "created_at"] = pd.Timestamp("2025-07-01 00:00:00")
    with pytest.raises(ValueError, match="(?i)fuga|shadow"):
        _build(df, synthetic_iana_map, synthetic_fid_currency)


def test_build_declares_observed_provenance(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency
):
    """Scenario: caso feliz — build completa y declara la procedencia observada.

    min=2025-01-01 00:00:00, max=2025-06-30 23:59:54 -> observed_* como
    ISO date YYYY-MM-DD, no timestamp.
    """
    df = synthetic_train_df.copy()
    df["created_at"] = df["created_at"].astype("datetime64[ns]")
    df.loc[df.index[0], "created_at"] = pd.Timestamp("2025-01-01 00:00:00")
    df.loc[df.index[-1], "created_at"] = pd.Timestamp("2025-06-30 23:59:54")
    stats = _build(df, synthetic_iana_map, synthetic_fid_currency)
    assert stats["observed_min_created_at"] == "2025-01-01"
    assert stats["observed_max_created_at"] == "2025-06-30"


def test_build_fails_if_train_df_empty(synthetic_iana_map, synthetic_fid_currency):
    """RISK challenge fix: train_df con columna created_at pero 0 filas.

    s.min()/s.max() dan NaT y NaT.date() vs date.fromisoformat levantaba
    TypeError opaco. El contrato D1/D2 exige fallar barato con ValueError
    accionable ANTES de agregar estadísticas. Debe ser el 7º mensaje de la
    familia D2 (train_df vacío), no un TypeError.
    """
    from fraud_detector.stats.builder import FacilityStatsBuilder

    empty = pd.DataFrame(
        {
            "facility_id": pd.Series([], dtype="int64"),
            "amount": pd.Series([], dtype="float64"),
            "currency": pd.Series([], dtype="object"),
            "created_at": pd.Series([], dtype="datetime64[ns]"),
        }
    )
    with pytest.raises(ValueError, match="(?i)vac|no hay filas"):
        FacilityStatsBuilder().build(
            empty, synthetic_iana_map, synthetic_fid_currency, **WINDOW_KWARGS
        )


def test_edge_classification_independent_of_process_tz(
    synthetic_train_df, synthetic_iana_map, synthetic_fid_currency, monkeypatch
):
    """Scenario: pago de borde a las 23:30 del último día de ventana no se
    malclasifica como fuga; resultado idéntico bajo cualquier TZ del proceso.

    max created_at = 2025-06-30 23:30:00 (naive). Se corre bajo TZ=UTC y
    TZ=America/La_Paz y se compara la salida temporal.
    """
    import time

    df = synthetic_train_df.copy()
    df["created_at"] = df["created_at"].astype("datetime64[ns]")
    df.loc[df.index[-1], "created_at"] = pd.Timestamp("2025-06-30 23:30:00")

    def _build_under_tz(tz_value: str):
        monkeypatch.setenv("TZ", tz_value)
        try:
            time.tzset()
        except AttributeError:  # pragma: no cover - non-POSIX
            pass
        return _build(df, synthetic_iana_map, synthetic_fid_currency)

    stats_utc = _build_under_tz("UTC")
    stats_lapaz = _build_under_tz("America/La_Paz")

    for key in ("observed_min_created_at", "observed_max_created_at"):
        assert stats_utc[key] == stats_lapaz[key]
    assert stats_utc["observed_max_created_at"] == "2025-06-30"


# ---------------------------------------------------------------------------
# Mapping de monto fuente (scenario: universo/amount-alias-interno)
# ---------------------------------------------------------------------------


def test_artifact_declares_amount_source(built_stats):
    """El artefacto declara de dónde proviene la columna interna 'amount'."""
    assert "reservation_paid_out" in built_stats["amount_source"]


# ---------------------------------------------------------------------------
# Umbral exacto de inclusión (scenario: universo/umbral-exacto-30)
# ---------------------------------------------------------------------------


def _n_rows_df(n: int, facility_id: int = 3001) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = [
        {"facility_id": facility_id, "amount": float(rng.normal(40, 8)), "currency": "USD"}
        for _ in range(n)
    ]
    df = pd.DataFrame(rows)
    df["created_at"] = _created_at_within_window(len(df))
    return df


def test_facility_with_exactly_30_payments_gets_own_stats():
    stats = _build(_n_rows_df(30), {3001: "America/New_York"}, {3001: "USD"})
    assert stats["facilities"]["3001"]["fallback_level"] == "facility"
    assert stats["facilities"]["3001"]["n"] == 30


def test_facility_with_29_payments_falls_back():
    stats = _build(_n_rows_df(29), {3001: "America/New_York"}, {3001: "USD"})
    assert stats["facilities"]["3001"]["fallback_level"] in {"currency", "global"}
    assert stats["facilities"]["3001"]["n"] == 29


# ---------------------------------------------------------------------------
# iana_tz desde la columna replicada (scenarios: iana/*)
# ---------------------------------------------------------------------------


def test_iana_tz_passthrough_from_column(built_stats):
    """iana_tz es la columna tzinfo_identifier tal cual, sin mapeo Rails."""
    assert built_stats["facilities"]["1003"]["iana_tz"] == "America/Denver"


def test_unseen_identifier_does_not_break_build(synthetic_train_df, synthetic_fid_currency):
    """Un identificador no presente en ningún diccionario propio no rompe el build."""
    iana_map = {
        1001: "America/Argentina/Ushuaia",
        1002: "Pacific/Kiritimati",
        1003: "America/Denver",
    }
    stats = _build(synthetic_train_df, iana_map, synthetic_fid_currency)
    assert stats["facilities"]["1001"]["iana_tz"] == "America/Argentina/Ushuaia"


def test_empty_tzinfo_yields_null_iana_without_abort(built_stats):
    """tzinfo_identifier vacío -> iana_tz nulo, el build completa (no aborta)."""
    entry = built_stats["facilities"]["1005"]
    assert entry["iana_tz"] is None
    assert entry["fallback_level"] in {"currency", "global"}


def test_none_tzinfo_yields_null_iana(synthetic_train_df, synthetic_fid_currency):
    stats = _build(
        synthetic_train_df,
        {1001: "America/New_York", 1002: None, 1003: "America/Denver"},
        synthetic_fid_currency,
    )
    assert stats["facilities"]["1002"]["iana_tz"] is None


# ---------------------------------------------------------------------------
# Comportamiento preservado del builder (fallbacks, IQR guard, cobertura)
# ---------------------------------------------------------------------------


def test_facility_a_fallback_level(built_stats):
    assert built_stats["facilities"]["1001"]["fallback_level"] == "facility"


def test_facility_b_fallback_level(built_stats):
    assert built_stats["facilities"]["1002"]["fallback_level"] in {"currency", "global"}


def test_facility_c_iqr_zero_guarded(built_stats):
    entry = built_stats["facilities"]["1003"]
    assert entry["iqr"] == 0.0
    assert entry["iqr_guarded"] == 1.0


def test_facility_d_tz_only_present(built_stats):
    assert "1004" in built_stats["facilities"]
    entry = built_stats["facilities"]["1004"]
    assert entry["iana_tz"] == "America/Chicago"
    assert entry["fallback_level"] in {"currency", "global"}


def test_coverage_equals_iana_map(built_stats, synthetic_iana_map):
    assert len(built_stats["facilities"]) == len(synthetic_iana_map)


def test_schema_version(built_stats):
    assert built_stats["schema_version"] == "facility-stats-v1"


def test_built_at_is_utc_iso(built_stats):
    assert built_stats["built_at"].endswith("Z")


def test_global_fallback_present(built_stats):
    gf = built_stats["global_fallback"]
    for field in ("median", "iqr", "iqr_guarded", "mean", "n", "fallback_level"):
        assert field in gf
    assert gf["fallback_level"] == "global"
    assert gf["iqr_guarded"] >= 1.0


def test_currency_fallbacks_usd_present(built_stats):
    assert "USD" in built_stats["currency_fallbacks"]
    usd = built_stats["currency_fallbacks"]["USD"]
    for field in ("median", "iqr", "iqr_guarded", "mean", "n", "fallback_level"):
        assert field in usd
    assert usd["fallback_level"] == "currency"


def test_currency_fallback_threshold():
    """Only currencies with n >= _MIN_CURRENCY_N appear in currency_fallbacks (plus USD)."""
    from fraud_detector.stats.builder import _MIN_CURRENCY_N

    rng = np.random.default_rng(99)
    rows = []
    for _ in range(1500):
        rows.append({"facility_id": 2001, "amount": float(rng.normal(100, 20)), "currency": "EUR"})
    for _ in range(500):
        rows.append({"facility_id": 2002, "amount": float(rng.normal(80, 10)), "currency": "GBP"})
    for _ in range(10):
        rows.append({"facility_id": 2003, "amount": float(rng.normal(50, 5)), "currency": "USD"})

    df = pd.DataFrame(rows)
    df["created_at"] = _created_at_within_window(len(df))
    iana_map = {2001: "America/New_York", 2002: "America/Los_Angeles", 2003: "Etc/UTC"}
    fid_currency = {2001: "EUR", 2002: "GBP", 2003: "USD"}

    stats = _build(df, iana_map, fid_currency)
    cf = set(stats["currency_fallbacks"].keys())

    assert 1500 >= _MIN_CURRENCY_N
    assert 500 < _MIN_CURRENCY_N
    assert "EUR" in cf
    assert "GBP" not in cf
    assert "USD" in cf


# ---------------------------------------------------------------------------
# Validator (scenarios: universo/validator-rechaza-legado, amount-alias, iana nulo)
# ---------------------------------------------------------------------------


@pytest.fixture
def validator_inputs(built_stats, synthetic_train_df, synthetic_iana_map):
    tz_df = pd.DataFrame({"facility_id": list(synthetic_iana_map.keys())})
    return built_stats, synthetic_train_df, tz_df


def test_validator_passes_canonical_artifact(validator_inputs):
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    assert validate_universe_filter(stats, train_df, tz_df) is True


def test_validator_rejects_legacy_universe_filter(validator_inputs):
    """El string legado sin user_id != 0 debe ser rechazado explícitamente."""
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    legacy = dict(stats)
    legacy["universe_filter"] = (
        "_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL"
    )
    with pytest.raises(AssertionError, match="user_id != 0"):
        validate_universe_filter(legacy, train_df, tz_df)


def test_validator_requires_stats_window(validator_inputs):
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    broken = dict(stats)
    broken.pop("stats_window_start")
    with pytest.raises(AssertionError, match="stats_window"):
        validate_universe_filter(broken, train_df, tz_df)


def test_validator_rejects_empty_amount_source(validator_inputs):
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    broken = dict(stats)
    broken["amount_source"] = ""
    with pytest.raises(AssertionError, match="amount_source"):
        validate_universe_filter(broken, train_df, tz_df)


def test_validator_rejects_ambiguous_amount_source(validator_inputs):
    """'amount' sin columna fuente declarada es mapping ambiguo."""
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    broken = dict(stats)
    broken["amount_source"] = "amount"
    with pytest.raises(AssertionError, match="amount_source"):
        validate_universe_filter(broken, train_df, tz_df)


def test_validator_tolerates_null_iana_tz(validator_inputs):
    """iana_tz nulo (tzinfo_identifier vacío) no aborta la validación."""
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    assert stats["facilities"]["1005"]["iana_tz"] is None
    assert validate_universe_filter(stats, train_df, tz_df) is True


def test_validator_rejects_missing_iana_key(validator_inputs):
    """La llave iana_tz debe existir en cada entrada (None permitido, ausencia no)."""
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    broken = json.loads(json.dumps(stats))
    del broken["facilities"]["1001"]["iana_tz"]
    with pytest.raises(AssertionError, match="iana_tz"):
        validate_universe_filter(broken, train_df, tz_df)


# ---------------------------------------------------------------------------
# Validator anti-fuga: rango observado + rama legacy explícita
# (change anti-fuga-builder-enforce, D5). specs/anti-fuga-stats/spec.md.
# ---------------------------------------------------------------------------


def _sample_df_with_created_at(max_ts: str, n: int = 155) -> pd.DataFrame:
    """sample_df sintético con created_at cuyo máximo es max_ts."""
    ts = pd.date_range(start="2025-01-01", end=max_ts, periods=n)
    return pd.DataFrame(
        {
            "facility_id": [1001] * n,
            "amount": [50.0] * n,
            "currency": ["USD"] * n,
            "created_at": ts,
        }
    )


def test_validator_fails_if_observed_max_overlaps_shadow(validator_inputs):
    """Scenario: validator falla si el máximo observado solapa el shadow.

    observed_max_created_at=2025-07-01, stats_window_end=2025-06-30,
    shadow=2025-07-01, sample_df máximo el 2025-07-01.
    """
    from fraud_detector.stats.validator import validate_universe_filter

    stats, _train_df, tz_df = validator_inputs
    broken = json.loads(json.dumps(stats))
    broken["observed_max_created_at"] = "2025-07-01"
    sample_df = _sample_df_with_created_at("2025-07-01 00:00:00")
    with pytest.raises(AssertionError, match="(?i)shadow|fuga"):
        validate_universe_filter(broken, sample_df, tz_df)


def test_validator_fails_if_observed_mismatch_sample_df(validator_inputs):
    """Scenario: validator falla si observed_* no coincide con sample_df.

    observed_max_created_at declarado 2025-06-30 pero sample_df máximo 2025-07-10.
    """
    from fraud_detector.stats.validator import validate_universe_filter

    stats, _train_df, tz_df = validator_inputs
    sample_df = _sample_df_with_created_at("2025-07-10 00:00:00")
    with pytest.raises(AssertionError, match="(?i)coincide|no coincide|observed"):
        validate_universe_filter(stats, sample_df, tz_df)


def test_validator_fails_new_artifact_sample_df_without_created_at(validator_inputs):
    """D5: artefacto nuevo (con observed_*) pero sample_df SIN created_at ->
    revalidación de procedencia imposible -> FALLA con mensaje accionable."""
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    sample_df = train_df.drop(columns=["created_at"])
    with pytest.raises(AssertionError, match="created_at"):
        validate_universe_filter(stats, sample_df, tz_df)


def test_validator_legacy_artifact_without_observed_passes(validator_inputs):
    """Scenario: artefacto legado sin observed_* no rompe.

    Declara stats_window_*/shadow pero sin observed_* -> rama legacy explícita.
    """
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    legacy = json.loads(json.dumps(stats))
    legacy.pop("observed_min_created_at")
    legacy.pop("observed_max_created_at")
    assert validate_universe_filter(legacy, train_df, tz_df) is True


def test_validator_partial_observed_is_malformed(validator_inputs):
    """D5: exactamente UNA clave observed_* -> malformado (no cae en legacy)."""
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    malformed = json.loads(json.dumps(stats))
    malformed.pop("observed_min_created_at")  # queda solo observed_max_created_at
    with pytest.raises(AssertionError, match="(?i)observed|malformado"):
        validate_universe_filter(malformed, train_df, tz_df)


def test_validator_new_artifact_window_end_not_before_shadow(validator_inputs):
    """D5 punto 2: check stats_window_end < shadow_period_start (hoy ausente).

    observed_* autoconsistente pero stats_window_end >= shadow_period_start.
    """
    from fraud_detector.stats.validator import validate_universe_filter

    stats, _train_df, tz_df = validator_inputs
    broken = json.loads(json.dumps(stats))
    # observed_max=2025-06-30 sigue dentro de la ventana y < shadow, pero la
    # ventana declarada solapa el shadow: stats_window_end >= shadow_period_start.
    broken["stats_window_end"] = "2025-07-15"
    broken["shadow_period_start"] = "2025-07-01"
    sample_df = _sample_df_with_created_at("2025-06-30 23:59:54")
    with pytest.raises(AssertionError, match="(?i)shadow"):
        validate_universe_filter(broken, sample_df, tz_df)


def test_validator_legacy_artifact_window_end_not_before_shadow_rejected(
    validator_inputs,
):
    """NOTE 2 challenge fix: el check de ORDEN de ventana (end < shadow,
    metadata-only) es UNIVERSAL — aplica también a la rama legacy.

    Un artefacto legado (sin observed_*) con stats_window_end >=
    shadow_period_start es un artefacto mal formado y DEBE ser rechazado,
    aunque omita los checks de procedencia observed_*. Esto convierte el
    endurecimiento (antes silencioso) en contrato explícito.
    """
    from fraud_detector.stats.validator import validate_universe_filter

    stats, train_df, tz_df = validator_inputs
    legacy = json.loads(json.dumps(stats))
    legacy.pop("observed_min_created_at")
    legacy.pop("observed_max_created_at")
    # end >= shadow: la ventana declarada solapa el shadow.
    legacy["stats_window_end"] = "2025-07-15"
    legacy["shadow_period_start"] = "2025-07-01"
    with pytest.raises(AssertionError, match="(?i)shadow"):
        validate_universe_filter(legacy, train_df, tz_df)


# ---------------------------------------------------------------------------
# Integration tests contra el artefacto materializado
# ---------------------------------------------------------------------------


def _load_stats_or_none():
    if not STATS_JSON.exists():
        return None
    with open(STATS_JSON) as f:
        return json.load(f)


_STATS = _load_stats_or_none()
_LEGACY_ARTIFACT = _STATS is not None and "stats_window_start" not in _STATS
_INTEGRATION_SKIP = _STATS is None or _LEGACY_ARTIFACT or not IANA_PARQUET.exists()
_INTEGRATION_REASON = (
    "artefacto legado pre-refresco (fnv1-03) o snapshot facility_iana.parquet ausente"
)

# change anti-fuga-builder-enforce (D7): el artefacto vigente NO declara
# observed_* (se generó bajo el gate declarativo anterior). La ausencia de
# observed_* es la señal de "legacy para procedencia observada": el validator
# debe tratarlo por su rama legacy SIN regenerar el artefacto. El próximo
# refresco lo generará con observed_* y caerá en la rama estricta.
_ARTIFACT_HAS_OBSERVED = _STATS is not None and "observed_min_created_at" in _STATS


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_schema_version():
    assert _STATS["schema_version"] == "facility-stats-v1"


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_declares_canonical_universe():
    assert "user_id != 0" in _STATS["universe_filter"]
    assert _STATS["stats_window_end"] < _STATS.get("shadow_period_start", "9999")


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_facility_coverage():
    iana_df = pd.read_parquet(IANA_PARQUET)
    expected = iana_df["facility_id"].nunique()
    assert _STATS["n_facilities"] == expected
    assert len(_STATS["facilities"]) == _STATS["n_facilities"]


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_has_facility_level_entries():
    n_facility = sum(1 for e in _STATS["facilities"].values() if e["fallback_level"] == "facility")
    assert n_facility > 0


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_iqr_guarded_count():
    n_guarded = sum(1 for e in _STATS["facilities"].values() if e.get("iqr") == 0.0)
    assert n_guarded > 0


@pytest.mark.skipif(
    _INTEGRATION_SKIP or not TRAIN_PARQUET.exists(),
    reason=_INTEGRATION_REASON + " o train parquet ausente",
)
def test_integration_validate_universe_filter():
    from fraud_detector.stats.validator import validate_universe_filter

    train_df = pd.read_parquet(TRAIN_PARQUET, columns=["amount", "facility_id", "currency"])
    iana_df = pd.read_parquet(IANA_PARQUET)
    assert validate_universe_filter(_STATS, train_df, iana_df) is True


@pytest.mark.skipif(_STATS is None, reason="artefacto no materializado")
def test_integration_vigente_artifact_uses_legacy_branch():
    """D7: el artefacto vigente (sin observed_*) NO cae en la rama de artefacto
    nuevo del validator; pasa por la rama legacy SIN regenerarse.

    Se valida con un sample_df SIN created_at: si el vigente cayera en la rama
    nueva, el validator exigiría created_at y FALLARÍA. Que pase confirma legacy.
    """
    from fraud_detector.stats.validator import validate_universe_filter

    assert not _ARTIFACT_HAS_OBSERVED, (
        "el artefacto vigente NO debe declarar observed_* (no se regenera en "
        "este change); su ausencia es la señal de la rama legacy"
    )
    if not IANA_PARQUET.exists() or not TRAIN_PARQUET.exists():
        pytest.skip("snapshot iana o train parquet ausente")
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=["amount", "facility_id", "currency"])
    iana_df = pd.read_parquet(IANA_PARQUET)
    assert "created_at" not in train_df.columns
    # Rama legacy: no exige created_at en sample_df -> pasa sin regenerar.
    assert validate_universe_filter(_STATS, train_df, iana_df) is True


@pytest.mark.skipif(not STATS_JSON.exists(), reason="artefacto no materializado")
def test_materialized_artifact_has_mandated_currencies():
    """Materialized artifact must contain the 9 mandated currency fallbacks (plan 02-01)."""
    with open(STATS_JSON) as f:
        stats = json.load(f)
    cf = set(stats["currency_fallbacks"])
    mandated = {"AUD", "ILS", "GTQ", "PKR", "HKD", "AED", "BWP", "SGD", "COP"}
    missing = mandated - cf
    assert not missing, f"currency_fallbacks missing mandated currencies: {missing}"
