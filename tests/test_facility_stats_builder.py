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


@pytest.fixture
def synthetic_train_df():
    """Synthetic DataFrame with four facilities.

    A: n=100, normal spread  -> fallback_level='facility'
    B: n=5,   n < MIN_N      -> fallback_level in {'currency','global'}
    C: n=50,  all same amount (IQR=0) -> iqr_guarded=1.0
    """
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(100):
        rows.append({"facility_id": 1001, "amount": float(rng.normal(50, 15)), "currency": "USD"})
    for _ in range(5):
        rows.append({"facility_id": 1002, "amount": float(rng.normal(30, 5)), "currency": "USD"})
    for _ in range(50):
        rows.append({"facility_id": 1003, "amount": 25.0, "currency": "CAD"})
    return pd.DataFrame(rows)


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
    return pd.DataFrame(rows)


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
    n_facility = sum(
        1 for e in _STATS["facilities"].values() if e["fallback_level"] == "facility"
    )
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


@pytest.mark.skipif(
    not STATS_JSON.exists(), reason="artefacto no materializado"
)
def test_materialized_artifact_has_mandated_currencies():
    """Materialized artifact must contain the 9 mandated currency fallbacks (plan 02-01)."""
    with open(STATS_JSON) as f:
        stats = json.load(f)
    cf = set(stats["currency_fallbacks"])
    mandated = {"AUD", "ILS", "GTQ", "PKR", "HKD", "AED", "BWP", "SGD", "COP"}
    missing = mandated - cf
    assert not missing, f"currency_fallbacks missing mandated currencies: {missing}"
