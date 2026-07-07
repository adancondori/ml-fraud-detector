"""Tests for the facility stats builder, validator, and tz_mapping.

Covers:
- RAILS_TO_IANA completeness (64 entries, all IANA names loadable)
- FacilityStatsBuilder: fallback chain, IQR guard, tz-only facilities
- validate_universe_filter: schema, row count tolerance, facility coverage
- Integration test against the materialized artifact (skipped if files absent)
- _MIN_CURRENCY_N threshold criterion (02-01)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parents[1]
TZ_PARQUET = ROOT / "output" / "revision" / "facility_tz.parquet"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train_features_enriched.parquet"
STATS_JSON = ROOT / "output" / "models" / "facility_stats_v1.json"


# ---------------------------------------------------------------------------
# Task 1 — tz_mapping tests
# ---------------------------------------------------------------------------


def test_tz_mapping_count():
    """RAILS_TO_IANA must have exactly 64 entries."""
    from fraud_detector.stats.tz_mapping import RAILS_TO_IANA

    assert len(RAILS_TO_IANA) == 64, f"Expected 64 entries, got {len(RAILS_TO_IANA)}"


def test_tz_mapping_all_iana_loadable():
    """Every IANA value in RAILS_TO_IANA must be loadable by zoneinfo."""
    from fraud_detector.stats.tz_mapping import RAILS_TO_IANA

    for rails, iana in RAILS_TO_IANA.items():
        try:
            ZoneInfo(iana)
        except Exception as exc:
            pytest.fail(f"ZoneInfo('{iana}') failed for Rails '{rails}': {exc}")


@pytest.mark.skipif(not TZ_PARQUET.exists(), reason="facility_tz.parquet not available")
def test_tz_mapping_covers_parquet():
    """RAILS_TO_IANA must cover ALL timezone names present in facility_tz.parquet."""
    from fraud_detector.stats.tz_mapping import RAILS_TO_IANA

    tz_df = pd.read_parquet(TZ_PARQUET)
    parquet_zones = set(tz_df["time_zone"].dropna().unique())
    missing = parquet_zones - set(RAILS_TO_IANA.keys())
    assert not missing, f"Rails zones in parquet not covered by RAILS_TO_IANA: {missing}"


def test_resolve_iana_known():
    """resolve_iana returns the correct IANA name for a known Rails name."""
    from fraud_detector.stats.tz_mapping import resolve_iana

    assert resolve_iana("Eastern Time (US & Canada)") == "America/New_York"
    assert resolve_iana("UTC") == "Etc/UTC"


def test_resolve_iana_unknown_fallback():
    """resolve_iana returns 'Etc/UTC' for unknown Rails names (no KeyError)."""
    from fraud_detector.stats.tz_mapping import resolve_iana

    assert resolve_iana("Some Unknown Zone") == "Etc/UTC"
    assert resolve_iana("") == "Etc/UTC"


# ---------------------------------------------------------------------------
# Task 2 — FacilityStatsBuilder tests
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
    # Facility A — 100 rows, amounts around 50 with spread
    for _ in range(100):
        rows.append({"facility_id": 1001, "amount": float(rng.normal(50, 15)), "currency": "USD"})
    # Facility B — 5 rows, low count
    for _ in range(5):
        rows.append({"facility_id": 1002, "amount": float(rng.normal(30, 5)), "currency": "USD"})
    # Facility C — 50 rows, all same amount (IQR=0)
    for _ in range(50):
        rows.append({"facility_id": 1003, "amount": 25.0, "currency": "CAD"})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_tz_map():
    """tz_map including facility D which has NO rows in train_df."""
    return {
        1001: "Eastern Time (US & Canada)",
        1002: "Pacific Time (US & Canada)",
        1003: "Mountain Time (US & Canada)",
        1004: "Central Time (US & Canada)",  # facility D: tz-only, no train rows
    }


@pytest.fixture
def synthetic_fid_currency(synthetic_train_df):
    """fid_currency derived from the synthetic train DataFrame."""
    return (
        synthetic_train_df.groupby("facility_id")["currency"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )


@pytest.fixture
def built_stats(synthetic_train_df, synthetic_tz_map, synthetic_fid_currency):
    from fraud_detector.stats.builder import FacilityStatsBuilder

    return FacilityStatsBuilder().build(
        synthetic_train_df, synthetic_tz_map, synthetic_fid_currency
    )


def test_builder_min_n():
    """FacilityStatsBuilder.MIN_N must be 30."""
    from fraud_detector.stats.builder import FacilityStatsBuilder

    assert FacilityStatsBuilder.MIN_N == 30


def test_facility_a_fallback_level(built_stats):
    """Facility A (n=100) must have fallback_level='facility'."""
    entry = built_stats["facilities"]["1001"]
    assert entry["fallback_level"] == "facility"


def test_facility_b_fallback_level(built_stats):
    """Facility B (n=5) must have fallback_level in {'currency','global'}."""
    entry = built_stats["facilities"]["1002"]
    assert entry["fallback_level"] in {"currency", "global"}


def test_facility_c_iqr_zero_guarded(built_stats):
    """Facility C (all same amount, IQR=0) must have iqr==0.0 but iqr_guarded==1.0."""
    entry = built_stats["facilities"]["1003"]
    assert entry["iqr"] == 0.0, f"Expected iqr=0.0, got {entry['iqr']}"
    assert entry["iqr_guarded"] == 1.0, f"Expected iqr_guarded=1.0, got {entry['iqr_guarded']}"


def test_facility_d_tz_only_present(built_stats, synthetic_tz_map):
    """Facility D (tz-only, no train rows) must appear in stats with iana_tz resolved."""
    assert (
        "1004" in built_stats["facilities"]
    ), "Facility D must be in stats even with no train rows"
    entry = built_stats["facilities"]["1004"]
    assert entry.get("iana_tz"), "Facility D must have non-empty iana_tz"
    assert entry["fallback_level"] in {"currency", "global"}


def test_coverage_equals_tz_map(built_stats, synthetic_tz_map):
    """len(stats['facilities']) must equal len(tz_map) — not just groupby count."""
    assert len(built_stats["facilities"]) == len(
        synthetic_tz_map
    ), f"Expected {len(synthetic_tz_map)} facilities, got {len(built_stats['facilities'])}"


def test_all_entries_have_iana_tz(built_stats):
    """Every facility entry must have a non-empty iana_tz field."""
    for fid, entry in built_stats["facilities"].items():
        assert entry.get("iana_tz"), f"Facility {fid} has empty/missing iana_tz"


def test_schema_version(built_stats):
    """stats must contain schema_version='facility-stats-v1'."""
    assert built_stats["schema_version"] == "facility-stats-v1"


def test_universe_filter_in_stats(built_stats):
    """stats must record the exact universe_filter string."""
    expected = "_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL"
    assert built_stats["universe_filter"] == expected


def test_global_fallback_present(built_stats):
    """stats must contain a global_fallback with required fields."""
    gf = built_stats["global_fallback"]
    for field in ("median", "iqr", "iqr_guarded", "mean", "n", "fallback_level"):
        assert field in gf, f"global_fallback missing field '{field}'"
    assert gf["fallback_level"] == "global"
    assert gf["iqr_guarded"] >= 1.0


def test_currency_fallbacks_usd_present(built_stats):
    """currency_fallbacks must include at least USD."""
    assert "USD" in built_stats["currency_fallbacks"]
    usd = built_stats["currency_fallbacks"]["USD"]
    for field in ("median", "iqr", "iqr_guarded", "mean", "n", "fallback_level"):
        assert field in usd, f"currency_fallback USD missing field '{field}'"
    assert usd["fallback_level"] == "currency"


def test_currency_fallback_threshold():
    """Only currencies with n >= _MIN_CURRENCY_N appear in currency_fallbacks (plus USD always).

    Constructs a synthetic train_df with:
      - EUR: 1500 rows -> above threshold -> must appear
      - GBP:  500 rows -> below threshold -> must NOT appear (unless it is USD)
      - USD:   10 rows -> below threshold, but USD is always included
    """
    from fraud_detector.stats.builder import FacilityStatsBuilder, _MIN_CURRENCY_N

    rng = np.random.default_rng(99)
    rows = []
    # EUR: 1500 rows, well above _MIN_CURRENCY_N
    for _ in range(1500):
        rows.append({"facility_id": 2001, "amount": float(rng.normal(100, 20)), "currency": "EUR"})
    # GBP: 500 rows, below _MIN_CURRENCY_N
    for _ in range(500):
        rows.append({"facility_id": 2002, "amount": float(rng.normal(80, 10)), "currency": "GBP"})
    # USD: 10 rows, below _MIN_CURRENCY_N but always included
    for _ in range(10):
        rows.append({"facility_id": 2003, "amount": float(rng.normal(50, 5)), "currency": "USD"})

    df = pd.DataFrame(rows)
    tz_map = {
        2001: "Eastern Time (US & Canada)",
        2002: "Pacific Time (US & Canada)",
        2003: "UTC",
    }
    fid_currency = {2001: "EUR", 2002: "GBP", 2003: "USD"}

    stats = FacilityStatsBuilder().build(df, tz_map, fid_currency)
    cf = set(stats["currency_fallbacks"].keys())

    assert 1500 >= _MIN_CURRENCY_N, "Test assumption: EUR (1500) is >= threshold"
    assert 500 < _MIN_CURRENCY_N, "Test assumption: GBP (500) is < threshold"

    assert "EUR" in cf, f"EUR (n=1500 >= {_MIN_CURRENCY_N}) must be in currency_fallbacks"
    assert "GBP" not in cf, f"GBP (n=500 < {_MIN_CURRENCY_N}) must NOT be in currency_fallbacks"
    assert "USD" in cf, "USD must always be in currency_fallbacks regardless of row count"


@pytest.mark.skipif(
    not Path("output/models/facility_stats_v1.json").exists(),
    reason="artefacto no materializado",
)
def test_materialized_artifact_has_mandated_currencies():
    """Materialized artifact must contain the 9 newly mandated currency fallbacks.

    AUD, ILS, GTQ, PKR, HKD, AED, BWP, SGD, COP were added in plan 02-01
    by switching from top-5 to _MIN_CURRENCY_N=1000 threshold criterion.
    """
    with open("output/models/facility_stats_v1.json") as f:
        stats = json.load(f)
    cf = set(stats["currency_fallbacks"])
    mandated = {"AUD", "ILS", "GTQ", "PKR", "HKD", "AED", "BWP", "SGD", "COP"}
    missing = mandated - cf
    assert not missing, f"currency_fallbacks missing mandated currencies: {missing}"


# ---------------------------------------------------------------------------
# Task 3 — Integration test against the materialized artifact
# ---------------------------------------------------------------------------

_INTEGRATION_SKIP = not (STATS_JSON.exists() and TZ_PARQUET.exists() and TRAIN_PARQUET.exists())
_INTEGRATION_REASON = "artifact or parquets not available"


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_schema_version():
    """Materialized artifact must have schema_version='facility-stats-v1'."""
    with open(STATS_JSON) as f:
        s = json.load(f)
    assert s["schema_version"] == "facility-stats-v1"


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_facility_coverage():
    """Artifact must cover all 1876 facilities from facility_tz.parquet."""
    with open(STATS_JSON) as f:
        s = json.load(f)
    tz_df = pd.read_parquet(TZ_PARQUET)
    expected = tz_df["facility_id"].nunique()
    assert s["n_facilities"] == expected, f"n_facilities={s['n_facilities']} != expected {expected}"
    assert (
        len(s["facilities"]) == s["n_facilities"]
    ), f"len(facilities)={len(s['facilities'])} != n_facilities={s['n_facilities']}"


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_has_facility_level_entries():
    """Artifact must have at least some facilities with fallback_level='facility'."""
    with open(STATS_JSON) as f:
        s = json.load(f)
    n_facility = sum(1 for e in s["facilities"].values() if e["fallback_level"] == "facility")
    assert n_facility > 0, "No facilities with fallback_level='facility' found"


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_all_have_iana_tz():
    """Every facility in the artifact must have a non-empty iana_tz."""
    with open(STATS_JSON) as f:
        s = json.load(f)
    missing = [fid for fid, e in s["facilities"].items() if not e.get("iana_tz")]
    assert not missing, f"Facilities with missing iana_tz: {missing[:10]}"


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_iqr_guarded_count():
    """At least one facility must have iqr_guarded==1.0 (the IQR=0 cases)."""
    with open(STATS_JSON) as f:
        s = json.load(f)
    n_guarded = sum(1 for e in s["facilities"].values() if e.get("iqr") == 0.0)
    assert n_guarded > 0, "Expected at least one facility with iqr=0.0 (iqr_guarded=1.0)"


@pytest.mark.skipif(_INTEGRATION_SKIP, reason=_INTEGRATION_REASON)
def test_integration_validate_universe_filter():
    """validate_universe_filter must pass against train parquet and tz parquet."""
    from fraud_detector.stats.validator import validate_universe_filter

    with open(STATS_JSON) as f:
        s = json.load(f)
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=["amount", "facility_id", "currency"])
    tz_df = pd.read_parquet(TZ_PARQUET)
    result = validate_universe_filter(s, train_df, tz_df)
    assert result is True
