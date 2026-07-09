"""Tests del reporte comparativo de stats old vs new (frame-normalization-v1).

Scenarios (specs/artefactos-stats — Requirement: Reporte comparativo obligatorio):
- reporte/material: Δmediana 15% en facility n>=1000 -> material_change=true
- reporte/borde-exacto: Δ exactamente 0.10 (criterio estrictamente mayor) -> false
- reporte/facility-nueva: listada como `new`, fuera del criterio Δmedian; computa
  como cambio de fallback_level solo si antes recibía scoring por fallback
"""

from __future__ import annotations

import pytest

from fraud_detector.stats.compare import (
    MEDIAN_DELTA_THRESHOLD,
    MIN_N_FOR_MEDIAN_CRITERION,
    compare_stats,
    render_summary_md,
)


def _entry(n, median, mean=None, iqr=5.0, level="facility"):
    return {
        "n": n,
        "median": median,
        "mean": mean if mean is not None else median,
        "iqr": iqr,
        "iqr_guarded": max(iqr, 1.0) if iqr is not None else 1.0,
        "iana_tz": "America/New_York",
        "fallback_level": level,
    }


def _stats(facilities):
    return {"schema_version": "facility-stats-v1", "facilities": facilities}


# ---------------------------------------------------------------------------
# Scenario: cambio material detectado
# ---------------------------------------------------------------------------


def test_material_change_on_median_delta_above_threshold():
    """Facility n=5000 pasa de mediana 100 a 115 (15% > 10%) -> material."""
    old = _stats({"1": _entry(5000, 100.0)})
    new = _stats({"1": _entry(5000, 115.0)})

    report = compare_stats(old, new)

    assert report["material_change"] is True
    fac = report["facilities"]["1"]
    assert fac["median_rel_delta"] == pytest.approx(0.15)
    assert fac["exceeds_median_threshold"] is True
    assert any("1" in reason for reason in report["material_reasons"])


def test_threshold_declared_in_report():
    """El umbral usado está declarado en el propio reporte."""
    report = compare_stats(_stats({}), _stats({}))
    assert report["threshold"]["median_rel_delta"] == MEDIAN_DELTA_THRESHOLD == 0.10
    assert report["threshold"]["min_n"] == MIN_N_FOR_MEDIAN_CRITERION == 1000
    assert report["threshold"]["comparison"] == "strictly_greater"


def test_per_facility_deltas_present():
    """Cada facility común reporta mean/median/iqr/n/fallback_level de ambas versiones."""
    old = _stats({"7": _entry(2000, 100.0, mean=110.0, iqr=20.0)})
    new = _stats({"7": _entry(2100, 104.0, mean=112.0, iqr=22.0)})

    fac = compare_stats(old, new)["facilities"]["7"]

    for side in ("old", "new"):
        for field in ("mean", "median", "iqr", "n", "fallback_level"):
            assert field in fac[side], f"{side}.{field} ausente"
    assert fac["delta"]["median"] == pytest.approx(4.0)
    assert fac["delta"]["mean"] == pytest.approx(2.0)
    assert fac["delta"]["iqr"] == pytest.approx(2.0)
    assert fac["delta"]["n"] == 100


# ---------------------------------------------------------------------------
# Scenario: borde exacto del umbral (estrictamente mayor)
# ---------------------------------------------------------------------------


def test_exact_threshold_is_not_material():
    """Δ exactamente 0.10 y sin cambios de fallback_level -> material_change=false."""
    old = _stats({"1": _entry(5000, 100.0)})
    new = _stats({"1": _entry(5000, 110.0)})  # |Δ|/old = 0.10 exacto

    report = compare_stats(old, new)

    assert report["facilities"]["1"]["median_rel_delta"] == pytest.approx(0.10)
    assert report["facilities"]["1"]["exceeds_median_threshold"] is False
    assert report["material_change"] is False


def test_small_facility_large_delta_not_material():
    """Δ grande en facility con n < 1000 no dispara el criterio de mediana."""
    old = _stats({"9": _entry(500, 100.0)})
    new = _stats({"9": _entry(500, 180.0)})

    report = compare_stats(old, new)

    assert report["facilities"]["9"]["exceeds_median_threshold"] is False
    assert report["material_change"] is False


def test_fallback_level_change_is_material():
    """Un cambio de fallback_level en facility común es material por sí solo."""
    old = _stats({"3": _entry(40, 50.0, level="facility")})
    new = _stats({"3": _entry(25, 50.0, level="currency")})

    report = compare_stats(old, new)

    assert report["facilities"]["3"]["fallback_level_changed"] is True
    assert report["material_change"] is True


# ---------------------------------------------------------------------------
# Scenario: facility nueva sin versión previa
# ---------------------------------------------------------------------------


def test_new_facility_listed_and_excluded_from_median_criterion():
    """Facility nueva se lista como `new` y no computa en Δmedian."""
    old = _stats({"1": _entry(5000, 100.0)})
    new = _stats({"1": _entry(5000, 100.0), "2": _entry(2000, 999.0, level="currency")})

    report = compare_stats(old, new)

    fac = report["facilities"]["2"]
    assert fac["status"] == "new"
    assert fac["median_rel_delta"] is None
    assert fac["exceeds_median_threshold"] is False


def test_new_facility_still_on_fallback_is_not_material():
    """Nueva con fallback_level currency/global: antes fallback, sigue fallback -> no material."""
    old = _stats({"1": _entry(5000, 100.0)})
    new = _stats({"1": _entry(5000, 100.0), "2": _entry(0, 40.0, level="currency")})

    report = compare_stats(old, new)

    assert report["facilities"]["2"]["fallback_level_changed"] is False
    assert report["material_change"] is False


def test_new_facility_with_own_stats_counts_as_fallback_change():
    """Nueva con stats propias: antes recibía scoring por fallback -> cambio de nivel."""
    old = _stats({"1": _entry(5000, 100.0)})
    new = _stats({"1": _entry(5000, 100.0), "2": _entry(1500, 40.0, level="facility")})

    report = compare_stats(old, new)

    assert report["facilities"]["2"]["fallback_level_changed"] is True
    assert report["material_change"] is True


def test_removed_facility_listed():
    old = _stats({"1": _entry(5000, 100.0), "8": _entry(50, 20.0)})
    new = _stats({"1": _entry(5000, 100.0)})

    report = compare_stats(old, new)

    assert report["facilities"]["8"]["status"] == "removed"


# ---------------------------------------------------------------------------
# Resumen y render MD
# ---------------------------------------------------------------------------


def test_summary_counts():
    old = _stats({"1": _entry(5000, 100.0), "8": _entry(50, 20.0)})
    new = _stats({"1": _entry(5000, 120.0), "2": _entry(0, 40.0, level="global")})

    report = compare_stats(old, new)

    assert report["summary"]["n_common"] == 1
    assert report["summary"]["n_new"] == 1
    assert report["summary"]["n_removed"] == 1
    assert report["summary"]["n_median_exceeded"] == 1


def test_render_summary_md_contains_verdict_and_threshold():
    old = _stats({"1": _entry(5000, 100.0)})
    new = _stats({"1": _entry(5000, 115.0)})

    md = render_summary_md(compare_stats(old, new))

    assert "material_change" in md
    assert "0.1" in md  # umbral declarado
    assert "true" in md.lower()
