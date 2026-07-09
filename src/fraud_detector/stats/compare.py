"""Reporte comparativo de facility stats old vs new (frame-normalization-v1).

Spec (artefactos-stats — Requirement: Reporte comparativo obligatorio):
el refresco produce un reporte viejo vs nuevo con, por facility, mean/median/
iqr/n/fallback_level de ambas versiones y sus deltas, más un veredicto global
``material_change`` con el umbral declarado en el propio reporte.

Criterio de materialidad (umbral por defecto, decisión humana 2):
- ``material_change = true`` si alguna facility con n >= 1000 tiene
  ``|Δmedian| / median_old > 0.10`` (ESTRICTAMENTE mayor), o
- si alguna facility cambia de ``fallback_level``. Una facility nueva computa
  como cambio de fallback_level solo si antes recibía scoring por fallback
  (ausente del artefacto viejo == fallback implícito) y ahora tiene stats
  propias (``fallback_level == "facility"``).

Las facilities nuevas NO computan en el criterio de Δmedian (no hay base).
"""

from __future__ import annotations

import datetime
from typing import Optional

MEDIAN_DELTA_THRESHOLD = 0.10  # estrictamente mayor
MIN_N_FOR_MEDIAN_CRITERION = 1_000

_SIDE_FIELDS = ("mean", "median", "iqr", "n", "fallback_level")


def compare_stats(old_stats: dict, new_stats: dict) -> dict:
    """Compara dos artefactos facility-stats y produce el reporte comparativo.

    Args:
        old_stats: dict del artefacto vigente (baseline).
        new_stats: dict del artefacto candidato.

    Returns:
        Reporte dict con threshold declarado, entrada por facility
        (status common/new/removed, valores old/new, deltas, flags) y
        veredicto global ``material_change`` con sus razones.
    """
    old_facilities = old_stats.get("facilities", {})
    new_facilities = new_stats.get("facilities", {})

    facilities_report: dict[str, dict] = {}
    material_reasons: list[str] = []

    all_fids = sorted(set(old_facilities) | set(new_facilities), key=str)
    n_common = n_new = n_removed = 0
    n_fallback_changes = n_median_exceeded = 0

    for fid in all_fids:
        old_entry = old_facilities.get(fid)
        new_entry = new_facilities.get(fid)

        if old_entry is not None and new_entry is not None:
            status = "common"
            n_common += 1
            median_rel_delta = _median_rel_delta(old_entry, new_entry)
            exceeds = _exceeds_median_threshold(old_entry, new_entry, median_rel_delta)
            fallback_changed = old_entry.get("fallback_level") != new_entry.get(
                "fallback_level"
            )
            delta = _deltas(old_entry, new_entry)
        elif new_entry is not None:
            # Facility nueva: fuera del criterio de Δmedian; antes recibía
            # scoring por fallback (implícito) — computa como cambio de nivel
            # solo si ahora tiene stats propias.
            status = "new"
            n_new += 1
            median_rel_delta = None
            exceeds = False
            fallback_changed = new_entry.get("fallback_level") == "facility"
            delta = None
        else:
            status = "removed"
            n_removed += 1
            median_rel_delta = None
            exceeds = False
            fallback_changed = False
            delta = None

        if exceeds:
            n_median_exceeded += 1
            material_reasons.append(
                f"facility {fid}: |Δmedian|/median_old="
                f"{median_rel_delta:.4f} > {MEDIAN_DELTA_THRESHOLD} con n >= "
                f"{MIN_N_FOR_MEDIAN_CRITERION}"
            )
        if fallback_changed:
            n_fallback_changes += 1
            old_level = old_entry.get("fallback_level") if old_entry else "fallback (implícito)"
            new_level = new_entry.get("fallback_level") if new_entry else None
            material_reasons.append(
                f"facility {fid}: fallback_level '{old_level}' -> '{new_level}'"
            )

        facilities_report[str(fid)] = {
            "status": status,
            "old": _side(old_entry),
            "new": _side(new_entry),
            "delta": delta,
            "median_rel_delta": median_rel_delta,
            "exceeds_median_threshold": exceeds,
            "fallback_level_changed": fallback_changed,
        }

    return {
        "schema_version": "stats-compare-v1",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "threshold": {
            "median_rel_delta": MEDIAN_DELTA_THRESHOLD,
            "min_n": MIN_N_FOR_MEDIAN_CRITERION,
            "comparison": "strictly_greater",
            "fallback_level_change_is_material": True,
        },
        "material_change": n_median_exceeded > 0 or n_fallback_changes > 0,
        "material_reasons": material_reasons,
        "summary": {
            "n_common": n_common,
            "n_new": n_new,
            "n_removed": n_removed,
            "n_fallback_changes": n_fallback_changes,
            "n_median_exceeded": n_median_exceeded,
        },
        "facilities": facilities_report,
    }


def render_summary_md(report: dict, evidence: Optional[dict] = None) -> str:
    """Renderiza el resumen MD del reporte comparativo (+ evidencia opcional)."""
    threshold = report["threshold"]
    summary = report["summary"]
    verdict = "true" if report["material_change"] else "false"

    lines = [
        "# Reporte comparativo de facility stats (old vs new)",
        "",
        f"- Generado: {report['generated_at']}",
        f"- **material_change: {verdict}**",
        (
            f"- Umbral: |Δmedian|/median_old > {threshold['median_rel_delta']} "
            f"(estrictamente mayor) en facilities con n >= {threshold['min_n']}, "
            f"o cambio de fallback_level"
        ),
        "",
        "## Resumen",
        "",
        "| Facilities comunes | Nuevas | Removidas | Cambios fallback | Δmedian excedidas |",
        "|---|---|---|---|---|",
        (
            f"| {summary['n_common']} | {summary['n_new']} | {summary['n_removed']} "
            f"| {summary['n_fallback_changes']} | {summary['n_median_exceeded']} |"
        ),
        "",
    ]

    if report["material_reasons"]:
        lines += ["## Razones de materialidad", ""]
        lines += [f"- {reason}" for reason in report["material_reasons"]]
        lines.append("")

    if evidence:
        lines += ["## Evidencia de procedencia", ""]
        lines += [f"- **{key}**: {value}" for key, value in evidence.items()]
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _side(entry: Optional[dict]) -> Optional[dict]:
    if entry is None:
        return None
    return {field: entry.get(field) for field in _SIDE_FIELDS}


def _median_rel_delta(old_entry: dict, new_entry: dict) -> Optional[float]:
    old_median = old_entry.get("median")
    new_median = new_entry.get("median")
    if old_median is None or new_median is None or old_median == 0:
        return None
    return abs(new_median - old_median) / old_median


def _exceeds_median_threshold(
    old_entry: dict, new_entry: dict, median_rel_delta: Optional[float]
) -> bool:
    if median_rel_delta is None:
        return False
    n_max = max(int(old_entry.get("n") or 0), int(new_entry.get("n") or 0))
    if n_max < MIN_N_FOR_MEDIAN_CRITERION:
        return False
    return median_rel_delta > MEDIAN_DELTA_THRESHOLD


def _deltas(old_entry: dict, new_entry: dict) -> dict:
    delta: dict[str, Optional[float]] = {}
    for field in ("median", "mean", "iqr", "n"):
        old_value = old_entry.get(field)
        new_value = new_entry.get(field)
        if old_value is None or new_value is None:
            delta[field] = None
        else:
            delta[field] = new_value - old_value
    return delta
