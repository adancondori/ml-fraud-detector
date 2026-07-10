"""validate_universe_filter: verify the facility stats artifact metadata.

Checks that (contrato frame-normalization-v1, spec artefactos-stats):
  1. universe_filter string matches the canonical 4-predicate filter
     (FINAL, _peerdb_is_deleted=0, NOT IN ('reversal','free'), user_id != 0).
     Legacy strings without ``user_id != 0`` are rejected explicitly.
  2. stats_window_start / stats_window_end are declared, parse as dates, and
     stats_window_end < shadow_period_start (anti-fuga temporal, simétrico con
     el builder). For NEW artifacts (con observed_*), el rango observado se
     re-verifica contra la ventana/shadow y contra sample_df["created_at"].
     Artefactos legados (sin observed_*) pasan por una rama legacy explícita.
  3. amount_source declares the source column mapping (must reference
     ``reservation_paid_out``; an empty or bare ``amount`` mapping is ambiguous).
  4. train_rows in the artifact matches len(sample_df) within 0.1% tolerance.
  5. schema_version == 'facility-stats-v1'.
  6. Facility coverage: len(facilities) == tz_df['facility_id'].nunique()
     (i.e. every live facility has an entry) and n_facilities matches.
  7. Every facility entry declares the ``iana_tz`` key. ``None`` is allowed
     (tzinfo_identifier vacío degrada por la cadena de fallback), pero la
     ausencia de la llave es un artefacto malformado.

Raises AssertionError with a descriptive message on any violation.
Returns True if all checks pass (convenient for scripting and testing).
"""

from __future__ import annotations

import datetime

import pandas as pd

from fraud_detector.stats.builder import CANONICAL_UNIVERSE_FILTER

EXPECTED_UNIVERSE_FILTER = CANONICAL_UNIVERSE_FILTER
EXPECTED_SCHEMA_VERSION = "facility-stats-v1"
ROW_COUNT_TOLERANCE = 0.001  # 0.1%


def validate_universe_filter(
    stats: dict,
    sample_df: pd.DataFrame,
    tz_df: pd.DataFrame,
) -> bool:
    """Verify the stats artifact universe filter, window, row count and coverage.

    Args:
        stats: The dict loaded from facility_stats_v1.json.
        sample_df: The training DataFrame (or a representative sample) used to
            check that train_rows in the artifact is consistent with the actual
            parquet. Must have at least one row.
        tz_df: DataFrame with a ``facility_id`` column covering all live
            facilities (snapshot de facilities.tzinfo_identifier). Used to
            compute the expected facility count.

    Returns:
        True if all assertions pass.

    Raises:
        AssertionError: if any check fails, with a descriptive message.
    """
    # 1. Schema version
    assert stats.get("schema_version") == EXPECTED_SCHEMA_VERSION, (
        f"schema_version mismatch: got '{stats.get('schema_version')}', "
        f"expected '{EXPECTED_SCHEMA_VERSION}'"
    )

    # 2. Universe filter: explicit rejection of the legacy string first,
    #    then exact match against the canonical 4-predicate filter.
    actual_filter = stats.get("universe_filter", "")
    assert "user_id != 0" in actual_filter, (
        f"universe_filter legado o incompleto: falta el predicado 'user_id != 0' "
        f"(got: '{actual_filter}')"
    )
    assert actual_filter == EXPECTED_UNIVERSE_FILTER, (
        f"universe_filter mismatch:\n"
        f"  got:      '{actual_filter}'\n"
        f"  expected: '{EXPECTED_UNIVERSE_FILTER}'"
    )

    # 3. Ventana temporal declarada (anti-fuga). Se parsean fechas (no strings)
    #    y se re-verifica end < shadow_period_start (simetría con el builder).
    #
    #    DISTINCIÓN DE ALCANCE (challenge NOTE 2, endurecimiento declarado):
    #    el check de ORDEN de ventana (stats_window_end < shadow_period_start)
    #    es metadata-only y UNIVERSAL — todo artefacto bien formado, nuevo o
    #    legacy, debe cumplirlo; un artefacto legacy con end >= shadow es un
    #    artefacto mal formado y se rechaza aquí. En cambio, los checks de
    #    PROCEDENCIA observed_* (rango real dentro de la ventana + coincidencia
    #    con sample_df["created_at"], en _validate_observed_provenance) son SOLO
    #    para artefactos nuevos (con observed_*); la rama legacy los omite.
    window_start = stats.get("stats_window_start")
    window_end = stats.get("stats_window_end")
    assert window_start and window_end, (
        f"stats_window_start/stats_window_end deben estar declaradas: "
        f"got start='{window_start}', end='{window_end}'"
    )
    start_date = datetime.date.fromisoformat(str(window_start))
    end_date = datetime.date.fromisoformat(str(window_end))
    assert (
        start_date < end_date
    ), f"stats_window invertida: start='{window_start}' >= end='{window_end}'"
    shadow_start = stats.get("shadow_period_start")
    if shadow_start:
        shadow_date = datetime.date.fromisoformat(str(shadow_start))
        assert end_date < shadow_date, (
            f"la ventana de stats solapa el período de shadow: "
            f"stats_window_end='{window_end}' >= shadow_period_start='{shadow_start}' "
            f"(se exige end < shadow)"
        )

    # 3b. Procedencia temporal observada.
    #   - Artefacto NUEVO: declara AMBAS claves observed_* -> rama estricta.
    #   - Artefacto LEGADO: declara NINGUNA -> rama legacy explícita.
    #   - Exactamente UNA clave -> artefacto malformado (no cae en legacy).
    _validate_observed_provenance(stats, sample_df, shadow_start)

    # 4. Mapping de monto fuente declarado y no ambiguo (design D9)
    amount_source = stats.get("amount_source", "")
    assert amount_source, "amount_source vacío: el mapping de monto fuente es obligatorio"
    assert "reservation_paid_out" in amount_source, (
        f"amount_source ambiguo: debe declarar la columna fuente "
        f"reservation_paid_out (got: '{amount_source}')"
    )

    # 5. Row count within tolerance
    artifact_rows = stats.get("train_rows", 0)
    parquet_rows = len(sample_df)
    if artifact_rows > 0:
        relative_diff = abs(artifact_rows - parquet_rows) / artifact_rows
        assert relative_diff < ROW_COUNT_TOLERANCE, (
            f"train_rows mismatch: artifact={artifact_rows}, "
            f"sample_df={parquet_rows}, "
            f"relative_diff={relative_diff:.4%} > tolerance {ROW_COUNT_TOLERANCE:.4%}"
        )

    # 6. Facility coverage: artifact must cover ALL facilities in tz_df
    expected_n = int(tz_df["facility_id"].nunique())
    actual_n_facilities = len(stats.get("facilities", {}))
    assert actual_n_facilities == expected_n, (
        f"facility coverage incomplete: len(facilities)={actual_n_facilities} "
        f"!= tz_df.facility_id.nunique()={expected_n}. "
        f"Builder likely iterated only train_df.groupby() instead of iana_map."
    )

    artifact_n_facilities = stats.get("n_facilities", -1)
    assert (
        artifact_n_facilities == expected_n
    ), f"n_facilities={artifact_n_facilities} != expected {expected_n}"

    # 7. Every facility entry must declare the iana_tz key (None allowed:
    #    tzinfo_identifier vacío degrada por la cadena de fallback en scoring).
    missing_key = [
        fid for fid, entry in stats.get("facilities", {}).items() if "iana_tz" not in entry
    ]
    assert not missing_key, (
        f"{len(missing_key)} facilities sin la llave iana_tz (artefacto malformado): "
        f"{missing_key[:10]}{'...' if len(missing_key) > 10 else ''}"
    )

    return True


def _validate_observed_provenance(
    stats: dict,
    sample_df: pd.DataFrame,
    shadow_start: str | None,
) -> None:
    """Re-verifica la procedencia temporal observada del artefacto.

    Ramifica por PRESENCIA de ambas claves observed_*:
      - Ambas presentes (artefacto nuevo): rango observado dentro de la ventana,
        estrictamente antes del shadow, y coincidencia con sample_df["created_at"].
      - Ninguna (artefacto legado): rama legacy explícita; no exige observed_*.
      - Exactamente una: artefacto malformado -> AssertionError.
    """
    has_min = "observed_min_created_at" in stats
    has_max = "observed_max_created_at" in stats

    if not has_min and not has_max:
        # Rama legacy explícita: artefacto sin procedencia observada. Los checks
        # temporales quedan limitados a la metadata declarada (ventana + shadow).
        return

    assert has_min and has_max, (
        "artefacto malformado: declara solo una de observed_min_created_at / "
        "observed_max_created_at; un artefacto nuevo DEBE declarar ambas "
        "(la ausencia de ambas es la única condición de la rama legacy)"
    )

    window_start = datetime.date.fromisoformat(str(stats["stats_window_start"]))
    window_end = datetime.date.fromisoformat(str(stats["stats_window_end"]))
    observed_min = datetime.date.fromisoformat(str(stats["observed_min_created_at"]))
    observed_max = datetime.date.fromisoformat(str(stats["observed_max_created_at"]))

    assert observed_min >= window_start, (
        f"observed_min_created_at fuera de ventana: "
        f"{observed_min.isoformat()} < stats_window_start={window_start.isoformat()}"
    )
    if shadow_start:
        shadow_date = datetime.date.fromisoformat(str(shadow_start))
        assert observed_max < shadow_date, (
            f"observed_max_created_at solapa el shadow (fuga temporal): "
            f"{observed_max.isoformat()} >= shadow_period_start={shadow_date.isoformat()}"
        )
    assert observed_max <= window_end, (
        f"observed_max_created_at fuera de ventana: "
        f"{observed_max.isoformat()} > stats_window_end={window_end.isoformat()}"
    )

    # Coincidencia con los datos realmente pasados (ata procedencia a filas).
    assert "created_at" in sample_df.columns, (
        "sample_df no contiene la columna 'created_at': no puedo revalidar la "
        "procedencia temporal de un artefacto nuevo (con observed_*)"
    )
    s = pd.to_datetime(sample_df["created_at"], errors="raise")
    sample_min = s.min().date()
    sample_max = s.max().date()
    assert observed_min == sample_min and observed_max == sample_max, (
        f"observed_* no coincide con sample_df['created_at']: "
        f"declarado min={observed_min.isoformat()}/max={observed_max.isoformat()}, "
        f"sample min={sample_min.isoformat()}/max={sample_max.isoformat()}"
    )
