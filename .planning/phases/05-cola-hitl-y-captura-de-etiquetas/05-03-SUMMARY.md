---
phase: 05-cola-hitl-y-captura-de-etiquetas
plan: 03
subsystem: hitl
tags: [clickhouse, pandas, argparse, hitl, false-negatives, shadow_new, below-p50]

# Dependency graph
requires:
  - phase: 04-shadow-dual-run-y-validacion-de-sesgo
    provides: anomaly_scores tabla con scoring_mode shadow_new; _build_ch_client pattern; shadow_monitor query pattern
  - phase: 05-cola-hitl-y-captura-de-etiquetas/05-01
    provides: HitlQueueQuery reparto logic (capacity + below_p50_pct); VALID_CATEGORIES vocabulario
provides:
  - hitl_queue_builder.py: exportador operativo parametrizado (--top-k, --below-p50-pct, --capacity, --output)
  - compute_counts: función pura que replica HitlQueueQuery reparto (05-01) en Python
  - resolve_p50: quantile(0.5) sobre shadow_new con fallback 0.5
  - build_hitl_queue: DataFrame con top_k + below_p50 rows, hitl_queue_source tag, top_factors
  - docs/hitl_false_negative_methodology.md: justificación del muestreo defensivo >=20%
  - 12 tests con ClickHouse mockeado (sin datos shadow reales)
affects:
  - operacion-hitl (cuando shadow acumule >=2 semanas de datos)
  - 06-hipotesis-y-reporte (puede referenciar metodología FN como limitación del recall del modelo)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - _build_ch_client copiado exactamente de shadow_gate.py (env ANOMALY_SCORES_CH_*)
    - query via ch_client.query(sql).result_rows / .column_names (shadow_monitor.py)
    - MockQueryResult stub que despacha por substring SQL (quantile / ORDER BY percentile DESC / ORDER BY rand())
    - compute_counts pura sin I/O — fácil de testear aisladamente

key-files:
  created:
    - scripts/hitl_queue_builder.py
    - tests/test_hitl_queue_builder.py
    - docs/hitl_false_negative_methodology.md
  modified: []

key-decisions:
  - "compute_counts replica exactamente el reparto de HitlQueueQuery (05-01): capacity mode = floor(capacity*(1-pct)) top + remainder below; absolute mode = top_k + max(1, ceil(top_k*pct)) below"
  - "resolve_p50 usa quantile(0.5)(percentile) sin FINAL (shadow_monitor no usa FINAL; coherencia de patrón)"
  - "top_factors exportado como String JSON crudo desde anomaly_scores; no re-derivado"
  - "Script es OPERATIVO (ClickHouse live) — no unificado con hitl_export_alerts.py/hitl_ingest_labels.py (pipeline offline/parquet)"
  - "output default: output/hitl_queue_<timestamp>.csv cuando --output no se pasa"
  - "grep causal: 'determina' en contexto administrativo (coordinador determina capacidad) — aceptable, confirmado por lectura humana"

patterns-established:
  - "Separación operativo/offline: hitl_queue_builder.py (live ClickHouse) coexiste sin tocar hitl_export_alerts.py/hitl_ingest_labels.py"
  - "ENV vars para todo parámetro operativo: HITL_TOP_K, HITL_BELOW_P50_PCT, ANOMALY_SCORES_CH_*"
  - "Mock de ClickHouse client por substring SQL — patrón reutilizable para otros scripts de scripts/"

# Metrics
duration: 4min
completed: 2026-07-06
---

# Phase 05 Plan 03: HITL Queue Builder Summary

**Exportador operativo `hitl_queue_builder.py` (top-k shadow_new + muestreo below-p50 >=20%) con `compute_counts` que replica HitlQueueQuery, y metodología documentada de estimación de falsos negativos como cota inferior correlacional sobre el proxy**

## Performance

- **Duration:** 4 min
- **Started:** 2026-07-06T19:30:47Z
- **Completed:** 2026-07-06T19:34:41Z
- **Tasks:** 3/3
- **Files modified:** 3 creados

## Accomplishments

- `hitl_queue_builder.py` completamente parametrizado (`--top-k`, `--below-p50-pct`, `--capacity`, `--output`); filtra `scoring_mode='shadow_new'`; exporta `top_factors` y `hitl_queue_source`; default output a `output/hitl_queue_<timestamp>.csv`
- 12 tests verdes con ClickHouse mockeado — verifican filtro shadow_new, ORDER BY percentile DESC, rand() below-p50, reparto capacity/absoluto, hitl_queue_source marks, top_factors presentes, fallback p50=0.5, escritura CSV/JSON, creación de directorios padre
- `docs/hitl_false_negative_methodology.md` en español: motivación, estrategia >=20%, estimación FN como cota inferior correlacional sobre el proxy, vocabulario 4 categorías, trazabilidad por fila, estado diferido operacional

## Task Commits

1. **Task 1: hitl_queue_builder.py** — `35f6e20` (feat)
2. **Task 2: Tests mockeados** — `cd274b5` (test)
3. **Task 3: Metodología FN + lint fix** — `62b1f77` (docs)

## Files Created/Modified

- `scripts/hitl_queue_builder.py` — Exportador operativo HITL-01 (lado scorer): `_build_ch_client`, `resolve_p50`, `compute_counts`, `build_hitl_queue`, `write_output`, `main`
- `tests/test_hitl_queue_builder.py` — 12 tests con MockQueryResult; despacho por substring SQL
- `docs/hitl_false_negative_methodology.md` — Justificación del muestreo defensivo below-p50 >=20%

## Decisions Made

- `compute_counts` es función pura que replica exactamente la lógica de HitlQueueQuery (05-01): capacity mode usa `floor(capacity*(1-pct))` top + remainder below; absolute mode usa `top_k + max(1, ceil(top_k*pct))` below.
- `resolve_p50` no usa FINAL en la query (coherente con shadow_monitor.py que tampoco lo usa).
- `top_factors` se exporta como String JSON crudo de la columna de anomaly_scores; no se re-deriva en el builder.
- Script separado de `hitl_export_alerts.py`/`hitl_ingest_labels.py` (Pitfall 1): conviven sin solapamiento. Solo se reutiliza el vocabulario VALID_CATEGORIES para el header del CSV.
- El match de `grep -iE "determina"` en el documento de metodología corresponde al uso administrativo "el coordinador determina la capacidad" — no es lenguaje causal en sentido metodológico; confirmado por lectura humana.

## Deviations from Plan

None — plan ejecutado exactamente como se especificó.

**[Rule 3 - Blocking]** El import de `pytest` en `test_hitl_queue_builder.py` generaba `F401` en lint; removido. `df` no utilizado en `test_build_queue_capacity_mode` generaba `F841`; reemplazado por llamada directa sin asignación. Ambos fixes son menores de lint, incluidos en el commit de Task 3.

## Issues Encountered

None — `make format` reformateó el test file (Black alineó indentación), lo cual es comportamiento esperado.

## User Setup Required

None — no se requiere configuración de servicios externos para este plan.

## Next Phase Readiness

El exportador y los tests están listos y verificados con datos sintéticos. Para operar en producción:

1. `SCORING_MODE=shadow_dual` debe acumular **>=14 días / >=500 filas** en `anomaly_scores` (SHAD-03 PENDING_DATA)
2. Confirmar **capacidad de revisión del equipo HITL** (bloqueante Pre-Fase 5 en STATE.md) para calibrar `--capacity` y `--below-p50-pct`

Cuando ambas condiciones se cumplan:
```bash
python scripts/hitl_queue_builder.py --capacity <N> --below-p50-pct 0.20 --output output/hitl_queue_$(date +%Y%m%d).csv
```

---
*Phase: 05-cola-hitl-y-captura-de-etiquetas*
*Completed: 2026-07-06*
