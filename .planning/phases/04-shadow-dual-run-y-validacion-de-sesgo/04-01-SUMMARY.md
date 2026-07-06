---
phase: 04-shadow-dual-run-y-validacion-de-sesgo
plan: 01
subsystem: infra
tags: [clickhouse, isolation-forest, shadow-scoring, dual-run, dedup-token, batch-scorer]

requires:
  - phase: 03-wiring-del-scorer-e-integracion-platform
    provides: SingleTransactionScorer con dispatch frame-v1, BatchScorer con _INSERT_COLUMNS (23), ScoringResult con calibration_segment/fallback_level/frame_flags

provides:
  - Migración DDL idempotente: 3 columnas frame-v1 en anomaly_scores (DEFAULT '')
  - load_artifacts(metadata_filename=...) — carga champion (model_metadata.json) o challenger (model_metadata_frame_v1.json) desde el mismo directorio
  - Lifespan dual en main.py: scorer_champion (IF-40-v1) + scorer_new (frame-v1) en _state cuando SCORING_MODE=shadow_dual
  - ShadowDualRunner.score_pair: champion + challenger con UserContext compartido; fallo parcial aislado; nunca lanza excepción
  - BatchScorer modo dual: 2 filas/pago con scoring_mode=shadow_old/shadow_new; tokens shadow-old-/shadow-new- (sin dedup de ClickHouse); _INSERT_COLUMNS extendido a 26
  - Retrocompat activo: modo active (scorer_shadow=None) con 26 columnas (' ' para frame-v1 cols)
  - 57 tests verdes (incluyendo retrocompat IF-40 + frame-v1 dispatch + dual-run suite)

affects:
  - 04-02 (shadow gate y métricas SHAD-02/SHAD-03 — consumen las 2 filas/pago producidas por SHAD-01)
  - scripts/shadow_monitor.py (consume scoring_mode=shadow_old/shadow_new y columnas frame-v1)
  - scorer/routers/score.py (puede exponer scoring_mode del _state en /health)

tech-stack:
  added: []
  patterns:
    - "Dedup token prefijado por modelo (shadow-old-/shadow-new-) para evitar deduplicación de segundo INSERT en ClickHouse MergeTree"
    - "UserContext compartido entre champion y challenger: contexto factual = model-independent; construido una vez, pasado a ambos scorers"
    - "Partial-failure isolation en dual-run: cada INSERT en try/except independiente; fallo de uno no afecta al otro"
    - "metadata_filename override en _load_metadata: fallbacks legacy solo cuando filename es el default"

key-files:
  created:
    - docker/clickhouse/migrations/02_anomaly_scores_frame_v1.sql
    - scorer/shadow/__init__.py
    - scorer/shadow/dual_runner.py
    - tests/test_shadow_dual_runner.py
  modified:
    - scorer/artifact_loader.py
    - scorer/main.py
    - scorer/batch/scorer.py
    - tests/test_artifact_loader.py

key-decisions:
  - "metadata_filename override: legacy fallbacks solo cuando filename == 'model_metadata.json'; override explícito falla con FileNotFoundError si ausente"
  - "_score_all_dual usa features_json={} vacío: los vectores de features son internos a cada scorer; re-extraerlos en el batch runner duplicaría lógica"
  - "_INSERT_COLUMNS: 23→26; active mode pasa '' para las 3 columnas frame-v1 (compatible con DEFAULT '' del DDL)"
  - "scorer_champion/_new en _state solo cuando scoring_mode=shadow_dual; scorer activo siempre presente independientemente"

patterns-established:
  - "Pattern dual-INSERT ClickHouse: dos INSERT separados con tokens shadow-old-/shadow-new- distintos; assert_write_target_is_safe primera llamada en _insert_chunks_dual"
  - "ShadowDualRunner como clase thin wrapper: recibe dos scorers ya instanciados; no sabe nada de ClickHouse; fácil de testear con mocks"

duration: 8min
completed: 2026-07-06
---

# Phase 04 Plan 01: Shadow Dual-Run Infrastructure (SHAD-01) Summary

**DDL migration con 3 columnas frame-v1 + ShadowDualRunner con contexto compartido + BatchScorer dual mode (2 filas/pago, 26 cols, tokens shadow-old-/shadow-new-)**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-07-06T18:08:21Z
- **Completed:** 2026-07-06T18:16:50Z
- **Tasks:** 2
- **Files modified:** 8 (4 created, 4 modified)

## Accomplishments

- Migración DDL idempotente `02_anomaly_scores_frame_v1.sql` — `ADD COLUMN IF NOT EXISTS` para `calibration_segment`, `fallback_level`, `frame_flags` con `DEFAULT ''`; re-ejecutable sin error sobre volumen existente
- `ShadowDualRunner.score_pair` — champion (IF-40) + challenger (frame-v1) con el mismo `UserContext` (sin duplicar los 6 queries bulk); fallo parcial aislado por try/except independiente; nunca lanza excepción
- `BatchScorer` modo dual — `_INSERT_COLUMNS` extendido a 26; `_score_all_dual` + `_insert_chunks_dual` con tokens `shadow-old-*` / `shadow-new-*` distintos; `assert_write_target_is_safe` primera llamada del path dual; modo active retrocompatible con 26 columnas
- `load_artifacts(metadata_filename=...)` — override sin romper fallbacks legacy IF-40/IF-31; lifespan carga scorer_champion + scorer_new cuando `SCORING_MODE=shadow_dual`

## Task Commits

1. **Task 1: DDL migration + metadata_filename override + lifespan dual** - `114dddc` (feat)
2. **Task 2: ShadowDualRunner + BatchScorer dual mode** - `daad698` (feat)

**Plan metadata:** (en este commit — docs)

## Files Created/Modified

- `docker/clickhouse/migrations/02_anomaly_scores_frame_v1.sql` — ALTER TABLE ADD COLUMN IF NOT EXISTS (3 columnas frame-v1, idempotente)
- `scorer/artifact_loader.py` — `_load_metadata(metadata_filename)` override; `load_artifacts` propaga el param; legacy fallbacks condicionados al filename default
- `scorer/main.py` — `ScorerSettings` añade `scoring_mode`, `shadow_champion_metadata`, `shadow_new_metadata`; lifespan carga dual cuando `scoring_mode=shadow_dual`
- `scorer/shadow/__init__.py` — package marker
- `scorer/shadow/dual_runner.py` — `ShadowDualRunner`: `score_pair` con aislamiento de fallo parcial, contexto compartido, delta logging
- `scorer/batch/scorer.py` — `_INSERT_COLUMNS` 23→26; `BatchScorer.__init__` añade `scorer_shadow=None`; `_score_all` añade `''` para cols frame-v1; `_score_all_dual` + `_insert_chunks_dual` nuevos
- `tests/test_artifact_loader.py` — `TestMetadataFilenameOverride` (4 tests): default regression, override explícito, frame-v1 load, missing override raises
- `tests/test_shadow_dual_runner.py` — 16 tests SHAD-01: aislamiento parcial (a/b), 2 filas/pago (c), 26 columnas (d), model_version (e), dedup tokens (f), guardrail (g), retrocompat active (h)

## Decisions Made

- **metadata_filename legacy guard:** Los fallbacks legacy (`final_feature_list.json` → IF-40; default → IF-31) solo se activan cuando `metadata_filename == "model_metadata.json"`. Un override explícito que no exista lanza `FileNotFoundError` en lugar de retornar silenciosamente un modelo incorrecto.
- **features_json vacío en dual-run:** `_score_all_dual` almacena `{}` en `features_json` para las filas shadow. Los vectores de features son internos a cada `SingleTransactionScorer`; re-extraerlos en el runner duplicaría la lógica del feature calculator. Las queries de monitoreo SHAD-02 no necesitan los vectores raw.
- **scorer activo independiente del dual:** El `scorer` activo (cargado sin cambios en el lifespan) no es reemplazado por `scorer_champion`. Los tres scorers (`scorer`, `scorer_champion`, `scorer_new`) coexisten en `_state` cuando `scoring_mode=shadow_dual`. El scorer activo sigue respondiendo rutas RT; el dual-runner es exclusivamente del path batch.
- **E2E del DDL diferido al runbook:** Docker/ClickHouse no está arriba en el entorno de ejecución. La migración se valida con el comando manual documentado abajo; los tests Python cubren toda la lógica Python.

## Deviations from Plan

None — plan ejecutado exactamente como escrito.

## Issues Encountered

Tres lint fixes menores tras la implementación inicial:
1. E305 — blank line after function definition en `scorer.py` (antes de `_INSERT_COLUMNS`)
2. F841 — variable `feature_names_r` asignada pero no usada en `_score_all_dual` (removida; `features_json` almacena `{}`)
3. F401 — imports no usados en `test_shadow_dual_runner.py` (`call`, `assert_write_target_is_safe`)

Todos resueltos antes del commit de Task 2.

## Runbook

### Activar la migración DDL (ClickHouse local)

```bash
# Idempotente — seguro de re-ejecutar sobre volumen existente
docker exec -i clickhouse clickhouse-client --multiquery \
  < docker/clickhouse/migrations/02_anomaly_scores_frame_v1.sql

# Verificar las 3 columnas nuevas
docker exec clickhouse clickhouse-client \
  -q "DESCRIBE pbp_productionDB_optimized.anomaly_scores FORMAT TabSeparated" \
  | grep -E "calibration_segment|fallback_level|frame_flags"
```

### Activar shadow dual-run (scorer FastAPI)

```bash
# Variables de entorno necesarias
export SCORING_MODE=shadow_dual
export SHADOW_CHAMPION_METADATA=model_metadata.json
export SHADOW_NEW_METADATA=model_metadata_frame_v1.json

# Ambos modelos deben estar en MODEL_DIR (output/models/)
uvicorn scorer.main:app --host 0.0.0.0 --port 8000
```

El lifespan verificará:
- `scorer_champion._model_version == 'IF-40-v1'`
- `scorer_new._model_version == 'frame-v1'`

y fallará con `AssertionError` si la configuración está incorrecta.

### Verificar dedup tokens en producción

Los tokens del dual-run tienen la forma:
- `shadow-old-{cursor_iso}-{cursor_end_iso}-IF40v1-chunk-{i}`
- `shadow-new-{cursor_iso}-{cursor_end_iso}-framev1-chunk-{i}`

```sql
-- Contar filas por modelo en la ventana shadow
SELECT scoring_mode, model_version, count()
FROM pbp_productionDB_optimized.anomaly_scores
WHERE scoring_mode IN ('shadow_old', 'shadow_new')
GROUP BY scoring_mode, model_version
ORDER BY scoring_mode;
```

## Next Phase Readiness

- **04-02 (SHAD-02/SHAD-03):** La infraestructura dual-run está completa. El BatchScorer produce 2 filas/pago con los campos necesarios para las 4 métricas de monitoreo (alert rate, top-5% winsorized, off-hours local, Jaccard@100). El script `shadow_gate.py` puede implementarse en 04-02 apuntando a `scoring_mode IN ('shadow_old', 'shadow_new')`.
- **Prerequisito operativo:** Ejecutar la migración DDL en ClickHouse local antes de la primera corrida dual (`02_anomaly_scores_frame_v1.sql`).
- **Gate temporal:** SHAD-03 requiere ≥2 semanas de datos shadow — documentar en 04-02 como checkpoint diferido.

---
*Phase: 04-shadow-dual-run-y-validacion-de-sesgo*
*Completed: 2026-07-06*
