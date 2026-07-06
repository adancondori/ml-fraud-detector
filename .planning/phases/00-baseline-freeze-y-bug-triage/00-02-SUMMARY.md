---
phase: 00-baseline-freeze-y-bug-triage
plan: "02"
subsystem: data-pipeline
tags: [pandas, currency, feature-engineering, data-quality, threshold-management, gitignore]

# Dependency graph
requires:
  - phase: 00-baseline-freeze-y-bug-triage/00-01
    provides: scoring fixes (getattr + NaT) that unblock baseline freeze

provides:
  - Currency EMPTY/'' sanitization in loader.py and engineering.py (preventive)
  - DataManager._sanitize_currency() static helper with warning + count logging
  - FS-frame-operational-v1 materialized as output/models/final_feature_list_operational.json (39 features)
  - IF-31 legacy threshold path in run_fase7_evaluation.py marked deprecated
  - 7 tests covering sanitization logic, warning behavior, and data quality confirmation

affects:
  - 00-03 (consumes final_feature_list_operational.json and data quality count)
  - Fase 1 (facility stats computation uses sanitized currency from loader.py)
  - Any scorer that might load thresholds.json expecting it to be operative

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Static helper pattern: extract testable logic from inline method (_sanitize_currency)"
    - "Loguru capture via add/remove sink for testing warning emission"
    - "Self-describing JSON artifact with version metadata (feature_set_id, derived_from, excluded)"

key-files:
  created:
    - output/models/final_feature_list_operational.json
    - tests/test_currency_sanitize_phase0.py
  modified:
    - src/fraud_detector/data/loader.py
    - src/fraud_detector/features/engineering.py
    - scripts/run_fase7_evaluation.py
    - .gitignore

key-decisions:
  - "Extraer _sanitize_currency como helper estático (en lugar de inline) para testabilidad sin mock de normalizer"
  - "Captura de loguru en tests via add/remove sink (caplog de pytest no captura loguru por defecto)"
  - "Threshold legacy: marcar con LEGACY_DEPRECATED + legacy_do_not_use_for_IF40:true (no re-generar desde val — mínimo impacto)"
  - "Agregar exception !output/models/final_feature_list_operational.json a .gitignore para commitear el artefacto versionado"

patterns-established:
  - "Preventive data quality fix: fix + report 0 rows = confirmación positiva de limpieza"
  - "Artifact self-description: feature_set_id, derived_from, excluded, exclusion_reason en el JSON"
  - "Legacy marker: threshold_source con sufijo _LEGACY_DEPRECATED + flag boolean + WARNING log"

# Metrics
duration: 6min
completed: "2026-07-06"
---

# Phase 0 Plan 02: Saneamiento de Moneda, Feature Set Operativo y Threshold Legacy Summary

**Currency EMPTY sanitized in loader+engineering via static helper with warning, FS-frame-operational-v1 materialized as 39-feature versioned JSON, and IF-31 legacy threshold path marked DEPRECATED to prevent silent test-set threshold promotion**

## Performance

- **Duration:** 6 min
- **Started:** 2026-07-06T05:06:04Z
- **Completed:** 2026-07-06T05:12:34Z
- **Tasks:** 3 (completadas)
- **Files modified:** 6 (loader.py, engineering.py, run_fase7_evaluation.py, test nuevo, final_feature_list_operational.json, .gitignore)

## Accomplishments

- Currency EMPTY/'' sanitizada en los dos puntos del pipeline (extracción y feature engineering) con warning logueado y conteo de filas afectadas; 0 filas afectadas en splits actuales (fix preventivo confirmado)
- FS-frame-operational-v1 materializada como artefacto JSON versionado y autodescriptivo con 39 features (capture_delay_seconds excluida); commiteada en git mediante exception en .gitignore
- Ruta legacy IF-31 en run_fase7_evaluation.py marcada con threshold_source=percentile_95_test_set_LEGACY_DEPRECATED + legacy_do_not_use_for_IF40:true; thresholds_v2.json intacto (0.024223975402714343, percentile_95_validation_set)
- 7 tests verdes: mapeo EMPTY/''/None → USD, monedas válidas sin alterar, warning con conteo=3 via loguru sink, sin warning cuando clean, confirmación 0 EMPTY en train y val parquets

## Task Commits

1. **Task 1: Sanear moneda EMPTY/'' en loader.py y engineering.py** - `70d59e7` (fix)
2. **Task 2: Test de saneamiento de moneda + extracción de helper** - `f32da3c` (test)
3. **Task 3: Materializar FS-frame-operational-v1 + threshold legacy** - `4c470e6` (feat)

**Plan metadata:** *(pending docs commit)*

## Files Created/Modified

- `src/fraud_detector/data/loader.py` - Añadido `_sanitize_currency()` static helper; `_postprocess_extraction` delega al helper; warning con conteo de filas EMPTY
- `src/fraud_detector/features/engineering.py` - Línea 646: `.replace({"EMPTY": "USD", "": "USD"})` encadenado tras fillna/upper
- `tests/test_currency_sanitize_phase0.py` - 7 tests: mapeo de valores, monedas válidas, helper directo, warning via loguru sink, confirmación datos actuales
- `output/models/final_feature_list_operational.json` - FS-frame-operational-v1: 39 features, capture_delay_seconds excluida, artefacto autodescriptivo
- `scripts/run_fase7_evaluation.py` - `save_thresholds()` con docstring legacy, threshold_source=LEGACY_DEPRECATED, legacy_do_not_use_for_IF40:true, WARNING log
- `.gitignore` - Exception para `output/models/final_feature_list_operational.json`

## Data Quality Report

| Split | Filas EMPTY en currency | Status |
|-------|------------------------|--------|
| train_raw.parquet | 0 | Confirmado (3.137.086 filas) |
| val_raw.parquet | 0 | Confirmado (1.130.118 filas) |
| test_raw.parquet | No verificado* | Preventive |

*test_raw.parquet cargado en test que hace skip si no existe en el entorno. El fix es preventivo para futuras extracciones ClickHouse donde `currency.utils.fallback_rate` devolvería 1.0 para EMPTY (como si fuera USD), corrompiendo `log_amount`, `amount_usd_ratio` y `staff_amount_zscore`.

## FS-frame-operational-v1: 39 Features

Derivado de `final_feature_list.json` (40 features) excluyendo `capture_delay_seconds`.

**Excluida:** `capture_delay_seconds` — train/serve skew: valor real en batch (min=-86400, max=86400, mean=-52847, zero_pct=6.2%), ~0 en real-time. AUC flag 0.511. Decisión de proyecto ya tomada.

Ruta del artefacto: `output/models/final_feature_list_operational.json`

## Threshold Legacy: Decisión Tomada

**Opción elegida:** Marcar `thresholds.json` como legacy-deprecated en lugar de re-generar desde val set.

**Justificación:**
- `thresholds_v2.json` (IF-40) ya está correcto (`percentile_95_validation_set`, threshold=0.024223975402714343). No requiere ninguna acción.
- `thresholds.json` (IF-31) tiene `threshold_source: "percentile_95_test_set"` — se marcó como deprecated sin re-generar para minimizar impacto sobre tests existentes y el orquestador `run_fase7_evaluation.py`.
- La alternativa (re-generar desde val) requeriría cargar val_features en el script, lo que es un cambio de mayor alcance fuera del scope de Fase 0.

**Resultado:** `run_fase7_evaluation.py::save_thresholds()` ahora produce un artefacto con `legacy_do_not_use_for_IF40: true` y emite un WARNING en cada llamada. Cualquier código que lea `thresholds.json` sin verificar este flag tiene una señal explícita de que no debe usarse para IF-40.

## Decisions Made

1. **_sanitize_currency como helper estático** en lugar de inline: permite testear la lógica de warning directamente sin necesidad de mockear `CurrencyNormalizer`; reutilizable. Los tests usan `loguru.add()/remove()` para capturar el warning porque `caplog` de pytest no intercepta loguru por defecto.

2. **Threshold legacy: deprecar, no re-generar**: Re-generar el threshold de IF-31 desde val requeriría cargar y scorear val_features (~1.13M filas) en `run_fase7_evaluation.py`, lo que cambia el flujo del orquestador. Marcar el artefacto con flags explícitos + WARNING log tiene el mismo efecto de seguridad con cero riesgo de regresión.

3. **Exception .gitignore para final_feature_list_operational.json**: El artefacto es small JSON (<2KB), versionado, y consumido por 00-03. La exception sigue el mismo patrón que `thresholds_v2.json` ya existente en el gitignore.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Loguru no capturado por caplog de pytest**

- **Found during:** Task 2 (tests de warning)
- **Issue:** `caplog.at_level(logging.WARNING)` no captura mensajes de loguru — el logger del proyecto usa loguru, no stdlib logging. El test fallaba con "0 warnings capturados" aunque el warning se veía en stderr.
- **Fix:** Usar `loguru_logger.add(lambda msg: captured.append(msg), level="WARNING")` + `loguru_logger.remove(sink_id)` para captura directa. Patrón estándar para test de loguru sin caplog.
- **Files modified:** `tests/test_currency_sanitize_phase0.py`
- **Verification:** 7/7 tests verdes
- **Committed in:** f32da3c (Task 2)

**2. [Rule 3 - Blocking] output/models/ gitignored via `*.json`**

- **Found during:** Task 3 (git add de final_feature_list_operational.json)
- **Issue:** `.gitignore` tiene `*.json` global; solo `thresholds_v2.json` y `model_metadata.json` tienen exceptions. El artefacto nuevo era ignorado.
- **Fix:** Agregar `!output/models/final_feature_list_operational.json` a la lista de exceptions en .gitignore.
- **Files modified:** `.gitignore`
- **Verification:** `git add output/models/final_feature_list_operational.json` exitoso
- **Committed in:** 4c470e6 (Task 3)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Ambos fixes necesarios para completar las tareas. Sin scope creep.

## Issues Encountered

- `test_staff_zscore_parity` en `test_parity_phase0.py` (de plan 00-01) falla con max_delta=2.06 — pre-existente antes de este plan. El fix de 00-01 redujo el delta de 1030 a 2.06; la paridad perfecta (<1e-6) aún pendiente. No es una regresión de 00-02.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **00-03 (Baseline Freeze)** puede consumir:
  - `output/models/final_feature_list_operational.json` para FS-frame-operational-v1
  - Conteo de calidad: 0 filas EMPTY en train/val actuales
  - `thresholds_v2.json` como threshold operativo (intacto)
- **Fase 1** (facility stats) está protegida contra corrupción de currency EMPTY en futuras extracciones ClickHouse
- **Blocker pendiente (de 00-01):** `test_staff_zscore_parity` aún falla — la paridad de staff zscore no está al nivel <1e-6 todavía

---
*Phase: 00-baseline-freeze-y-bug-triage*
*Completed: 2026-07-06*
