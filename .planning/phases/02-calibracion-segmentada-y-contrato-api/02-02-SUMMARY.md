---
phase: 02-calibracion-segmentada-y-contrato-api
plan: 02
subsystem: calibration
tags: [isolation-forest, segmented-thresholds, anomaly-detection, scikit-learn, numpy, tdd]

# Dependency graph
requires:
  - phase: 02-01
    provides: facility_stats_v1.json regenerado con 14 currency fallbacks (_MIN_CURRENCY_N=1000)
  - phase: 01-03
    provides: isolation_forest_frame_v1.joblib + scaler_frame_v1.joblib + add_frame_features_from_artifact
provides:
  - SegmentedThresholdCalibrator con MIN_N=200, LUT 1001pts global / 201pts segmento, cadena facility→currency→global
  - SegmentedThresholdClassifier con 5-tupla (is_anomaly, risk_level, percentile, fallback_level, calibration_segment)
  - thresholds_segmented_v1.json: 452 facilities, 17 monedas, global p95=0.04359 (frame-v1 verificado)
  - Script offline calibrate_segmented_thresholds.py con guardrail [0.040, 0.048]
  - 26 tests deterministas TDD sin carga de modelo
affects:
  - 02-03 (contrato API usa SegmentedThresholdClassifier para resolver segmento por transacción)
  - Fase 3 (re-cableado del scorer en vivo para consumir thresholds_segmented_v1.json)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SegmentedThresholdCalibrator.fit(): percentile por segmento con guarda MIN_N, LUT compacta 201pts"
    - "Fallback chain: facility_id → currency → global (resolución en orden de granularidad)"
    - "Guardrail de score: global p95 en [0.040, 0.048] para detectar mezcla IF-40 vs frame-v1"
    - "Module-level _compute_percentile helper compartido por ThresholdClassifier y SegmentedThresholdClassifier"
    - "Script offline de materialización: importa helpers de retrain_frame_v1.py sin ejecutar main()"

key-files:
  created:
    - src/fraud_detector/calibration/__init__.py
    - src/fraud_detector/calibration/segmented.py
    - scripts/calibrate_segmented_thresholds.py
    - output/models/thresholds_segmented_v1.json
    - tests/test_calibration_segmented.py
  modified:
    - src/fraud_detector/scoring/classifier.py

key-decisions:
  - "LUT global tiene 1001 puntos (consistencia con thresholds_v2.json); LUT de segmento tiene 201 puntos (JSON compacto)"
  - "SegmentedThresholdClassifier NO se importa en scorer.py (re-cableado es Fase 3); ThresholdClassifier legacy intacto"
  - "Guardrail [0.040, 0.048] en script offline para detectar mezcla IF-40/frame-v1 antes de escribir el artefacto"
  - "calibrate_segmented_thresholds.py importa add_frame_features_from_artifact desde retrain_frame_v1.py vía sys.path.insert(0, 'scripts') — no reimplementa lógica de features"
  - "by_currency tiene 17 entries: MXN (n=88) e INR (n=2) caen a global como se esperaba"

patterns-established:
  - "Segmented calibration pattern: fit() returns plain dict JSON-serializable (no state, no pickle)"
  - "TDD without model loading: tests use synthetic numpy arrays únicamente"

# Metrics
duration: 4min
completed: 2026-07-06
---

# Phase 2 Plan 02: Segmented Threshold Calibration Summary

**SegmentedThresholdCalibrator + Classifier con cadena facility→currency→global; thresholds_segmented_v1.json materializado desde scores frame-v1 (p95=0.04359), 452 facilities + 17 monedas, guardrail anti-IF-40 verificado**

## Performance

- **Duration:** 4 min
- **Started:** 2026-07-06T14:19:01Z
- **Completed:** 2026-07-06T14:23:21Z
- **Tasks:** 3 (Tasks 1+2 en un ciclo TDD, Task 3 independiente)
- **Files modified:** 6

## Accomplishments

- `SegmentedThresholdCalibrator` implementado con MIN_N=200, LUT 1001pts global/201pts segmento, keys raíz backward-compat (`binary_threshold`, `score_percentiles`)
- `SegmentedThresholdClassifier` añadido a classifier.py con cadena facility→currency→global; `ThresholdClassifier` IF-40 intacto; helper `_compute_percentile` extraído a nivel de módulo
- `thresholds_segmented_v1.json` materializado sobre 1,130,117 filas val: global p95=0.04359 (GUARDRAIL PASS), 452 facilities y 17 monedas con n≥200 cada una; MXN (n=88) e INR (n=2) correctamente excluidas
- 26 tests deterministas TDD completamente en verde sin cargar ningún modelo de disco
- Scorer en vivo (`scorer.py`) sin modificar; `SegmentedThresholdClassifier` disponible pero no conectado (re-cableado es Fase 3)

## Task Commits

Cada tarea fue commiteada atómicamente:

1. **Tasks 1+2: SegmentedThresholdCalibrator + SegmentedThresholdClassifier** - `7be925f` (feat)
2. **Task 3: materializar thresholds_segmented_v1.json** - `7cbfeba` (feat)

**Plan metadata:** (docs commit — ver a continuación)

_Nota: Tasks 1 y 2 se desarrollaron en un único ciclo TDD porque el archivo de tests importa ambas clases a nivel de módulo._

## Files Created/Modified

- `src/fraud_detector/calibration/__init__.py` — módulo nuevo, docstring de paquete
- `src/fraud_detector/calibration/segmented.py` — `SegmentedThresholdCalibrator.fit()` con MIN_N=200
- `src/fraud_detector/scoring/classifier.py` — `SegmentedThresholdClassifier` añadido; `_compute_percentile` helper extraído; `ThresholdClassifier` intacto
- `scripts/calibrate_segmented_thresholds.py` — script offline con guardrail [0.040, 0.048]
- `output/models/thresholds_segmented_v1.json` — artefacto calibrado (452 facilities, 17 monedas)
- `tests/test_calibration_segmented.py` — 26 tests TDD deterministas

## Decisions Made

- **LUT 1001 global / 201 segmento**: 1001 para consistencia con `thresholds_v2.json`; 201 por segmento mantiene el JSON a ~28MB en lugar de ~140MB, suficiente resolución para percentil.
- **ThresholdClassifier intacto**: el plan exige explícitamente que el scorer en vivo no sea re-cableado en esta fase. `SegmentedThresholdClassifier` existe y está testeado pero no se importa desde `scorer.py`.
- **Guardrail [0.040, 0.048]**: rango calculado desde el conocimiento previo (frame-v1 p95 ≈ 0.0436, IF-40 p95 ≈ 0.024). El script aborta con `RuntimeError` si el p95 global cae fuera de rango.
- **Import de retrain_frame_v1 via sys.path**: más seguro que reimplementar la lógica de features; `main()` de retrain está protegido con `if __name__ == "__main__"`, por lo que el import es seguro.
- **by_currency = 17 entries**: MXN (n=88) e INR (n=2) caen a global; los 17 restantes tienen n≥200. Confirmado con assertion en el script.

## Deviations from Plan

None — plan ejecutado exactamente como fue escrito. Los tests se desarrollaron simultáneamente para Tasks 1 y 2 porque el archivo importa ambas clases al nivel de módulo (no se puede ejecutar parcialmente), pero esto no constituye una desviación del plan dado que el plan los describe como un único ciclo TDD en la misma sección `<task>`.

## Issues Encountered

None. El import `from retrain_frame_v1 import ...` funcionó directamente con `sys.path.insert(0, "scripts")` sin necesidad de `importlib` alternativo.

## User Setup Required

None — no se requieren servicios externos. El script lee artefactos locales y escribe `output/models/thresholds_segmented_v1.json`.

## Next Phase Readiness

- `thresholds_segmented_v1.json` disponible para cualquier componente que quiera resolver umbrales por segmento
- `SegmentedThresholdClassifier` listo para conectar en `SingleTransactionScorer` (Fase 3)
- Contrato de la 5-tupla `(is_anomaly, risk_level, percentile, fallback_level, calibration_segment)` establecido y testeado
- Bloqueadores: ninguno; el scorer en vivo no se modifica hasta Fase 3

---
*Phase: 02-calibracion-segmentada-y-contrato-api*
*Completed: 2026-07-06*
