---
phase: 02-calibracion-segmentada-y-contrato-api
plan: "01"
subsystem: stats
tags: [facility-stats, currency-fallbacks, isolation-forest, parity, parquet, json-artifact]

# Dependency graph
requires:
  - phase: 01-artefacto-de-stats-y-feature-calculator
    provides: FacilityStatsBuilder, facility_stats_v1.json (5 currency fallbacks), test_parity_phase1.py guardrail
provides:
  - facility_stats_v1.json regenerado con 14 currency fallbacks (todos con n>=1000 en train)
  - _MIN_CURRENCY_N=1000 como criterio explícito de umbral (sustituye top-5 por volumen)
  - test_currency_fallback_threshold: test unitario sintético del nuevo criterio
  - test_materialized_artifact_has_mandated_currencies: guardrail de integración sobre el artefacto
  - fid 1214/1232/1373 promovidos de fallback_level='global' a 'currency'
affects:
  - 02-02-calibracion-segmentada (consume currency_fallbacks para segmentar calibración)
  - 03-contrato-api (scorer en vivo usa facility_stats_v1.json)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Umbral de mínimo de n para fallbacks de moneda: _MIN_CURRENCY_N=1000 (constante de módulo, autodescriptiva en artefacto via min_currency_n_threshold)"
    - "Artefacto JSON autodescriptivo: min_n_threshold (facility) + min_currency_n_threshold (currency) coexisten en el dict"
    - "EMPTY excluido explícitamente antes del umbral de moneda (previene fallback stats contaminadas)"

key-files:
  created:
    - output/models/facility_stats_v1.json (regenerado — commiteado con git add -f)
  modified:
    - src/fraud_detector/stats/builder.py
    - tests/test_facility_stats_builder.py

key-decisions:
  - "_MIN_CURRENCY_N=1000 reemplaza top-5-por-volumen: criterio basado en suficiencia estadística, no popularidad relativa"
  - "USD incluido siempre como garantía, independientemente del umbral (línea de seguridad para scorer)"
  - "facility_stats_v1.json commiteado forzado (output/models/ está en .gitignore) para reproducibilidad"

patterns-established:
  - "Constante de umbral de moneda en módulo: _MIN_CURRENCY_N (análoga a MIN_N=30 para facility)"
  - "Guardrail de integración skipif artefacto no existe: permite correr tests sin artefacto en CI sin fallos"

# Metrics
duration: 2min
completed: "2026-07-06"
---

# Phase 02 Plan 01: Currency Fallbacks Extension Summary

**facility_stats_v1.json regenerado con 14 monedas de fallback (AED/AUD/BWP/CAD/COP/GTQ/HKD/HNL/ILS/MYR/NIO/PKR/SGD/USD) mediante criterio _MIN_CURRENCY_N=1000; fid 1214/1232/1373 promovidos de global a currency; paridad batch↔real-time mantenida en 0.0 (35 tests PASS)**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-07-06T14:14:24Z
- **Completed:** 2026-07-06T14:16:32Z
- **Tasks:** 3/3
- **Files modified:** 3

## Accomplishments

- `_N_CURRENCY_FALLBACKS=5` eliminado; `_MIN_CURRENCY_N=1000` gobierna la selección de monedas con fallback dedicado
- `facility_stats_v1.json` regenerado: 14 currency_fallbacks (5 previas + AUD, ILS, GTQ, PKR, HKD, AED, BWP, SGD, COP), fallback=global reducido de 7 a 4
- Las 3 facilities que estaban en global (fid 1214 GTQ, 1232 AUD, 1373 HKD) pasaron a fallback_level='currency'
- `test_parity_phase1.py` re-corrido contra el artefacto regenerado: 11/11 PASS, maxdiff <1e-8
- 2 nuevos tests en `test_facility_stats_builder.py`: test unitario sintético del criterio + guardrail de integración sobre artefacto materializado

## Task Commits

1. **Task 1: Reemplazar criterio de currency_fallbacks por umbral de n en train** - `2c4c035` (feat)
2. **Task 2: Regenerar facility_stats_v1.json y actualizar el test del builder** - `dc653c1` (feat)
3. **Task 3: Re-correr paridad de Fase 1 sobre el artefacto regenerado** — sin commit propio (solo ejecución del guardrail; Task 2 ya materializa el artefacto)

**Plan metadata:** (a generar en commit docs(02-01))

## Parity Report (Task 3)

- **Test:** `tests/test_parity_phase1.py` — 11 tests
- **Result:** 11/11 PASS
- **maxdiff:** <1e-8 (tol exacta del test)
- **Pagos evaluados:** ≥100 pagos estratificados de ≥20 facilities del val set
- **Conclusión:** ampliar currency_fallbacks cambia magnitude stats de algunas facilities (global→currency), pero la asimetría batch/real-time es cero — ambas rutas usan el mismo artefacto y la misma lógica de lookup.

## Artifact Summary

| Métrica | Antes (01-03) | Después (02-01) |
|---------|--------------|-----------------|
| currency_fallbacks | 5 (USD, CAD, MYR, HNL, NIO) | 14 (+AED, AUD, BWP, COP, GTQ, HKD, ILS, PKR, SGD) |
| fallback=facility | 580 | 580 (sin cambio) |
| fallback=currency | ~1289 | 1292 |
| fallback=global | 7 | 4 |
| min_currency_n_threshold | (no existía) | 1000 |

## Files Created/Modified

- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/src/fraud_detector/stats/builder.py` — `_N_CURRENCY_FALLBACKS` → `_MIN_CURRENCY_N=1000`; bloque de selección de monedas reescrito; `min_currency_n_threshold` en dict de retorno
- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/tests/test_facility_stats_builder.py` — `test_currency_fallback_threshold` + `test_materialized_artifact_has_mandated_currencies` añadidos
- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/output/models/facility_stats_v1.json` — artefacto regenerado (commitado con `git add -f`)

## Decisions Made

- `_MIN_CURRENCY_N=1000` reemplaza top-5-por-volumen: criterio basado en suficiencia estadística (n>=1000 = base robusta para median/mean/IQR), no en popularidad relativa de la moneda. Esto garantiza que monedas con mucho volumen pero potencialmente ruidosas no se coelen por orden de frecuencia.
- USD siempre incluido como garantía de fallback de última instancia para el scorer, independientemente de si supera el umbral.
- `EMPTY` excluido explícitamente antes del filtro de umbral (ya santizado en loader.py desde 00-02, pero la exclusión explícita en el builder es defensiva).
- `facility_stats_v1.json` commiteado con `git add -f` (está en .gitignore como output de modelo) para reproducibilidad del artefacto y trazabilidad en git.

## Deviations from Plan

None - plan ejecutado exactamente como estaba escrito.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `facility_stats_v1.json` con 14 currency_fallbacks listo para 02-02 (calibración segmentada)
- Las 3 facilities previamente en global (fid 1214/1232/1373) ahora en currency, lo que mejora el marco de referencia para detección de anomalías en GTQ, AUD, HKD
- Guardrail de paridad re-confirmado: cualquier cambio futuro al artefacto puede re-correr `test_parity_phase1.py` como gate de regresión
- Bloqueante potencial de 02-02: el min-n para calibración segmentada (100 vs 200) aún pendiente de tabular distribución de tamaño de segmentos en val set (ya documentado en STATE.md desde Fase 1)

---
*Phase: 02-calibracion-segmentada-y-contrato-api*
*Completed: 2026-07-06*
