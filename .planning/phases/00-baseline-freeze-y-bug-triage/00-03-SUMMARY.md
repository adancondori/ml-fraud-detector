---
phase: 00-baseline-freeze-y-bug-triage
plan: "03"
subsystem: baseline
tags: [python, pandas, isolation-forest, bias-reduction, golden-set, baseline-freeze, feature-engineering]

# Dependency graph
requires:
  - phase: 00-baseline-freeze-y-bug-triage/00-01
    provides: scorer real-time post-fix (getattr corregido, paridad batch<->RT exacta)
  - phase: 00-baseline-freeze-y-bug-triage/00-02
    provides: FS-frame-operational-v1 (final_feature_list_operational.json), thresholds_v2.json operativo, 0 filas EMPTY confirmado

provides:
  - output/golden_set_v0.parquet: 14831 pagos estratificados por facility_id (680 facilities, seed=42, reproducible)
  - output/baseline_v0.json: documento de baseline congelado post-fix con gate de sesgo formal, métricas actuales, flags de circularidad
  - scripts/build_baseline_v0.py: script reproducible que construye golden set y computa métricas post-fix
  - Gate formal: top-5% monto ratio <4x y off-hours local ~4-5% (reducción de sesgo, no AUC)
  - Punto de partida medido: top-5% ratio = 11.79x, off-hours UTC = 29.78%
  - AUC pure_fraud = 0.836 etiquetado explícitamente como diagnostic_circular_not_a_gate_metric
  - Fase 0 cerrada: BASE-01..06 satisfechos

affects:
  - Fase 1 (calibración facility stats, reducción de sesgo mide contra este baseline)
  - Cualquier reporte o tabla de tesis que cite el punto de partida de sesgo
  - Modelos alternativos (LOF, OC-SVM) que necesiten comparar su sesgo contra IF-40-v1

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Golden set estratificado por facility_id con seed fijo: reproducibilidad byte-a-byte"
    - "Baseline JSON como documento versionado autodescriptivo con flags metodológicos explícitos"
    - "AUC circular: etiqueta explicit diagnostic_circular_not_a_gate_metric en el artefacto"
    - "Gate de sesgo > AUC: criterio de éxito fijado en artefacto frozen, no en código"

key-files:
  created:
    - output/golden_set_v0.parquet
    - output/baseline_v0.json
    - scripts/build_baseline_v0.py
  modified: []

key-decisions:
  - "Gate de éxito = reducción de sesgo (top-5% monto <4x, off-hours local ~4-5%), nunca AUC"
  - "Punto de partida top-5% ratio = 11.79x sobre IF-40 post-fix (no 15.7x del research — cohorte distinta)"
  - "AUC Tipo A = 0.493 (diagnóstico honesto; IF no se alinea con reembolsos sobre FS-operational)"
  - "AUC pure_fraud = 0.836 marcado circular: 4 features del modelo definen el proxy; alta circularidad"
  - "Golden set: 14831 filas (680 facilities x 25 por facility), no truncado a 500 mínimo"
  - "Nota orquestador aceptada: corregidas features en gate_criteria de pure_fraud (same_amount_count_1h, user_account_age_days, user_txn_count_1h, is_third_party_payment)"

patterns-established:
  - "Baseline freeze: artefacto JSON con frozen_at, gate_metric, gate_criteria y classification de AUC"
  - "Punto de partida documentado con nota explicativa si difiere del valor citado en research"

# Metrics
duration: ~15min (incluyendo checkpoint de aprobación humana)
completed: "2026-07-06"
---

# Phase 0 Plan 03: Baseline Freeze Summary

**Baseline v0 congelado post-fix: IF-40 sobre FS-operational-v1 parte de top-5% ratio=11.79x y off-hours UTC=29.78%; gate fijado como reducción de sesgo (<4x, ~4-5% local); AUC pure_fraud=0.836 marcado circular; Fase 0 cerrada**

## Performance

- **Duration:** ~15 min (tareas automáticas ~10 min + checkpoint humano)
- **Started:** 2026-07-06T05:14:32Z
- **Completed:** 2026-07-06T05:24:50Z
- **Tasks:** 3 (2 auto + 1 checkpoint:human-verify)
- **Files modified:** 3 (scripts/build_baseline_v0.py creado, output/golden_set_v0.parquet creado, output/baseline_v0.json creado)

## Accomplishments

- Construido `output/golden_set_v0.parquet`: 14831 pagos estratificados por facility_id (680 facilities distintas, 25 por facility, seed=42), reproducible byte-a-byte desde `val_features_enriched.parquet`
- Computadas métricas POST-fix del scorer IF-40 corregido (post 00-01): top-5% monto ratio = 11.79x (mean top-5% = $3685.71 vs global $312.68 USD), off-hours UTC = 29.78%, tipo_a_rate = 6.08%
- Materializado `output/baseline_v0.json` con gate formal (bias_reduction), criterios explícitos (<4x monto, ~4-5% off-hours local), links a artefactos de 00-01/00-02, y AUC pure_fraud etiquetado como diagnostic_circular_not_a_gate_metric
- Checkpoint humano aprobado: baseline v0 queda congelado; Fase 0 cerrada
- Corrección del orquestador incorporada en baseline_v0.json (commit 42ac6df): features correctas de pure_fraud en nota de gate_criteria + aclaración punto de partida 11.79x vs 15.7x

## Task Commits

Cada tarea commiteada atómicamente:

1. **Task 1: Construir set dorado reproducible >=500 pagos estratificados** - `4d1efc9` (feat)
2. **Task 2: Materializar baseline_v0.json con gate de sesgo formal** - `d670ecb` (feat)
3. **Corrección orquestador: gate_criteria pure_fraud + nota 11.79x** - `42ac6df` (fix)
4. **Task 3: checkpoint:human-verify** - Aprobado por humano (no genera commit de código)

**Plan metadata:** *(este commit — docs(00-03))*

## Files Created/Modified

- `scripts/build_baseline_v0.py` — Script reproducible: `build_golden_set()` + `build_baseline_document()`. Lee val_features_enriched.parquet, muestrea estratificado por facility_id (seed=42), computa métricas post-fix sobre val completo (1.13M filas), escribe baseline_v0.json
- `output/golden_set_v0.parquet` — 14831 filas, 680 facilities, columnas para paridad y sesgo (facility_id, status, currency, user_role, amount, created_at, facility_avg_amount, amount_facility_ratio, staff_amount_zscore, capture_delay_seconds)
- `output/baseline_v0.json` — Documento de baseline congelado con: scorer_model=IF-40-v1, feature_set=FS-frame-operational-v1, threshold=0.024223975402714343, gate_metric=bias_reduction, métricas post-fix, data_quality_report (00-02), parity_test_result PASS (00-01), bugs_fixed (5 bugs 00-01/00-02), AUC circular etiquetado

## Métricas del Baseline Congelado (punto de partida para Fase 1)

| Métrica | Valor baseline v0 | Objetivo Fase 1 | Notas |
|---------|------------------|-----------------|-------|
| top-5% monto ratio | **11.79x** | <4x | Media top-5%=$3685.71 vs global=$312.68 (USD-normalized, val, IF-40 post-fix) |
| off-hours UTC rate | **29.78%** | ~4-5% local | Offset de timezone por facility pendiente (Fase 1). 39.21% en top-5% vs 29.78% global |
| AUC vs Tipo A (val) | 0.493 | diagnóstico | <0.5: IF no alinea con reembolsos sobre FS-operational-v1. No criterio |
| AUC vs pure_fraud (val) | 0.836 | diagnóstico circular | 4 features del modelo definen el proxy. No criterio de éxito ni progreso |
| Tipo A rate (val) | 6.08% | — | 68655/1130117 filas; proxy Tipo A solo para diagnóstico |

**Nota importante sobre top-5% ratio:** El valor 15.7x citado en el research (00-RESEARCH.md) proviene de una cohorte distinta (USD, feature set V0 contaminado). El valor correcto post-fix sobre IF-40 / FS-operational-v1 / todas las monedas es **11.79x**. Fase 1 compara contra este 11.79x.

## Set Dorado: Stats

- **Path:** `output/golden_set_v0.parquet`
- **Filas:** 14831
- **Facilities:** 680 (≥20 requeridas, 680 cubiertas)
- **Estrategia:** 25 filas por facility, seed=42, relleno hasta mínimo si es necesario
- **Reproducible:** Byte-a-byte con seed=42 desde val_features_enriched.parquet
- **Val set de origen:** data/processed/val_features_enriched.parquet (1130117 filas, Jul-Ago 2025)

## Satisfacción de Criterios BASE-01..06

| Criterio | Satisfecho por |
|----------|---------------|
| BASE-01: Scorer RT libre de train/serve skew | 00-01 (getattr fix + paridad exacta 0.00e+00) |
| BASE-02: currency EMPTY sanitizada | 00-02 (loader.py + engineering.py, 0 filas afectadas) |
| BASE-03: FS-frame-operational-v1 cerrada | 00-02 (final_feature_list_operational.json, 39 features) |
| BASE-04: threshold legacy deprecado | 00-02 (thresholds.json LEGACY_DEPRECATED; thresholds_v2.json intacto) |
| BASE-05: Set dorado reproducible >=500 pagos | 00-03 (14831 filas, 680 facilities, seed=42) |
| BASE-06: Baseline congelado con gate formal | 00-03 (baseline_v0.json, gate=bias_reduction, AUC circular etiquetado) |

## Decisions Made

1. **Gate de éxito = reducción de sesgo, no AUC**: El AUC vs pure_fraud (0.836) es parcialmente autovalidación (4 features del modelo definen el proxy). El criterio real de progreso es reducción del sesgo de monto (top-5% ratio <4x) y corrección de off-hours (de UTC a hora local, ~4-5%). Documentado explícitamente en baseline_v0.json con clasificación `diagnostic_circular_not_a_gate_metric`.

2. **Punto de partida 11.79x, no 15.7x**: El valor 15.7x del research provenía de una corrida con feature set V0 contaminado y cohorte solo USD. Sobre IF-40 post-fix con FS-operational-v1 y todas las monedas, el ratio es 11.79x. La corrección del orquestador en 42ac6df añadió esta nota aclaradora al baseline.

3. **Golden set 14831 filas, no truncado a 500**: La estratificación 25-por-facility sobre 680 facilities produce 14831 filas, muy por encima del mínimo de 500. No se truncó; más cobertura de facilities mejora la confianza del test de paridad y las métricas de sesgo.

4. **Features de pure_fraud corregidas en nota (orquestador)**: La nota original listaba features incorrectas. Las 4 features del modelo que definen el proxy pure_fraud son: same_amount_count_1h, user_account_age_days, user_txn_count_1h, is_third_party_payment. Esto es crítico para la documentación de la circularidad.

## Deviations from Plan

### Corrección del orquestador (post-checkpoint)

**1. [Corrección externa] Nota de gate_criteria listaba features incorrectas de pure_fraud**

- **Encontrado por:** Orquestador durante revisión del checkpoint
- **Problema:** La nota en `gate_criteria.notes` listaba features distintas a las 4 features del modelo que realmente definen el proxy pure_fraud
- **Fix:** Commit 42ac6df (orquestador): actualizar `gate_criteria.notes` con las features correctas (same_amount_count_1h, user_account_age_days, user_txn_count_1h, is_third_party_payment) y añadir `top5pct_amount_ratio_note` aclarando el punto de partida 11.79x vs 15.7x
- **Archivos modificados:** `output/baseline_v0.json`
- **Compromiso:** NO revertir; ya commiteado y aprobado por humano

---

**Total deviations:** 1 corrección post-checkpoint (externa, por orquestador)
**Impact on plan:** Corrección necesaria para exactitud metodológica. Sin scope creep.

## Issues Encountered

Ninguno bloqueante. Las métricas resultaron distintas a las del research (11.79x vs 15.7x) por diferencia de cohorte — documentado y aclarado en el baseline.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Fase 1 lista para iniciar**: Tiene baseline congelado con punto de partida exacto (11.79x monto, 29.78% off-hours UTC)
- **Gate de comparación establecido**: Fase 1 debe mostrar reducción a <4x (monto) y ~4-5% (off-hours local con timezone IANA por facility)
- **Artefactos disponibles para Fase 1:**
  - `output/baseline_v0.json` (referencia de gate)
  - `output/golden_set_v0.parquet` (14831 filas para evaluación de sesgo)
  - `output/models/final_feature_list_operational.json` (FS-frame-operational-v1)
  - `output/models/thresholds_v2.json` (threshold operativo)
  - `tests/test_parity_phase0.py` (guardrail de paridad activo)
- **Concern abierto**: off-hours local requiere `facility_time_zone_iana` en el payload Rails (lead time confirmado en STATE.md para Pre-Fase 4, pero Fase 1 puede estimar con timezone por país como proxy)

---
*Phase: 00-baseline-freeze-y-bug-triage*
*Completed: 2026-07-06*
