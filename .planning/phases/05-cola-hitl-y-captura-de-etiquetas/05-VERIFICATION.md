---
phase: 05-cola-hitl-y-captura-de-etiquetas
verified: 2026-07-06T20:00:00Z
status: passed
score: 7/7 must-haves verified
---

# Phase 05: Cola HITL y Captura de Etiquetas — Verification Report

**Phase Goal:** Cola top-k de frame-v1 con `top_factors`; etiquetas con procedencia completa; muestreo >=20% below-p50.
**Verified:** 2026-07-06
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Cola top-k filtra `scoring_mode='shadow_new'`, ordena por `percentile DESC`, incluye `top_factors` parseado | VERIFIED | `hitl_queue_query.rb` line 63/71/72: tres SQLs con `WHERE scoring_mode = 'shadow_new'` y `ORDER BY percentile DESC`; `build_row` llama `parse_json_array(row["top_factors"])` |
| 2 | `TABLE` default SIN FINAL; `FINAL` viene del ENV operador (no hardcodeado) | VERIFIED | Line 8: `ENV.fetch("ANOMALY_DETECTION_CLICKHOUSE_TABLE", "pbp_productionDB_optimized.anomaly_scores")` — sin FINAL en el default; `stub_const` en spec para test FINAL |
| 3 | Parámetros `top_k`, `below_p50_pct`, `capacity` configurables via ENV/params | VERIFIED | `DEFAULT_TOP_K = ENV.fetch("HITL_TOP_K", 100).to_i`, `DEFAULT_BELOW_P50_PCT = ENV.fetch("HITL_BELOW_P50_PCT", 0.20).to_f`; lógica `capacity` en `initialize` |
| 4 | Muestreo >=20% below-p50 implementado y parametrizable en ambos repos | VERIFIED | Ruby: `@below_k_count = [1, (top_k * @below_p50_pct).ceil].max`; Python: `max(1, math.ceil(top_k * below_p50_pct))` — fórmula idéntica |
| 5 | Migración retrocompatible con 6 columnas nullable; `reviewer_id` = alias de `performed_by_id` | VERIFIED | Migración `20260706193114` con 6 ADD COLUMN nullable+default; `alias_attribute :reviewer_id, :performed_by_id` en `triage_action.rb`; columnas confirmadas en `structure.sql` |
| 6 | `HitlLabelService` registra 6 campos de procedencia (incluyendo `Time.current`); rollback en fallo | VERIFIED | `hitl_label_service.rb` lines 44-51: `update!` con los 6 campos; `reviewed_at: Time.current`; dentro de `ActiveRecord::Base.transaction` |
| 7 | SC-4: `model_version_at_label` + `hitl_queue_source` permiten trazar qué modelo generó el score | VERIFIED | Ambas columnas en migración + `update!` en service; `model_version_at_label: score_row["model_version"]`; `hitl_queue_source: form.hitl_queue_source` |

**Score:** 7/7 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `platform/packs/anomaly_detection/app/queries/anomaly_detection/hitl_queue_query.rb` | HitlQueueQuery READ-only top-k + below-p50 | VERIFIED | 125 lines; clase completa con TABLE, SCORE_COLUMNS, 3 SQLs, parse_json_array, hitl_queue_source |
| `platform/packs/anomaly_detection/spec/queries/anomaly_detection/hitl_queue_query_spec.rb` | 12 ejemplos RSpec con mocks | VERIFIED | 266 lines; 12 ejemplos cubriendo shadow_new, FINAL via stub_const, ORDER BY, LIMIT, below-p50, hitl_queue_source, top_factors, meta, capacity, failure, edge-case top_k_count=0 |
| `platform/packs/anomaly_detection/db/migrate/20260706193114_add_hitl_fields_to_anomaly_detection_triage_actions.rb` | Migración 6 columnas HITL nullable | VERIFIED | Timestamp Rails-generated (193114, no redondo); 6 ADD COLUMN con tipos correctos; guard `table_exists?`; up/down reversible |
| `platform/packs/anomaly_detection/app/forms/anomaly_detection/hitl_label_form.rb` | HitlLabelForm con vocabulario 4 categorías | VERIFIED | 42 lines; `VALID_REVIEWER_LABELS`, `VALID_QUEUE_SOURCES`, validación inclusion, valid_transition vía `TriageTransitionForm::VALID_TRANSITIONS` |
| `platform/packs/anomaly_detection/app/services/anomaly_detection/hitl_label_service.rb` | HitlLabelService enriquece TriageAction | VERIFIED | 71 lines; envuelve TriageTransitionService en transaction; captura status_change con fallback defensivo; update! con 6 campos |
| `platform/packs/anomaly_detection/spec/forms/anomaly_detection/hitl_label_form_spec.rb` | 16 ejemplos form | VERIFIED | 138 lines; 16 it-blocks |
| `platform/packs/anomaly_detection/spec/services/anomaly_detection/hitl_label_service_spec.rb` | 16 ejemplos service | VERIFIED | 151 lines; 16 it-blocks |
| `ml-fraud-detector/scripts/hitl_queue_builder.py` | Exportador CLI parametrizado | VERIFIED | 395 lines; `_build_ch_client`, `resolve_p50`, `compute_counts`, `build_hitl_queue`, `write_output`, `main`; 4 args CLI |
| `ml-fraud-detector/tests/test_hitl_queue_builder.py` | 12 tests con ClickHouse mockeado | VERIFIED | 12 tests, 12/12 pasan (`pytest -q`: `12 passed in 0.37s`) |
| `ml-fraud-detector/docs/hitl_false_negative_methodology.md` | Metodología sin lenguaje causal | VERIFIED (con nota) | Existe; explica motivación, estrategia >=20%, estimación FN como cota inferior correlacional, vocabulario 4 categorías, trazabilidad, estado diferido. Ver nota causal abajo. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `hitl_queue_query.rb` | `Analytics::ClickhouseClient.select` | 3 llamadas (p50_sql, top_k_sql, below_p50_sql) | WIRED | Lines 46, 53; cada SQL con `scoring_mode='shadow_new'` |
| `hitl_queue_query.rb SQL` | `anomaly_scores` (TABLE) | `FROM #{TABLE}` en los 3 SQLs | WIRED | TABLE usa ENV; default sin FINAL |
| `hitl_label_service.rb` | `TriageTransitionService` | `TriageTransitionService.call(form: triage_form)` dentro de transaction | WIRED | Line 39; retorno capturado como status_change con fallback defensivo |
| `hitl_label_service.rb` | `TriageAction` columnas HITL | `status_change_action.update!(reviewer_label:, reviewed_at: Time.current, ...)` | WIRED | Lines 44-51; los 6 campos |
| `hitl_label_form.rb` | `VALID_REVIEWER_LABELS` | `validates :reviewer_label, inclusion: { in: VALID_REVIEWER_LABELS }` | WIRED | Line 26 |
| `hitl_queue_builder.py` | `_build_ch_client` | `ch_client = _build_ch_client()` en `main` | WIRED | Line 362; env `ANOMALY_SCORES_CH_*` |
| `hitl_queue_builder.py SQL` | `anomaly_scores` | `WHERE scoring_mode = 'shadow_new'` en las 3 queries (p50, top-k, below) | WIRED | Lines 131, 228, 242 |
| `compute_counts` (Python) | `HitlQueueQuery` (Ruby) | Misma fórmula `floor/ceil + max(1,...)` | WIRED (cross-repo) | Ruby lines 24-28 == Python lines 174-178 |

---

## Requirements Coverage

| Criterio | Status | Notas |
|----------|--------|-------|
| HITL-01: cola top-k `scoring_mode='shadow_new'` ORDER BY percentile DESC + `top_factors` | SATISFIED | Ambos repos: Ruby HitlQueueQuery + Python hitl_queue_builder |
| HITL-02: migración 6 campos nullable + HitlLabelForm (4 categorías) + HitlLabelService (procedencia completa) | SATISFIED | Timestamp Rails-generated; `reviewer_id`=`performed_by_id` (sin columna duplicada); Time.current; transaction |
| HITL-03: muestreo >=20% below-p50 parametrizable + docs/hitl_false_negative_methodology.md | SATISFIED | `[1, ceil].max` garantiza mínimo 1; default 20%; doc existe con metodología correlacional |
| SC-4: `model_version_at_label` + `hitl_queue_source` para trazar modelo generador | SATISFIED | Ambas columnas en migración, en service update!, y en Python builder |

---

## Anti-Patterns Found

| File | Pattern | Severity | Assessment |
|------|---------|----------|------------|
| `hitl_queue_query.rb` | Ninguno | — | CLEAN |
| `hitl_label_form.rb` | Ninguno | — | CLEAN |
| `hitl_label_service.rb` | Ninguno | — | CLEAN |
| `hitl_queue_builder.py` | Ninguno | — | CLEAN |

---

## Observación: Lenguaje Causal en Metodología

El grep de causal encontró `"determina"` en `docs/hitl_false_negative_methodology.md` (línea 97):

> *"La capacidad N la determina el coordinador del equipo revisor en función de la disponibilidad semanal."*

Este uso es administrativo/organizacional, no metodológico. La frase describe quién decide un parámetro operativo, no una relación causal entre variables del modelo. Confirmado por lectura en contexto: no hay afirmaciones del tipo "el modelo predice/causa/determina anomalías". El documento usa consistentemente "asociación", "capacidad discriminativa", "estimación correlacional sobre el proxy". No constituye un gap.

---

## Observación Pre-existente: Inconsistencia en `schema_migrations` de Platform

Según el SUMMARY de 05-02, las migraciones `20260617040336` (add_assigned_to) y `20260617040338` (seed_settings) están marcadas como `down` en `schema_migrations` pero sus efectos (columnas/datos) ya existen en la BD. Esta inconsistencia es **pre-existente** a la Fase 5 y se documentó explícitamente en el SUMMARY como bloqueante para `rails db:migrate` sin VERSION. No es un gap de esta fase. Debe resolverse antes de cualquier `db:migrate` global.

---

## Operación Diferida — Fuera de Alcance Día-1

El poblado real de la cola (`SCORING_MODE=shadow_dual` con >=14 días / >=500 filas en `anomaly_scores`) queda diferido a la fase de operación HITL. Esta condición es explícita en los PLANs (scope W5 / gate PENDING_DATA Fase 4) y no constituye un gap de esta fase. La infraestructura verificable día-1 (queries mockeadas, specs con datos sintéticos) está completamente operativa.

---

## Human Verification Required

### 1. Lenguaje causal en metodología (confirmación final)

**Test:** Leer `docs/hitl_false_negative_methodology.md` completo.
**Expected:** Sin afirmaciones causales metodológicas. El uso de "determina" (línea 97) debe leerse en contexto administrativo.
**Why human:** El grep heurístico produjo un match; la confirmación requiere lectura contextual.

---

## Summary

La Fase 5 (última del milestone) alcanzó su objetivo: la infraestructura de cola HITL y captura de etiquetas es funcional y verificable día-1 sin datos shadow reales.

- **7/7 truths verificadas** contra código real (no contra SUMMARY).
- **12/12 Python tests pasan** (`pytest tests/test_hitl_queue_builder.py -q`).
- **Rails specs**: 12 ejemplos en HitlQueueQuery spec, 16 en HitlLabelForm spec, 16 en HitlLabelService spec — todos creados y presentes en working tree (sin commit per regla 7 platform).
- **Migración aplicada** con timestamp Rails-generated (20260706193114); 6 columnas confirmadas en `structure.sql`.
- **Cross-repo consistency**: `compute_counts` Python replica exactamente la lógica Ruby (`floor/ceil + max(1,...)`).
- **SC-4**: `model_version_at_label` + `hitl_queue_source` presentes en migración, service y Python builder.
- **Anti-patterns**: cero en los 4 archivos de implementación.
- **Observación documentada**: inconsistencia pre-existente en `schema_migrations` (no gap de Fase 5).

---

*Verified: 2026-07-06*
*Verifier: Claude (so-verifier)*
