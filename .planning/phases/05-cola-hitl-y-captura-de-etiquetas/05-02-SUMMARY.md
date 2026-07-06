---
phase: 05-cola-hitl-y-captura-de-etiquetas
plan: "02"
subsystem: hitl
tags: [rails, activerecord, migration, activemodel, tdd, triage, hitl]

# Dependency graph
requires:
  - phase: 05-01
    provides: HitlQueueQuery (cola HITL top-k / below-p50) — lógica de selección de items a revisar
  - phase: 03-02
    provides: TriageTransitionService, TriageAction, AlertManager — infraestructura de triage existente
provides:
  - Migración AddHitlFieldsToAnomalyDetectionTriageActions (timestamp 20260706193114, 6 columnas nullable)
  - HitlLabelForm (vocabulario 4 categorías + validación de hitl_queue_source + transición heredada)
  - HitlLabelService (enriquece TriageAction status_change con procedencia ML completa, transacción con rollback)
  - alias_attribute :reviewer_id en TriageAction (performed_by_id — sin columna duplicada)
affects:
  - 05-03 (endpoint/UI de captura POST-shadow — diferido)
  - cualquier fase que lea labels HITL desde TriageAction

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "HitlLabelForm hereda TriageTransitionForm::VALID_TRANSITIONS via referencia directa (no duplica constante)"
    - "HitlLabelService envuelve TriageTransitionService en una transacción y captura el retorno directo (auditado)"
    - "reviewer_saw_factors: TINYINT(1) almacenado como 0/1 — Rails lo lee como boolean (ActiveRecord mapping)"
    - "Defensive fallback en HitlLabelService: verifica que el retorno sea status_change antes de enriquecer"

key-files:
  created:
    - platform/packs/anomaly_detection/db/migrate/20260706193114_add_hitl_fields_to_anomaly_detection_triage_actions.rb
    - platform/packs/anomaly_detection/app/forms/anomaly_detection/hitl_label_form.rb
    - platform/packs/anomaly_detection/app/services/anomaly_detection/hitl_label_service.rb
    - platform/packs/anomaly_detection/spec/forms/anomaly_detection/hitl_label_form_spec.rb
    - platform/packs/anomaly_detection/spec/services/anomaly_detection/hitl_label_service_spec.rb
  modified:
    - platform/packs/anomaly_detection/app/models/anomaly_detection/triage_action.rb
    - platform/db/structure.sql

key-decisions:
  - "Timestamp de migración 20260706193114 generado por Rails (bundle exec rails g migration), no hand-picked. Movido de db/migrate/ raíz al pack conservando timestamp."
  - "reviewer_id NO es columna nueva — es alias_attribute de performed_by_id en TriageAction. Sin duplicación."
  - "reviewed_at se fija en HitlLabelService con Time.current, no en el form ni con Time.now (regla timezone)."
  - "reviewer_saw_factors es TINYINT(1) NOT NULL DEFAULT 0. Rails lo mapea a boolean en lectura; el service escribe 0/1 vía update!."
  - "TriageTransitionService.call retorna directamente la TriageAction status_change (auditado: última expresión del bloque transaction es create_status_action). HitlLabelService captura ese retorno con fallback defensivo si cambia en el futuro."
  - "Endpoint/UI de captura diferido POST-shadow (scope W5 / gate PENDING_DATA Fase 4). HITL-02 satisfecho a nivel infra+service verificable día-1 con FactoryBot."
  - "OPENSEARCH_URL debe tener esquema http:// para que rails_helper.rb pueda parsear el URI (bug pre-existente de entorno). Los specs del pack se corren con OPENSEARCH_URL=http://127.0.0.1:9200."

patterns-established:
  - "HitlLabelForm: ActiveModel::Model con VALID_REVIEWER_LABELS frozen + VALID_QUEUE_SOURCES + valid_transition via VALID_TRANSITIONS del TriageTransitionForm"
  - "HitlLabelService < ApplicationService: initialize(form:, score_row:), retorno nil si form inválido, todo en transaction"
  - "Migración pack: idioma execute(SQL.squish) con guard table_exists?, up/down explícitos — mismo patrón que CreateAnomalyDetectionTriageActions"

# Metrics
duration: 8min
completed: 2026-07-06
---

# Phase 05 Plan 02: HitlLabelService — Captura de Etiquetas HITL con Procedencia ML Summary

**Migración nullable de 6 columnas HITL (timestamp Rails 20260706193114) + HitlLabelForm (vocabulario tesis 4 categorías) + HitlLabelService (envuelve TriageTransitionService, enriquece TriageAction con score/modelo/factores/cola, transacción con rollback) — 32 specs verdes, sin controller (diferido POST-shadow)**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-06T19:31:03Z
- **Completed:** 2026-07-06T19:38:59Z
- **Tasks:** 3 de 3
- **Files modificados/creados:** 7 (Rails) + 1 (structure.sql)

## Accomplishments

- Migración `AddHitlFieldsToAnomalyDetectionTriageActions` generada con timestamp Rails (20260706193114), movida al pack, aplicada y reversibilidad verificada (rollback + re-migrate exitosos).
- `HitlLabelForm` con `VALID_REVIEWER_LABELS = %w[sospecha_fraude anomalia_operativa falso_positivo indeterminado]`, validación de `hitl_queue_source` (allow_nil), hereda `VALID_TRANSITIONS` de `TriageTransitionForm`.
- `HitlLabelService` que envuelve `TriageTransitionService.call`, captura la `TriageAction` status_change (con fallback defensivo), y la enriquece con los 6 campos de procedencia en la misma transacción. Rollback total ante form inválido o fallo de enrich.
- `alias_attribute :reviewer_id, :performed_by_id` en `TriageAction` — sin columna duplicada.
- 32 specs TDD verdes (16 form + 16 service). 78 ejemplos totales (forms + services) sin regresiones. Rubocop limpio.

## Task Commits

Codigo Rails sin commit (regla 7 platform — diff presentado para revisión del usuario). Solo el commit de metadata en ml-fraud-detector.

1. **Task 1: Migración + modelo** — Sin commit Rails. Archivos: `20260706193114_add_hitl_fields_to_anomaly_detection_triage_actions.rb`, `triage_action.rb` (alias_attribute), `structure.sql`.
2. **Task 2: HitlLabelForm TDD** — Sin commit Rails. Archivos: `hitl_label_form.rb`, `hitl_label_form_spec.rb`.
3. **Task 3: HitlLabelService TDD** — Sin commit Rails. Archivos: `hitl_label_service.rb`, `hitl_label_service_spec.rb`.

**Plan metadata:** [hash del commit de metadata en ml-fraud-detector]

## Files Created/Modified (Rails — sin commit, pendientes revisión)

| Archivo | Operación | Descripción |
|---------|-----------|-------------|
| `platform/packs/anomaly_detection/db/migrate/20260706193114_add_hitl_fields_to_anomaly_detection_triage_actions.rb` | CREADO | Migración con 6 columnas HITL nullable + guard table_exists? + up/down explícitos |
| `platform/packs/anomaly_detection/app/forms/anomaly_detection/hitl_label_form.rb` | CREADO | HitlLabelForm con VALID_REVIEWER_LABELS, VALID_QUEUE_SOURCES, valid_transition |
| `platform/packs/anomaly_detection/app/services/anomaly_detection/hitl_label_service.rb` | CREADO | HitlLabelService que envuelve TriageTransitionService y enriquece la TriageAction |
| `platform/packs/anomaly_detection/spec/forms/anomaly_detection/hitl_label_form_spec.rb` | CREADO | 16 ejemplos TDD (adversarial primero): vocabulario, queue_source, presence, casting |
| `platform/packs/anomaly_detection/spec/services/anomaly_detection/hitl_label_service_spec.rb` | CREADO | 16 ejemplos TDD (adversarial primero): rollback, status_change targeting, Timecop, reviewer_id |
| `platform/packs/anomaly_detection/app/models/anomaly_detection/triage_action.rb` | MODIFICADO | Añadido `alias_attribute :reviewer_id, :performed_by_id` |
| `platform/db/structure.sql` | MODIFICADO | Actualizado por db:migrate (columnas HITL en anomaly_detection_triage_actions) |

## Columnas de la Migración (6 campos HITL-02)

| Columna | Tipo | Restricción | Propósito |
|---------|------|-------------|-----------|
| `reviewer_label` | VARCHAR(100) | NULL | Categoría asignada por el revisor (vocabulario 4 valores) |
| `reviewed_at` | DATETIME(6) | NULL | Timestamp de la revisión — fijado con Time.current en el service |
| `score_at_label_time` | DECIMAL(5,4) | NULL | Score de anomalía que vio el revisor (de anomaly_scores) |
| `model_version_at_label` | VARCHAR(100) | NULL | Versión del modelo activo cuando se etiquetó |
| `reviewer_saw_factors` | TINYINT(1) | NOT NULL DEFAULT 0 | Si el revisor visualizó los factores explicativos (0=no, 1=sí) |
| `hitl_queue_source` | VARCHAR(50) | NULL | Procedencia de la cola: 'top_k' o 'below_p50' |

**Nota:** `reviewer_id` (FK al usuario revisor) NO es columna nueva — es `alias_attribute :reviewer_id, :performed_by_id` en el modelo.

## VALID_REVIEWER_LABELS (alineado con tesis y hitl_ingest_labels.py)

```ruby
VALID_REVIEWER_LABELS = %w[sospecha_fraude anomalia_operativa falso_positivo indeterminado].freeze
```

Excluye `_correccion_` (sentinel solo válido en el script de ingest Python).

## Firma de HitlLabelService

```ruby
HitlLabelService.call(form: hitl_label_form, score_row: hash)
# Returns: AnomalyDetection::TriageAction (status_change, enriquecida) o nil si form inválido
```

`score_row` espera `{ "raw_score" => Float, "model_version" => String }` (subset de fila de anomaly_scores).

## Decisiones Made

- **Timestamp Rails (no hand-picked):** `bundle exec rails g migration AddHitlFieldsToAnomalyDetectionTriageActions` generó `20260706193114`. Archivo movido de `db/migrate/` raíz a `packs/anomaly_detection/db/migrate/` conservando el timestamp. Comando `bundle exec rails db:migrate` fallaba por conflicto pre-existente en migración `20260617040336` (assigned_to_id ya existía) — se ejecutó `db:migrate:up VERSION=20260706193114` para aplicar solo la nueva migración.
- **reviewer_id sin columna nueva:** `alias_attribute :reviewer_id, :performed_by_id` evita duplicar la FK. El form/service usa `performed_by`; la API/UI puede exponer `reviewer_id`.
- **reviewer_saw_factors como TINYINT(1):** Rails lo mapea a boolean en lectura (TrueClass/FalseClass). Los specs verifican con `be(true)`/`be(false)` post-reload, no con `eq(1)`/`eq(0)`.
- **Scope W5 preservado:** Sin controller ni endpoint HTTP. `HitlQueueQuery → HitlLabelService` queda diferido a la fase de operación HITL POST-shadow.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Conflicto pre-existente en db:migrate bloqueaba la aplicación de la nueva migración**

- **Found during:** Task 1 (migración)
- **Issue:** `bundle exec rails db:migrate` fallaba con `Duplicate column name 'assigned_to_id'` porque la migración `20260617040336_add_assigned_to_to_anomaly_detection_alerts.rb` estaba `down` en schema_migrations pero la columna ya existía en la BD (inconsistencia pre-existente no introducida por este plan).
- **Fix:** Se ejecutó `bundle exec rails db:migrate:up VERSION=20260706193114` para aplicar únicamente la migración HITL sin intentar re-aplicar las anteriores en conflicto.
- **Files modified:** Ninguno adicional — la migración se aplicó, structure.sql actualizado.
- **Verification:** `rails runner "puts AnomalyDetection::TriageAction.column_names.include?('reviewer_label')"` → `true`.

**2. [Rule 1 - Bug] reviewer_saw_factors: expectativas de spec con eq(1)/eq(0) incorrectas (TINYINT(1) → boolean)**

- **Found during:** Task 3 (HitlLabelService spec — GREEN phase)
- **Issue:** Los specs del service esperaban `eq(1)` y `eq(0)` pero Rails/MySQL mapea TINYINT(1) a TrueClass/FalseClass en lectura. 2 ejemplos fallaban.
- **Fix:** Se corrigieron las expectativas a `be(true)` / `be(false)` con `.reload` explícito. El service sigue escribiendo `0`/`1` vía `update!` (correcto para el tipo de columna).
- **Files modified:** `spec/services/anomaly_detection/hitl_label_service_spec.rb`
- **Verification:** 16 ejemplos del service spec verdes.

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug en test)
**Impact:** Ambas correcciones necesarias para correcta operación. Sin scope creep.

## Issues Encountered

- `OPENSEARCH_URL=127.0.0.1:9200` (sin esquema) en el entorno local provoca `URI::InvalidURIError` al cargar `rails_helper.rb`. Workaround: prefixar `OPENSEARCH_URL=http://127.0.0.1:9200` al invocar rspec. Issue pre-existente, no introducido por este plan.

## Scope POST-shadow diferido (W5)

El endpoint/UI de captura que conecta `HitlQueueQuery` (05-01) → `HitlLabelService` queda **diferido** a la fase de operación HITL. Razón: el gate de Fase 4 está en `PENDING_DATA` (requiere ≥2 semanas de shadow data). No existe volumen real que revisar todavía.

HITL-02 queda satisfecho a nivel **infra + service**:
- La cola puede generarse vía `HitlQueueQuery` (05-01)
- Las etiquetas pueden capturarse vía `HitlLabelService.call(form:, score_row:)` día-1 usando FactoryBot
- La superficie HTTP (controller, endpoint REST/GraphQL, UI) se planificará cuando el shadow acumule datos suficientes

## Next Phase Readiness

- **05-03 (si existe):** Puede construir sobre `HitlLabelService.call` directamente. La firma está estable.
- **Endpoint/UI POST-shadow:** Necesita definir el protocolo de autenticación/autorización del revisor antes de implementar.
- **Preocupación abierta:** La migración `20260617040336` (add_assigned_to) y `20260617040338` (seed_settings) están `down` en schema_migrations pero sus efectos ya existen en la BD. Requiere limpieza manual de schema_migrations o re-aplicación controlada — no bloqueante para HITL-02 pero debe resolverse antes de `rails db:migrate` sin VERSION.

---
*Phase: 05-cola-hitl-y-captura-de-etiquetas*
*Completed: 2026-07-06*
