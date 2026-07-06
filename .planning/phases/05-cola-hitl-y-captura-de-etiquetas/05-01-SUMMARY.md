---
phase: 05-cola-hitl-y-captura-de-etiquetas
plan: 01
subsystem: anomaly_detection-rails-query
tags: [ruby, rails, clickhouse, hitl, shadow_new, percentile, top_factors, tdd, rspec]

requires:
  - phase: 04-shadow-dual-run
    provides: "anomaly_scores con scoring_mode='shadow_new', percentile y top_factors persistidos"
provides:
  - "AnomalyDetection::HitlQueueQuery: cola READ-only ClickHouse top-k shadow_new + muestra below-p50"
  - "Spec TDD con 12 ejemplos verdes, mocks de Analytics::ClickhouseClient.select"
affects:
  - "05-02 (label capture): consume la misma tabla anomaly_scores shadow_new"
  - "05-03 (Python builder): debe replicar lógica capacity/below_p50_pct para consistencia"
  - "05-04+ (UI HITL): HitlQueueQuery es la fuente de datos del revisor"

tech-stack:
  added: []
  patterns:
    - "TABLE = ENV.fetch('ANOMALY_DETECTION_CLICKHOUSE_TABLE', '...anomaly_scores') SIN FINAL en default"
    - "3 llamadas ClickHouse independientes: p50 resolver → top-k → below-p50"
    - "stub_const TABLE para testear el path con FINAL sin depender del ENV del desarrollador"
    - "SCORE_COLUMNS constante compartida entre top_k_sql y below_p50_sql (DRY)"

key-files:
  created:
    - "platform/packs/anomaly_detection/app/queries/anomaly_detection/hitl_queue_query.rb"
    - "platform/packs/anomaly_detection/spec/queries/anomaly_detection/hitl_queue_query_spec.rb"
  modified: []

key-decisions:
  - "TABLE default SIN FINAL — idéntico a los 12 queries existentes del pack; FINAL viene del ENV operador"
  - "stub_const TABLE para test FINAL — más robusto que depender del ENV del desarrollador en test"
  - "3 queries separadas (p50, top-k, below-p50) en lugar de 1 subquery — legibilidad y aislamiento de fallos"
  - "SCORE_COLUMNS constante para evitar duplicación entre top_k_sql y below_p50_sql"
  - "fallback p50=0.5 cuando ClickHouse falla — cola degrada gracefully sin exception"
  - "below_k_count mínimo 1 siempre (via [1, ceil].max o capacity-top_k_count)"

patterns-established:
  - "HitlQueueQuery sigue exactamente el patrón TransactionQuery: TABLE, ENV.fetch, private SQL methods, build_row"
  - "parse_json_array con rescue JSON::ParserError idéntico a TransactionDetailQuery"
  - "positive_integer helper para inputs numéricos — mismo que TransactionQuery"

duration: 4min
completed: 2026-07-06
---

# Phase 05 Plan 01: HitlQueueQuery Summary

**Cola READ-only ClickHouse top-k shadow_new ordenada por percentile DESC + muestra below-p50 con top_factors parseados, ENV-parametrizada (HITL_TOP_K, HITL_BELOW_P50_PCT), 12 specs TDD verdes**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-07-06T19:31:13Z
- **Completed:** 2026-07-06T19:34:55Z
- **Tasks:** 3 (RED + GREEN + REFACTOR)
- **Files created:** 2 (Rails, no committed per regla 7 platform)

## Accomplishments

- `AnomalyDetection::HitlQueueQuery` implementado en `packs/anomaly_detection/app/queries/anomaly_detection/` siguiendo el patrón exacto de TransactionQuery/TransactionDetailQuery
- 12 ejemplos TDD verdes cubriendo: filtro shadow_new, FINAL via ENV, ORDER BY percentile DESC, LIMIT configurable, below-p50 con p50 resuelto y rand(), hitl_queue_source tag, parse top_factors, meta hash, capacity ratio, failure graceful, edge case top_k_count=0
- Suite completa de queries del pack: 156 ejemplos, 4 fallos preexistentes (FINAL tests que requieren ENV del operador), cero regresiones

## Task Commits

Nota: Código Rails NO commitado per regla 7 platform — diff presentado al usuario para revisión.

1. **Task 1 (RED):** spec con 11+ ejemplos fallando por `uninitialized constant AnomalyDetection::HitlQueueQuery`
2. **Task 2 (GREEN):** implementación HitlQueueQuery — 11 ejemplos verdes, rubocop limpio
3. **Task 3 (REFACTOR):** edge case top_k_count=0 añadido al spec — 12 ejemplos verdes, suite sin regresiones

**Plan metadata:** commit docs(05-01) en ml-fraud-detector (ver abajo)

## Files Created (Rails — NO committed, pendientes revisión)

- `/Users/eidan/Documentation/Personal/Master/Perfil/platform/packs/anomaly_detection/app/queries/anomaly_detection/hitl_queue_query.rb`
  — `AnomalyDetection::HitlQueueQuery` con TABLE, DEFAULT_TOP_K, DEFAULT_BELOW_P50_PCT, SCORE_COLUMNS; métodos: call, fetch_p50, fetch_rows, p50_sql, top_k_sql, below_p50_sql, build_row, parse_json_array, positive_integer, clamp_pct

- `/Users/eidan/Documentation/Personal/Master/Perfil/platform/packs/anomaly_detection/spec/queries/anomaly_detection/hitl_queue_query_spec.rb`
  — 12 ejemplos RSpec con mocks de Analytics::ClickhouseClient.select, datos sintéticos shadow_new verificables día-1 sin datos reales

## Constructor y Formato de Retorno

```ruby
# Constructor
HitlQueueQuery.new(
  top_k: Integer,          # ENV HITL_TOP_K, default 100
  below_p50_pct: Float,    # ENV HITL_BELOW_P50_PCT, default 0.20
  capacity: Integer        # opcional; si presente: reparte top_k/below_k
)

# Retorno de #call
{
  queue: [
    {
      payment_id:, facility_id:, facility_name:, user_id:,
      scored_at:, payment_created_at:, amount_usd:, raw_score:,
      percentile:, risk_level:, is_anomaly:, model_version:,
      top_factors: Array,       # parseado de JSON, [] si vacío/error
      hitl_queue_source: String # "top_k" | "below_p50"
    }
  ],
  meta: {
    top_k_count: Integer,
    below_p50_count: Integer,
    p50: Float
  }
}
```

## ENV Parameters

| Variable | Default | Descripción |
|---|---|---|
| `ANOMALY_DETECTION_CLICKHOUSE_TABLE` | `pbp_productionDB_optimized.anomaly_scores` | Tabla ClickHouse (sin FINAL en default; operador añade FINAL en ENV) |
| `HITL_TOP_K` | `100` | Número de filas top-k cuando no se usa capacity |
| `HITL_BELOW_P50_PCT` | `0.20` | Fracción de la cola para muestra below-p50 |

## Lógica de Reparto capacity/below_p50_pct

```
Si capacity presente:
  top_k_count  = floor(capacity * (1 - below_p50_pct))
  below_k_count = capacity - top_k_count

Si capacity nil:
  top_k_count  = top_k (ENV HITL_TOP_K o param)
  below_k_count = max(1, ceil(top_k * below_p50_pct))
```

**Nota para 05-03 (Python builder):** El builder Python debe replicar exactamente esta lógica de reparto para consistencia con la cola Rails. Usar la misma fórmula `floor/ceil` y el mismo fallback `max(1, ...)`.

## Decisions Made

1. **TABLE default SIN FINAL** — Idéntico a los 12 queries del pack (auditado). El `FINAL` lo aporta el ENV del operador en producción/staging. El código nunca hardcodea FINAL.

2. **stub_const para test FINAL** — En lugar de depender del ENV del desarrollador (que puede no incluir FINAL en `.env` local), el ejemplo 2 del spec usa `stub_const("AnomalyDetection::HitlQueueQuery::TABLE", "...anomaly_scores FINAL")` para simular explícitamente el ENV de producción. Más robusto que `ENV` override porque sobrevive a la evaluación del constante al momento de `require`.

3. **3 queries separadas** — p50 resolver + top-k + below-p50 como 3 llamadas independientes a `Analytics::ClickhouseClient.select`. Alternativa: 1 subquery compleja. Elegido: separación para legibilidad y aislamiento de fallos (si p50 falla → fallback 0.5; no aborta toda la cola).

4. **SCORE_COLUMNS constante** — Columnas SELECT extraídas a constante para evitar duplicación entre `top_k_sql` y `below_p50_sql`. DRY sin over-engineering.

5. **fallback p50=0.5** — Cuando ClickHouse falla en la query p50, usar 0.5 como proxy del percentil mediano. La cola below-p50 filtra `percentile < 0.5` que es conservador pero funcional.

## Deviations from Plan

None — plan ejecutado exactamente como escrito. La lógica del esqueleto del plan fue correcta; el único ajuste fue agregar el edge case `top_k_count=0` al spec en Task 3 (REFACTOR) para cobertura explícita del caso límite `LIMIT 0`.

## Issues Encountered

- El entorno de test no carga `ANOMALY_DETECTION_CLICKHOUSE_TABLE` con FINAL (el `.env` local tiene el valor sin FINAL). Los 4 tests `uses FINAL` pre-existentes en la suite fallan en este entorno — confirmado como comportamiento preexistente (fallan también sin el código nuevo). Solución en el spec nuevo: `stub_const` para fiabilidad independiente del entorno.
- `OPENSEARCH_URL` requiere formato `http://` para cargar rails_helper en test — necesario como `OPENSEARCH_URL="http://127.0.0.1:9200"` al correr specs.

## Next Phase Readiness

- **05-02 (Capture labels):** La tabla `anomaly_scores` ya tiene `scoring_mode='shadow_new'` con `payment_id` como clave — listo para añadir columnas de etiqueta HITL.
- **05-03 (Python builder):** Replicar lógica capacity/below_p50_pct (documentada arriba). La query de ingesta Python debe filtrar `scoring_mode='shadow_new'` e incluir `top_factors`.
- **Datos reales:** HitlQueueQuery verifica desde día 1 con mocks. En producción, requiere shadow_dual activo ≥2 semanas (condición de 04-02).

---
*Phase: 05-cola-hitl-y-captura-de-etiquetas*
*Completed: 2026-07-06*
