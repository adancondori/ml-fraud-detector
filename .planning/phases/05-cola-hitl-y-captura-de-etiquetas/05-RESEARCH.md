# Phase 5: Cola HITL y Captura de Etiquetas — Research

**Researched:** 2026-07-06
**Domain:** Human-in-the-Loop review queue, label schema, defensive sampling — Rails/MySQL + Python/ClickHouse
**Confidence:** HIGH — derivado íntegramente de auditoría directa del código fuente. Sin especulación de training data.

---

## Summary

La Fase 5 tiene un contexto muy específico: el gate SHAD-03 de Fase 4 está PENDING_DATA (no hay ≥2 semanas de shadow real). Por lo tanto la fase debe entregar la **infraestructura** de la cola HITL, el schema de etiquetas y la lógica de muestreo — completamente probados con datos sintéticos/pequeños desde día-1 — y el **poblado real** de la cola queda diferido a cuando el shadow acumule datos.

La auditoría revela tres hallazgos críticos que determinan las decisiones de arquitectura:

**Hallazgo 1 — `top_factors` YA está persistido en ClickHouse.** La columna `top_factors String` existe en el DDL base (`02_anomaly_scores.sql`), el batch scorer la popula con `json.dumps(factors)` (lista de `FactorItem` con `feature`, `value`, `z_score`, `direction`), y la query `TransactionDetailQuery` ya la lee con `parse_json_array`. No hay nada que derivar; el campo ya existe en cada fila de `anomaly_scores` con `model_version='shadow_new'`.

**Hallazgo 2 — Los 6 campos HITL-02 NO existen en `TriageAction`.** El schema actual de `anomaly_detection_triage_actions` tiene: `id`, `alert_id`, `performed_by_id`, `action_type`, `from_status`, `to_status`, `note`, `metadata` (JSON), `created_at`, `updated_at`. Faltan los 6 campos de procedencia HITL: `reviewer_label`, `reviewed_at`, `reviewer_id` (ya capturado como `performed_by_id` pero sin alias explícito), `score_at_label_time`, `model_version_at_label`, `reviewer_saw_factors`. La estrategia es: **una sola migración Rails** que añade los 6 campos a `anomaly_detection_triage_actions`, preservando retrocompatibilidad.

**Hallazgo 3 — Los scripts HITL existentes son offline/parquet, NO live.** `scripts/hitl_export_alerts.py` y `scripts/hitl_ingest_labels.py` operan sobre parquets del test set histórico — pipeline académico de tesis. La Fase 5 construye la infraestructura **operativa** sobre `anomaly_scores` vivo (ClickHouse `model_version='shadow_new'`) y `Alert`/`TriageAction` (MySQL Rails). Son dos sistemas distintos con propósitos distintos que coexisten sin conflicto.

**Primary recommendation:** Extender `TriageAction` con los 6 campos HITL-02 vía migración Rails; construir una nueva query ClickHouse `HitlQueueQuery` que filtra `scoring_mode='shadow_new'` y sirve el top-k + muestra defensiva; crear `HitlLabelService` que wrap el `TriageTransitionService` capturando la procedencia completa. Todo verificable en día-1 con datos sintéticos.

---

## Standard Stack

### Core (ya presente, no instalar nada)

| Librería / Framework | Versión | Propósito | Por qué estándar |
|----------------------|---------|-----------|-----------------|
| Rails 6.1 + ActiveRecord | 6.1.7.9 | Migración MySQL, modelo `TriageAction`, servicio de etiquetado | Stack del pack `anomaly_detection` ya existente |
| MySQL 8.0 | 8.0.42 | Tabla `anomaly_detection_triage_actions` — labels con procedencia | Ya en uso para `Alert` y `TriageAction` |
| `clickhouse-connect` (Python) | instalado en venv | Query `anomaly_scores` para construir la cola top-k | Ya usado en `BatchScorer`, `TransactionDetailQuery` |
| `pandas` | instalado | Lógica de muestreo defensivo (p50 percentile, sample_below) | Ya en uso en `shadow_monitor.py` |
| `pydantic` | instalado | Schema de exportación/ingestión HITL | Ya en `scorer/schemas.py` |
| RSpec + FactoryBot | existente | Tests de migración, servicio, query | Ya en uso en el pack |

### Supporting

| Librería | Propósito | Cuándo usar |
|----------|-----------|------------|
| `Analytics::ClickhouseClient` (Rails) | SELECT sobre `anomaly_scores` desde Rails | `HitlQueueQuery` hereda el patrón de `TransactionQuery` |
| `ApplicationService` (Rails) | Service object para `HitlLabelService` | Mismo patrón que `TriageTransitionService` |
| `ActiveModel::Model` | Form object `HitlLabelForm` | Mismo patrón que `TriageTransitionForm` |

### Alternativas consideradas

| En vez de | Podría usar | Tradeoff |
|-----------|-------------|----------|
| Extender `TriageAction` con 6 columnas | Nueva tabla `hitl_labels` separada | Columnas adicionales en tabla existente evita JOIN; retrocompatible vía `allow_nil` / defaults NULL. Nueva tabla sería más limpia pero rompe el flujo de triage existente sin ganancia real |
| Cola en ClickHouse (vista/tabla) | Cola en MySQL (`Alert` + query) | ClickHouse es append-only y adecuado para consulta analítica; MySQL es mejor para mutable state (assignment, status). La cola vive en ClickHouse (READ-ONLY via query), las etiquetas viven en MySQL (WRITE) — separación de responsabilidades clara |
| Scripts Python para muestreo defensivo | Lógica en Ruby | Python ya tiene pandas + numpy instalados y es el home del análisis. El script exporta CSV/JSON parametrizado; Rails lo consume. Mantiene separación scorer↔platform |

---

## Architecture Patterns

### Recommended Project Structure (additions for Phase 5)

```
platform/packs/anomaly_detection/
├── app/
│   ├── forms/anomaly_detection/
│   │   └── hitl_label_form.rb          # (NEW) form con los 6 campos HITL-02
│   ├── queries/anomaly_detection/
│   │   └── hitl_queue_query.rb         # (NEW) top-k + muestra defensiva desde anomaly_scores
│   ├── services/anomaly_detection/
│   │   └── hitl_label_service.rb       # (NEW) wrap TriageTransitionService + captura procedencia
│   └── controllers/anomaly_detection/
│       └── hitl_queue_controller.rb    # (NEW) endpoint GET /hitl_queue + POST /hitl_labels
├── db/migrate/
│   └── YYYYMMDDHHMMSS_add_hitl_fields_to_triage_actions.rb  # (NEW) 6 columnas HITL-02
└── spec/
    ├── queries/anomaly_detection/hitl_queue_query_spec.rb
    ├── forms/anomaly_detection/hitl_label_form_spec.rb
    └── services/anomaly_detection/hitl_label_service_spec.rb

ml-fraud-detector/
└── scripts/
    └── hitl_queue_builder.py           # (NEW) exporta cola top-k + muestra defensiva desde CH local
                                        # parametrizado por --top-k K --below-p50-pct 0.20 --capacity N
```

### Pattern 1: HitlQueueQuery — ClickHouse top-k + muestra defensiva

**Qué es:** Query sobre `anomaly_scores` que filtra `scoring_mode='shadow_new'` (frame-v1), ordena por `percentile DESC`, toma los top-k, y aparte toma una muestra aleatoria de filas con `percentile < p50` para cubrir falsos negativos.

**Cuándo usar:** Para construir la cola de revisión en día-1 (con datos sintéticos) y en producción.

**Estructura:**

```ruby
# Source: auditoría de TransactionQuery + TransactionDetailQuery
module AnomalyDetection
  class HitlQueueQuery
    TABLE = ENV.fetch("ANOMALY_DETECTION_CLICKHOUSE_TABLE",
                      "pbp_productionDB_optimized.anomaly_scores")

    # params: top_k (default 100), below_p50_pct (default 0.20), capacity (default nil)
    # Si capacity es nil: top_k_count = top_k, below_count = (top_k * below_p50_pct).ceil
    # Si capacity es N: top_k_count = (N * 0.80).floor, below_count = (N * 0.20).ceil
    def call
      top_rows = fetch_top_k
      below_rows = fetch_below_p50_sample
      { queue: top_rows + below_rows,
        meta: { top_k_count: top_rows.size, below_p50_count: below_rows.size } }
    end

    private

    def top_k_sql
      <<~SQL
        SELECT payment_id, facility_id, user_id, scored_at, percentile,
               risk_level, raw_score, amount_usd, top_factors, model_version
        FROM #{TABLE}
        WHERE scoring_mode = 'shadow_new'
          AND is_anomaly = 1
        ORDER BY percentile DESC
        LIMIT #{@top_k_count}
      SQL
    end

    def below_p50_sql(p50_value)
      <<~SQL
        SELECT payment_id, facility_id, user_id, scored_at, percentile,
               risk_level, raw_score, amount_usd, top_factors, model_version
        FROM #{TABLE}
        WHERE scoring_mode = 'shadow_new'
          AND percentile < #{p50_value}
        ORDER BY rand()         -- muestreo pseudoaleatorio; ClickHouse soporta rand()
        LIMIT #{@below_k_count}
      SQL
    end
  end
end
```

**Importante:** El `p50` de `percentile` se resuelve con `SELECT quantile(0.5)(percentile)` sobre `scoring_mode='shadow_new'` en una query previa. En ClickHouse esto es una operación de agregación O(log n) con índice de merge; no es costoso.

### Pattern 2: Migración Rails — 6 columnas HITL-02 en `triage_actions`

**Qué es:** Una migración que añade las 6 columnas de procedencia al schema existente de `anomaly_detection_triage_actions`. Todas admiten NULL para retrocompatibilidad.

```ruby
# Source: auditoría de CreateAnomalyDetectionTriageActions (20260617040332)
class AddHitlFieldsToAnomalyDetectionTriageActions < ActiveRecord::Migration[6.1]
  def up
    return unless table_exists?(:anomaly_detection_triage_actions)

    execute(<<-SQL.squish)
      ALTER TABLE anomaly_detection_triage_actions
        ADD COLUMN reviewer_label       VARCHAR(100) NULL,
        ADD COLUMN reviewed_at          DATETIME(6) NULL,
        ADD COLUMN score_at_label_time  DECIMAL(5,4) NULL,
        ADD COLUMN model_version_at_label VARCHAR(100) NULL,
        ADD COLUMN reviewer_saw_factors TINYINT(1) DEFAULT 0,
        ADD COLUMN hitl_queue_source    VARCHAR(50) NULL
      -- hitl_queue_source: 'top_k' | 'below_p50' | NULL (non-HITL triage)
    SQL
  end

  def down
    # ...
  end
end
```

**Nota:** `reviewer_id` ya está capturado por `performed_by_id` (FK a `users`). No se duplica. `reviewed_at` es distinto de `created_at` de la acción porque puede haber lag entre que la Cola asigna y el revisor completa la revisión.

### Pattern 3: HitlLabelService — captura de procedencia completa

**Qué es:** Service que extiende `TriageTransitionService` capturando los 6 campos de procedencia antes de llamar al service existente.

```ruby
module AnomalyDetection
  class HitlLabelService < ApplicationService
    def initialize(form:, score_row:)
      @form = form           # HitlLabelForm — incluye los campos HITL + new_status
      @score_row = score_row # hash de la fila de anomaly_scores al momento de etiquetar
    end

    def call
      ActiveRecord::Base.transaction do
        triage_form = build_triage_form
        return unless triage_form.valid?

        result = TriageTransitionService.call(form: triage_form)
        enrich_triage_action(result)
        result
      end
    end

    private

    def enrich_triage_action(triage_action)
      triage_action.update!(
        reviewer_label:        @form.reviewer_label,
        reviewed_at:           Time.current,
        score_at_label_time:   @score_row["raw_score"],
        model_version_at_label: @score_row["model_version"],
        reviewer_saw_factors:  @form.reviewer_saw_factors ? 1 : 0,
        hitl_queue_source:     @form.hitl_queue_source
      )
    end
  end
end
```

### Pattern 4: `hitl_queue_builder.py` — exportación parametrizada (scorer/Python)

**Qué es:** Script Python que construye la cola desde ClickHouse local (WRITE target), parametrizado por `--top-k`, `--below-p50-pct` y `--capacity`. Produce CSV o JSON consultable sin depender de datos shadow reales. Con datos sintéticos, `--capacity 10` es suficiente para verificar el contrato.

```python
# Source: patrón de scripts/shadow_monitor.py (usa _build_ch_client)
def build_hitl_queue(ch_client, top_k: int, below_p50_pct: float, capacity: int | None):
    """Retorna (df_top_k, df_below_p50) desde scoring_mode='shadow_new'."""
    if capacity:
        top_k_count = int(capacity * (1 - below_p50_pct))
        below_k_count = capacity - top_k_count
    else:
        top_k_count = top_k
        below_k_count = max(1, int(top_k * below_p50_pct))

    # 1. Resolver p50
    p50_result = ch_client.query(P50_SQL)
    p50 = p50_result.result_rows[0][0] if p50_result.result_rows else 50.0

    # 2. Top-k
    df_top = ch_client.query_df(TOP_K_SQL.format(top_k_count=top_k_count))

    # 3. Muestra below-p50
    df_below = ch_client.query_df(BELOW_P50_SQL.format(p50=p50, below_k_count=below_k_count))

    df_top["hitl_queue_source"] = "top_k"
    df_below["hitl_queue_source"] = "below_p50"
    return pd.concat([df_top, df_below], ignore_index=True)
```

### Anti-Patterns to Avoid

- **No persistir `top_factors` como columna separada en ClickHouse:** Ya existe como String JSON en `anomaly_scores`. Duplicarla sería redundante y desincronizada.
- **No hardcodear `top_k=100` en el código:** El contexto indica que la capacidad del equipo es desconocida. El parámetro debe ser configurable via ENV var o argumento.
- **No usar `ORDER BY rand()` en tablas grandes sin LIMIT:** En ClickHouse `rand()` es costoso sin límite; siempre aplicar `LIMIT` antes o después del ORDER. Para muestras grandes usar `sample(N)` nativo de ClickHouse MergeTree.
- **No confundir `scoring_mode` con `model_version`:** `scoring_mode='shadow_new'` es la fila del challenger frame-v1; `model_version` identifica el artefacto específico. La cola filtra por `scoring_mode='shadow_new'` (campo de comportamiento de scoring), no por `model_version` directamente.
- **No usar la columna `is_anomaly=1` como único filtro para la cola top-k:** `is_anomaly` usa el umbral calibrado del champion. Para la cola del challenger (frame-v1), ordenar por `percentile DESC` es más correcto — permite revisar incluso rows con `is_anomaly=0` que tengan alto percentile (zona gris).

---

## Don't Hand-Roll

| Problema | No construir | Usar en cambio | Por qué |
|----------|-------------|----------------|---------|
| Muestreo aleatorio en ClickHouse | SQL `ORDER BY rand()` sobre toda la tabla | `sample(N)` de MergeTree + `LIMIT` | `sample()` usa el índice de granularidad; es mucho más eficiente en tablas grandes |
| Validación del formulario de etiqueta | Validación manual en el controller | `ActiveModel::Model` + `validates` | Mismo patrón que `TriageTransitionForm`; reutiliza el error handling del controller existente |
| Timestamps de "momento de etiquetado" | Capturar `reviewed_at` manualmente en el controller | `Time.current` en el service, no en el form | Evita timezone bugs; el CLAUDE.md del platform prohíbe `Time.now` explícitamente |
| Schema de exportación CSV para revisores | Columnas ad-hoc | Schema fijo derivado de `top_factors` (FactorItem: feature, value, z_score, direction) | `FactorItem` ya es el contrato del scorer; reusar garantiza que el revisor ve los factores correctos |

**Key insight:** La superficie de triage existente (Alert → TriageAction → TriageTransitionService → TriageTransitionForm) está bien construida y cubre el workflow completo. La Fase 5 lo **extiende** con 6 campos adicionales y una nueva query de cola; no lo reemplaza.

---

## Common Pitfalls

### Pitfall 1: Confundir los dos sistemas HITL

**Qué falla:** Intentar unificar `scripts/hitl_export_alerts.py` (pipeline offline/parquet de tesis) con la cola operativa sobre ClickHouse live.
**Por qué ocurre:** Ambos se llaman "HITL" y ambos exportan alertas, pero tienen propósitos completamente distintos.
**Cómo evitar:** Los scripts existentes (`hitl_export_alerts.py`, `hitl_ingest_labels.py`) operan sobre `data/processed/test_features_enriched.parquet` — son para la evaluación académica del capítulo de resultados de tesis. El nuevo `hitl_queue_builder.py` opera sobre `anomaly_scores` vivo con `scoring_mode='shadow_new'`. Mantenerlos separados.
**Warning signs:** Si alguien propone modificar `hitl_export_alerts.py` para consumir ClickHouse, está unificando lo que debe estar separado.

### Pitfall 2: `reviewer_id` duplicado — `performed_by_id` ya existe

**Qué falla:** Añadir una columna `reviewer_id` redundante a `TriageAction` cuando ya existe `performed_by_id` (FK a `users`).
**Por qué ocurre:** El spec HITL-02 lista `reviewer_id` como campo requerido, y se da por sentado que no existe.
**Cómo evitar:** `performed_by_id` ES el `reviewer_id`. En la aplicación, exponer como `reviewer_id` en la API/JSON pero no duplicar la columna. Si se necesita distinguir "quién inició el triage" de "quién etiquetó en la cola HITL", usar `hitl_queue_source` para marcar la procedencia.

### Pitfall 3: Cola vacía en día-1 (PENDING_DATA)

**Qué falla:** Tests que fallan porque `anomaly_scores` no tiene filas con `scoring_mode='shadow_new'` en el entorno local.
**Por qué ocurre:** El shadow scoring real está PENDING_DATA.
**Cómo evitar:** La verificación día-1 usa datos sintéticos: insertar 10-20 filas en ClickHouse local con `scoring_mode='shadow_new'` y percentiles variados. El plan debe incluir un fixture SQL de datos sintéticos. El test de `HitlQueueQuery` debe mockear `Analytics::ClickhouseClient.select` igual que hacen los specs de `TransactionQuery` y `DashboardKpiQuery`.
**Warning signs:** Tests que requieren datos reales de ClickHouse. Todos los tests de queries ClickHouse en el pack usan stubs.

### Pitfall 4: Hardcodear la razón 80/20 top-k vs. below-p50

**Qué falla:** Ratio fijo 80/20 codificado en el servicio.
**Por qué ocurre:** El context indica que la capacidad del equipo es desconocida.
**Cómo evitar:** `below_p50_pct` es un parámetro ENV `HITL_BELOW_P50_PCT` (default 0.20). El servicio acepta `capacity:` como argumento y calcula `top_k_count = (capacity * (1 - below_p50_pct)).floor`. Si `capacity` es nil, usa `top_k` como valor absoluto.

### Pitfall 5: `top_factors` en ClickHouse dual-run tiene `{}` para shadow_old

**Qué falla:** Para filas `scoring_mode='shadow_old'` en el dual-run, `features_json` es `'{}'` (vacío) — ver `_build_row` en `batch/scorer.py` línea 604. `top_factors` sin embargo SÍ se popula correctamente porque viene de `result.factors`.
**Por qué ocurre:** Limitación de diseño documentada en el código: "Feature vectors are internal to each scorer; they are not re-extracted here to avoid duplicating calculator logic."
**Cómo evitar:** La cola HITL filtra `scoring_mode='shadow_new'` (challenger frame-v1), donde `top_factors` está garantizado no-vacío. No filtrar por `shadow_old`.

### Pitfall 6: Timestamp migration — Rails convención

**Qué falla:** El timestamp del archivo de migración es generado manualmente con valor round (120000, etc.) que colisiona con otras migraciones.
**Por qué ocurre:** El CLAUDE.md del platform es explícito: "NEVER hand-pick the timestamp... use `bundle exec rails g migration`".
**Cómo evitar:** Generar el archivo con `bundle exec rails g migration AddHitlFieldsToAnomalyDetectionTriageActions` desde el directorio del pack o con `date -u +%Y%m%d%H%M%S`.

---

## Code Examples

### 1. Query p50 en ClickHouse para muestreo defensivo

```sql
-- Source: auditoría de score_distribution_query.rb (patrón quantile)
SELECT quantile(0.5)(percentile) AS p50
FROM pbp_productionDB_optimized.anomaly_scores
WHERE scoring_mode = 'shadow_new'
```

ClickHouse `quantile()` es una función de agregado aproximado con error < 1%. Para exactitud exacta usar `quantileExact()` — más lento pero correcto para p50 en datasets pequeños.

### 2. Cola top-k en ClickHouse (Rails, sigue patrón TransactionQuery)

```ruby
# Source: auditoría de transaction_query.rb + transaction_detail_query.rb
def top_k_sql
  <<~SQL
    SELECT
      payment_id,
      facility_id,
      user_id,
      scored_at,
      amount_usd,
      raw_score,
      percentile,
      risk_level,
      is_anomaly,
      model_version,
      top_factors
    FROM #{TABLE}
    WHERE scoring_mode = 'shadow_new'
      AND percentile >= (
        SELECT quantile(0.5)(percentile)
        FROM #{TABLE}
        WHERE scoring_mode = 'shadow_new'
      )
    ORDER BY percentile DESC
    LIMIT #{@top_k_count}
  SQL
end
```

**Alternativa más eficiente:** Resolver p50 en query separada (2 queries) para evitar el subquery en cada fila. `TransactionQuery` ya sigue el patrón de count_sql + data_sql separados.

### 3. Muestreo below-p50 con `rand()` limitado

```sql
-- Source: ClickHouse docs — rand() es función escalar, ORDER BY rand() funciona con LIMIT
SELECT
  payment_id, facility_id, user_id, scored_at, amount_usd,
  raw_score, percentile, risk_level, is_anomaly, model_version, top_factors
FROM pbp_productionDB_optimized.anomaly_scores
WHERE scoring_mode = 'shadow_new'
  AND percentile < {p50_value:Float32}
ORDER BY rand()
LIMIT {below_k_count:UInt32}
```

Para tablas grandes (>1M filas), reemplazar `ORDER BY rand()` con:
```sql
WHERE scoring_mode = 'shadow_new'
  AND percentile < {p50_value:Float32}
  AND cityHash64(payment_id) % {sample_modulo:UInt64} = 0
LIMIT {below_k_count:UInt32}
```
`cityHash64(payment_id) % N = 0` es determinístico y eficiente para N grande.

### 4. Migración Rails — patrón del pack (MySQL raw DDL)

```ruby
# Source: auditoría de 20260617040332_create_anomaly_detection_triage_actions.rb
class AddHitlFieldsToAnomalyDetectionTriageActions < ActiveRecord::Migration[6.1]
  def up
    return unless table_exists?(:anomaly_detection_triage_actions)

    execute(<<-SQL.squish)
      ALTER TABLE anomaly_detection_triage_actions
        ADD COLUMN IF NOT EXISTS reviewer_label          VARCHAR(100) NULL,
        ADD COLUMN IF NOT EXISTS reviewed_at             DATETIME(6) NULL,
        ADD COLUMN IF NOT EXISTS score_at_label_time     DECIMAL(5,4) NULL,
        ADD COLUMN IF NOT EXISTS model_version_at_label  VARCHAR(100) NULL,
        ADD COLUMN IF NOT EXISTS reviewer_saw_factors    TINYINT(1) NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS hitl_queue_source       VARCHAR(50) NULL
    SQL
  end

  def down
    execute(<<-SQL.squish)
      ALTER TABLE anomaly_detection_triage_actions
        DROP COLUMN IF EXISTS reviewer_label,
        DROP COLUMN IF EXISTS reviewed_at,
        DROP COLUMN IF EXISTS score_at_label_time,
        DROP COLUMN IF EXISTS model_version_at_label,
        DROP COLUMN IF EXISTS reviewer_saw_factors,
        DROP COLUMN IF EXISTS hitl_queue_source
    SQL
  end
end
```

**Nota MySQL 8.0:** `ADD COLUMN IF NOT EXISTS` está soportado en MySQL 8.0. El patrón con `execute(<<-SQL.squish)` es el idioma del pack (ver migración existente).

### 5. Datos sintéticos para tests día-1

```sql
-- Insertar en ClickHouse local para tests de HitlQueueQuery
INSERT INTO pbp_productionDB_optimized.anomaly_scores
  (payment_id, facility_id, user_id, scored_at, payment_created_at,
   amount_usd, raw_score, percentile, risk_level, is_anomaly,
   model_version, top_factors, features_json, scoring_mode,
   feature_version, threshold_version, latency_ms, error,
   gateway, payment_method, currency, source_enum,
   calibration_segment, fallback_level, frame_flags)
VALUES
  -- Top-k: alta percentile
  (1001, 1, 100, now(), now(), 150.0, 0.72, 0.95, 'high', 1,
   'frame-v1', '[{"feature":"log_amount","value":5.0,"z_score":3.1,"direction":"high"}]',
   '{}', 'shadow_new', 'base-40', 'v2', 5.0, '', 'stripe', 'card', 'USD', 'web',
   'gateway_stripe', '', ''),
  -- Below-p50: baja percentile (0.30 < p50~0.50)
  (1002, 2, 101, now(), now(), 20.0, 0.35, 0.30, 'low', 0,
   'frame-v1', '[]', '{}', 'shadow_new', 'base-40', 'v2', 3.0, '',
   'card_connect', 'card', 'USD', 'pos', 'gateway_cc', '', '')
```

---

## State of the Art

| Enfoque anterior | Enfoque Fase 5 | Cuándo cambió | Impacto |
|-----------------|----------------|--------------|---------|
| HITL offline sobre parquets (`hitl_export_alerts.py`) | Cola operativa sobre ClickHouse `scoring_mode='shadow_new'` | Fase 5 | Sistema live que alimenta etiquetas reales vs. evaluación académica histórica |
| `TriageAction` sin procedencia ML | `TriageAction` + 6 campos HITL-02 | Migración Fase 5 | Etiquetas con trazabilidad completa: qué score vio el revisor, qué versión del modelo, si vio los factores |
| Capacidad HITL desconocida → bloqueante | `capacity` como parámetro ENV | Fase 5 | La infraestructura está lista independientemente del volumen operativo |

**Deprecated/outdated:**
- El uso de `is_anomaly=1` como filtro de la cola: para el challenger frame-v1, usar `percentile DESC` con `scoring_mode='shadow_new'` es más correcto. `is_anomaly` en el challenger refleja el umbral del challenger que puede diferir del champion.

---

## Open Questions

1. **¿La cola HITL debe incluir solo `scoring_mode='shadow_new'` o también `shadow_old`?**
   - What we know: El challenger frame-v1 es el que se quiere promover; revisar el champion en paralelo tiene valor para comparación pero duplica el trabajo.
   - What's unclear: Si el protocolo de revisión requiere validación comparativa champion vs. challenger.
   - Recommendation: La cola filtra solo `shadow_new`. Si se quiere comparación, unir `shadow_old` en la query detail — no en la cola de asignación.

2. **¿Dónde persisten las etiquetas HITL-02 a largo plazo — solo MySQL o también ClickHouse?**
   - What we know: Las etiquetas en MySQL (`TriageAction` extendido) son el source of truth operativo. Los scripts offline de tesis consumen parquets.
   - What's unclear: Si habrá un pipeline de retroalimentación que inyecte las etiquetas MySQL en ClickHouse para análisis de drift.
   - Recommendation: Fase 5 persiste en MySQL únicamente. Un script futuro puede exportar a parquet para análisis. No scope de Fase 5.

3. **`reviewer_label` — valores permitidos**
   - What we know: `hitl_ingest_labels.py` define `VALID_CATEGORIES = {"sospecha_fraude", "anomalia_operativa", "falso_positivo", "indeterminado", "_correccion_"}`. El HITL operativo debería alinearse con esto para consistencia con la tesis.
   - Recommendation: Usar el mismo vocabulario en el `HitlLabelForm` con `validates :reviewer_label, inclusion: { in: %w[sospecha_fraude anomalia_operativa falso_positivo indeterminado] }`.

---

## Sources

### Primary (HIGH confidence)
- Código fuente auditado directamente:
  - `platform/packs/anomaly_detection/db/migrate/20260617040332_create_anomaly_detection_triage_actions.rb` — schema actual de triage_actions (sin campos HITL-02)
  - `platform/packs/anomaly_detection/db/migrate/20260617040330_create_anomaly_detection_alerts.rb` — schema de alerts
  - `platform/packs/anomaly_detection/app/models/anomaly_detection/triage_action.rb` — modelo actual
  - `platform/packs/anomaly_detection/app/forms/anomaly_detection/triage_transition_form.rb` — form pattern
  - `platform/packs/anomaly_detection/app/services/anomaly_detection/triage_transition_service.rb` — service pattern
  - `platform/packs/anomaly_detection/app/queries/anomaly_detection/transaction_query.rb` — query ClickHouse pattern
  - `platform/packs/anomaly_detection/app/queries/anomaly_detection/transaction_detail_query.rb` — top_factors ya leído
  - `ml-fraud-detector/docker/clickhouse/init/02_anomaly_scores.sql` — DDL completo con top_factors String
  - `ml-fraud-detector/docker/clickhouse/migrations/02_anomaly_scores_frame_v1.sql` — frame_flags, calibration_segment, fallback_level
  - `ml-fraud-detector/scorer/batch/scorer.py` — _INSERT_COLUMNS, json.dumps(factors) para top_factors
  - `ml-fraud-detector/scorer/schemas.py` — FactorItem(feature, value, z_score, direction)
  - `ml-fraud-detector/scripts/hitl_export_alerts.py` — pipeline HITL offline (NO tocar)
  - `ml-fraud-detector/scripts/hitl_ingest_labels.py` — VALID_CATEGORIES para reviewer_label
  - `ml-fraud-detector/scripts/shadow_gate.py` — patrón _build_ch_client para hitl_queue_builder.py

### Tertiary (LOW confidence, no verificado con fuentes externas)
- Comportamiento de `ADD COLUMN IF NOT EXISTS` en MySQL 8.0: asumido soportado basado en MySQL 8.0 docs generales. Verificar antes de aplicar en producción.
- `ORDER BY rand()` vs `sample()` en ClickHouse para tablas grandes: recomendación basada en conocimiento general de ClickHouse MergeTree. Para verificar: [ClickHouse docs — SAMPLE clause](https://clickhouse.com/docs/en/sql-reference/statements/select/sample).

---

## Metadata

**Confidence breakdown:**
- Schema de triage_actions y qué falta: HIGH — auditado directamente el DDL de migración
- `top_factors` persistido en ClickHouse: HIGH — verificado en DDL + código de inserción
- Patrón de queries ClickHouse desde Rails: HIGH — 3 query classes auditadas (TransactionQuery, TransactionDetailQuery, ScoreDistributionQuery)
- Lógica de muestreo defensivo (p50 + rand): MEDIUM — ClickHouse `quantile()` y `rand()` son funciones documentadas; eficiencia a escala es LOW-MEDIUM sin benchmark
- `ADD COLUMN IF NOT EXISTS` en MySQL 8.0: MEDIUM — asumido por MySQL 8.0 feature set; no verificado con docs oficiales en esta sesión

**Research date:** 2026-07-06
**Valid until:** 2026-08-06 (30 días — stack estable)
