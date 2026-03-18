# Fase 1. Datos y ClickHouse

## Proposito

Construir el snapshot oficial del estudio desde ClickHouse y congelarlo de manera reproducible.

## Hallazgos validados

### Tabla fuente

- base: `pbp_productionDB_optimized.payments`
- engine: `SharedReplacingMergeTree`
- `ORDER BY`: `(facility_id, created_at, id)`
- sin `PARTITION BY`

### Implicacion principal

La tabla fisica contiene multiples versiones de filas. Sin `FINAL`, los conteos quedan inflados. El snapshot de tesis debe usar `FINAL`.

## Reglas ClickHouse aplicadas

- Per `schema-pk-plan-before-creation`, no crear tablas derivadas nuevas sin documentar antes los patrones de consulta.
- Per `schema-pk-prioritize-filters`, cualquier tabla derivada futura debe priorizar columnas realmente filtradas por el pipeline.
- Per `schema-pk-filter-on-orderby`, hay que entender que filtrar por `created_at` sin el prefijo completo no es ideal, aunque `EXPLAIN` mostro pruning suficiente para el corte anual.
- Per `query-join-filter-before`, cualquier join auxiliar debe filtrar primero y unir despues.
- Per `query-mv-incremental`, no crear materialized views salvo que la tesis realmente lo necesite y haya permiso de escritura.

## SQL canonico del snapshot

```sql
SELECT
    id,
    user_id,
    facility_id,
    facility_name,
    created_at,
    captured_at,
    payment_method,
    gateway,
    source_enum,
    status,
    reservation_paid_out,
    discount,
    tax,
    tip,
    card_brand,
    reversed_id,
    debit_refund,
    _peerdb_version
FROM pbp_productionDB_optimized.payments FINAL
WHERE created_at >= %(start)s
  AND created_at < %(end)s
  AND payment_method != 'reversal'
  AND payment_method != 'free'
  AND user_id != 0
  AND _peerdb_is_deleted = 0
ORDER BY created_at, id
```

## Historia previa obligatoria para features temporales

El universo de evaluacion es 2025, pero las features con ventanas retrospectivas no pueden comenzar "en cero" en cada split. Se requieren dos estrategias adicionales:

### Warm history para ventanas moviles

Minimo extraer:

- `2024-12-01` a `2024-12-31` para ventanas de 30 dias;
- o un rango mayor si una feature lo exige.

Uso:

- no entra al universo evaluado;
- si entra al calculo de rolling windows para enero 2025 y para el borde train/val/test.

### Historia de cuenta o primera observacion

Para `user_account_age_days` se necesita una de estas fuentes:

1. `users.created_at` desde una dimension de usuarios, si existe y es confiable.
2. `first_seen_at` del usuario calculado en historial transaccional disponible, documentando el sesgo de censura si no cubre toda la vida de la cuenta.

Sin resolver este punto, la feature queda incompleta.

## Joins auxiliares permitidos

Si se requieren dimensiones pequenas, deben cumplir:

- filtro previo en subconsulta;
- join sobre dataset ya reducido por periodo o ids necesarios;
- preferencia por extraer la dimension chica y unir localmente.

Per `query-join-filter-before`, no se debe unir tablas completas y filtrar despues. Per `query-join-choose-algorithm`, si se mantiene el join en ClickHouse, hay que fijar algoritmo o usar `auto` conscientemente y garantizar que la tabla derecha sea la menor.

## Conteos objetivo

### Totales anuales

- total: `6,784,695`
- strict proxy: `429,442`
- wide proxy: `512,609`

### Splits

- train: `3,137,086`
- val: `1,130,118`
- test: `2,517,491`

## Decisiones tecnicas

### Extraccion

- extraer por split, no cargar el anio entero si no hace falta;
- persistir en `parquet` con compresion `snappy`;
- guardar manifest con fecha de extraccion, conteos y filtros usados.

### Validacion de calidad

Validar al extraer:

- tipos de columnas;
- rango temporal;
- unicidad razonable de `id` post `FINAL`;
- nulos en columnas obligatorias;
- dominios de `status`, `payment_method`, `source_enum`, `gateway`.
- cardinalidad y tipo de `gateway`, `payment_method`, `source_enum`, `card_brand` para mantenerlos como columnas categoricas compactas.
- presencia de strings vacios versus nulos.
- consistencia de `currency`.
- monotonia o no de `created_at` dentro de cada extract.

Per `schema-types-lowcardinality`, si se crea cualquier tabla derivada local o persistente en ClickHouse, estas columnas deben conservar `LowCardinality(String)` cuando el dominio sea pequeno. Per `schema-types-avoid-nullable`, si se materializan staging tables, usar `DEFAULT` cuando el nulo no sea semantico.

### Archivos esperados

- `data/processed/train_raw.parquet`
- `data/processed/val_raw.parquet`
- `data/processed/test_raw.parquet`
- `output/manifests/dataset_manifest.json`
- `output/manifests/query_snapshot.sql`

## Tareas operativas

1. Reescribir el extractor actual y eliminar la fabricacion de `is_fraud`.
2. Corregir `scripts/test_clickhouse_connection.py` para no consultar columnas inexistentes.
3. Crear `scripts/verify_counts.py`.
4. Guardar `EXPLAIN indexes = 1` de la consulta base como evidencia tecnica.
5. Congelar el snapshot local y trabajar desde ahi.
6. Definir si las dimensiones auxiliares se unen en ClickHouse o fuera de ClickHouse.
7. Construir `warm_history.parquet` si las ventanas se calculan localmente.
8. Dejar reintentos y checkpoints por split y por mes.

## Riesgos

### Riesgo de performance

El uso de `FINAL` encarece la lectura.

Mitigacion:

- extraccion por split;
- cache local;
- evitar repetir lecturas completas.
- si el split anual falla, extraer por mes y consolidar.

### Riesgo de bordes temporales

Las features de `val` y `test` pueden quedar mal calculadas si se procesan aisladas.

Mitigacion:

- warm history para rolling windows;
- pruebas especificas del cambio de split;
- manifest que documente el rango real usado para cada feature set.

### Riesgo de drift del origen

La base productiva puede seguir cambiando.

Mitigacion:

- congelar artefactos parquet;
- registrar fecha y hora de extraccion.

## Gate de salida

No pasar a Fase 2 ni Fase 3 hasta que:

- el snapshot reproduzca los conteos objetivo;
- el manifest exista;
- se haya eliminado `is_fraud` del flujo de extraccion.
- exista una estrategia cerrada para `warm history` y `user_account_age_days`.
