# ETL y Lineage

## Proposito

Definir el pipeline de datos completo, idempotente y trazable desde ClickHouse hasta los artefactos finales de tesis.

## Capas del pipeline

### Capa 0. Source

- `pbp_productionDB_optimized.payments`
- tablas auxiliares pequenas, si aplican

### Capa 1. Raw extracted

Archivos parquet casi espejo del origen, ya deduplicados con `FINAL`, separados por:

- `warm_history`
- `train_raw`
- `val_raw`
- `test_raw`
- `dim_users` si aplica
- `dim_facilities` si aplica

### Capa 2. Canonical

Dataset ya normalizado semantica y tecnicamente:

- tipos saneados;
- dominios validados;
- columnas renombradas;
- proxies calculados;
- currency policy cerrada.

### Capa 3. Feature

Dataset con features oficiales de tesis:

- 20 features;
- 19 features sin `user_reversal_ratio_30d`.

### Capa 4. Model input

Matrices finales post imputacion y escalado.

### Capa 5. Results

- scores;
- metricas;
- tablas;
- figuras;
- manifests de corrida.

## Reglas del ETL

### Idempotencia

Cada etapa debe poder rerunearse sin corromper artefactos previos.

Reglas:

- no sobrescribir por defecto sin `--force`;
- escribir primero a archivo temporal y luego renombrar;
- registrar checksum o row count;
- mantener manifests con version de snapshot, seed y schema hash.

### Reanudacion

Si la extraccion se corta:

- reanudar por split o por mes;
- no reiniciar desde cero si ya hay artefactos validos;
- verificar integridad antes de reusar.

### Naming

Recomendado:

- `2025-03_train_raw.parquet` para fragmentos;
- `train_raw.parquet` para consolidado;
- `dataset_manifest.json`;
- `run_manifest_<timestamp>.json`.

## Historia previa y lookback

### Warm history

Obligatoria para:

- `user_txn_count_1h`
- `user_txn_count_24h`
- `user_amount_24h`
- `user_distinct_facilities_30d`
- `user_reversal_ratio_30d`

Regla:

- las filas de `warm_history` nunca entran a las metricas finales;
- solo existen para iniciar ventanas retrospectivas.

### Account age

Se resuelve por una de dos vias:

1. dimension `users` con `created_at`;
2. `first_seen_at` del historial transaccional disponible.

Si se usa opcion 2, debe documentarse la limitacion de censura izquierda.

## Joins auxiliares

### Politica recomendada

- extraer datasets principales sin join pesado;
- extraer dimensiones pequenas aparte;
- unir localmente si el volumen lo permite;
- si el join queda en ClickHouse, filtrar primero y reducir cardinalidad.

Per `query-join-filter-before`, el filtro debe ocurrir antes del join. Per `query-join-choose-algorithm`, si el join es grande o restringido por memoria, hay que fijar algoritmo conscientemente.

## Persistencia local y formatos

### Parquet

- usar compresion `snappy`;
- mantener schema estable;
- considerar particionado por split o mes para reanudacion.

### CSV

- solo para export pequeno y debugging;
- no como formato principal de trabajo.

## Validaciones de integridad por capa

### Raw extracted

- row count;
- min/max `created_at`;
- conteo por status;
- conteo por `payment_method`;
- unicidad de `id`.

### Canonical

- proxies;
- dominios;
- tipos;
- filas descartadas y motivo.

### Feature

- null rates;
- infinito/NaN;
- ranges;
- schema freeze.

### Results

- todas las tablas y figuras referencian el mismo run manifest;
- las metricas citan snapshot y seed.

## Si se crean tablas derivadas en ClickHouse

Solo si de verdad hacen falta.

Reglas:

- Per `schema-pk-plan-before-creation`, documentar antes los patrones de consulta.
- Per `schema-pk-cardinality-order`, ordenar key de menor a mayor cardinalidad.
- Per `schema-pk-prioritize-filters`, priorizar filtros reales del ETL.
- Per `schema-partition-low-cardinality`, no partir por claves explosivas.
- Per `schema-partition-lifecycle`, particionar por tiempo solo si ayuda a lifecycle.
- Per `insert-batch-size`, insertar en lotes 10K-100K filas.
- Per `insert-optimize-avoid-final`, no ejecutar `OPTIMIZE TABLE ... FINAL`; usar `FINAL` solo en `SELECT` cuando aplique.
- Per `schema-types-lowcardinality`, preservar `LowCardinality` en strings repetidos.
- Per `schema-types-avoid-nullable`, evitar `Nullable` si un `DEFAULT` es suficiente.

## Artefactos obligatorios

- `output/manifests/dataset_manifest.json`
- `output/manifests/feature_manifest.json`
- `output/manifests/run_manifest_*.json`
- `output/manifests/lineage.md`

## Gate de salida

No continuar a modelado final si:

- el lineage no esta claro;
- no existe warm history;
- no hay manifests por capa.
