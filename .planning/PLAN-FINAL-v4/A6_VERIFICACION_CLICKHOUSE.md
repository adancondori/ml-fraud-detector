# A6 Verificacion ClickHouse v4

## Reglas aplicadas

Reglas revisadas de `clickhouse-best-practices`:

- `query-join-filter-before`: filtrar pagos antes de joins.
- `query-join-use-any`: usar `ANY JOIN` cuando basta una fila.
- `schema-pk-filter-on-orderby`: filtrar por columnas de `ORDER BY` cuando sea posible.
- `query-join-choose-algorithm`: evitar joins grandes sin preagregar.
- `query-index-skipping-indices`: considerar indices solo despues de optimizar filtros/joins.

## Patrones SQL

### Base de pagos

```sql
WITH p AS (
    SELECT *
    FROM pbp_productionDB_optimized.payments FINAL
    WHERE created_at >= '2025-01-01'
      AND created_at < '2026-01-01'
      AND payment_method != 'reversal'
      AND payment_method != 'free'
      AND user_id != 0
      AND _peerdb_is_deleted = 0
)
SELECT ...
FROM p
```

### Join dimension one-to-one

```sql
LEFT ANY JOIN (
    SELECT id, created_at
    FROM pbp_productionDB_optimized.users FINAL
    WHERE _peerdb_is_deleted = 0
) u ON p.user_id = u.id
```

### Join one-to-many preagregado simple

```sql
LEFT ANY JOIN (
    SELECT
        reservation_id,
        uniqExact(user_id) AS participant_count,
        countIf(user_affiliation_enum = 'teacher') AS teacher_rows
    FROM pbp_productionDB_optimized.reservations_users FINAL
    WHERE created_at >= '2025-01-01'
      AND created_at < '2026-01-01'
      AND _peerdb_is_deleted = 0
    GROUP BY reservation_id
) ru ON p.reservation_id = ru.reservation_id
```

Este patron solo es valido para dimensiones historicamente estables. Para `reservations_users` puede introducir soft leakage porque agrega el estado actual de participantes. No usarlo en V4-CLEAN salvo como sensibilidad.

### Join as-of para participantes de reserva

Para V4-CLEAN, los agregados de `reservations_users` deben respetar:

```text
reservations_users.created_at <= payments.created_at
```

Patron recomendado:

1. Filtrar pagos primero (`p`) por split y universo.
2. Filtrar `reservations_users` a columnas minimas y rango 2025.
3. Agregar por `payment_id` despues de aplicar condicion temporal.
4. Si el join resulta costoso, materializar temporalmente el subset offline o excluir estas features.

Ejemplo conceptual:

```sql
WITH p AS (
    SELECT id, reservation_id, created_at
    FROM pbp_productionDB_optimized.payments FINAL
    WHERE created_at >= '2025-10-01'
      AND created_at < '2026-01-01'
      AND reservation_id != 0
      AND payment_method != 'reversal'
      AND payment_method != 'free'
      AND user_id != 0
      AND _peerdb_is_deleted = 0
),
ru AS (
    SELECT reservation_id, user_id, user_affiliation_enum, free_pass, resident, created_at
    FROM pbp_productionDB_optimized.reservations_users FINAL
    WHERE created_at >= '2025-01-01'
      AND created_at < '2026-01-01'
      AND _peerdb_is_deleted = 0
)
SELECT
    p.id AS payment_id,
    uniqExactIf(ru.user_id, ru.created_at <= p.created_at) AS participant_count,
    countIf(ru.created_at <= p.created_at AND ru.user_affiliation_enum = 'teacher') AS teacher_rows
FROM p
LEFT JOIN ru ON p.reservation_id = ru.reservation_id
GROUP BY p.id
```

Aplicar `query-join-filter-before`: ambos lados deben estar filtrados antes del join. Aplicar `query-join-choose-algorithm`: si el join as-of consume demasiada memoria, moverlo a job offline o particionar por split/mes.

## Restricciones del usuario ClickHouse

El usuario read-only no permite:

```sql
SETTINGS join_algorithm='auto'
```

No incluir settings mutables en queries finales si fallan por readonly.

## FINAL

Usar `FINAL` en:

- `payments`
- `reservations`
- `reservations_users`
- `user_tokens`
- `payment_discounts`
- `coupons`
- `comments`
- `user_penalties`

No usar `FINAL` en `audit_logs`, porque es `SharedMergeTree` y no lo soporta.

## Gate SQL

Antes de extraccion final:

- verificar row counts;
- verificar columnas;
- probar query con LIMIT;
- guardar query completa en manifest.
