# 02 Datos Snapshot v4

## Universo

Fuente principal:

- `pbp_productionDB_optimized.payments`

Filtros:

```sql
created_at >= '2025-01-01'
AND created_at < '2026-01-01'
AND payment_method != 'reversal'
AND payment_method != 'free'
AND user_id != 0
AND _peerdb_is_deleted = 0
```

Resultado de referencia test Sep-Dic 2025:

- 2,517,473 pagos.
- 158,893 Tipo A.
- Tasa Tipo A: 6.3116%.

## Columnas nuevas desde payments

Agregar al SQL canonico:

- `reservation_id`
- `user_token_id`
- `payment_source`
- `no_player_selected`
- `comments_count` solo para sensibilidad
- `captured_at` ya existe, pero definir si se usa en T0 o T1
- `batch_id`, `host_stat` solo sensibilidad

### Mapeo de monto

El feature `amount` no es una columna literal del modelo SQL final; debe mapearse explicitamente:

```text
amount = p.reservation_paid_out
```

Si `reservation_paid_out` no estuviera disponible en una fuente puntual, usar fallback documentado:

```text
amount = p.original_amount_paid_out
```

Todas las razones (`discount_ratio`, `tax_ratio`, `tip_ratio`) usan `NULLIF(amount, 0)` o epsilon equivalente.

## Tablas fuente adicionales

| Tabla | Uso | Estado |
|---|---|---|
| `reservations` | lead time, admin booking, tipo, booked_from | Principal |
| `reservations_users` | participantes, invitado, profesor, guest/free pass | Principal |
| `user_tokens` | edad token, default, accttype, gateway mismatch | Principal |
| `payment_discounts` | cupon/descuento por pago | Principal |
| `coupons` | tipo, edad y alcance del cupon | Principal |
| `failed_payment_logs` | fallos previos | Sensibilidad |
| `comments` | comentarios previos | Sensibilidad |
| `audit_logs` | eventos de usuario | Sensibilidad, baja cobertura |
| `user_penalties` | penalidades previas | Sensibilidad, baja cobertura |
| `membership_payments` | solo para excluir leakage | No principal |
| `users` | `role` solo si se audita mutabilidad | Sensibilidad |

`user_role` no entra a V4-CLEAN por defecto porque no existe en `payments`. Si se decide usarlo, el SQL debe agregar:

```sql
LEFT ANY JOIN (
    SELECT id, role
    FROM pbp_productionDB_optimized.users FINAL
    WHERE _peerdb_is_deleted = 0
) u ON p.user_id = u.id
```

La feature solo pasa a principal si se verifica que `role` no cambia de forma material despues del pago.

## Cupones

No usar `payments.coupon_id` como fuente principal porque su cobertura 2025 es ~0.011%.

Fuente correcta:

```sql
LEFT ANY JOIN (
    SELECT payment_id, discount_id, discount_type
    FROM pbp_productionDB_optimized.payment_discounts FINAL
    WHERE _peerdb_is_deleted = 0
) pd ON p.id = pd.payment_id
LEFT ANY JOIN (
    SELECT id, amount, percentual, kind, expiration_date, created_at, kind_mapping
    FROM pbp_productionDB_optimized.coupons FINAL
    WHERE _peerdb_is_deleted = 0
) c ON pd.discount_id = c.id
```

## Captura

`captured_at` cubre aproximadamente 41.7% de pagos 2025. Por tanto:

- V4-CLEAN no depende de `captured_at`.
- V4-T1 se evalua como subset o sensibilidad.
- Features candidatas permitidas: `is_captured`, `payment_to_capture_seconds`, `payment_to_capture_bucket`.
- No imputar `captured_at = created_at` para el resultado principal.

## Participantes de reserva

`reservations_users` es fuente de soft leakage si se agrega por estado actual. Para V4-CLEAN:

```text
participant features = solo filas con reservations_users.created_at <= payment.created_at
```

Features afectadas:

- `participant_count`
- `has_invited`
- `has_teacher` / `teacher_rows`
- `has_free_pass`
- `has_resident`

Implementacion recomendada:

1. Filtrar pagos 2025 primero.
2. Pre-filtrar `reservations_users` a 2025 y columnas minimas.
3. Calcular agregados as-of por `payment_id` en paso offline o query dedicada.
4. Si el costo del join as-of es alto, excluir estas features de V4-CLEAN y moverlas a sensibilidad.

Esta regla sigue `query-join-filter-before`: no unir tablas completas y filtrar despues.

## Reserva mutable

`reservation.date` debe tratarse como estable solo despues de verificacion. Si se detectan cambios de fecha por versiones:

- usar snapshot point-in-time si esta disponible; o
- excluir `reservation_lead_days` del modelo principal; o
- reportar sensibilidad sin lead time.

Validacion preliminar 2025: no se observaron cambios de `date` entre versiones fisicas disponibles, pero debe quedar como test automatizado.

## Reglas de extraccion

1. Filtrar pagos primero y luego hacer joins.
2. Usar `LEFT ANY JOIN` cuando una dimension debe aportar una sola fila.
3. Preagregar tablas one-to-many antes de unir.
4. Usar `FINAL` en SharedReplacingMergeTree.
5. No cambiar settings como `join_algorithm` porque el usuario ClickHouse esta en read-only.

## SQL canonico Fase 1

El query final debe partir de pagos filtrados y unir dimensiones ya filtradas. Para V4-CLEAN strict, `reservations_users` se agrega por `payment_id` usando condicion as-of.

```sql
WITH p AS (
    SELECT
        id AS payment_id,
        user_id,
        facility_id,
        created_at,
        status,
        source_enum,
        payment_method,
        gateway,
        card_brand,
        currency,
        category,
        paid_by_manager,
        reservation_paid_out,
        original_amount_paid_out,
        discount,
        tax,
        tip,
        reservation_id,
        user_token_id,
        captured_at,
        payment_source
    FROM pbp_productionDB_optimized.payments FINAL
    WHERE created_at >= {start}
      AND created_at <  {end}
      AND payment_method NOT IN ('reversal', 'free')
      AND user_id != 0
      AND _peerdb_is_deleted = 0
),
r AS (
    SELECT
        id,
        created_at AS reservation_created_at,
        date AS res_date,
        admin_booked,
        booked_from,
        generated_by_court,
        kind_enum,
        reservation_type
    FROM pbp_productionDB_optimized.reservations FINAL
    WHERE created_at >= '2024-12-01'
      AND created_at < '2026-02-01'
      AND _peerdb_is_deleted = 0
),
ru AS (
    SELECT
        reservation_id AS ru_reservation_id,
        user_id AS ru_user_id,
        created_at AS ru_created_at,
        user_affiliation_enum,
        free_pass,
        resident
    FROM pbp_productionDB_optimized.reservations_users FINAL
    WHERE created_at >= '2024-12-01'
      AND created_at < {end}
      AND _peerdb_is_deleted = 0
),
t AS (
    SELECT
        id,
        created_at AS token_created_at,
        is_default AS token_is_default,
        accttype AS token_accttype,
        gateway AS token_gateway,
        length(last4) > 0 AS token_has_last4
    FROM pbp_productionDB_optimized.user_tokens FINAL
    WHERE _peerdb_is_deleted = 0
),
pd_disc AS (
    SELECT
        payment_id AS pd_payment_id,
        anyLast(discount_id) AS discount_id,
        anyLast(discount_type) AS discount_type
    FROM pbp_productionDB_optimized.payment_discounts FINAL
    WHERE _peerdb_is_deleted = 0
    GROUP BY pd_payment_id
)
SELECT
    p.payment_id,
    p.user_id,
    p.facility_id,
    p.created_at,
    p.status,
    p.source_enum,
    p.payment_method,
    p.gateway,
    p.card_brand,
    p.currency,
    p.category,
    p.paid_by_manager,
    p.reservation_paid_out,
    p.original_amount_paid_out,
    p.discount,
    p.tax,
    p.tip,
    p.reservation_id,
    p.user_token_id,
    p.captured_at,
    p.payment_source,
    r.reservation_created_at,
    r.res_date,
    r.admin_booked,
    r.booked_from,
    r.generated_by_court,
    r.kind_enum,
    r.reservation_type,
    uniqExactIf(ru.ru_user_id, ru.ru_created_at <= p.created_at) AS participant_count,
    countIf(ru.user_affiliation_enum = 'teacher' AND ru.ru_created_at <= p.created_at) AS teacher_rows,
    maxIf(1, ru.user_affiliation_enum = 'invited' AND ru.ru_created_at <= p.created_at) AS has_invited,
    maxIf(1, ru.free_pass AND ru.ru_created_at <= p.created_at) AS has_free_pass,
    maxIf(1, ru.resident AND ru.ru_created_at <= p.created_at) AS has_resident,
    t.token_created_at,
    t.token_is_default,
    t.token_accttype,
    t.token_gateway,
    t.token_has_last4,
    pd_disc.discount_type AS coupon_discount_type
FROM p
LEFT ANY JOIN r       ON p.reservation_id = r.id
LEFT JOIN ru          ON p.reservation_id = ru.ru_reservation_id
LEFT ANY JOIN t       ON p.user_token_id  = t.id
LEFT ANY JOIN pd_disc ON p.payment_id     = pd_disc.pd_payment_id
GROUP BY
    p.payment_id, p.user_id, p.facility_id, p.created_at, p.status,
    p.source_enum, p.payment_method, p.gateway, p.card_brand, p.currency,
    p.category, p.paid_by_manager, p.reservation_paid_out,
    p.original_amount_paid_out, p.discount, p.tax, p.tip, p.reservation_id,
    p.user_token_id, p.captured_at, p.payment_source,
    r.reservation_created_at, r.res_date, r.admin_booked, r.booked_from,
    r.generated_by_court, r.kind_enum, r.reservation_type,
    t.token_created_at, t.token_is_default, t.token_accttype,
    t.token_gateway, t.token_has_last4, pd_disc.discount_type
```

Este patron aplica `query-join-filter-before` al filtrar `p` y `ru` antes del join. Aplica `query-join-use-any` solo en dimensiones one-to-one (`reservations`, `user_tokens`, `payment_discounts` ya preagregado). El join con `ru` no usa `ANY` porque necesita todas las filas historicas del participante para agregarlas con `uniqExactIf`.

Si este query supera memoria o tiempo, el fallback aceptado para Gate A0 es `V4-CLEAN-NO-RU`: las columnas de participantes se fijan en cero y el resultado queda strict por exclusion.

## Artefactos

```text
data/processed/v4/
├── train_raw.parquet
├── val_raw.parquet
├── discovery_sep_raw.parquet
├── test_oct_dec_raw.parquet
└── manifests/
    ├── query_snapshot.sql
    ├── dataset_manifest.json
    └── source_tables_manifest.json
```

## Gate

No avanzar si:

- falta alguna columna principal;
- hay duplicados de `id` > 0.01%;
- la tasa Tipo A por split no se calcula;
- el manifest no guarda conteos, fechas, query hash y version v4.
