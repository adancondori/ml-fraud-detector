# 04 Feature Engineering v4

## Principio

Toda feature debe ser point-in-time. Para cada columna se documenta el momento minimo en el que existe.

## Conteo v4

| Catalogo | Features logicas | Uso |
|---|---:|---|
| V4-CANDIDATE | 65 | Inventario completo |
| V4-CLEAN | 62 | Modelo principal defendible |
| V4-CLEAN-BOOSTED | 62 logicas, ~71 columnas finales | Sensibilidad/operativo con target encoding |
| V4-FULL-SENS | 65 logicas, ~74 columnas finales | Sensibilidad con historial de refund |

El modelo principal es **V4-CLEAN**. Las 3 features de historial directo de reembolso no entran al resultado central.

## Grupos principales V4-CLEAN

### A. Transaccion base

- `amount`
- `amount_usd`
- `log_amount_usd`
- `amount_usd_ratio`
- `discount_ratio`
- `has_tip`
- `tax_ratio`
- `tip_ratio`

Definicion:

```text
amount_raw = reservation_paid_out
amount_usd = amount_raw * fx_rate(currency, payment.created_at::date)
log_amount_usd = log1p(amount_usd)
amount_usd_ratio = amount_usd / median(amount_usd en train)
discount_ratio = discount_usd / NULLIF(amount_usd, 0)
tax_ratio = tax_usd / NULLIF(amount_usd, 0)
tip_ratio = tip_usd / NULLIF(amount_usd, 0)
has_tip = tip > 0
```

Fallback permitido si la fuente no trae `reservation_paid_out`:

```text
amount_raw = original_amount_paid_out
```

`amount_raw` y `log_amount` nominal multi-moneda no son features principales. Solo pueden aparecer en benchmarks exploratorios marcados como `nominal_amount_benchmark`.

### B. Tiempo

- `hour_sin`
- `hour_cos`
- `day_of_week`
- `is_weekend`
- `is_off_hours`
- `month`

### C. Usuario rolling 2025

- `user_txn_count_1h`
- `user_txn_count_24h`
- `user_txn_count_7d`
- `time_since_last_txn`
- `user_amount_24h`
- `user_amount_7d`
- `user_distinct_facilities_30d`
- `user_distinct_methods`
- `user_discount_ratio_30d`

Motor de computo:

- Implementacion principal: pandas despues de la extraccion snapshot, ordenando por `user_id, created_at, payment_id`.
- Todas las ventanas excluyen la fila actual con `shift(1)` antes del rolling.
- Si memoria local no alcanza para 6.7M filas, particionar por hash de `user_id`; nunca por mes, porque una ventana puede cruzar el borde mensual.
- ClickHouse queda permitido para aceleracion, pero el resultado debe reconciliarse contra una muestra pandas golden.

### D. Historial de reembolso directo

Features sensibles. No entran al modelo principal V4-CLEAN:

- `user_reversal_ratio_30d`
- `user_reversal_count_30d`
- `user_refund_count_90d`

Estas features solo pueden usarse en `V4-FULL-SENS`, con transacciones previas y nunca la fila actual. Se reportan para sensibilidad, no para la conclusion principal, porque son un proxy retrasado de la variable criterio.

### E. Contexto categorico limpio

Frequency encoding train-only para el modelo principal:

- `category`
- `facility_id`
- `currency`
- `payment_method`
- `gateway`
- `source_enum`
- `payment_source`
- `card_brand`

Regla: fit solo en train. Val, discovery y test usan mappings congelados.

Target encoding queda permitido solo en `V4-CLEAN-BOOSTED` y debe reportarse separado porque usa la etiqueta de train para construir senales agregadas.

`user_role` no esta en `payments`; vive en `users.role`. No entra a V4-CLEAN hasta documentar el join a `users` y auditar mutabilidad point-in-time. Si se usa, reportarlo como sensibilidad.

### F. Reserva point-in-time

Permitidas:

- `reservation_type`
- `kind_enum`
- `admin_booked`
- `booked_from`
- `generated_by_court`
- `reservation_lead_days`
- `reservation_lead_bucket`
- `payment_after_booking_minutes`
- `payment_after_booking_bucket`
- `participant_count`
- `has_invited`
- `has_teacher`
- `has_guest_name`
- `has_free_pass`
- `has_resident`

Reglas:

- `reservation_lead_days` usa `reservation.date` solo si la auditoria de mutabilidad confirma que no cambia materialmente.
- `payment_after_booking_minutes = payment.created_at - reservation.created_at`. No usa `captured_at`; las features de captura son T1/sensibilidad.
- Features provenientes de `reservations_users` deben calcularse as-of: `reservations_users.created_at <= payment.created_at`.
- Si no hay implementacion as-of eficiente, `participant_count`, `has_invited`, `has_teacher`, `has_free_pass`, `has_resident` y `teacher_rows` pasan a sensibilidad, no a V4-CLEAN.

Prohibidas como principales:

- `incident_enum`
- `reservation_status`
- `payment_completed`
- `allow_cancel` si cambia despues del pago.

### G. Token/card

- `has_user_token`
- `token_age_days`
- `token_age_bucket`
- `token_is_default`
- `token_accttype_norm`
- `token_gateway_mismatch`
- `has_token_last4`
- `user_token_count_facility_2025_prior`

Definicion de `user_token_count_facility_2025_prior`:

```sql
countDistinctIf(
    user_tokens.id,
    user_tokens.user_id = payments.user_id
    AND user_tokens.facility_id = payments.facility_id
    AND user_tokens.created_at < payments.created_at
)
```

Si `user_tokens.facility_id` no existe o no es confiable, reemplazar por `user_token_count_2025_prior` y documentar el cambio en `feature_catalog.csv`.

### H. Cupon/descuento

- `has_coupon`
- `coupon_value_type`
- `coupon_age_days`
- `coupon_age_bucket`
- `coupon_for_reservation`
- `coupon_for_clinic`
- `coupon_for_all`
- `discount_amount_bucket`

Fuente obligatoria:

- `payment_discounts.payment_id`.
- `payment_discounts.discount_id`.
- `coupons.id`.

No usar `payments.coupon_id` como fuente principal por cobertura casi nula.

### I. Sensibilidad

No entran al modelo principal hasta superar auditoria:

- `prior_failed_payment_30d`
- `prior_user_comments_30d`
- `prior_reservation_comments`
- `prior_audit_refund_30d`
- `prior_payment_method_added_30d`
- `prior_user_penalty`
- `is_captured`
- `payment_to_capture_seconds`
- `payment_to_capture_bucket`

### J. Normalizacion USD

Todas las features monetarias principales deben estar en USD antes de entrenar:

- `amount_usd`
- `log_amount_usd`
- `amount_usd_ratio`
- `discount_ratio`
- `tax_ratio`
- `tip_ratio`

Si una moneda no tiene tasa disponible:

- registrar `missing_exchange_rate=1`;
- imputar con politica documentada;
- reportar conteo por moneda en `imputation_manifest.json`.

No usar montos nominales multi-moneda como features principales.

## Variantes obligatorias

| Variante | Descripcion |
|---|---|
| V4-CLEAN | 62 features logicas, sin grupo D y sin target encoding |
| V4-CLEAN-BOOSTED | V4-CLEAN + target encoding train-only |
| V4-FULL-SENS | V4-CLEAN + grupo D, solo sensibilidad |
| V4-CLEAN-NO-RU | V4-CLEAN sin features de `reservations_users` |
| V4-T0 | Solo datos disponibles al crear pago |
| V4-T1 | Incluye datos disponibles despues de captura |
| V4-LEGACY-31 | 31 features actuales para comparabilidad |

## Artefactos

```text
data/processed/v4/{split}_features.parquet
output/v4/features/feature_catalog.csv
output/v4/features/leakage_contract.csv
output/v4/models/feature_pipeline.joblib
```

## Gate

No se acepta una feature sin:

- nombre estable;
- tabla fuente;
- timestamp de disponibilidad;
- prueba anti-leakage;
- test unitario.
