# 03 EDA Capitulo 2 v4

## Objetivo

Reemplazar EDA descriptivo generico por EDA orientado a senales que elevan discriminacion de Tipo A.

## Analisis obligatorios

1. Tasa Tipo A mensual.
2. Tasa Tipo A por `source_enum`.
3. Tasa Tipo A por `payment_method`.
4. Tasa Tipo A por `gateway`.
5. Tasa Tipo A por `category`.
6. Tasa Tipo A por `facility_id` y `facility_group_id`.
7. Tasa Tipo A por lead time de reserva.
8. Tasa Tipo A por participantes de reserva.
9. Tasa Tipo A por edad de token.
10. Tasa Tipo A por cupon.

## Hallazgos base ya medidos

| Senal | Hallazgo |
|---|---|
| `source_enum` app | 12-13% refund, lift cercano a 2 |
| `payment_method=prepaid/user_package` | 12-14% refund, lift cercano a 2 |
| lead time 8-30d | ~35% refund |
| lead time 31d+ | ~37% refund |
| `admin_booked=false` | ~22% refund |
| reserva con 1 participante | ~16% refund |
| `has_coupon` | ~9.5% refund |
| token edad 8-30d | lift ~1.2 |

## Leakage audit en EDA

Separar cada variable en:

- **Disponible T0:** al crear el pago.
- **Disponible T1:** despues de captura/autorizacion.
- **Posterior/no usable:** cambia por cancelacion, refund o auditoria posterior.

Variables posteriores identificadas:

- `reservation_status`
- `incident_enum`
- `payment_completed`
- `membership_state`
- `payments.comments_count`

## Artefactos

```text
output/v4/eda/
├── tipo_a_monthly.csv
├── lift_by_source_enum.csv
├── lift_by_payment_method.csv
├── lift_by_reservation_context.csv
├── lift_by_token_context.csv
├── leakage_audit_candidates.csv
└── figures/*.png
```

## Gate

El EDA debe mostrar lift y cobertura. Una feature con lift alto pero cobertura muy baja queda en sensibilidad, no en modelo principal.
