# 11 Tests, Cleanup e Integracion v4

## Tests unitarios nuevos

- SQL contiene columnas v4.
- SQL usa `FINAL` donde aplica.
- Target encoder fit solo en train.
- Unknown categories transforman sin error.
- No hay `status` en matriz de features.
- Rolling features excluyen fila actual.
- Reservation features no usan columnas prohibidas.
- Reservation-user features aplican `created_at <= payment.created_at` o quedan fuera de V4-CLEAN.
- Test de mutabilidad de `reservation.date` no detecta cambios materiales.
- Token features manejan token faltante.
- Coupon features manejan cupon faltante.
- Coupon features se calculan desde `payment_discounts`, no desde `payments.coupon_id`.
- Thresholds se calculan desde validation.

## Tests de integracion

- Pipeline v4 smoke con sample.
- Train HGB con sample y produce scores.
- Evaluation produce AUC/AP/top-k.
- Report genera tablas.
- Scorer single reproduce batch en muestra.

## Tests anti-leakage

Debe fallar si una feature usa:

- `reservation_status`.
- `incident_enum`.
- `payment_completed`.
- `membership_state`.
- `status`.
- `comments_count` como feature principal.
- `last_change_at`.
- `updated_at`.
- `deleted_at`.
- `total_paid_out`.
- `most_recent_date`.
- `approval_status`.
- `status_enum`.
- `reservations_users.payment_id`.

Tests adicionales:

- SIMPLE-RULE baseline se calcula y se guarda.
- Monthly stability incluye Oct, Nov y Dic por separado.
- Sampling de negativos es reproducible con seed y, si aplica, estratificado por mes/clase.
- Monedas se normalizan a USD o se reportan con `missing_exchange_rate`.

Test minimo:

```python
PROHIBITED_COLUMNS = {
    "status", "incident_enum", "reservation_status", "payment_completed",
    "membership_state", "comments_count", "last_change_at", "updated_at",
    "deleted_at", "total_paid_out", "most_recent_date", "approval_status",
    "status_enum", "payment_id"
}
```

## Cleanup

No eliminar artefactos historicos sin autorizacion. Guardar v4 en:

```text
data/processed/v4/
output/v4/
```

## Gate

Antes de cerrar:

```bash
make test
```

Debe pasar. Si no pasa, documentar tests fallidos y razon.
