# 12 Single Transaction Scorer v4

## Objetivo

Actualizar el scorer individual para que use las mismas features, encoders y modelo del batch v4.

## Problemas actuales a corregir

- El scorer individual pasa `ch_connector=None`.
- El calculador single no carga bien contexto de facility/staff.
- El threshold actual viene de test, debe venir de validation.
- Batch y single no estan validados con equivalencia estricta.

## Entrada

Transaccion o `payment_id`.

Campos minimos:

- `id`
- `user_id`
- `facility_id`
- `created_at`
- `amount`
- `payment_method`
- `gateway`
- `source_enum`
- `category`
- `currency`

## Contexto requerido

Consultar o recuperar de feature store:

- historial usuario 2025 previo;
- contexto reserva;
- participantes reserva;
- token/card;
- cupon;
- mappings de target/frequency encoding.

## Salida

```json
{
  "payment_id": 123,
  "risk_score": 0.87,
  "rank_percentile": 0.99,
  "alert": true,
  "threshold_source": "validation",
  "model_version": "v4",
  "feature_version": "v4"
}
```

## Equivalencia batch/single

Validar con al menos 1000 pagos de test final:

- diferencia absoluta score <= 0.01 para 95% de casos;
- misma decision alert/no alert para >= 98%;
- sin errores en usuarios cold-start.

## Gate

No activar scorer si no existe equivalencia batch/single.
