# A4 Gobernanza, Privacidad y Contratos v4

## Principios

1. No exponer PII en artefactos.
2. No guardar nombres, emails, telefonos ni direcciones.
3. Usar IDs internos solo cuando sean necesarios para joins o trazabilidad.
4. Reportar resultados agregados.

## Contrato de datos

Campos permitidos en features:

- IDs tecnicos: `id`, `user_id`, `facility_id`, `reservation_id`, `user_token_id`.
- Montos normalizados.
- Categorias operativas.
- Conteos historicos.
- Flags anonimos.

Campos prohibidos:

- email.
- nombre.
- telefono.
- direccion.
- comentarios textuales.
- notas libres.
- IPs.

## Contrato de modelo

El modelo v4 debe declarar:

- `model_type`.
- `feature_version`.
- `label_definition`.
- `train_period`.
- `validation_period`.
- `test_period`.
- `threshold_source`.

## Contrato de score

El score representa:

```text
riesgo relativo de reembolso Tipo A
```

No representa:

```text
probabilidad legal de fraude
```

## Auditoria

Cada corrida debe conservar manifest con:

- fecha;
- query hash;
- feature list;
- metricas;
- thresholds;
- version de dependencias.
