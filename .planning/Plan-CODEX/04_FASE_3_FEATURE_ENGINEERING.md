# Fase 3. Feature Engineering

## Proposito

Implementar el catalogo oficial de features de la tesis con garantias de validez temporal y metodologica.

## Catalogo oficial

### Transaccionales

- `reservation_paid_out`
- `log_amount`
- `amount_usd_ratio`
- `discount_ratio`
- `has_tip`

### Temporales

- `hour_sin`
- `hour_cos`
- `day_of_week`
- `is_weekend`
- `is_off_hours`

### Velocidad

- `user_txn_count_1h`
- `user_txn_count_24h`
- `time_since_last_txn`
- `user_amount_24h`

### Comportamiento

- `user_distinct_facilities_30d`
- `user_distinct_methods`
- `user_reversal_ratio_30d`
- `user_account_age_days`

### Contextuales

- `facility_avg_amount`
- `amount_facility_ratio`

## Variante obligatoria

Ademas del modelo de 20 features, debe existir una version de 19 features sin:

- `user_reversal_ratio_30d`

Razon:
es la feature con mayor riesgo de correlacion mecanica con el proxy.

## Reglas anti-leakage

- ordenar siempre por entidad temporal antes de calcular ventanas;
- usar `closed='left'` o `shift(1)` en toda agregacion retrospectiva;
- no incluir la fila actual dentro de su propio agregado;
- no usar `status` ni eventos posteriores como predictor;
- si una entidad requiere historial previo no disponible, inicializar con valor neutro y documentarlo.
- no reiniciar historiales artificialmente al comienzo de `val` o `test`.

## Definiciones que deben quedar escritas

Antes de implementar, cada feature debe tener:

- formula exacta;
- columnas fuente;
- periodo historico utilizado;
- manejo de nulos y division por cero;
- estrategia para cold-start;
- prueba minima de sanidad.

### Casos especialmente delicados

#### `amount_usd_ratio`

Definir contra que promedio divide:

- promedio global train?
- promedio historico hasta la fila?

La definicion debe ser unica y mantenerse igual en todos los splits.

#### `user_distinct_methods`

Definir si es:

- historico total hasta la fecha;
- o en ventana fija.

#### `facility_avg_amount`

Debe evitar fuga temporal. No puede calcularse con el promedio de todo el dataset incluyendo futuro. Debe ser historico hasta la fila o baseline congelado en train con analisis de tradeoff documentado.

## Requisitos de implementacion

- no usar loops fila por fila;
- preferir operaciones vectorizadas, `groupby`, `rolling`, `expanding`, merges controlados;
- controlar memoria por split;
- loggear columnas creadas, null rates y tiempos de ejecucion.
- construir primero una implementacion "tiny sample" y despues escalar al snapshot completo.
- serializar por etapas para no recalcular toda la ingenieria si una sola feature falla.

## Validaciones obligatorias

- test unitario que demuestre que una observacion no mira al futuro;
- test de consistencia de columnas entre splits;
- test de rango o dominio para features clave;
- comparacion de performance entre version de 20 y 19 features.
- test de borde entre `2025-06-30` y `2025-07-01`;
- test de borde entre `2025-08-31` y `2025-09-01`;
- test de cold-start para usuarios sin historial;
- test de robustez ante `reservation_paid_out = 0` o `discount > amount`.

## Entregables

- `src/fraud_detector/features/engineering.py`
- `src/fraud_detector/features/catalogs.py`
- `tests/test_features.py`
- `data/processed/train_features.parquet`
- `data/processed/val_features.parquet`
- `data/processed/test_features.parquet`

## Gate de salida

No pasar a Fase 4 hasta que:

- existan los parquets con features;
- el catalogo este congelado;
- los tests de leakage pasen;
- la version de 19 features tambien exista.
- los bordes de split esten probados;
- cada feature tenga formula documentada en catalogo.
