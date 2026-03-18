# Fase 0. Contrato y Alcance

## Proposito

Traducir la tesis a un contrato tecnico verificable para que ninguna fase posterior trabaje con supuestos ambiguos.

## Preguntas que esta fase debe cerrar

- Cual es el universo exacto del estudio?
- Cual es el proxy principal y cual es el proxy de sensibilidad?
- Cuales son los splits temporales oficiales?
- Cual es el catalogo oficial de features?
- Cuales metricas validan cada hipotesis?
- Cual es el alcance real del proyecto y que queda fuera?

## Definiciones fijas

### Universo del estudio

Transacciones 2025 de `payments` deduplicadas con `FINAL`, excluyendo:

- `payment_method = 'reversal'`
- `payment_method = 'free'`
- `user_id = 0`
- `_peerdb_is_deleted = 0` como condicion de inclusion valida

### Proxy principal

`status IN ('totally_refunded', 'refunded_to_credit')`

### Proxy amplio

`status IN ('totally_refunded', 'refunded_to_credit', 'partially_refunded')`

### Splits

- train: `2025-01-01` a `2025-06-30`
- validation: `2025-07-01` a `2025-08-31`
- test: `2025-09-01` a `2025-12-31`

### Algoritmos

- principal: `Isolation Forest`
- comparadores: `LOF`, `One-Class SVM`

### Hipotesis

- HE1: separacion estadistica de scores
- HE2: capacidad discriminativa
- HE3: concentracion top-k
- HE4: comparacion de metodos

## Fuera de alcance

- despliegue en produccion;
- scoring en tiempo real;
- API de inferencia;
- dashboards operativos;
- deteccion legal de fraude;
- deep learning si no es necesario para la defensa de la tesis.

## Entregables

- documento `thesis_contract` o equivalente;
- parametros fijos en `config/config.py`;
- tabla de trazabilidad `objetivo -> hipotesis -> modulo -> output`.

## Tareas operativas

1. Refactorizar configuracion para eliminar restos de aprendizaje supervisado.
2. Eliminar conceptos ajenos al estudio: `SMOTE`, `fraud_threshold`, `model_type` supervisado, `MLflow` como dependencia central.
3. Crear constantes compartidas para proxy, splits y rutas de salida.
4. Documentar que el tuning en validation es supervision indirecta ex post, no entrenamiento supervisado.

## Gate de salida

No pasar a Fase 1 hasta que:

- exista una unica fuente de verdad para filtros y proxy;
- el README del proyecto ya describa correctamente la tesis actual;
- no haya ninguna ruta critica dependiente de `is_fraud`.
