# Fase 4. Preprocesamiento y Datasets

## Proposito

Transformar el snapshot con features en matrices consistentes y reutilizables para los tres algoritmos.

## Decisiones metodologicas

- el ajuste del preprocesamiento se hace solo sobre train;
- validation y test usan exclusivamente transformaciones aprendidas en train;
- el esquema final de columnas debe ser identico entre splits;
- se privilegia simplicidad y trazabilidad sobre sofisticacion innecesaria.

## Tareas

### Imputacion

- categoricas: `Unknown`
- numericas: mediana de train

### Escalado

- `StandardScaler` comun para comparabilidad entre `Isolation Forest`, `LOF` y `OC-SVM`

### Encoding

- si alguna categorica entra al modelo, definir una sola estrategia permitida:
  - ordinal controlado;
  - frequency encoding;
  - one-hot solo si el dominio es acotado y la explosion dimensional no rompe `LOF`/`OC-SVM`.

La tesis debe dejar cerrada esta decision; no puede quedar implicita.

### Serializacion

- guardar `scaler.joblib`;
- guardar metadata de columnas finales;
- guardar matrices listas o parquet transformado.

## Controles

- revisar que no aparezcan columnas extra en val/test;
- registrar shape y orden de columnas;
- congelar el schema final antes de iniciar tuning.
- verificar que no existan infinitos, NaN residuales o overflow por conversion de decimales.
- documentar downcast a `float32` si se usa para memoria.

## Archivos esperados

- `output/models/scaler.joblib`
- `output/manifests/features_schema.json`
- `data/processed/train_model_input.parquet`
- `data/processed/val_model_input.parquet`
- `data/processed/test_model_input.parquet`

## Pruebas

- test de fit solo en train;
- test de transform consistente en val/test;
- test de ausencia de columnas faltantes;
- test de reproducibilidad con `random_seed`.
- test de densidad/sparsity si se usa encoding categorico.
- smoke test de memoria con lote real.

## Gate de salida

No pasar a Fase 5 hasta que:

- el schema final este congelado;
- scaler y metadata existan;
- todas las matrices sean consumibles por los tres modelos.
