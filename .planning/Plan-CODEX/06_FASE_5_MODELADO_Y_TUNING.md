# Fase 5. Modelado y Tuning

## Proposito

Entrenar el modelo principal y los comparadores bajo un procedimiento metodologicamente limpio y computacionalmente viable.

## Modelo principal

### Isolation Forest

Parametros de arranque:

- `n_estimators = 300`
- `max_samples = 1024`
- `contamination` entre `0.02` y `0.08`
- `max_features` entre `0.5` y `1.0`
- `random_state = 42`

Tuning en validation:

- metrica primaria: `AUC-ROC` contra proxy estricto;
- metrica secundaria: `Average Precision`;
- guardar todas las combinaciones probadas.
- registrar tambien runtime y memoria por combinacion.

## Comparadores

### LOF

- `novelty=True`
- grid de `n_neighbors`
- contamination alineado a la tasa del proxy

### One-Class SVM

- `kernel='rbf'`
- `nu` cercano a la tasa base
- subsample controlado de train

## Repeticiones y estabilidad

`Isolation Forest` es estocastico. Para el resultado final se recomienda:

- una pasada de desarrollo con seed fija;
- corrida final con al menos 3 seeds (`42`, `52`, `62`) o justificacion explicita de por que una sola seed es suficiente;
- reporte del promedio y rango de variacion si la dispersion no es trivial.

## Decisiones obligatorias

- test set no participa en tuning;
- si `One-Class SVM` resulta demasiado costoso, se mantiene como baseline documentado con subsampling;
- no introducir nuevos algoritmos antes de cerrar la tesis base.
- toda comparacion final debe usar exactamente el mismo conjunto de filas y columnas.
- cualquier fila descartada por un modelo debe descartarse para todos en la comparacion principal.

## Artefactos

- `output/models/isolation_forest_best.joblib`
- `output/models/lof.joblib`
- `output/models/ocsvm.joblib`
- `output/metrics/grid_search_if.csv`
- `output/scores/if_val.parquet`
- `output/scores/if_test.parquet`
- `output/scores/lof_test.parquet`
- `output/scores/ocsvm_test.parquet`

## Tareas operativas

1. Reescribir trainer actual para modelos no supervisados.
2. Homogeneizar `score = -decision_function(X)` para que score alto signifique mayor anomalia.
3. Guardar scores junto a `id`, `created_at`, proxies y metadatos necesarios para reporting.
4. Medir tiempos y memoria de entrenamiento.
5. Guardar semilla, hash del schema y version del snapshot en cada artefacto.

## Gate de salida

No pasar a Fase 6 hasta que:

- existan scores de test de los tres modelos;
- `Isolation Forest` tenga hiperparametros fijados;
- el pipeline de scoring sea reproducible.
- exista registro de estabilidad por seed para el modelo principal o una excepcion metodologicamente justificada.
