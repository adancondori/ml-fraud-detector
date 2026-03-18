# Protocolo de Experimentos

## Proposito

Cerrar la forma exacta en que se ejecutan, registran y comparan los experimentos.

## Modos de ejecucion

### Modo 1. Smoke

Objetivo:

- probar integridad del pipeline;
- usar muestra pequena;
- validar que todas las etapas corren.

No usar para reportar resultados de tesis.

### Modo 2. Full thesis

Objetivo:

- generar resultados oficiales;
- usar snapshot completo;
- exportar artefactos finales.

## Versiones y reproducibilidad

Guardar por corrida:

- timestamp;
- git SHA o equivalente;
- version del snapshot;
- seed;
- schema hash;
- hiperparametros;
- hostname o resumen de hardware;
- duracion por etapa.

## Seeds

### Desarrollo

- seed unica: `42`

### Corrida final

- al menos `42`, `52`, `62`
- o justificar por escrito una sola seed si el costo computacional obliga

## Protocolo del tuning

### Isolation Forest

- tuning solo en validation;
- proxy estricto como criterio principal;
- AUC-ROC primaria;
- AP secundaria;
- runtime como desempate tecnico si el desempeno es muy parecido.

### LOF

- tuning ligero, no exhaustivo si el costo es prohibitivo;
- misma base de validacion.

### One-Class SVM

- tuning minimo y transparente;
- subsample fijo y versionado.

## Baselines de sanidad

No necesariamente para la tesis principal, pero si para validar que el pipeline tiene sentido:

- ranking aleatorio;
- ranking por monto;
- ranking por `user_reversal_ratio_30d` si se usa solo como sanity check fuera del modelo de 19 features;
- z-score simple de monto.

Si `Isolation Forest` no supera al menos estos sanity baselines en la corrida interna, hay que investigar antes de reportar.

## Aceptacion minima de la corrida final

- snapshot correcto;
- scores guardados;
- HE1-HE4 ejecutadas;
- sensibilidad ejecutada;
- tablas y figuras exportadas;
- checklist de edge cases pasada.

## Artefactos obligatorios

- `run_manifest_<timestamp>.json`
- `grid_search_if.csv`
- `model_params_<timestamp>.json`
- `runtime_metrics_<timestamp>.json`

## Prohibiciones

- cambiar el proxy despues de mirar test sin actualizar contrato;
- ajustar features mirando el test set;
- comparar modelos sobre datasets distintos;
- regenerar tablas manualmente.
