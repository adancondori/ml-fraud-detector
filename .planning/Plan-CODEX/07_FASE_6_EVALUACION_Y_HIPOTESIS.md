# Fase 6. Evaluacion y Hipotesis

## Proposito

Responder OE3 y OE4 y cerrar formalmente HE1-HE4.

## Hipotesis y evidencia requerida

### HE1

Separacion de scores entre proxy+ y proxy-.

Metricas:

- Mann-Whitney U
- p-value
- rank-biserial correlation

### HE2

Capacidad discriminativa del modelo principal.

Metricas:

- AUC-ROC
- Average Precision
- comparacion contra tasa base

### HE3

Concentracion de casos proxy+ en top-k.

Metricas:

- Precision@5%
- Recall@5%
- Enrichment Factor

### HE4

Comparacion entre `Isolation Forest`, `LOF` y `One-Class SVM`.

Metricas:

- AUC-ROC
- AP
- Precision@5%
- EF

## Regla de comparacion justa

La comparacion principal debe cumplir:

- mismo snapshot;
- mismas filas;
- mismo set de features;
- mismo proxy;
- misma ventana temporal;
- misma regla de orientacion del score.

Sin eso, HE4 queda debil.

## Bootstrap

Calcular IC 95% para:

- AUC-ROC
- AP
- Precision@5%
- EF

Numero de iteraciones recomendado:

- `1000`

## Comparaciones adicionales recomendadas

- diferencia de AUC con bootstrap pareado;
- diferencia de AP con bootstrap pareado;
- tabla de runtime y memoria para contexto ingenieril.

No son estrictamente necesarias para aprobar la tesis, pero fortalecen la defensa.

## Visualizaciones obligatorias

- curva ROC
- curva PR
- boxplot o violin de scores por proxy
- grafico de enrichment por top-k
- tabla comparativa de modelos

## Tareas operativas

1. Reescribir `metrics.py` como evaluador de hipotesis.
2. Implementar estadistica y bootstrap separadas del training.
3. Consolidar un archivo `final_results.json`.
4. Generar tablas LaTeX directamente desde resultados.
5. Guardar tambien un `comparison_long.csv` consumible por notebooks y LaTeX.

## Salidas

- `output/metrics/final_results.json`
- `output/tables/he1_stats.tex`
- `output/tables/he2_he3_metrics.tex`
- `output/tables/he4_model_comparison.tex`
- `output/figures/roc_comparison.pdf`
- `output/figures/pr_comparison.pdf`
- `output/figures/score_distribution.pdf`

## Gate de salida

No pasar a Fase 7 hasta que:

- cada hipotesis tenga estado `respaldada` o `rechazada`;
- existan IC bootstrap;
- las tablas y figuras principales ya esten exportadas.
- exista comparacion justa y documentada entre modelos.
