# 07 Evaluacion e Hipotesis v4

## Evaluacion principal

Periodo principal:

- Octubre-diciembre 2025.

Periodo secundario:

- Septiembre-diciembre 2025 para comparabilidad legacy.

Gate A0 no es resultado final de tesis. Las metricas de esta fase deben recalcularse con pipeline final, USD normalizado y catalogo v4 completo.

## Metricas

Obligatorias:

- ROC AUC.
- Average Precision.
- AP/base.
- Precision@0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%.
- Recall@k.
- Enrichment factor@k.
- Bootstrap CI 95%.
- AUC mensual.
- Regla manual simple como baseline.
- Estabilidad mensual: AUC minimo y rango mensual.

Bootstrap CI 95%:

- 1,000 resamples.
- Estratificado por mes y clase Tipo A.
- Seed fija: 42.
- Intervalo por percentiles 2.5% y 97.5%.
- Reportar CI para ROC AUC, AP, AP/base y P@1%.

## Hipotesis v4

### HE1

El ranker supervisado v4 discrimina pagos Tipo A mejor que baseline aleatorio.

PASS si:

- AUC > 0.5 con CI 95% sin cruzar 0.5.

### HE2

El ranker V4-CLEAN alcanza utilidad operativa minima sin features circulares.

PASS si:

- AUC >= 0.70.
- AP/base >= 3.
- P@1% >= 40%.

Como SIMPLE-RULE ya alcanza AUC cercano a 0.70, HE2 debe discutirse con enfasis en P@1%, AP/base y carga operativa, no solo ROC AUC.

### HE3

El ranker v4 supera baselines relevantes, no solo al sistema no supervisado actual.

PASS si:

- Delta AUC >= 0.10 vs IF actual.
- Delta AUC >= 0.03 vs SIMPLE-RULE.
- Delta P@1% >= 10 puntos porcentuales vs SIMPLE-RULE, o mejora clara en AP/base.

### HE4

El resultado no depende de historial directo de reembolso ni de target encoding.

PASS si V4-CLEAN:

- AUC >= 0.75.
- AP/base >= 2.5.

`V4-CLEAN-BOOSTED` y `V4-FULL-SENS` se reportan como sensibilidad, no como prueba principal.

## Thresholds

Threshold operativo se calibra en validation, no en test.

Guardar:

- threshold por percentil.
- threshold por capacidad operativa mensual.
- expected alerts/month.
- expected precision.

## Estabilidad mensual

Para Oct, Nov y Dic 2025:

- cada mes debe tener AUC >= 0.75;
- rango mensual max(AUC)-min(AUC) <= 0.05;
- si no cumple, se reporta como drift temporal y se reevalua el gate operativo.

## Validacion humana del proxy

Antes de afirmar utilidad antifraude, realizar revision manual ciega:

- 100 pagos Tipo A de alto score;
- 100 pagos Tipo A aleatorios;
- 100 pagos no Tipo A de alto score.

Clasificacion minima:

- fraude/sospecha operacional;
- cancelacion legitima;
- error administrativo;
- indeterminado.

Sin esta muestra, el lenguaje debe limitarse a "riesgo de reembolso Tipo A", no "fraude".

## Artefactos

```text
output/v4/evaluation/
├── final_test_scores.parquet
├── legacy_test_scores.parquet
├── metrics_final.json
├── metrics_legacy.json
├── monthly_stability.csv
├── topk_metrics.csv
├── bootstrap_ci.csv
└── thresholds.json
```

## Gate final

No declarar v4 exitosa si no pasa HE2 en octubre-diciembre 2025.
