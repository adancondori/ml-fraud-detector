# 06 Modelado y Tuning v4

## Modelo principal

Usar `HistGradientBoostingClassifier` de scikit-learn como primera implementacion. El modelo principal es `V4-CLEAN`: sin historial directo de reembolso y sin target encoding.

## Pre-benchmark obligatorio

Antes de Fase 1, crear y ejecutar:

```bash
python scripts/run_hgb_benchmark.py --variant clean-strict --period legacy_sep_dec
```

Debe guardar:

```text
output/v4/benchmarks/baseline_hgb_clean_strict.json
output/v4/benchmarks/baseline_hgb_clean_strict_scores.parquet
output/v4/benchmarks/feature_list_clean_strict.json
output/v4/benchmarks/simple_rule_baseline.json
```

Gate A0 ejecutado:

| Variante | Val AUC | Test Sep-Dic AUC | Test Oct-Dic AUC | P@1% Oct-Dic | AP/base Oct-Dic |
|---|---:|---:|---:|---:|---:|
| V4-CLEAN-NO-RU | 0.8420 | 0.8301 | 0.8285 | 71.3% | 5.64 |
| V4-CLEAN strict as-of RU | 0.8460 | 0.8352 | 0.8339 | 73.4% | 5.78 |
| SIMPLE-RULE | 0.7000 | 0.7049 | 0.7052 | 35.8% | 2.29 |

El POC no estricto con `reservations_users` en estado actual alcanzo AUC ~0.83 y P@1% ~81%, pero ya no es necesario como cifra central. La variante strict as-of supera los gates y la variante NO-RU confirma que el resultado no depende de participantes.

Parametros iniciales:

```text
max_iter: [120, 180, 240]
learning_rate: [0.03, 0.05, 0.08]
max_leaf_nodes: [15, 31, 63]
l2_regularization: [0.0, 0.05, 0.2]
min_samples_leaf: [20, 50, 100]
```

## Entrenamiento

Modo rapido:

- Todos los positivos train.
- Negativos muestreados 1:5.
- Seed fija 42.
- Muestreo estratificado por mes y clase si no se usa train completo.

Modo final:

- Entrenar con todos los datos o muestreo 1:10 si memoria/tiempo lo exige.
- Reportar sensibilidad de sampling.
- Guardar indices o hash de muestreo en `training_manifest.json`.

## Baselines

Mantener:

- Isolation Forest actual.
- Isolation Forest con features v4 numericas.
- LOF.
- One-Class SVM.
- Logistic Regression supervisada.
- Amount z-score.
- Random baseline.

## Variantes v4

| Variante | Uso | Benchmark Sep-Dic 2025 |
|---|---|---:|
| V4-CLEAN strict as-of RU | Principal inicial, no circular | AUC 0.8352 |
| V4-CLEAN-NO-RU | Control sin participantes | AUC 0.8301 |
| V4-CLEAN-BOOSTED | Sensibilidad/operativo con target encoding | AUC ~0.818 |
| V4-FULL-SENS | Sensibilidad con historial de refund | AUC ~0.822 |
| V4-LEGACY-31 IF | Baseline historico | AUC ~0.576 Tipo A |
| SIMPLE-RULE | Baseline manual interpretable | AUC 0.7049 |

El benchmark V4-CLEAN strict incluye reserva/token/cupon basicos y cierra viabilidad. La implementacion final todavia debe reemplazar montos nominales por USD normalizado.

## Baseline manual simple

La comparacion contra IF no basta, porque HGB supervisado vs IF no supervisado es metodologicamente trivial. Agregar regla manual:

```python
score = 0
score += source_enum in ("pbp_app", "white_label_app")
score += payment_method in ("prepaid", "user_package")
score += reservation_lead_days >= 8
score += admin_booked is False
```

Medicion preliminar Oct-Dic 2025:

- AUC Sep-Dic: 0.7049.
- AUC Oct-Dic: 0.7052.
- P@1% Oct-Dic: 35.8%.

HGB-CLEAN debe superar esta regla por margen util, no solo a IF.

## Seleccion

Solo validation define:

- hiperparametros;
- variantes V4-CLEAN, V4-CLEAN-BOOSTED y V4-FULL-SENS;
- threshold operativo;
- top-k recomendado.

Septiembre no define hiperparametros si sera usado como discovery. Si influye en features, test final debe ser octubre-diciembre.

## Artefactos

```text
output/v4/models/
├── risk_ranker_hgb.joblib
├── logistic_baseline.joblib
├── isolation_forest_baseline.joblib
├── best_params_hgb.json
├── training_manifest.json
└── validation_scores.parquet
```

## Gate

Validation debe cumplir:

- AUC >= 0.78.
- AP/base >= 3.5.
- P@1% >= 35%.

Si no cumple, revisar features antes de ampliar grid.
