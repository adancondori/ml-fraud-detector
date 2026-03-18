# Plan Maestro Unificado — PLAN-FINAL

> Sintesis de Plan-CODEX (estrategia, gobernanza, edge cases) y Plan-CLAUDE (implementacion detallada, codigo, correcciones anti-leakage).

## Objetivo final

Completar la implementacion empirica de la tesis de deteccion de anomalias transaccionales en pagos digitales usando Isolation Forest, comparada contra LOF y One-Class SVM, con resultados reproducibles e integrables en `Tesis-Latex`.

## Resultado esperado

1. Snapshot de datos 2025 deduplicado y congelado con warm history.
2. Pipeline reproducible de extraccion a reporting (`run_pipeline.py`).
3. Tablas LaTeX y figuras PDF/PNG para Capitulo 2 y Capitulo 3.
4. Hipotesis HE1-HE4 contestadas con evidencia y bootstrap CI 95%.
5. Analisis de sensibilidad (proxy estricto/amplio, Feature #17, estabilidad temporal).
6. Analisis post-hoc de anomalias por centro, actor operativo y moneda (concentracion de descuentos).
7. Repositorio limpio, probado y defendible.

## Indice de fases

| Fase | Archivo | Descripcion | Gate |
|------|---------|-------------|------|
| 0 | `01_CONTRATO_ALCANCE.md` | Contrato tesis-codigo, definiciones fijas | — |
| 1 | `02_DATOS_SNAPSHOT.md` | Extraccion ClickHouse, snapshot, warm history | Gate A |
| 2 | `03_EDA_CAPITULO2.md` | Diagnostico y EDA para OE2 | — |
| 3 | `04_FEATURE_ENGINEERING.md` | 20 features oficiales + variante 19 | Gate B |
| 4 | `05_PREPROCESAMIENTO.md` | StandardScaler fit en train | — |
| 5 | `06_MODELADO_TUNING.md` | IF + LOF + OC-SVM con grid search | Gate C |
| 6 | `07_EVALUACION_HIPOTESIS.md` | HE1-HE4, bootstrap, temporal | Gate D |
| 7 | `08_SENSIBILIDAD.md` | Proxy, Feature #17, SHAP, per-status, **post-hoc centro/actor/moneda** | — |
| 8 | `09_REPORTING.md` | Tablas LaTeX + figuras PDF/PNG | — |
| 9 | `10_ORQUESTADOR.md` | `run_pipeline.py` con CLI | — |
| 10 | `11_TESTS_CLEANUP_INTEGRACION.md` | Tests, limpieza, integracion tesis | Gate E |

## Documentos transversales

| Doc | Archivo | Proposito |
|-----|---------|-----------|
| T1 | `A1_ETL_LINEAGE_EDGE_CASES.md` | ETL idempotente, lineage, 20 edge cases |
| T2 | `A2_PROTOCOLO_RUNBOOK.md` | Protocolo de experimentos + runbook de ejecucion |
| T3 | `A3_RIESGOS_CHECKLIST.md` | Riesgos, decisiones cerradas, checklist por fase |
| T4 | `A4_GOBERNANZA_PRIVACIDAD_CONTRATOS.md` | Gobernanza de datos, privacidad, data contracts y contratos de artefactos |
| T5 | `A5_AUDITORIA_END_TO_END.md` | Auditoria final del plan y criterios de suficiencia end-to-end |

## Arquitectura objetivo del repositorio

```
ml-fraud-detector/
├── config/
│   └── config.py                        # REESCRIBIR (Pydantic Settings, sin supervisado)
├── src/fraud_detector/
│   ├── data/
│   │   ├── clickhouse_connector.py      # MANTENER (con ajuste menor)
│   │   └── loader.py                    # REESCRIBIR → DataManager
│   ├── features/
│   │   ├── engineering.py               # REESCRIBIR → FeatureEngineer (20 features)
│   │   └── preprocessor.py             # REESCRIBIR → UnsupervisedPreprocessor
│   ├── models/
│   │   └── trainer.py                   # REESCRIBIR → AnomalyModelTrainer
│   ├── evaluation/
│   │   ├── metrics.py                   # REESCRIBIR → HypothesisEvaluator
│   │   └── posthoc_analysis.py          # NUEVO → PostHocAnalyzer (centro/actor/moneda)
│   ├── reporting/                       # NUEVO
│   │   ├── __init__.py
│   │   ├── latex_tables.py             # ThesisTableGenerator
│   │   └── figures.py                  # ThesisFigureGenerator
│   └── utils/
│       └── logger.py                    # MANTENER
├── notebooks/
│   ├── 01_eda_capitulo2.ipynb          # NUEVO
│   └── 02_resultados_capitulo3.ipynb   # NUEVO
├── scripts/
│   ├── verify_counts.py                # NUEVO
│   └── inspect_clickhouse.py           # ACTUALIZAR
├── tests/                               # ACTUALIZAR + NUEVOS
├── run_pipeline.py                      # NUEVO (orquestador principal)
├── data/processed/                      # Parquets (gitignored)
├── output/                              # Tablas, figuras, modelos, scores (gitignored)
├── requirements.txt                     # ACTUALIZAR (sin supervisado)
├── .env.example                         # ACTUALIZAR
├── CLAUDE.md                            # ACTUALIZAR
└── README.md                            # ACTUALIZAR
```

**ELIMINAR:** `balancing.py`, `run_simple_rf.py`, `check_git_status.py`, `verify_setup.py`, `QUICKSTART.md`, `EXPERT_AUDIT.md`, `01_exploratory_analysis.ipynb`

## Flujo de datos

```
ClickHouse (6.7M txns) + Warm History (Dic 2024)
  │
  ▼
[Fase 1] → data/processed/{warm,train,val,test}_raw.parquet
            output/manifests/dataset_manifest.json
  │
  ▼
[Fase 3] → data/processed/{train,val,test}_features.parquet (20 cols + metadata)
            output/models/feature_engineer.joblib
  │
  ▼
[Fase 4] → output/scores/X_{train,val,test}.npy
            output/models/scaler.joblib
  │
  ▼
[Fase 5] → output/models/{isolation_forest,lof,ocsvm}.joblib
            output/grid_search_{if,lof,ocsvm}.csv
            output/models/best_params_{if,lof,ocsvm}.json
  │
  ▼
[Fase 6] → output/scores/test_scores.parquet (id + created_at + 3 scores)
            output/results.json (HE1-HE4, bootstrap, temporal)
  │
  ▼
[Fase 7] → output/results_sensitivity.json
            output/results_posthoc.json
            output/figures/shap_summary.{pdf,png}
  │
  ▼
[Fase 8] → output/tables/table_3_*.tex (incluyendo 3.20-3.23 post-hoc)
            output/figures/*.{pdf,png} (incluyendo post-hoc centro/actor/moneda)
```

## Gates obligatorios

### Gate A — Universo de estudio

El snapshot debe reproducir:

| Concepto | Valor esperado |
|----------|---------------|
| N total depurado | ~6,784,695 |
| Proxy estricto | ~429,442 (6.33%) |
| Proxy amplio | ~512,609 (7.55%) |
| Train (Ene-Jun) | ~3,137,086 |
| Val (Jul-Ago) | ~1,130,118 |
| Test (Sep-Dic) | ~2,517,491 |

No continuar si los conteos divergen mas de ±1%.

### Gate B — Validez metodologica (anti-leakage)

No se acepta ninguna feature que:
- use `status` de reembolso como predictor;
- use informacion futura (la fila actual o posteriores);
- dependa de la propia fila dentro de la ventana;
- convierta el proxy en senal circular.

Ademas, las features de `val` y `test` deben arrastrar historia previa correctamente (warm history).

### Gate C — Independencia del test

El test set NO se usa para seleccionar hiperparametros ni decisiones de ingenieria.

### Gate D — Robustez del resultado principal

Si el resultado colapsa al remover `user_reversal_ratio_30d` (delta AUC >= 0.02), el modelo de 20 features no puede presentarse como hallazgo principal.

### Gate E — Reporting y tests

No cerrar Cap 2/3 con tablas manuales; todo artefacto debe salir del pipeline. Tests automaticos pasan.

## Principios de implementacion

1. `FINAL` no es opcional para reproducir la tesis.
2. El proxy se usa solo para evaluacion, nunca para entrenar.
3. Toda feature debe ser auditada contra leakage temporal y circularidad.
4. Pipeline offline y reproducible, no API productiva.
5. Todo resultado de Cap 2 y Cap 3 sale del codigo.
6. El test set temporal permanece intocable hasta evaluacion final.
7. La comparacion entre modelos usa misma base, features, proxy y split.
8. Cada fase deja artefactos serializados para no repetir extracciones.
9. Score alto = mas anomalo (convencion: `-decision_function(X)`).

## Correcciones criticas incorporadas (de ambos planes)

| # | Correccion | Origen |
|---|-----------|--------|
| 1 | Anti-leakage features 15, 16: `cumulative_nunique_shifted` O(n) en vez de `expanding().apply(nunique)` | CLAUDE |
| 2 | Feature 17 shift(1) dentro del grupo, no global | CLAUDE |
| 3 | Feature 18: `user_account_age_days` usa `first_txn` del training set | CLAUDE |
| 4 | Filtros post-extraccion: `user_id > 0` y `_peerdb_is_deleted = 0` | CODEX |
| 5 | Contamination eliminada del grid search IF (no afecta ranking): 256 → 64 combos | CLAUDE |
| 6 | Grid search para LOF y OC-SVM (comparacion justa en HE4) | CLAUDE |
| 7 | Rank-biserial r corregida: positiva cuando anomaly > normal | CLAUDE |
| 8 | Holm-Bonferroni aplicada a HE1-HE4 | CLAUDE |
| 9 | Estabilidad temporal: AUC mensual en test set | CLAUDE |
| 10 | Multiples k: top-K evaluado a 1%, 2%, 5%, 10% | CLAUDE |
| 11 | Scores como parquet con `id` y `created_at` | CLAUDE |
| 12 | Warm history obligatoria (Dic 2024) para features con ventana | CODEX |
| 13 | Epsilon 1e-8 en ratios (era 0.01) | CLAUDE |
| 14 | Checkpoint/resume en grid search | CLAUDE |
| 15 | Feature 15 renombrada a `user_distinct_facilities_cumul` | CLAUDE |
| 16 | Multi-seed recomendado (42, 52, 62) con justificacion si solo una | CODEX |
| 17 | Sanity baselines (random, monto, z-score) para validacion interna | CODEX |
| 18 | Deduplicacion con `FINAL` obligatoria | CODEX |
| 19 | Dataset manifest JSON por split con trazabilidad | CODEX+CLAUDE |
| 20 | Prerequisite validation entre pasos del pipeline | CLAUDE |
| 21 | SQL canonico agrega `currency` y `paid_by_manager` para analisis post-hoc | CLAUDE |
| 22 | Analisis post-hoc: concentracion de anomalias por centro, manager, moneda y descuentos | CLAUDE |
| 23 | Tablas 3.20-3.23 y figuras post-hoc en Fase 8 Reporting | CLAUDE |
| 24 | Gate de identidad del actor agregado al post-hoc de managers/descuentos | CODEX |
| 25 | Gobernanza, privacidad y data contracts agregados al plan final | CODEX |

## Estimaciones

### Tiempo (end-to-end)

| Paso | Tiempo |
|------|--------|
| Extraccion (3 splits + warm) | 5-15 min |
| Feature engineering (por split) | 5-12 min |
| Preprocesamiento | < 1 min |
| Grid search IF (64 combos) | 10-15 min |
| Grid search LOF (3 combos) | 15-30 min |
| Grid search OC-SVM (6 combos) | 3 min |
| Scoring test | 2-5 min |
| Evaluacion + bootstrap (3 modelos) | 30-50 min |
| SHAP | 5-10 min |
| Sensibilidad completa | 20-30 min |
| Reportes | < 2 min |
| **Total** | **~2-3 horas** |

### Disco

| Componente | Tamano |
|-----------|--------|
| Parquets raw (3 splits + warm) | ~1.8 GB |
| Feature parquets | ~900 MB |
| Arrays numpy | ~1.5 GB |
| Modelos (IF ~200MB, LOF ~1GB, OC-SVM ~100MB) | ~1.5 GB |
| Otros (scores, tablas, figuras) | ~100 MB |
| **Total** | **~5.5 GB** |

### Memoria pico

~5-6 GB durante rolling groupby en feature engineering.

## Cronograma sugerido (10 semanas)

| Semanas | Fases | Meta |
|---------|-------|------|
| 1-2 | 0, 1 | Contrato cerrado, snapshot congelado |
| 3-4 | 2, 3 | EDA completa, 20 features implementadas |
| 5-6 | 4, 5 | Matrices listas, 3 modelos entrenados |
| 7-8 | 6, 7 | HE1-HE4 cerradas, sensibilidad completa |
| 9-10 | 8, 9, 10 | Reporting, tests, integracion Tesis-LaTeX |

## Secuencia minima viable

1. Cerrar contrato tesis-codigo.
2. Congelar snapshot con `FINAL` + warm history.
3. Completar EDA para Capitulo 2.
4. Implementar 20 features con anti-leakage.
5. Entrenar IF, LOF, OC-SVM.
6. Evaluar HE1-HE4 en test temporal.
7. Ejecutar sensibilidad.
8. Exportar tablas y figuras.
9. Pasar checklist de edge cases.
10. Integrar resultados en Tesis-LaTeX.

Si uno de estos diez hitos falla, la tesis aun no esta lista.
