# Pipeline de Deteccion de Anomalias No Supervisado - Plan Maestro

## Indice de Fases

| Fase | Archivo | Descripcion | Gate |
|------|---------|-------------|------|
| [Fase 1](fase1_config_y_data.md) | config.py, loader.py, requirements.txt, .env.example | Config + extraccion + validacion | Gate A |
| [Fase 2](fase2_feature_engineering.md) | engineering.py (20 features vectorizadas) | Feature engineering con anti-leakage | Gate B |
| [Fase 3](fase3_preprocessor.md) | preprocessor.py (StandardScaler) | Preprocesamiento | - |
| [Fase 4](fase4_modelos.md) | trainer.py (IF + LOF + OC-SVM + grid search) | Entrenamiento con tuning justo | Gate C |
| [Fase 5](fase5_evaluacion.md) | metrics.py (HE1-HE4 + bootstrap + temporal) | Evaluacion estadistica rigurosa | Gate D |
| [Fase 6](fase6_reporting.md) | latex_tables.py, figures.py | Tablas LaTeX + figuras PDF/PNG | - |
| [Fase 7](fase7_orquestador.md) | run_pipeline.py | Orquestador con CLI y validacion | - |
| [Fase 8](fase8_notebooks.md) | notebooks EDA + resultados | Notebooks para Cap 2 y Cap 3 | - |
| [Fase 9](fase9_tests_y_cleanup.md) | tests, cleanup, CLAUDE.md | Tests + limpieza + reproducibilidad | Gate E |

## Estructura Final

```
ml-fraud-detector/
├── config/
│   └── config.py                        # REESCRIBIR
├── src/fraud_detector/
│   ├── data/
│   │   ├── clickhouse_connector.py      # MANTENER
│   │   └── loader.py                    # REESCRIBIR
│   ├── features/
│   │   ├── engineering.py               # REESCRIBIR
│   │   └── preprocessor.py             # REESCRIBIR
│   ├── models/
│   │   └── trainer.py                   # REESCRIBIR
│   ├── evaluation/
│   │   └── metrics.py                   # REESCRIBIR
│   ├── reporting/                       # NUEVO
│   │   ├── __init__.py
│   │   ├── latex_tables.py
│   │   └── figures.py
│   └── utils/
│       └── logger.py                    # MANTENER
├── notebooks/
│   ├── 01_eda_capitulo2.ipynb          # NUEVO
│   └── 02_resultados_capitulo3.ipynb   # NUEVO
├── scripts/                             # MANTENER
├── tests/
│   ├── test_config.py                  # ACTUALIZAR
│   ├── test_features.py                # NUEVO
│   ├── test_preprocessor.py            # NUEVO
│   ├── test_loader.py                  # NUEVO
│   ├── test_models.py                  # NUEVO
│   ├── test_metrics.py                 # NUEVO
│   ├── test_reporting.py               # NUEVO
│   └── test_integration.py             # NUEVO
├── run_pipeline.py                      # NUEVO
├── requirements.txt                     # ACTUALIZAR
├── requirements-lock.txt               # NUEVO (post thesis run)
├── .env.example                         # ACTUALIZAR
└── CLAUDE.md                            # ACTUALIZAR
```

**ELIMINAR:** `balancing.py`, `run_simple_rf.py`, `check_git_status.py`, `verify_setup.py`, `QUICKSTART.md`, `EXPERT_AUDIT.md`, `01_exploratory_analysis.ipynb`

## Flujo de Datos

```
ClickHouse (6.7M txns)
  |
  v
[Step 1] --> data/processed/{train,val,test}_raw.parquet
  |           output/manifests/dataset_manifest.json
  v
[Step 2] --> data/processed/{train,val,test}_features.parquet (20 cols + metadata)
  |           output/models/feature_engineer.joblib
  v
[Step 3] --> output/scores/X_{train,val,test}.npy
  |           output/models/scaler.joblib
  v
[Step 4] --> output/models/{isolation_forest,lof,ocsvm}.joblib
  |           output/grid_search_{if,lof,ocsvm}.csv
  |           output/models/best_params_{if,lof,ocsvm}.json
  v
[Step 5] --> output/scores/test_scores.parquet (id + created_at + 3 score columns)
  v
[Step 6] --> output/results.json (HE1-HE4, sensitivity, bootstrap, temporal)
  v
[Step 7] --> output/figures/shap_summary.{pdf,png}
  v
[Step 8] --> output/tables/table_3_*.tex + output/figures/*.{pdf,png}
```

## Correcciones Criticas Incorporadas

1. **Anti-leakage fix** (features 15, 16, 17): `.shift(1)` reemplazado por shifting dentro del grupo o `cumulative_nunique_shifted` O(n)
2. **Feature 18 fix**: account age usa `first_txn` del training set, no del split actual
3. **`user_id > 0`** y **`_peerdb_is_deleted = 0`**: filtros post-extraccion
4. **Contamination eliminada** del grid search (no afecta ranking): 256 -> 64 combos
5. **Grid search para LOF y OC-SVM**: comparacion justa en HE4
6. **Rank-biserial r corregida**: positiva cuando anomaly > normal
7. **Holm-Bonferroni** aplicada a HE1-HE4
8. **Estabilidad temporal**: AUC mensual en test set
9. **Multiple k values**: top-K evaluado a 1%, 2%, 5%, 10%
10. **Scores como parquet** con `id` y `created_at` (no `.npy` desnudo)
11. **Dataset manifest** para reproducibilidad
12. **Checkpoint/resume** en grid search
13. **Epsilon 1e-8** en ratios (era 0.01)
14. **Validacion de prerrequisitos** entre pasos
15. **Feature 15 renombrada** a `user_distinct_facilities_cumul` (expanding, no 30d)

## Gestion de Memoria

- Extraer splits por separado (nunca 6.7M a la vez)
- Parquet snappy (~3x compresion)
- Downcast SELECTIVO: solo cols seguras (no `id`, no `reversed_id`)
- Feature engineering por split
- OC-SVM subsample: 100K
- SHAP: 5K del test
- `gc.collect()` despues de `del`
- Pico estimado: ~5-6 GB durante rolling groupby

## Espacio en Disco

Total estimado: ~5.5 GB (parquets 2.4 GB, numpy 1.5 GB, modelos 1.5 GB, otros 100 MB)

## Tiempo Estimado (end-to-end)

| Paso | Tiempo |
|------|--------|
| 1. Extraccion | 5-15 min (depende de red) |
| 2. Feature engineering | 5-12 min por split |
| 3. Preprocesamiento | < 1 min |
| 4. Grid search IF (64 combos) | 10-15 min |
| 4. Grid search LOF (3 combos) | 15-30 min |
| 4. Grid search OC-SVM (6 combos) | 3 min |
| 5. Scoring | 2-5 min |
| 6. Evaluacion + bootstrap (3 modelos) | 30-50 min |
| 7. SHAP | 5-10 min |
| 8. Reportes | < 2 min |
| **Total** | **~1.5-2.5 horas** |
