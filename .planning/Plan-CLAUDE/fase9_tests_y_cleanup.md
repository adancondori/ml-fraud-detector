# Fase 9: Tests, Cleanup y Actualizaciones

---

## 9.1 Tests

### test_features.py (ENHANCED from original)

- `test_fit_sets_statistics`
- `test_transform_produces_20_features`
- `test_transform_no_nans`
- `test_transform_without_fit_raises`
- `test_anti_leakage_velocity` (primera transaccion de un usuario tiene count=0)
- `test_facility_avg_from_train_only`
- `test_unseen_facility_uses_global_avg`
- `test_account_age_uses_train_first_txn` (NEW -- verifica fix de feature 18)
- `test_cumulative_nunique_no_cross_group_leakage` (NEW -- verifica fix de features 15/16)

### test_preprocessor.py (NEW -- no existia)

- `test_fit_transform_shape`
- `test_output_dtype_float32`
- `test_transform_without_fit_raises`
- `test_save_load_roundtrip` (verificar mismo output despues de load)
- `test_feature_names_preserved`
- `test_missing_column_raises_valueerror`

### test_loader.py (NEW -- no existia)

- `test_assign_proxy_labels_strict`
- `test_assign_proxy_labels_wide`
- `test_assign_proxy_labels_invalid_raises`
- `test_downcast_preserves_data_integrity`
- `test_load_split_missing_file_clear_error`

### test_models.py (ENHANCED)

- `test_train_isolation_forest`
- `test_train_lof`
- `test_train_ocsvm`
- `test_scoring_shape`
- `test_anomalies_score_higher`
- `test_grid_search_returns_results` (ajustado para no contamination)
- `test_save_load_scoring_equivalence` (NEW -- verificar mismos scores despues de load)

### test_metrics.py (ENHANCED)

- `test_he1_perfect_separation`
- `test_he2_perfect_auc`
- `test_he3_perfect_enrichment`
- `test_he2_random_auc`
- `test_bootstrap_ci_coherent` (lower <= mean <= upper)
- `test_compare_models`
- `test_full_evaluation_keys`
- `test_edge_case_all_same_class` (NEW)
- `test_edge_case_constant_scores` (NEW)
- `test_holm_bonferroni_correction` (NEW)
- `test_topk_multiple_k_values` (NEW)

### test_config.py (UPDATE existing)

- Eliminar test de `test_size` (campo removido)
- Agregar tests para `grid_n_estimators_list`, `grid_max_features_list` (manejo de "auto")
- Agregar tests para propiedades de directorios
- Agregar tests para parsing de `proxy_list`

### test_integration.py (NEW -- marcado `@pytest.mark.slow`)

- Crea dataset sintetico de 200 filas
- Ejecuta feature engineering, preprocessing, entrenamiento, scoring, evaluacion
- Verifica que todos los outputs se crean con shapes esperadas
- NO requiere ClickHouse

### test_reporting.py (NEW -- smoke tests)

- `test_table_generates_valid_latex` (contiene `\begin{table}`)
- `test_figure_generates_pdf` (archivo existe, tamano > 0)
- `test_latex_special_chars_escaped`

---

## 9.2 Archivos a ELIMINAR

| Archivo | Razon |
|---------|-------|
| `src/fraud_detector/data/balancing.py` | SMOTE/resampling, no aplica a no supervisado |
| `run_simple_rf.py` | Demo de Random Forest supervisado |
| `check_git_status.py` | Script de utilidad obsoleto |
| `verify_setup.py` | Script de verificacion obsoleto |
| `QUICKSTART.md` | Documentacion del scaffolding viejo |
| `EXPERT_AUDIT.md` | Auditoria del scaffolding viejo |
| `notebooks/01_exploratory_analysis.ipynb` | Reemplazado por `01_eda_capitulo2.ipynb` |

---

## 9.3 Archivos a ACTUALIZAR

### .gitignore -- adiciones:

```
output/
data/processed/*.parquet
*.npy
*.joblib
```

### Makefile -- cambios:

- **Eliminar**: `make mlflow`, `make git-check`, `make git-status`, `make verify`
- **Agregar**:
  - `make pipeline` -> `python run_pipeline.py`
  - `make pipeline-quick` -> `python run_pipeline.py --fast`
  - `make clean-output` -> `rm -rf output/ data/processed/*.parquet`

### pyproject.toml:

- Actualizar version a `1.0.0`
- Actualizar descripcion

### setup.py:

- Actualizar version, descripcion
- Actualizar `install_requires` (eliminar xgboost, lightgbm, mlflow, imbalanced-learn, etc.)

### .pre-commit-config.yaml:

- Eliminar `pandas-stubs` y `types-requests` de `mypy` `additional_dependencies`

### .github/workflows/ci.yml:

- Instalar desde `requirements.txt` actualizado
- Saltar tests de integracion (`-m "not integration"`)
- Actualizar versiones de actions

### CLAUDE.md:

- Nueva descripcion (deteccion de anomalias no supervisada)
- Nuevo stack (scikit-learn solamente, sin XGBoost/LightGBM/MLflow)
- Nueva estructura (reporting/, run_pipeline.py)
- Nuevos comandos (`python run_pipeline.py`, `--step`, `--skip-grid-search`)
- Variables de entorno actualizadas

### Archivos `__init__.py`:

- `fraud_detector/__init__.py`: version `"1.0.0"`
- `data/__init__.py`: exportar `DataManager`
- `features/__init__.py`: exportar `FeatureEngineer`, `FEATURE_NAMES`, `UnsupervisedPreprocessor`
- `models/__init__.py`: exportar `AnomalyModelTrainer`
- `evaluation/__init__.py`: exportar `HypothesisEvaluator`
- `reporting/__init__.py`: exportar `ThesisTableGenerator`, `ThesisFigureGenerator`

---

## 9.4 Reproducibilidad

- Despues del run final de la tesis: `pip freeze > requirements-lock.txt`
- Documentar version exacta de Python en `CLAUDE.md`
- Establecer `PYTHONHASHSEED=42` en `run_pipeline.py`
- Todos los random seeds explicitamente establecidos

---

## 9.5 Estimacion de espacio en disco

| Componente | Tamano estimado |
|------------|-----------------|
| Raw parquets (3 splits) | ~1.5 GB |
| Feature parquets | ~900 MB |
| Numpy arrays | ~1.5 GB |
| Modelos (IF ~200 MB, LOF ~1 GB, OC-SVM ~100 MB) | ~1.5 GB |
| Otros outputs (JSON, CSV, figuras, tablas) | ~50 MB |
| **Total** | **~5.5 GB** |

---

## Gate E: Reporting & Tests

1. `pytest tests/ -v` -- todos pasan
2. `pytest tests/test_integration.py -v` -- pasa (sin necesidad de ClickHouse)
3. Todas las tablas LaTeX compilan con `pdflatex`
4. Todas las figuras son PDF + PNG, tamano > 0
5. `results.json` es JSON valido con estructura completa
6. `CLAUDE.md` actualizado
7. No quedan archivos obsoletos
8. `.gitignore` cubre todas las rutas de output
