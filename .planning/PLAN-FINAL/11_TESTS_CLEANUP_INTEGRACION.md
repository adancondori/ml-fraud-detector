# Fase 10: Tests, Limpieza e Integración con la Tesis

Fase final que consolida la calidad del código, elimina artefactos obsoletos
y conecta los resultados del pipeline con el documento de tesis en LaTeX.

---

## Workflow TDD: Red-Green-Refactor

Toda implementación en este proyecto debe seguir el ciclo **Red-Green-Refactor** de Test-Driven Development:

1. **Red**: Escribir el test primero. El test debe fallar porque la funcionalidad aún no existe.
2. **Green**: Implementar el código mínimo necesario para que el test pase. Sin optimizar, sin refactorizar.
3. **Refactor**: Limpiar el código (eliminar duplicación, mejorar nombres, extraer funciones) manteniendo los tests en verde.

### Orden de implementación por fase

Para cada fase del pipeline (02-09), el flujo es:

```
1. Leer la especificación del plan (e.g., 02_DATA_EXTRACTION.md)
2. Crear/actualizar el archivo de test correspondiente (e.g., test_loader.py)
3. Escribir TODOS los test contracts especificados → ejecutar pytest → todo en ROJO
4. Implementar el módulo → ejecutar pytest → todo en VERDE
5. Refactorizar → ejecutar pytest → sigue en VERDE
6. Commit: "feat(fase-N): implementar [módulo] con tests"
```

### Regla de oro

> **Nunca escribir código de producción sin un test que lo exija.**

Si descubres un bug o un edge case durante la implementación, **primero** escribe un test que lo reproduzca (rojo), **después** corrígelo (verde).

---

## Fixtures compartidas (`conftest.py`)

El archivo `tests/conftest.py` debe definir fixtures reutilizables por todos los módulos de test:

```python
import numpy as np
import pandas as pd
import pytest

from fraud_detector.features.engineering import FeatureEngineer


@pytest.fixture
def sample_transactions():
    """200-row synthetic dataset mimicking production schema for unit tests.

    Includes columns: id, user_id, facility_id, amount, currency, status,
    created_at, payment_method, card_brand, card_country, plus derived fields.
    Contains ~6% proxy-positive rows (status='totally_refunded').
    """
    rng = np.random.default_rng(42)
    n = 200
    statuses = rng.choice(
        ["approved", "totally_refunded", "refunded_to_credit", "declined"],
        size=n,
        p=[0.85, 0.04, 0.02, 0.09],
    )
    return pd.DataFrame({
        "id": range(n),
        "user_id": rng.choice(range(30), size=n),
        "facility_id": rng.choice(range(5), size=n),
        "amount": rng.lognormal(mean=4.0, sigma=1.0, size=n).round(2),
        "currency": "USD",
        "status": statuses,
        "created_at": pd.date_range("2024-01-01", periods=n, freq="h"),
        "payment_method": rng.choice(["card", "wallet"], size=n),
        "card_brand": rng.choice(["visa", "mastercard", "amex"], size=n),
        "card_country": rng.choice(["US", "MX", "BR"], size=n),
    })


@pytest.fixture
def fitted_feature_engineer(sample_transactions):
    """Pre-fitted FeatureEngineer for testing transform without re-fitting."""
    fe = FeatureEngineer()
    fe.fit(sample_transactions)
    return fe


@pytest.fixture
def sample_scores():
    """Synthetic anomaly scores array (200 elements) for metric tests.

    Higher scores indicate more anomalous. ~6% of corresponding proxy labels are True.
    """
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=200)


@pytest.fixture
def sample_proxy_labels():
    """Binary proxy labels matching sample_scores fixture. ~6% positive rate."""
    rng = np.random.default_rng(42)
    return rng.choice([0, 1], size=200, p=[0.94, 0.06])
```

---

## Tests

### test_features.py (MEJORADO)

| Test                                      | Propósito                                                  |
|-------------------------------------------|------------------------------------------------------------|
| `test_fit_sets_statistics`                | Verificar que `fit()` calcula y almacena estadísticas      |
| `test_transform_produces_20_features`     | Salida tiene exactamente 20 columnas de features           |
| `test_transform_no_nans`                  | Sin NaN en la salida transformada                          |
| `test_transform_without_fit_raises`       | Error claro si se transforma sin ajustar primero           |
| `test_anti_leakage_velocity`              | Primera transacción de un usuario tiene `txn_count=0`      |
| `test_facility_avg_from_train_only`       | Promedio por facility calculado exclusivamente del train    |
| `test_unseen_facility_uses_global_avg`    | Facility no vista en train usa promedio global              |
| `test_account_age_uses_train_first_txn`   | Edad de cuenta referencia primera txn del training set     |
| `test_cumulative_nunique_no_cross_group_leakage` | Conteo acumulativo no mezcla entre grupos            |
| `test_split_boundary`                     | Fronteras correctas: Jun30→Jul1, Aug31→Sep1                |

### test_preprocessor.py (NUEVO)

| Test                                      | Propósito                                                  |
|-------------------------------------------|------------------------------------------------------------|
| `test_fit_transform_shape`                | Dimensiones correctas tras fit_transform                   |
| `test_output_dtype_float32`               | Salida en float32 para eficiencia                          |
| `test_transform_without_fit_raises`       | Error si se transforma sin fit                             |
| `test_save_load_roundtrip`                | Guardar y cargar produce resultados idénticos              |
| `test_feature_names_preserved`            | Nombres de features se mantienen tras serialización        |
| `test_missing_column_raises`              | Error claro si faltan columnas esperadas                   |

### test_loader.py (NUEVO)

| Test                                      | Propósito                                                  |
|-------------------------------------------|------------------------------------------------------------|
| `test_assign_proxy_labels_strict`         | Proxy estricto asigna correctamente                        |
| `test_assign_proxy_labels_wide`           | Proxy amplio incluye categorías adicionales                |
| `test_assign_proxy_labels_invalid_raises` | Tipo de proxy inválido lanza error                         |
| `test_downcast_preserves_data_integrity`  | Downcast de tipos no pierde datos                          |

### test_models.py (MEJORADO)

| Test                                      | Propósito                                                  |
|-------------------------------------------|------------------------------------------------------------|
| `test_train_isolation_forest`             | IF se entrena sin errores                                  |
| `test_train_lof`                          | LOF se entrena sin errores                                 |
| `test_train_ocsvm`                        | OC-SVM se entrena sin errores                              |
| `test_scoring_shape`                      | Scores tienen dimensión correcta                           |
| `test_anomalies_score_higher`             | Anomalías conocidas reciben scores más altos               |
| `test_grid_search_returns_results`        | Grid search produce DataFrame de resultados                |
| `test_save_load_scoring_equivalence`      | Modelo cargado produce mismos scores que original          |

### test_metrics.py (MEJORADO)

| Test                                      | Propósito                                                  |
|-------------------------------------------|------------------------------------------------------------|
| `test_he1_perfect_separation`             | HE1 pasa con separación perfecta                           |
| `test_he2_perfect_auc`                    | AUC-ROC = 1.0 con discriminación perfecta                  |
| `test_he3_perfect_enrichment`             | Enrichment Factor alto con separación perfecta             |
| `test_he2_random_auc`                     | AUC-ROC ≈ 0.5 con scores aleatorios                       |
| `test_bootstrap_ci_coherent`              | IC inferior ≤ puntual ≤ IC superior                        |
| `test_compare_models`                     | Comparación entre modelos produce ranking                  |
| `test_full_evaluation_keys`               | `results.json` contiene todas las claves esperadas         |
| `test_edge_case_all_same_class`           | Manejo correcto cuando todos son misma clase               |
| `test_edge_case_constant_scores`          | Manejo correcto con scores constantes                      |
| `test_holm_bonferroni_correction`         | Corrección de Holm-Bonferroni aplicada correctamente       |
| `test_topk_multiple_k_values`             | Top-k evaluado para múltiples valores de k                 |

### test_reporting.py (NUEVO, smoke tests)

| Test                                      | Propósito                                                  |
|-------------------------------------------|------------------------------------------------------------|
| `test_table_generates_valid_latex`        | Tabla genera LaTeX válido (sin errores de sintaxis)        |
| `test_figure_generates_pdf`               | Figura genera archivo PDF no vacío                         |
| `test_latex_special_chars_escaped`        | Caracteres especiales LaTeX escapados correctamente        |

### test_integration.py (NUEVO, `@pytest.mark.slow`)

- Dataset sintético de 200 filas → features → preprocessing → training → scoring → evaluation
- **No requiere conexión a ClickHouse**
- Valida el flujo completo end-to-end con datos controlados

### test_pipeline.py (NUEVO — tests del orquestador)

Tests unitarios para `run_pipeline.py` que validan la lógica de orquestación sin ejecutar el pipeline real:

| Test                                           | Propósito                                                        |
|------------------------------------------------|------------------------------------------------------------------|
| `test_prerequisite_validation_fails_on_missing` | `validate_prerequisites()` lanza `FileNotFoundError` si falta un artefacto |
| `test_should_run_all_steps_by_default`         | Sin flags, `should_run()` retorna `True` para todos los pasos   |
| `test_should_run_single_step`                  | Con `--step 3`, solo el paso 3 retorna `True`                   |
| `test_should_run_from_step`                    | Con `--from-step 5`, pasos 1-4 retornan `False`, 5-8 `True`    |
| `test_dry_run_validates_but_does_not_execute`  | Con `--dry-run`, se validan prerrequisitos pero no se ejecutan pasos |
| `test_step_wrapper_logs_timing`                | `run_step()` registra tiempo en formato `MM:SS`                  |
| `test_fast_flag_disables_grid_search`          | `--fast` implica `skip_grid_search=True` y `bootstrap_n=100`    |

```python
# Smoke test del pipeline completo (con mocks)
@pytest.mark.slow
def test_pipeline_smoke(tmp_path, sample_transactions):
    """Pipeline completo con datos sintéticos y dependencias mockeadas.

    No ejecuta ClickHouse. Verifica que los 8 pasos se invocan en orden
    y que los artefactos finales (`results.json`, `results_sensitivity.json`,
    `results_posthoc.json`, tablas y figuras) se generan.
    """
    ...
```

---

## Archivos a ELIMINAR

| Archivo                                    | Razón                                          |
|--------------------------------------------|------------------------------------------------|
| `src/fraud_detector/data/balancing.py`     | Pertenece a enfoque supervisado, no aplica     |
| `run_simple_rf.py`                         | Demo de Random Forest, obsoleto                |
| `check_git_status.py`                      | Utilidad temporal, no necesaria                |
| `verify_setup.py`                          | Reemplazado por `--dry-run` del orquestador    |
| `QUICKSTART.md`                            | Reemplazado por CLAUDE.md actualizado          |
| `EXPERT_AUDIT.md`                          | Documento temporal de revisión                 |
| `notebooks/01_exploratory_analysis.ipynb`  | EDA se genera programáticamente ahora          |

---

## Archivos a ACTUALIZAR

| Archivo          | Cambios                                                              |
|------------------|----------------------------------------------------------------------|
| `.gitignore`     | Agregar `output/`, `data/processed/*.parquet`, `*.npy`, `*.joblib`   |
| `Makefile`       | Agregar `make pipeline`, `make pipeline-quick`, `make clean-output`; eliminar `make mlflow` |
| `pyproject.toml` | Versión → 1.0.0, eliminar dependencias supervisadas                  |
| `setup.py`       | Versión → 1.0.0, eliminar dependencias supervisadas                  |
| `CLAUDE.md`      | Actualizar al stack de detección de anomalías no supervisado         |
| `__init__.py`    | Actualizar exports en todos los paquetes                             |

---

## Integración con la Tesis (Tesis-LaTeX)

### Capítulo 1 (Introducción)
- Ya completo, no requiere cambios.

### Capítulo 2 (Marco Teórico / EDA)
- Insertar tablas y figuras de EDA generadas en `output/`
- Vincular con `\input{}` las tablas LaTeX generadas automáticamente

### Capítulo 3 (Resultados)
- Insertar resultados del pipeline
- **Completar los 52 marcadores `[POR COMPLETAR]`** con datos reales de `results.json`

### Conclusiones
- Sintetizar OG, OE1-OE4 con evidencia real de los resultados
- Usar lenguaje correlacional, nunca causal

### Apéndices
- SQL de extracción
- Catálogo de features (20 + 19 variantes)
- Configuración del pipeline

### Compilación final
```bash
pdflatex → biber → pdflatex → pdflatex
```

---

## Reproducibilidad

- `pip freeze > requirements-lock.txt` después de la ejecución final
- `PYTHONHASHSEED=42` establecido en `run_pipeline.py`
- Todas las seeds explícitamente configuradas en cada componente

---

## Gate E: Criterios de aceptación

| #  | Criterio                                           | Verificación                          |
|----|---------------------------------------------------|---------------------------------------|
| 1  | `pytest tests/ -v` pasa                           | Todos los tests verdes                |
| 2  | Test de integración pasa (sin ClickHouse)          | `@pytest.mark.slow` ejecutado         |
| 3  | Todas las tablas LaTeX compilan                    | Sin errores de `pdflatex`             |
| 4  | Todas las figuras PDF/PNG existen, tamaño > 0      | Verificación de filesystem            |
| 5  | `results.json`, `results_sensitivity.json` y `results_posthoc.json` válidos y completos | Schema validation |
| 6  | `CLAUDE.md` actualizado                            | Refleja stack actual                  |
| 7  | No quedan archivos obsoletos                       | Lista de eliminación ejecutada        |
| 8  | Tesis-LaTeX compila con resultados reales          | Compilación completa sin errores      |
| 9  | Gate de gobernanza satisfecho para outputs post-hoc | `actor_identity_validated` y política interna/pública definidos |
