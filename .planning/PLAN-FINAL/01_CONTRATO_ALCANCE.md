# Fase 0. Contrato y Alcance

> Traducir la tesis a un contrato tecnico verificable para que ninguna fase posterior trabaje con supuestos ambiguos.

---

## 1. Universo del estudio

### SQL canonico (con FINAL obligatorio)

```sql
SELECT
    id,
    user_id,
    facility_id,
    facility_name,
    created_at,
    captured_at,
    payment_method,
    gateway,
    source_enum,
    status,
    reservation_paid_out,
    discount,
    tax,
    tip,
    card_brand,
    currency,
    paid_by_manager,
    reversed_id,
    debit_refund,
    _peerdb_version
FROM pbp_productionDB_optimized.payments FINAL
WHERE created_at >= %(start)s
  AND created_at < %(end)s
  AND payment_method != 'reversal'
  AND payment_method != 'free'
  AND user_id != 0
  AND _peerdb_is_deleted = 0
ORDER BY created_at, id
```

**Tabla fuente:** `SharedReplacingMergeTree`, `ORDER BY (facility_id, created_at, id)`, sin `PARTITION BY`.
`FINAL` es obligatorio porque la tabla contiene multiples versiones de filas; sin el, los conteos quedan inflados.

### Exclusiones

| Filtro | Razon |
|--------|-------|
| `payment_method != 'reversal'` | Movimientos contables internos, no transacciones de cliente |
| `payment_method != 'free'` | Reservaciones sin cobro |
| `user_id != 0` | Transacciones de sistema/anonimas sin perfil conductual |
| `_peerdb_is_deleted = 0` | Filas soft-deleted por replicacion PeerDB |

---

## 2. Proxy de anomalia

El estudio es **no supervisado**: los modelos entrenan sin etiquetas. El proxy se usa **exclusivamente para evaluacion** (nunca para entrenamiento).

### Proxy estricto (principal)

```
status IN ('totally_refunded', 'refunded_to_credit')
```

- N estimado: **~429,442** (6.33% del universo)
- Uso: evaluacion principal de HE1-HE4

### Proxy amplio (sensibilidad)

```
status IN ('totally_refunded', 'refunded_to_credit', 'partially_refunded')
```

- N estimado: **~512,609** (7.55% del universo)
- Uso: analisis de sensibilidad (verificar que conclusiones no dependen de la definicion del proxy)

> **Proxy != fraude.** Nunca afirmar que las anomalias detectadas equivalen a fraude. El proxy es una aproximacion basada en el estado transaccional. Usar lenguaje de asociacion, no causal.

---

## 3. Temporal splits

| Split | Inicio (inclusive) | Fin (exclusive) | N estimado | Proporcion |
|-------|--------------------|-----------------|------------|------------|
| Train | 2025-01-01 | 2025-07-01 | ~3,137,086 | 46.2% |
| Validation | 2025-07-01 | 2025-09-01 | ~1,130,118 | 16.7% |
| Test | 2025-09-01 | 2026-01-01 | ~2,517,491 | 37.1% |
| **Total** | | | **~6,784,695** | 100% |

Ademas:
- **Warm history:** 2024-12-01 a 2024-12-31 (para inicializar ventanas rolling de 30 dias en enero 2025)

Tolerancia de verificacion: los conteos reales no deben divergir mas de **+-1%** de los valores estimados.

---

## 4. Catalogo oficial de 20 features

### Grupo 1: Transaccionales (5)

| # | Feature | Descripcion |
|---|---------|-------------|
| 1 | `amount` | reservation_paid_out (monto de la transaccion) |
| 2 | `log_amount` | `log(1 + amount)` — normalizacion logaritmica |
| 3 | `amount_usd_ratio` | `amount / global_avg_amount` (fitted en train) |
| 4 | `discount_ratio` | `discount / (amount + 1e-8)` |
| 5 | `has_tip` | Indicador binario: `tip > 0` |

### Grupo 2: Temporales (5)

| # | Feature | Descripcion |
|---|---------|-------------|
| 6 | `hour_sin` | `sin(2*pi*hour/24)` — codificacion ciclica |
| 7 | `hour_cos` | `cos(2*pi*hour/24)` — codificacion ciclica |
| 8 | `day_of_week` | Dia de la semana (1=Lun, 7=Dom) |
| 9 | `is_weekend` | Indicador binario: Sab o Dom |
| 10 | `is_off_hours` | Indicador binario: horas 23-06 |

### Grupo 3: Velocidad (4)

| # | Feature | Descripcion |
|---|---------|-------------|
| 11 | `user_txn_count_1h` | Transacciones del usuario en ultima hora (excluye txn actual) |
| 12 | `user_txn_count_24h` | Transacciones del usuario en ultimas 24h (excluye txn actual) |
| 13 | `time_since_last_txn` | Segundos desde la transaccion anterior del usuario |
| 14 | `user_amount_24h` | Monto acumulado del usuario en 24h (excluye txn actual) |

### Grupo 4: Comportamiento (4)

| # | Feature | Descripcion |
|---|---------|-------------|
| 15 | `user_distinct_facilities_cumul` | Facilities distintas acumuladas del usuario (expanding, excluye txn actual) |
| 16 | `user_distinct_methods` | Metodos de pago distintos acumulados del usuario (expanding, excluye txn actual) |
| 17 | `user_reversal_ratio_30d` | Tasa de reembolso del usuario en 30 dias (shift per-group) — **SENSITIVITY** |
| 18 | `user_account_age_days` | Dias desde primera transaccion (first_txn fitted en train) |

### Grupo 5: Contextuales (2)

| # | Feature | Descripcion |
|---|---------|-------------|
| 19 | `facility_avg_amount` | Monto promedio de la facility (fitted en train) |
| 20 | `amount_facility_ratio` | `amount / (facility_avg_amount + 1e-8)` |

### Variante de 19 features

Se implementa obligatoriamente una version sin `user_reversal_ratio_30d` (Feature #17) por su riesgo de correlacion mecanica con el proxy. El analisis de sensibilidad (Gate D) compara ambas versiones.

---

## 5. Algoritmos

| Rol | Algoritmo | Justificacion |
|-----|-----------|---------------|
| **Principal** | Isolation Forest (IF) | Tesis propone IF como metodo principal; grid search de 64 combinaciones |
| **Comparador** | Local Outlier Factor (LOF) | Metodo basado en densidad local; grid de 3 combinaciones |
| **Comparador** | One-Class SVM (OC-SVM) | Metodo basado en frontera; grid de 6 combinaciones; subsample de 100K para viabilidad |

**Excluido:** DBSCAN — no produce ranking continuo de anomalia, incompatible con metricas de la tesis.

**Contamination:** eliminada del grid search de IF (no afecta el ranking de scores en metricas basadas en rango como AUC-ROC y AP). Se usa `contamination="auto"` en entrenamiento.

**Score convention:** score alto = mas anomalo. Se usa `-decision_function(X)` para IF y OC-SVM.

---

## 6. Hipotesis especificas (HE1-HE4) con umbrales cuantificados

### HE1 — Separacion estadistica de scores

> Los scores de anomalia de IF asignan valores significativamente mas altos a transacciones proxy+ que a transacciones proxy-.

| Metrica | Umbral | Detalle |
|---------|--------|---------|
| Mann-Whitney U | p < 0.05 | Test unilateral (alternative="greater") |
| Rank-biserial r | r_rb > 0.10 | `r = 2*U/(n1*n2) - 1`; positivo cuando anomaly > normal |

**HE1 pasa si:** p < 0.05 **Y** r_rb > 0.10 (ambos despues de correccion Holm-Bonferroni).

Metricas complementarias: CLES (Common Language Effect Size), KS statistic.

### HE2 — Capacidad discriminativa

> IF logra una capacidad discriminativa superior a la asignacion aleatoria y al baseline de tasa base.

| Metrica | Umbral |
|---------|--------|
| AUC-ROC | > 0.70 |
| Average Precision (AP) | > 6.33% (tasa base del proxy estricto) |

**HE2 pasa si:** AUC > 0.70 **Y** AP > tasa base proxy estricto.

Bootstrap CI 95% (N=1000) para ambas metricas.

### HE3 — Concentracion en top-K

> Las transacciones proxy+ se concentran desproporcionadamente en los percentiles mas altos de anomalia segun IF.

| Metrica | Umbral |
|---------|--------|
| Enrichment Factor (EF) | > 1 en top-5% |

**HE3 pasa si:** EF > 1 en top-5%.

Se evalua ademas en top-1%, top-2%, top-5%, top-10% (multiples k).

EF = (proporcion de proxy+ en top-k%) / (proporcion global de proxy+).

### HE4 — Comparacion IF vs. competidores

> IF presenta desempeno discriminativo comparable o superior a LOF y OC-SVM en la mayoria de las metricas principales.

| Metricas de comparacion |
|------------------------|
| AUC-ROC |
| Average Precision (AP) |
| Precision@5% |
| Enrichment Factor@5% |

**HE4 pasa si:** IF >= competidores en **>=3 de 4** metricas.

> **Alineacion con la tesis:** La tesis define el criterio como "la mayoria de las metricas" y la tabla `tab:he4-validacion` usa explicitamente `≥ 3/4` (ver `03_propuesta_validacion.tex:572`). El plan se alinea con este umbral. Si IF gana en solo 2/4, HE4 se rechaza pero el estudio sigue siendo valido — se documenta cual metodo fue superior.

Comparacion justa: mismo snapshot, mismas filas, mismo set de features, mismo proxy, misma ventana temporal, misma orientacion del score.

### Correcciones estadisticas

- **Holm-Bonferroni** aplicada al conjunto de tests HE1-HE4.
- **Estabilidad temporal:** AUC mensual en test set (Sep, Oct, Nov, Dic 2025).

---

## 6.1. Contratos de test (TDD red-green-refactor)

Cada clase del pipeline tiene un contrato de comportamiento que DEBE expresarse como tests **antes** de escribir la implementacion. El ciclo es:

1. **Red:** Escribir el test que describe el comportamiento esperado. Ejecutar y verificar que falla.
2. **Green:** Escribir la implementacion minima que haga pasar el test.
3. **Refactor:** Mejorar la implementacion sin romper tests existentes.

### Contratos por clase

| Clase | Test contract (escribir PRIMERO) | Comportamiento que valida |
|-------|----------------------------------|--------------------------|
| `DataManager` | `test_extract_returns_correct_columns` | El DataFrame extraido contiene exactamente las columnas del SQL canonico |
| `DataManager` | `test_extract_validates_row_counts` | Los conteos por split estan dentro de +-1% de los valores esperados |
| `DataManager` | `test_proxy_labels_strict` | `assign_proxy_labels(df, "strict")` marca exactamente los status `totally_refunded` y `refunded_to_credit` |
| `DataManager` | `test_proxy_labels_wide` | `assign_proxy_labels(df, "wide")` incluye ademas `partially_refunded` |
| `DataManager` | `test_downcast_preserves_large_ids` | `id` y `reversed_id` no se truncan a int32 (valores > 2^31 sobreviven) |
| `DataManager` | `test_manifest_contains_required_fields` | El JSON sidecar contiene: name, start_date, end_date, row_count, extracted_at, checksum_sha256 |
| `DataManager` | `test_atomic_write_survives_interruption` | Si el proceso muere durante la escritura, no queda un Parquet corrupto en la ruta final |
| `FeatureEngineer` | `test_first_txn_counts_zero` | La primera transaccion de cada usuario tiene velocity features == 0 |
| `FeatureEngineer` | `test_no_nans_after_transform` | Cero NaN en las 20 features despues de transform |
| `FeatureEngineer` | `test_fit_transform_produces_20_features` | El output contiene exactamente las 20 columnas de `FEATURE_NAMES` |
| `UnsupervisedPreprocessor` | `test_scaler_fitted_on_train_only` | Los parametros del scaler provienen exclusivamente del train set |
| `UnsupervisedPreprocessor` | `test_output_shape_preserved` | El numero de filas y features no cambia tras preprocesamiento |
| `AnomalyModelTrainer` | `test_score_orientation` | Scores mas altos corresponden a mayor anomalia (convencion `-decision_function`) |
| `AnomalyModelTrainer` | `test_grid_search_runs_all_combinations` | Para IF, se ejecutan las 64 combinaciones documentadas |
| `HypothesisEvaluator` | `test_he1_requires_both_conditions` | HE1 solo pasa si p < 0.05 Y r_rb > 0.10 (no basta una sola condicion) |
| `HypothesisEvaluator` | `test_he4_threshold_gte_3_of_4` | HE4 pasa con 3/4 metricas, falla con 2/4 |
| `HypothesisEvaluator` | `test_holm_bonferroni_applied` | Los p-values reportados incluyen correccion por comparaciones multiples |

> **Regla:** Ningun modulo pasa a implementacion hasta que sus tests de contrato esten escritos y fallen (fase Red). La implementacion se considera completa cuando todos los tests pasan (fase Green).

---

## 7. Fuera de alcance

| Excluido | Razon |
|----------|-------|
| Despliegue en produccion | Tesis metodologica, no operativa |
| Scoring en tiempo real | Requiere infraestructura de streaming |
| API de inferencia | Pipeline offline y reproducible |
| Algoritmos adicionales (deep learning, DBSCAN, etc.) | No requeridos por los objetivos de la tesis |
| Dashboards operativos | Fuera del alcance academico |
| Deteccion legal de fraude | El proxy no equivale a fraude |
| MLflow como dependencia central | Se elimina; reproducibilidad via manifests y seeds |

---

## 8. Matriz de trazabilidad: Objetivo - Hipotesis - Modulo - Output

| Objetivo | Hipotesis | Modulo / Clase | Output |
|----------|-----------|----------------|--------|
| OE1: Fundamentar referentes teoricos | — | `Tesis-Latex/capitulos/capitulo1.tex`, revision bibliografica | Marco teorico en tesis (Cap. 1) |
| OE2: Diagnosticar el estado transaccional actual | — | `01_eda_capitulo2.ipynb`, `ThesisTableGenerator`, `ThesisFigureGenerator` | `output/tables/cap2_*.tex`, `output/figures/cap2_*.pdf` |
| OE3: Construir pipeline y evaluar capacidad discriminativa de IF | HE1, HE2, HE3 | `DataManager`, `FeatureEngineer`, `UnsupervisedPreprocessor`, `AnomalyModelTrainer`, `HypothesisEvaluator` | `data/processed/*.parquet`, `output/models/*.joblib`, `output/results.json` (secciones he1, he2, he3, bootstrap_ci) |
| OE4: Comparar IF contra LOF y OC-SVM | HE4 | `HypothesisEvaluator` | `output/results.json` (seccion he4) |
| Sensibilidad | Gate D | `HypothesisEvaluator` | `output/results_sensitivity.json` |

---

## 9. Entregables de la Fase 0

| Entregable | Descripcion |
|------------|-------------|
| `config/config.py` | `Settings` reescrito: sin parametros supervisados, con splits temporales, proxy labels, grids de tuning |
| `scripts/verify_counts.py` | Script que ejecuta los conteos contra ClickHouse y verifica contra los valores esperados |
| Este documento (`01_CONTRATO_ALCANCE.md`) | Contrato tesis-codigo congelado |

### Cambios en config.py

**Eliminar:** `model_type`, `test_size`, `validation_size`, `mlflow_*`, `use_gpu`, `fraud_threshold`, `high_risk_threshold`, `auto_decline_threshold`, `api_host`, `api_port`, `database_url`, `use_smote`, `smote_sampling_strategy`, `use_class_weights`, `use_temporal_split`, `temporal_split_date`, `embargo_days`, `enable_drift_detection`, todo parametro supervisado.

**Agregar:** `train_start`, `train_end`, `val_end`, `test_end`, `strict_proxy_statuses`, `wide_proxy_statuses`, grids de IF/LOF/OC-SVM, `bootstrap_n`, `top_k_percents`, `shap_sample_size`.

---

## 10. Definition of Done

- [ ] `config/config.py` reescrito y cargable sin errores con `.env` actual
- [ ] No existe ninguna referencia a `is_fraud` en el codigo fuente
- [ ] No existe ninguna referencia a parametros supervisados (`SMOTE`, `fraud_threshold`, etc.)
- [ ] `verify_counts.py` ejecutable contra ClickHouse
- [ ] Los filtros del universo (SQL canonico) estan centralizados en una unica fuente de verdad
- [ ] La matriz de trazabilidad Objetivo-Hipotesis-Modulo-Output esta documentada
- [ ] Los umbrales de HE1-HE4 estan codificados como constantes (no hardcodeados en logica dispersa)
- [ ] README del proyecto describe correctamente la tesis actual (no supervisado, no fraude)

---

## 11. Gate de salida — Fase 0

**No pasar a Fase 1 hasta que:**

1. Exista una unica fuente de verdad para filtros SQL, proxy y splits temporales (`config.py`).
2. El README describa correctamente la tesis actual.
3. No haya ninguna ruta critica dependiente de `is_fraud`.
4. Los archivos a eliminar esten identificados: `balancing.py`, `run_simple_rf.py`, `check_git_status.py`, `verify_setup.py`, `QUICKSTART.md`, `EXPERT_AUDIT.md`, `01_exploratory_analysis.ipynb`.
5. Las dependencias supervisadas esten marcadas para remocion: `imbalanced-learn`, `xgboost`, `lightgbm`, `mlflow`, `optuna`.
6. Este documento este revisado y aprobado como contrato congelado.
