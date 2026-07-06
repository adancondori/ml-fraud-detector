# Codebase Concerns

**Analysis Date:** 2026-07-06

---

## 1. Arquitectura Dual / Migración a Medias

**Dos pipelines evaluativos independientes coexisten con resultados divergentes:**

- **Ruta legacy (FS-baseline-31 + proxy unificado):**
  - `run_pipeline.py` (steps 1–8) → `scripts/run_fase6_modeling.py` → `scripts/run_fase7_evaluation.py`
  - Modelo: `output/models/isolation_forest.joblib` (IF-31, 31 features con StandardScaler)
  - Proxy: `unified` (OR Tipo A+B+C+D+E = 10.23%)
  - Output: `output/results.json`

- **Ruta confirmatorio (FS-clean-A-29 / disjoint-30 + Tipo A):**
  - `scripts/eval_clean_honest.py`, `scripts/eval_with_raw_features.py`, `scripts/eval_fs_disjoint.py`, `scripts/validate_if40_pivot_disjoint.py`, `scripts/run_bootstrap_confirmatorio_ci.py`
  - Modelo: `output/models/isolation_forest_final.joblib` (IF-40, RobustScaler)
  - Proxy: `tipo_a` o `operational_proxy` (pure_fraud)
  - Output: `output/revision/bootstrap_confirmatorio.json`, `output/results_fs_disjoint.json`, `output/results_clean_honest.json`, `output/results_final.json`

**Impacto:** No existe un único número AUC canónico para la tesis. `run_pipeline.py` es el orquestador "oficial" pero produce evaluación sobre el proxy más contaminado (unified) con el modelo desactualizado (IF-31). La ruta confirmatorio está dispersa en scripts `eval_*` sin orquestador unificado. `revision_report.py` aparece en `SOURCES.txt` (`src/fraud_detector.egg-info/SOURCES.txt` línea 16) pero el archivo no existe en `src/fraud_detector/evaluation/`, lo mismo que `tests/test_revision_report.py` (línea 49) — módulo declarado pero no implementado.

**Fix approach:** Crear un orquestador confirmatorio (e.g., `run_confirmatorio.py`) que ejecute la cadena `eval_with_raw_features.py` → `eval_fs_disjoint.py` / `validate_if40_pivot_disjoint.py` → `run_bootstrap_confirmatorio_ci.py`. Deprecar `run_pipeline.py` o añadir un `--mode confirmatorio` flag.

---

## 2. Circularidad de Features / Proxy Leakage

**2a. Proxy Tipo A ← features de reversión (IF-31)**

- `user_reversal_ratio_30d` (F18): `BehavioralFeatures._reversal_ratio_30d()` en `src/fraud_detector/features/engineering.py` línea 292–296, usa `status.isin(["totally_refunded", "refunded_to_credit"])` — la misma condición que define Tipo A.
- `user_reversal_count_30d` (F33): `OperationalDiversityFeatures._reversal_count_30d()` en `src/fraud_detector/features/engineering.py` línea 484–489, misma condición.
- **Consecuencia:** IF-31 evaluado contra proxy Tipo A en `scripts/run_fase7_evaluation.py` tiene circularidad mecánica parcial. El análisis de sensibilidad (IF-30, sin F18) es necesario pero no suficiente: F33 también es circular y solo se elimina en IF-21 (ablación extrema).
- `eval_clean_honest.py` líneas 45–55 documenta y elimina ambas: `CIRCULAR_FEATURES = {"user_reversal_ratio_30d", "user_reversal_count_30d", "user_discount_ratio_30d", "user_txn_count_24h"}`. Esta es la definición correcta para el confirmatorio.

**2b. Proxy `pure_fraud` / `operational_proxy` ← features del modelo (IF-40)**

- `pure_fraud` = `same_amount_count_1h >= 3 OR (user_account_age_days < 14 AND user_txn_count_1h >= 3) OR (is_third_party_payment == 1 AND user_txn_count_1h >= 2)`.
- Las cuatro variables exactas del proxy (`same_amount_count_1h`, `user_account_age_days`, `user_txn_count_1h`, `is_third_party_payment`) son features de IF-40 (ver `output/models/final_feature_list.json` líneas 9, 14, 33, 34).
- Esta circularidad está documentada en `scripts/eval_fs_disjoint.py` líneas 50–63 (`PROXY_VARS`, `PROXY_RECODINGS`) y `scripts/validate_if40_pivot_disjoint.py` líneas 31–54. El AUC 0.841 reportado contra `pure_fraud` es **autovalidación parcial** — el modelo aprende exactamente las reglas que definen el criterio de evaluación.
- La evaluación honesta requiere la variante `FS-disjoint-30` (eliminando también recodings y `capture_delay_seconds`) o `FS-disjoint-35` al menos.

**Fix approach:** Para la tesis, el criterio de evaluación primario debe ser Tipo A (structurally independent) con IF-40 en variante `FS-disjoint-30`. El AUC confirmatorio debe ser `summary["tipo_a"]["auc_mean"]` de `validate_if40_pivot_disjoint.py`, no el AUC vs `pure_fraud`.

---

## 3. Train/Serve Skew: `capture_delay_seconds`

**`capture_delay_seconds` tiene distribución radicalmente distinta entre batch y real-time.**

- **En batch** (`scripts/eval_with_raw_features.py` líneas 150–155): se computa como `(captured_at - created_at).dt.total_seconds()` sobre parquets históricos donde `captured_at` está poblada — da valores reales de segundos/minutos.
- **En real-time** (`src/fraud_detector/scoring/features_enriched.py` líneas 104–116): `_capture_delay_seconds` devuelve `0.0` cuando `captured_at` es None. En el scorer FastAPI (`scorer/schemas.py` línea 19), `captured_at: Optional[datetime] = None` — será nulo en el hook `after_commit on:create` porque la captura ocurre después de la creación.
- **Resultado:** La feature tiene valor informativo en el batch de evaluación histórica, pero es siempre 0 para transacciones nuevas en el scorer. El modelo IF-40 aprende a discriminar con `capture_delay_seconds`, pero en producción recibe un valor constante.
- El audit en `scripts/validate_if40_pivot_disjoint.py` líneas 291–323 (`audit_capture_delay`) intenta cuantificar esto, pero el problema persiste en producción.
- El batch scorer (`scorer/batch/scorer.py` línea 141) sí pasa `captured_at` al payload; para pagos ya capturados el valor puede ser real. El skew real aplica solo al scoring en tiempo real (`/score` endpoint).

**Fix approach:** Eliminar `capture_delay_seconds` de IF-40 o reemplazarla por un flag binario `is_captured` (0/1) que sea estable en real-time. El modelo ya tiene la variante `FS-disjoint-30` que la elimina (`scripts/validate_if40_pivot_disjoint.py` línea 38).

**Bug adicional en NaT check:** `src/fraud_detector/scoring/features_enriched.py` línea 112: `if captured is pd.NaT or created is pd.NaT:` — la comparación de identidad `is pd.NaT` **no funciona** para `pd.Timestamp` creados desde strings; `pd.Timestamp(None)` devuelve `NaT` pero `ts is pd.NaT` puede ser `False` (verificado: `pd.Timestamp('NaT') is pd.NaT` → `True`, pero `pd.Timestamp(captured_at) is pd.NaT` cuando `captured_at` es un string malformado puede ser `False`). La forma correcta es `pd.isnull(captured)`.

---

## 4. Train/Serve Skew: Features Temporales en UTC sin Hora Local

**`is_off_hours` y features horarias se calculan en UTC tanto en entrenamiento como en scoring.**

- `src/fraud_detector/features/engineering.py` línea 152: `hour = out["created_at"].dt.hour` — sin conversión de timezone.
- `src/fraud_detector/scoring/features.py` línea 49: `ts = pd.Timestamp(payment["created_at"])` — sin timezone.
- Para facilities en América Central, Sudamérica, Asia (21 zonas de tiempo en el sistema): la "hora" UTC puede diferir 3–8 horas de la hora local, haciendo que `is_off_hours` clasifique incorrectamente transacciones diurnas como nocturnas para facilities en UTC-5/UTC-6.
- `scripts/exp_frames_improvement.py` y `scripts/exp_reference_frames.py` prueban la corrección con `time_zone` de la facility (obtenida de ClickHouse), pero estas correcciones no están integradas en el pipeline principal ni en el scorer.
- **Discrepancia adicional:** `engineering.py` línea 159 define off-hours como `hour.isin([23, 0, 1, 2, 3, 4, 5, 6])` (inclusive hora 6), mientras que `scoring/features.py` línea 93 define `1.0 if hour >= 23 or hour <= 6 else 0.0` (también inclusive hora 6). Son consistentes entre sí pero ambos en UTC.

**Fix approach:** Integrar `facility_tz.parquet` (ya disponible en `output/revision/facility_tz.parquet`) en `TemporalFeatures.transform()` y replicar la conversión en `SingleFeatureCalculator`.

---

## 5. Data Quality: Moneda `EMPTY`

**La normalización de moneda fallback devuelve rate=1.0 para strings no reconocidos.**

- `src/fraud_detector/utils/currency.py` líneas 49–52 (`fallback_rate`): `_FALLBACK_RATES.get(currency, 1.0)` — si `currency` es `"EMPTY"`, `""`, o cualquier string no registrado, devuelve 1.0 (trata como USD).
- `src/fraud_detector/data/loader.py` línea 175: `currency = out[currency_col].fillna("USD").astype(str).str.upper().replace("", "USD")` — solo reemplaza string vacío, no `"EMPTY"`.
- `src/fraud_detector/features/engineering.py` línea 646: `out["currency"] = out["currency"].fillna("USD").astype(str).str.upper()` — tampoco sanitiza `"EMPTY"`.
- **Resultado:** Facilities con moneda `EMPTY` producen amounts que parecen USD (sin conversión) cuando en realidad pueden ser moneda local. `log_amount`, `amount_usd_ratio`, `staff_amount_zscore` quedan con valores incorrectos para esas transacciones.
- `src/fraud_detector/utils/currency.py` línea 154–157 solo emite `logger.warning` cuando falla la búsqueda en el lookup; para la ruta `fallback_rate()` (línea 52) no hay ningún aviso.

**Fix approach:** Sanitizar `"EMPTY"` junto con `""` en `_postprocess_extraction()` de loader.py: `currency.replace({"EMPTY": "USD", "": "USD"})`.

---

## 6. Threshold Calibrado en Test Set

**El threshold operativo se deriva del test set, contaminando la evaluación.**

- `scripts/run_fase7_evaluation.py` líneas 115–128 (`save_thresholds`): `threshold = float(np.percentile(scores_if, 95))` sobre `scores_if` del test set. El JSON resultante incluye `"threshold_source": "percentile_95_test_set"`.
- El mismo test set se usa para reportar HE1–HE4 en `output/results.json`.
- Aunque el threshold se usa para el scorer operativo (no para las métricas de la tesis), reportar AUC y threshold derivados del mismo conjunto introduce circularidad: el threshold está optimizado para ese conjunto particular.
- El scorer `scorer/artifact_loader.py` línea 64 carga `thresholds_v2.json` para IF-40; `thresholds_v2.json` se genera en `scripts/calibrate_threshold_v2.py` (a verificar si usa val o test).

**Fix approach:** Calibrar threshold en val set. Separar `run_fase7_evaluation.py` en scoring (test) y threshold-calibration (val).

---

## 7. Holm-Bonferroni Aplicado con Placeholder

**`run_fase7_evaluation.py` pasa `auc_roc` como p-valor para HE2 en la corrección múltiple.**

- `scripts/run_fase7_evaluation.py` líneas 176–187: La lista `p_values` incluye `eval_if["he2"]["auc_roc"]` con el comentario `# placeholder — use actual p-value if available`. AUC-ROC puede valer 0.7–0.9, no es un p-valor.
- Sin embargo, la función `apply_holm_bonferroni` solo se aplica a `[eval_if["he1"]["p_value"]]` (lista de un elemento), por lo que la corrección es identidad — el array `p_values` construido no se usa. El impacto es nulo en los resultados pero indica código incompleto/confuso.

**Fix approach:** Eliminar las líneas 176–183 o documentar explícitamente que solo HE1 tiene p-valor estadístico. La variable `p_values` debería llamarse `he1_p_values` si es de un solo elemento.

---

## 8. Atributos Silenciados con `getattr` en `SingleFeatureCalculator`

**`scoring/features.py` usa nombres de atributos incorrectos, silenciados por `getattr`.**

- `src/fraud_detector/scoring/features.py` línea 27: `self._facility_avgs = getattr(fe._groups[4], "_facility_avg", {})` — pero `ContextualFeatures` en `engineering.py` línea 326 usa el atributo `_facility_avg_amount`, no `_facility_avg`. El `getattr` devuelve `{}` silenciosamente.
- Línea 28: `self._staff_stats = getattr(fe._groups[6], "_staff_stats", {})` — pero `StaffRoleFeatures` usa `_role_currency_stats` (línea 393 de engineering.py), no `_staff_stats`. El `getattr` devuelve `{}` silenciosamente.
- **Consecuencia:** El scorer real-time siempre usa `self._global_avg_amount` para `facility_avg_amount` (nunca la media per-facility) y el zscore de staff siempre usa global stats. El modelo fue entrenado con facility-specific averages. **Este es un silencioso train/serve skew activo en producción.**

**Fix approach:**
```python
# En scoring/features.py líneas 27–28:
self._facility_avgs = getattr(fe._groups[4], "_facility_avg_amount", {})
self._staff_stats = getattr(fe._groups[6], "_role_currency_stats", {})
```
También ajustar la lógica de lookup en `calculate()` para usar `(role, currency)` como clave en `_staff_stats`.

---

## 9. Hardcoded Database Name en `context.py` y `batch/scorer.py`

**La base de datos `pbp_productionDB_optimized` está hardcodeada en múltiples SQLs.**

- `src/fraud_detector/scoring/context.py` líneas 75, 91, 104, 118, 128, 135, 149, 166: todas las queries usan `pbp_productionDB_optimized.payments`, `pbp_productionDB_optimized.users`, `pbp_productionDB_optimized.facilities_users` directamente, ignorando `settings.clickhouse_database`.
- `scorer/batch/scorer.py` líneas 144, 156: misma hardcoding en `_FETCH_SQL` y `_CURSOR_END_SQL`.
- **Impacto:** Impossible levantar el scorer contra un ClickHouse de test/staging con nombre de BD diferente sin modificar el código fuente.

**Fix approach:** Parametrizar el database name en `UserContextProvider.__init__` y `BatchScorer.__init__`, interpolando `{database}` en los SQL templates igual que lo hace `CANONICAL_SQL` en `loader.py` línea 83.

---

## 10. Módulo `revision_report.py` Declarado pero Inexistente

**El paquete instalado declara un módulo que no existe en disco.**

- `src/fraud_detector.egg-info/SOURCES.txt` líneas 16 y 49 listan:
  - `src/fraud_detector/evaluation/revision_report.py`
  - `tests/test_revision_report.py`
- Ninguno de los dos archivos existe en `src/fraud_detector/evaluation/` (solo `__init__.py`, `hypothesis.py`, `metrics.py`).
- **Impacto:** `import fraud_detector.evaluation.revision_report` fallará en tiempo de ejecución; los tests de CI que intenten correr `test_revision_report.py` producirán errores de colección.

**Fix approach:** Crear el módulo con la implementación pendiente, o eliminar las entradas de SOURCES.txt y el test stub.

---

## 11. Áreas Frágiles

**`BehavioralFeatures._distinct_facilities_30d()` — O(n²) por usuario**

- `src/fraud_detector/features/engineering.py` líneas 236–247: loop anidado Python puro sobre cada transacción de cada usuario. Para usuarios con >1000 transacciones y datasets de 3M+ filas, el tiempo puede ser O(n²) dentro del grupo.
- Safe modification: no cambiar el algoritmo sin tests que verifiquen exactitud anti-leakage. Alternativa vectorizada: `_rolling_shifted_stat` con `nunique`.

**`OperationalDiversityFeatures._category_entropy_30d()` — mismo patrón O(n²)**

- `src/fraud_detector/features/engineering.py` líneas 465–481: loop Python puro, mismo patrón que `_distinct_facilities_30d`. Para el dataset completo (3M filas train) puede tomar 30–60 minutos.

**`per_user_same_amount_count()` en `eval_with_raw_features.py` — sliding window con reset incorrecto**

- `scripts/eval_with_raw_features.py` líneas 91–111: el comentario línea 109 dice "reset j tracking" pero el doble loop no resetea `j` entre grupos — `j` se declara fuera del loop de groups y acumula, lo que es correcto solo si el grupo está ordenado y `j` comienza en 0 para cada grupo. Como está dentro del `for (uid, amt), group in df.groupby(...)`, `j` se declara en la línea 103 dentro del loop de groups — sí se resetea. Sin embargo la lógica solo es correcta si `group` está ordenado por `created_at`, lo que no se garantiza explícitamente (solo se ordena el DataFrame completo en línea 130, antes del groupby). **Test coverage: ninguno.**

**`transform_with_warm_history()` — marker column `_is_split` puede colisionar**

- `src/fraud_detector/features/engineering.py` líneas 586–599: añade una columna temporal `_is_split` al DataFrame combinado. Si el DataFrame de entrada ya contiene una columna con ese nombre, la lógica falla silenciosamente o sobreescribe. No hay guard contra esta colisión.

---

## 12. Cobertura de Tests Insuficiente en Áreas Críticas

**Áreas sin tests o con tests vacíos:**

- `tests/test_revision_report.py` — referenciado en SOURCES.txt pero no existe.
- `tests/test_if40_artifacts.py` — existe pero no cubre el bug de atributos `_facility_avg` vs `_facility_avg_amount` (concern #8).
- Scoring end-to-end (`tests/test_scoring.py`) no verifica que `capture_delay_seconds=0` cuando `captured_at=None` coincide con el comportamiento de entrenamiento.
- No hay tests para `per_user_same_amount_count()` ni para la lógica de `build_proxies()` en múltiples scripts.
- No hay tests de integración que verifiquen que IF-40 scorer produce scores consistentes con los del pipeline de evaluación batch.

---

*Concerns audit: 2026-07-06*
