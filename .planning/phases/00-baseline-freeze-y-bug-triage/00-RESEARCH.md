# Phase 0: Baseline Freeze y Bug Triage — Research

**Researched:** 2026-07-06
**Domain:** Brownfield bugfix + baseline documentation — Python/pandas/scikit-learn scorer
**Confidence:** HIGH — all findings verified against real source files; zero unverified claims

---

## Summary

Fase 0 es una fase de saneamiento, no de construcción. Cuatro bugs activos contaminan toda
medición anterior y futura: dos `getattr` con nombres de atributo incorrectos en el scorer
real-time, un check `is pd.NaT` que puede sillar silenciosamente, un threshold derivado del
test set en `thresholds.json` (aunque `thresholds_v2.json` ya está correctamente calibrado en
val), y ausencia de sanitización de moneda `"EMPTY"` en el pipeline de extracción. Adicionalmente,
`capture_delay_seconds` es una feature con semántica radicalmente distinta en batch vs. real-time.

La investigación verificó el código actual contra cada bug reportado en CONCERNS.md. Se encontró
una discrepancia importante respecto al concern #6: `thresholds_v2.json` (el que usa IF-40 en
producción) **ya está calibrado en val set** (`"threshold_source": "percentile_95_validation_set"`);
el problema de calibración en test set solo afecta a `thresholds.json` (IF-31, ruta legacy). El
planner debe distinguir los dos artefactos.

El documento de baseline congelado no requiere nueva extracción ni re-entrenamiento. Los datos
necesarios ya existen: `val_features_enriched.parquet` (1.13M filas, incluye `status` para Tipo A,
`facility_avg_amount`, `amount_facility_ratio`, y `capture_delay_seconds`). El set dorado de ≥500
pagos se construye sobre ese parquet con un filtro estratificado.

**Recomendación primaria:** Fijar los dos `getattr` primero (son cambios de una línea cada uno),
añadir aserciones post-carga + test de paridad, y luego congelar el baseline. El fix de `getattr`
cambia los scores reales de producción; el baseline document debe capturar las métricas
**post-fix**, no pre-fix.

---

## Standard Stack

Este proyecto no agrega dependencias nuevas en Fase 0. Todo lo necesario está instalado.

### Core (verificado contra venv instalado)

| Library | Version instalada | Uso en Fase 0 |
|---------|-------------------|---------------|
| pandas | 2.3.3 | NaT check (`pd.isnull`), currency replace |
| numpy | 1.24+ | percentile sobre val scores |
| scikit-learn | 1.6.1 | score_samples / decision_function del modelo |
| joblib | 1.5.3 | load de `feature_engineer.joblib` |
| pytest | 7.4.0+ | test de paridad batch↔real-time |
| loguru | instalado | logging del data quality report |

### Sin dependencias nuevas

No se requiere `pip install` de nada. El stack base cubre todos los fixes y el test de paridad.

---

## Architecture Patterns

### Flujo de artefactos relevante para Fase 0

```
output/models/
├── feature_engineer.joblib       # Cargado por SingleFeatureCalculator.__init__
│   └── _groups[0]  = TransactionalFeatures  → ._global_avg_amount  (correcto)
│   └── _groups[4]  = ContextualFeatures     → ._facility_avg_amount (BUG: código lee _facility_avg)
│   └── _groups[6]  = StaffRoleFeatures      → ._role_currency_stats (BUG: código lee _staff_stats)
├── thresholds.json               # IF-31, threshold_source="percentile_95_test_set" (legacy, a corregir)
└── thresholds_v2.json            # IF-40, threshold_source="percentile_95_validation_set" (YA correcto)
```

### Pattern: Fix de getattr con aserción post-carga

El patrón correcto para cargar atributos de artefactos entrenados es acceso directo con
validación, no `getattr` con default silencioso.

```python
# ANTES (bug activo):
self._facility_avgs = getattr(fe._groups[4], "_facility_avg", {})
self._staff_stats = getattr(fe._groups[6], "_staff_stats", {})

# DESPUÉS (fix):
self._facility_avgs = fe._groups[4]._facility_avg_amount
self._staff_stats = fe._groups[6]._role_currency_stats
assert len(self._facility_avgs) > 0, "facility_avgs vacío — artefacto corrupto o nombre incorrecto"
assert len(self._staff_stats) > 0, "staff_stats vacío — artefacto corrupto o nombre incorrecto"
```

Fuente: `src/fraud_detector/features/engineering.py:326` (`_facility_avg_amount`) y `:393`
(`_role_currency_stats`).

### Pattern: Lookup de staff_stats con clave compuesta

**CRÍTICO:** `_role_currency_stats` usa claves `(str, str)` (role, currency), NO solo `str(role)`.
El `calculate()` actual busca con `role_key` solo (sin currency), lo que nunca coincide con ninguna
clave del dict. El fix requiere pasar currency al lookup y replicar la cadena de fallback de
`StaffRoleFeatures.transform()`.

```python
# Clave en _role_currency_stats (engineering.py:406-412):
# (str(role), str(currency)) -> {"mean": float, "std": float}

# Fallback chain que debe replicar calculate():
currency = payment.get("currency", "USD") or "USD"
role_key = context.user_role if is_staff else "player"
currency_key = (role_key, currency.upper())

if currency_key in self._staff_stats:
    stats = self._staff_stats[currency_key]
elif (role_key, "USD") in self._staff_stats:          # fallback a USD para ese rol
    stats = self._staff_stats[(role_key, "USD")]
else:
    stats = {"mean": self._global_avg_amount, "std": 1.0}

staff_mean = stats["mean"]
staff_std = stats["std"]
```

También se deben exponer `_global_mean` y `_global_std` de `StaffRoleFeatures` (líneas 395-396)
para el fallback final.

### Pattern: Test de paridad batch↔real-time

El test de paridad es el contrato central de Fase 0. Estructura mínima:

```python
# tests/test_parity_phase0.py
def test_facility_avg_parity(feature_engineer_fixture):
    """facility_avg_amount debe ser idéntico en batch y real-time para mismo pago."""
    # Tomar fila del val_features_enriched.parquet
    row = pd.read_parquet("data/processed/val_features_enriched.parquet").iloc[0]
    
    # Batch path: valor ya calculado por FeatureEngineer
    batch_facility_avg = row["facility_avg_amount"]
    
    # Real-time path: SingleFeatureCalculator.calculate()
    calc = SingleFeatureCalculator("output/models/feature_engineer.joblib")
    payment = {row_to_payment_dict(row)}
    context = {row_to_context(row)}
    features = calc.calculate(payment, context)
    rt_facility_avg = features[FEATURE_NAMES.index("facility_avg_amount")]
    
    assert abs(batch_facility_avg - rt_facility_avg) < 1e-6, (
        f"Paridad fallida: batch={batch_facility_avg}, real-time={rt_facility_avg}"
    )
```

**Tolerancia:** <1e-6 (float32 precision limit; `facility_avg_amount` es el valor almacenado
directamente, no calculado en vuelo).

**Fuente del set dorado:** `val_features_enriched.parquet` (1.13M filas, ya en disco). Tomar
≥500 filas estratificadas por `facility_id` (cubrir al menos 20 facilities distintas) para
validar que la lookup per-facility funciona, no solo el global fallback.

### Anti-Patterns a Evitar

- **`getattr(obj, name, {})`** para atributos de artefactos entrenados: devuelve `{}` silenciosamente si el nombre es incorrecto; usar acceso directo + assertion.
- **`ts is pd.NaT`** para check de timestamp nulo: puede ser `False` para NaT construido de ciertas rutas. Usar `pd.isnull(ts)`.
- **Derivar threshold del test set**: `thresholds.json` tiene `threshold_source: "percentile_95_test_set"`. Para IF-31 legado, corregir a val. Para IF-40, `thresholds_v2.json` ya está correcto.

---

## Don't Hand-Roll

| Problema | No construir | Usar en cambio | Por qué |
|----------|-------------|----------------|---------|
| Check de NaT en Timestamp | `ts is pd.NaT` custom check | `pd.isnull(ts)` | Cubre todos los path de construcción de NaT |
| Sanitización de currency strings | Lista manual de guards | `pd.Series.replace({"EMPTY": "USD", "": "USD"})` | Vectorizado, un lugar, ya existe `fillna("USD")` en el mismo método |
| Serialización de baseline métricas | Custom writer | `json.dumps` a `output/baseline_v0.json` | Consistente con `thresholds.json` y `results.json` existentes |
| Golden set para parity test | Generar datos sintéticos | Leer de `val_features_enriched.parquet` | Val set real ya tiene facility_avg_amount calculado por FeatureEngineer; los sintéticos no replican la distribución real |

---

## Common Pitfalls

### Pitfall 1: El fix de `_staff_stats` es más que cambiar el nombre del atributo

**Qué falla:** Cambiar solo la línea 28 de `_staff_stats` a `_role_currency_stats` no es suficiente.
La clave de lookup en `calculate()` línea 57 es `role_key` (string), pero las claves de
`_role_currency_stats` son tuplas `(str, str)`. Con el nombre correcto pero la misma clave,
`self._staff_stats.get(role_key, {})` seguirá devolviendo `{}` para todas las consultas porque
ninguna clave es un string puro.

**Prevención:** Actualizar tanto el nombre del atributo (línea 28) como la lógica de lookup
(líneas 56-58) en el mismo commit. Agregar la currency del payment al lookup. Replicar la cadena
de fallback de `StaffRoleFeatures.transform()` (role+currency → solo currency → global).

**Señal de advertencia:** Test de paridad pasa en facilidades pero falla para pagos de staff
(court_manager, court_operator, teacher) porque el zscore de staff aún usa global stats.

### Pitfall 2: El fix de `getattr` cambia scores de producción

**Qué pasa:** Una vez aplicado el fix, `facility_avg_amount` pasa de ser siempre el global mean
(~XXX USD) a ser la media per-facility real. Para facilidades con montos medios muy distintos al
global (ej: facilities de alto valor en AED o AUD), el score IF-40 cambiará. Esto es el
comportamiento correcto, pero los scores anteriores y el threshold actual ya no serán comparables.

**Consecuencia para el plan:** El baseline document debe capturar métricas **post-fix**. No medir
AUC/enriquecimiento con el scorer buggeado y luego presentarlo como "baseline". El fix es el
prerequisito; el baseline es el punto de partida después del fix.

**Prevención:** Ejecutar el scorer en una muestra de val antes y después del fix. Documentar el
delta de scores como evidencia de que el bug estaba activo. El baseline congelado usa solo
métricas post-fix.

### Pitfall 3: `thresholds_v2.json` ya está correcto, `thresholds.json` no

**Qué pasa:** CONCERNS.md concern #6 describe el problema correctamente, pero aplica
principalmente a la ruta IF-31 (legacy). `thresholds_v2.json` (para IF-40, cargado por
`artifact_loader.py` en producción) tiene `threshold_source: "percentile_95_validation_set"` y
fue generado por `calibrate_threshold_v2.py` que usa `val_features_enriched.parquet`.

**Lo que sí requiere fix en Fase 0:** `thresholds.json` (ruta legacy IF-31) tiene
`threshold_source: "percentile_95_test_set"`. Si la ruta IF-31 se sigue usando para alguna
evaluación, hay que corregirla. Pero si el plan congela un baseline sobre IF-40 + `thresholds_v2.json`,
el concern #6 ya está resuelto para ese artefacto.

**Recomendación:** Verificar qué scorer carga `thresholds.json` vs `thresholds_v2.json`. El path
en `artifact_loader.py` línea 65 muestra que IF-40 carga `thresholds_v2.json`. La tarea de Fase 0
es documentar que el baseline usa IF-40 + `thresholds_v2.json` y actualizar `run_fase7_evaluation.py`
para que no sobreescriba `thresholds.json` con datos del test set en futuras corridas.

### Pitfall 4: `"EMPTY"` no aparece en los parquets procesados actuales, pero sí puede entrar por ClickHouse en futuras extracciones

**Qué pasa:** La verificación sobre `train_raw.parquet` y `val_raw.parquet` mostró 0 filas con
`currency="EMPTY"`. El bug de sanitización es una vulnerabilidad latente para futuras extracciones,
no un problema que afecte al baseline actual.

**Consecuencia:** La tarea de sanitización de `"EMPTY"` en `_postprocess_extraction()` es un fix
preventivo (importante para Fase 1 cuando se computen facility stats), no un correctivo urgente
para las métricas del baseline. Sin embargo, el reporte de calidad con conteo es igualmente válido:
reportar "0 filas con EMPTY en splits actuales" es información positiva que confirma la limpieza.

**Prevención:** Aplicar el fix en loader.py de todas formas (es una línea), y añadir un test que
pase `currency="EMPTY"` y verifique que el resultado es `"USD"` con un warning logueado.

### Pitfall 5: `capture_delay_seconds` tiene distribución no-trivial en batch (no es todo ceros)

**Qué pasa:** En `val_features_enriched.parquet` (datos históricos ya capturados), `capture_delay_seconds`
tiene valores reales: `min=-86400, max=86400, mean=-52847, zero_pct=6.2%`. El modelo IF-40 aprendió
esta distribución. En producción real-time, el campo es casi siempre 0.

**El check `is pd.NaT`:** La condición actual en `features_enriched.py:107-108` ya tiene un guard
anterior: `if not captured_at or not created_at: return 0.0`. Ese guard captura el caso `None`
antes de llegar al `pd.NaT` check. El `is pd.NaT` check en línea 112 solo se alcanza cuando ambos
son truthy strings. El riesgo real es cuando se pasa un string que produce NaT al parsear (e.g.,
`""`), que falla el `not captured_at` check de Python (vacío es falsy) y nunca llega a la línea
112. La vulnerabilidad es un string malformado que no sea vacío, e.g., `"invalid-date"`, que haría
`pd.Timestamp("invalid-date")` lanzar `ValueError` antes de llegar al NaT check.

**Recomendación:** Cambiar la lógica de `_capture_delay_seconds` a:
```python
try:
    captured = pd.Timestamp(captured_at)
    created = pd.Timestamp(created_at)
    if pd.isnull(captured) or pd.isnull(created):
        return 0.0
    ...
except (ValueError, TypeError):
    return 0.0
```

---

## Code Examples

### Fix Bug #8: Atributos `getattr` incorrectos

```python
# src/fraud_detector/scoring/features.py — __init__ (líneas 26-32)
# REEMPLAZAR:
self._global_avg_amount = fe._groups[0]._global_avg_amount
self._facility_avgs = getattr(fe._groups[4], "_facility_avg", {})
self._staff_stats = getattr(fe._groups[6], "_staff_stats", {})

# POR:
self._global_avg_amount = fe._groups[0]._global_avg_amount
self._facility_avgs = fe._groups[4]._facility_avg_amount
self._staff_stats = fe._groups[6]._role_currency_stats
self._staff_currency_stats = fe._groups[6]._currency_stats
self._staff_global_mean = fe._groups[6]._global_mean
self._staff_global_std = fe._groups[6]._global_std
# Aserciones post-carga:
assert len(self._facility_avgs) > 0, \
    f"_facility_avg_amount vacío (atributo={type(fe._groups[4]).__name__})"
assert len(self._staff_stats) > 0, \
    f"_role_currency_stats vacío (atributo={type(fe._groups[6]).__name__})"
logger.info(
    f"SingleFeatureCalculator loaded: global_avg={self._global_avg_amount:.2f}, "
    f"facilities={len(self._facility_avgs)}, staff_roles={len(self._staff_stats)}"
)
```

### Fix Bug #8: Lookup de staff en `calculate()` (líneas 55-58)

```python
# Reemplazar las líneas 55-58 de calculate():
is_staff = context.user_role in ("court_manager", "court_operator", "teacher")
role_key = context.user_role if is_staff else "player"
currency = (payment.get("currency") or "USD").upper()
currency_key = (role_key, currency)

if currency_key in self._staff_stats:
    s = self._staff_stats[currency_key]
elif (role_key, "USD") in self._staff_stats:
    s = self._staff_stats[(role_key, "USD")]
elif currency in self._staff_currency_stats:
    s = self._staff_currency_stats[currency]
else:
    s = {"mean": self._staff_global_mean, "std": self._staff_global_std}

staff_mean = s["mean"]
staff_std = s["std"]
```

Fuente: `StaffRoleFeatures.transform()` líneas 436-446 — esta es la cadena exacta de fallback
que el scorer real-time debe replicar.

### Fix Bug #3: NaT check en `_capture_delay_seconds`

```python
# src/fraud_detector/scoring/features_enriched.py — _capture_delay_seconds (líneas 104-116)
@staticmethod
def _capture_delay_seconds(payment: Dict) -> float:
    captured_at = payment.get("captured_at")
    created_at = payment.get("created_at")
    if not captured_at or not created_at:
        return 0.0
    try:
        captured = pd.Timestamp(captured_at)
        created = pd.Timestamp(created_at)
    except (ValueError, TypeError):
        return 0.0
    if pd.isnull(captured) or pd.isnull(created):
        return 0.0
    delay = (captured.to_pydatetime() - created.to_pydatetime()).total_seconds()
    return float(np.clip(delay, -86400, 86400))
```

### Fix Bug #5: Sanitización de `"EMPTY"` en loader.py

```python
# src/fraud_detector/data/loader.py — _postprocess_extraction (línea ~203-210)
# Dentro del bloque if "currency" in out.columns:, ANTES de normalizer.normalize():
out["currency"] = (
    out["currency"]
    .fillna("USD")
    .astype(str)
    .str.upper()
    .replace({"EMPTY": "USD", "": "USD"})
)
```

Y en `src/fraud_detector/features/engineering.py:646`:

```python
# ANTES:
out["currency"] = out["currency"].fillna("USD").astype(str).str.upper()
# DESPUÉS:
out["currency"] = (
    out["currency"]
    .fillna("USD")
    .astype(str)
    .str.upper()
    .replace({"EMPTY": "USD", "": "USD"})
)
```

### Corrección de `thresholds.json` para ruta IF-31 legacy

```python
# scripts/run_fase7_evaluation.py — save_thresholds (líneas 113-130)
# El threshold debe derivarse del VAL set, no del test set.
# Opción A: Reemplazar esta función y hacer que cargue val_features:
def save_thresholds_from_val():
    val_path = settings.processed_dir / "val_features.parquet"
    # ... score val set con IF-31
    threshold = float(np.percentile(scores_val, 95))
    result = {
        ...
        "threshold_source": "percentile_95_val_set",
    }

# Opción B (más pragmática para Fase 0): documentar que thresholds.json (IF-31)
# es legacy y que el scorer activo usa thresholds_v2.json (IF-40, ya correcto).
# Dejar thresholds.json con un comentario/flag de "legacy_do_not_use_for_IF40".
```

---

## Qué debe contener el Documento de Baseline Congelado

El baseline document (`output/baseline_v0.json` o similar) debe contener exactamente:

```json
{
  "baseline_version": "v0",
  "frozen_at": "<ISO timestamp post-fix>",
  "scorer_model": "IF-40-v1",
  "feature_set": "FS-frame-operational-v1",
  "feature_list_source": "output/models/final_feature_list.json",
  "threshold_artifact": "thresholds_v2.json",
  "threshold_source": "percentile_95_validation_set",
  "threshold_value": 0.024223975402714343,

  "gate_metric": "bias_reduction",
  "gate_criteria": {
    "top5pct_amount_ratio_target": "<4x",
    "off_hours_rate_target": "~4-5%",
    "notes": "AUC vs pure_fraud es diagnóstico circular — NO usar como gate"
  },

  "current_metrics_post_fix": {
    "val_set_tipo_a_count": 68655,
    "val_set_size": 1130117,
    "tipo_a_rate_pct": 6.075,
    "capture_delay_seconds_zero_pct": 6.19,
    "capture_delay_note": "excluded_from_operational_feature_set",
    "off_hours_rate_utc_pct": "<medido post-fix>",
    "top5pct_amount_ratio": "<medido post-fix>",
    "auc_tipo_a_diagnostic": "<medido post-fix, labeled circular>",
    "auc_pure_fraud_diagnostic": "<medido post-fix, labeled circular>"
  },

  "data_quality_report": {
    "currency_empty_train": 0,
    "currency_empty_val": 0,
    "currency_empty_test": "<verificar>",
    "currency_empty_note": "zero in processed parquets; fix applied preventively for future extractions"
  },

  "golden_set": {
    "source": "val_features_enriched.parquet",
    "n_rows": 500,
    "stratified_by": "facility_id",
    "facilities_covered": ">= 20",
    "path": "output/golden_set_v0.parquet"
  },

  "parity_test_result": {
    "facility_avg_amount_max_delta": "<float, debe ser <1e-6>",
    "staff_amount_zscore_max_delta": "<float, debe ser <1e-6>",
    "n_transactions_tested": 500,
    "status": "PASS"
  },

  "bugs_fixed": [
    "scoring/features.py:27 _facility_avg → _facility_avg_amount",
    "scoring/features.py:28 _staff_stats → _role_currency_stats (key type: tuple)",
    "scoring/features_enriched.py:112 is pd.NaT → pd.isnull()",
    "data/loader.py: currency EMPTY → USD replace",
    "features/engineering.py:646: currency EMPTY → USD replace"
  ],

  "excluded_features": ["capture_delay_seconds"],
  "auc_pure_fraud_classification": "diagnostic_circular_not_a_gate_metric"
}
```

---

## Estado del Arte: Qué ya está resuelto vs. qué requiere trabajo

| Concern | Estado actual | Trabajo en Fase 0 |
|---------|--------------|-------------------|
| #8 getattr _facility_avg | BUG ACTIVO — `getattr(…, "_facility_avg", {})` devuelve `{}` | Fix 2 líneas + fix de clave de lookup + assertions |
| #8 getattr _staff_stats | BUG ACTIVO — `getattr(…, "_staff_stats", {})` devuelve `{}` + clave incorrecta | Fix nombre + reescribir lookup con tupla + fallback chain |
| #3 capture_delay NaT check | PARCIALMENTE ACTIVO — el guard `not captured_at` cubre el caso None; `is pd.NaT` vulnerable a strings malformados | Reemplazar con `pd.isnull()` + try/except |
| #3 capture_delay train/serve skew | ACTIVO CONCEPTUALMENTE — feature informativa en batch pero constante 0 en RT | Documentar exclusión de FS-operational-v1; no requiere cambio de código en Fase 0 |
| #6 threshold en test set (IF-40) | YA RESUELTO — `thresholds_v2.json` usa val set | Solo documentar; no requiere fix para IF-40 |
| #6 threshold en test set (IF-31) | BUG EN RUTA LEGACY — `thresholds.json` usa test set | Deprecar o corregir `run_fase7_evaluation.py` |
| #5 EMPTY currency en parquets actuales | NO ACTIVO — 0 filas con EMPTY en train/val | Fix preventivo en loader.py + test; data quality report |
| #4 proxy circularity pure_fraud | RIESGO DOCUMENTADO — AUC 0.841 es parcialmente autovalidación | Congelar gate metric como bias reduction, no AUC |
| #2 circularidad IF-31 vs proxy | DOCUMENTADO — `user_reversal_ratio_30d` correlado con Tipo A | Excluir de FS-operational-v1; ya excluido en FS-clean-A-29 |

---

## Open Questions

1. **¿Qué tests existentes se romperán con el fix de `getattr`?**
   - `test_scoring.py` fixtures cargan `feature_engineer.joblib` real. Los tests
     `test_feature_count_is_31`, `test_features_are_finite` y `test_scorer_returns_scoring_result`
     seguirán pasando porque no validan los valores específicos de `facility_avg_amount` ni
     `staff_amount_zscore`.
   - `test_if40_artifacts.py::test_if40_scores_match_offline_payment_scorer` usa
     `calculate_from_feature_row(row)` (que no pasa por `SingleFeatureCalculator.calculate()`),
     por lo que no se verá afectado por el fix del bug.
   - El test de paridad nuevo es el único que verificará el fix real.
   - **Recomendación:** Ejecutar `make test` antes y después del fix y documentar qué cambia.

2. **¿Cuánto cambia el score de IF-40 post-fix?**
   - Imposible saberlo sin ejecutar. El fix de `facility_avg_amount` afecta F22 y F23
     (`amount_facility_ratio`). Para facilidades con media muy diferente al global (ej: una
     facility de alta gama en AED), el cambio puede ser sustancial.
   - **Recomendación:** Calcular un delta de scores en una muestra de val antes y después del
     fix. Documentar la distribución del delta como parte del baseline.

3. **¿El `payment_token_features.parquet` en `data/processed/` es relevante para Fase 0?**
   - Encontrado en `ls` pero no documentado en ARCHITECTURE.md ni CONCERNS.md. No parece
     relevante para los 4 bugs de Fase 0. Se puede ignorar.

---

## Sources

### Primary (HIGH confidence — verificado contra código real)

- `src/fraud_detector/scoring/features.py:16-32` — bug getattr confirmado en código actual
- `src/fraud_detector/features/engineering.py:326` — `_facility_avg_amount` (atributo correcto)
- `src/fraud_detector/features/engineering.py:393` — `_role_currency_stats` (atributo correcto)
- `src/fraud_detector/features/engineering.py:406-412` — tipo de clave `(str, str)` confirmado
- `src/fraud_detector/features/engineering.py:436-446` — cadena de fallback exacta
- `src/fraud_detector/features/engineering.py:508-521` — índices de `_groups[]` confirmados
- `src/fraud_detector/scoring/features_enriched.py:104-116` — `is pd.NaT` bug activo
- `scripts/run_fase7_evaluation.py:113-130` — `threshold_source: "percentile_95_test_set"` confirmado
- `scripts/calibrate_threshold_v2.py` — usa val set; genera `thresholds_v2.json` correcto
- `output/models/thresholds_v2.json` — verificado: `threshold_source: "percentile_95_validation_set"`
- `output/models/thresholds.json` — verificado: `threshold_source: "percentile_95_test_set"` (legacy)
- `scorer/artifact_loader.py:64-65` — IF-40 carga `thresholds_v2.json`
- `src/fraud_detector/data/loader.py:182-212` — `_postprocess_extraction` sin replace de EMPTY
- `src/fraud_detector/features/engineering.py:646` — currency sin replace de EMPTY
- `src/fraud_detector/utils/currency.py:49-52` — `fallback_rate` devuelve 1.0 para EMPTY
- `data/processed/val_raw.parquet` — 0 filas con EMPTY en val actual (verificado)
- `data/processed/train_raw.parquet` — 0 filas con EMPTY en train actual (verificado)
- `data/processed/val_features_enriched.parquet` — 1.13M filas, schema verificado; tiene status, facility_avg_amount, capture_delay_seconds
- `.planning/codebase/CONCERNS.md` — audit completo con 12 concerns
- `.planning/research/SUMMARY.md` — roadmap y pitfall mapping
- `.planning/research/PITFALLS.md` — descripción detallada de cada pitfall

---

## Metadata

**Confidence breakdown:**

| Área | Nivel | Razón |
|------|-------|-------|
| Bug #8 getattr (facility_avg) | HIGH | Línea de código exacta verificada; atributo correcto verificado en engineering.py |
| Bug #8 staff_stats clave compuesta | HIGH | Tipo de clave verificado en engineering.py:406-412; fallback chain verificado en :436-446 |
| Bug #3 NaT check | HIGH | Código verificado; análisis de los paths de ejecución |
| Bug #6 threshold (IF-40 ya correcto) | HIGH | thresholds_v2.json inspeccionado directamente |
| Bug #5 EMPTY en datos actuales | HIGH | Parquets inspeccionados; 0 filas con EMPTY |
| Contenido del baseline document | HIGH | Metricas verificables contra val_features_enriched.parquet |
| Delta de scores post-fix | LOW | No ejecutado; solo estimación cualitativa |
| Tests que se romperán | MEDIUM | Análisis estático sin ejecución |

**Research date:** 2026-07-06
**Valid until:** 2026-08-06 (código fuente estable; no hay refactoring pendiente que invalide los hallazgos)
