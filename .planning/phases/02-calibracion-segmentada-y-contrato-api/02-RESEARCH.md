# Phase 2: Calibración Segmentada y Contrato API — Research

**Researched:** 2026-07-06
**Domain:** Segmented threshold calibration (IsolationForest, frame-v1), versioned artifact extension, Pydantic v2 API contract
**Confidence:** HIGH — all findings verified against the real codebase, actual artifact data, and live val scores

---

## Summary

Esta fase tiene tres componentes mecánicamente independientes que comparten el mismo artefacto raíz (`facility_stats_v1.json`): (1) extender `currency_fallbacks` en el builder de Fase 1 para cubrir todas las monedas con suficiente volumen, (2) construir `thresholds_segmented_v1.json` como un artefacto de umbrales calibrados sobre el val set con cadena de fallback facility→currency→global y guarda de n≥200, y (3) extender `Artifacts` y los esquemas Pydantic del scorer para que los nuevos campos sean observables sin defaults silenciosos.

Los scores de val para el modelo frame-v1 se calcularon offline durante este research (`-model.decision_function(X_val_scaled)` sobre 1,130,117 filas). El global p95 es 0.0436. En el val set hay 452 facilities con n≥200 (cubren 99.1% de las transacciones), y 17 monedas con n≥200 (MXN n=88 e INR n=2 caen a global). Estos números determinan directamente la distribución de `fallback_level` esperada.

El concepto de `currency_group` que aparece en el objetivo se puede simplificar: dado que cada moneda tiene suficiente n en val para ser su propia clave de segmento, `currency_group` = código de moneda directamente. No se necesita clustering. La cadena facility→currency→global es ya completa con este modelo.

**Primary recommendation:** Implementar la fase en tres tareas independientes (builder extension, calibrador, contrato API), unidas únicamente por la regeneración de `facility_stats_v1.json` como paso previo. Toda la calibración usa el val set; el test set permanece intocable.

---

## Standard Stack

### Core (ya instalado, sin cambios)

| Library | Version verificada | Purpose | Notes |
|---------|--------------------|---------|-------|
| scikit-learn | 1.6.1 | IsolationForest.decision_function | frame-v1 model ya entrenado |
| numpy | installed | percentile, searchsorted, LUT | calibración de umbrales |
| pandas | 2.3.3 | groupby segmentos, apply features | |
| pydantic | 2.12.5 | API schemas, Optional[str]=None semantics | Pydantic v2 — ver nota |
| joblib | 1.5.3 | carga de model/scaler (no de stats dict) | stats dict → JSON |
| scipy.stats | installed | ks_2samp (para currency_group analysis) | solo en offline calibration |
| json | stdlib | serialización de todos los artefactos nuevos | |

### Nota Pydantic v2 crítica

En Pydantic v2 (instalado: 2.12.5), `Optional[str]` sin default es un campo **requerido** (lanza `ValidationError` si ausente). `Optional[str] = None` es opcional y por default es `None`. Esta diferencia es la base del contrato API-01: nunca poner `= "UTC"` o `= "USD"` como default, sino `= None`, y detectar el `None` en el scorer para emitir `frame_flags`.

### Artefactos existentes que se extienden (no se reemplazan)

| Artefacto | Path | Status en Fase 2 |
|-----------|------|------------------|
| `facility_stats_v1.json` | `output/models/` | Se regenera con currency_fallbacks ampliado |
| `model_metadata_frame_v1.json` | `output/models/` | Se añade `artifact_files` key |
| `Artifacts` dataclass | `scorer/artifact_loader.py` | Se añaden campos opcionales `facility_stats`, `thresholds_segmented` |
| `ScoreRequest` | `scorer/schemas.py` | Se cambian `currency` default y se añaden campos nuevos |
| `ScoreResponse` | `scorer/schemas.py` | Se añaden `calibration_segment`, `fallback_level`, `frame_flags` |

---

## Architecture Patterns

### Recommended Project Structure (archivos nuevos en Fase 2)

```
src/fraud_detector/
├── stats/
│   ├── builder.py          # MODIFY: _N_CURRENCY_FALLBACKS → threshold n>=1000
│   ├── tz_mapping.py       # sin cambios
│   └── validator.py        # sin cambios
├── calibration/            # NEW MODULE (entero)
│   ├── __init__.py
│   └── segmented.py        # SegmentedThresholdCalibrator + json serde
└── scoring/
    └── classifier.py       # MODIFY: add SegmentedThresholdClassifier

scorer/
├── artifact_loader.py      # MODIFY: Artifacts + facility_stats + thresholds_segmented
└── schemas.py              # MODIFY: ScoreRequest currency, ScoreResponse new fields

output/models/
├── facility_stats_v1.json          # REGENERATED (extended currency_fallbacks)
└── thresholds_segmented_v1.json    # NEW

scripts/
└── calibrate_segmented_thresholds.py  # NEW offline script

tests/
├── test_facility_stats_builder.py  # EXTEND: test new currency_fallbacks
└── test_calibration_segmented.py   # NEW
```

---

### Pattern 1: Currency Fallbacks Extension (builder.py)

**What:** Cambiar `_N_CURRENCY_FALLBACKS = 5` (top-N por volumen) por un criterio basado en umbral de n en train.

**Hallazgo verificado:** Las monedas con train_n ≥ 1000 son:
```
Ya en fallback: USD(2,384,422), CAD(173,424), MYR(148,405), HNL(112,845), NIO(92,180)
Añadir:  AUD(67,003), ILS(53,658), GTQ(32,739), PKR(26,547), HKD(21,053),
         AED(8,545), BWP(6,887), SGD(5,471), COP(2,650)
Quedan en global: JPY(962), RWF(166), INR(63), MXN(54), EUR(9)
```

**Implementación:**
```python
# src/fraud_detector/stats/builder.py
# Reemplazar _N_CURRENCY_FALLBACKS = 5 con umbral explícito:
_MIN_CURRENCY_N = 1_000  # monedas con menos rows en train caen a global

# En FacilityStatsBuilder.build():
top_currencies = (
    train_df[train_df["currency"] != "EMPTY"]
    .groupby("currency")
    .size()
    .pipe(lambda s: s[s >= _MIN_CURRENCY_N])
    .index.tolist()
)
if "USD" not in top_currencies:
    top_currencies.append("USD")  # siempre incluir USD
currency_fallbacks = self._compute_currency_fallbacks(train_df, top_currencies)
```

**Impacto en paridad:** Solo afecta 3 facilities de los 7 actualmente en `global` (fid=1214 GTQ→currency, fid=1232 AUD→currency, fid=1373 HKD→currency). Las 580 facilities con `fallback_level='facility'` (n≥30) no cambian en nada. `test_parity_phase1.py` debe re-correr después de regenerar para confirmar que sigue en PASS.

---

### Pattern 2: SegmentedThresholdCalibrator

**What:** Clase offline que calcula umbrales por segmento (facility, currency, global) desde scores de val. Produce `thresholds_segmented_v1.json`.

**Inputs reales:**
- Scores frame-v1 val: 1,130,117 valores en [-0.1419, 0.2254], global p95 = 0.0436
- Segmentos: 452 facilities con n≥200, 17 monedas con n≥200
- Convención de score: -model.decision_function(X_scaled) → mayor = más anómalo

**Estructura de `thresholds_segmented_v1.json`:**

```json
{
  "schema_version": "thresholds-segmented-v1",
  "built_at": "2026-...",
  "model_version": "frame-v1",
  "feature_version": "frame-v1",
  "calibration_source": "validation_set",
  "calibration_rows": 1130117,
  "min_n_threshold": 200,
  "percentile": 95,
  "binary_threshold": 0.043588,
  "score_percentiles": [/* 1001 puntos de p0 a p100 */],
  "threshold_version": "segmented-v1",
  "by_facility": {
    "1234": {
      "binary_threshold": 0.051,
      "n": 823,
      "fallback_level": "facility",
      "score_percentiles": [/* 1001 puntos */]
    }
  },
  "by_currency": {
    "USD": {
      "binary_threshold": 0.024,
      "n": 830112,
      "fallback_level": "currency",
      "score_percentiles": [/* 1001 puntos */]
    },
    "MYR": {
      "binary_threshold": 0.0995,
      "n": 58205,
      "fallback_level": "currency",
      "score_percentiles": [/* 1001 puntos */]
    }
    // ... 17 monedas total (MXN y INR no tienen entrada por n<200)
  }
}
```

**Diseño de `binary_threshold` + `score_percentiles` al nivel top del JSON:** Esto es intencionalmente backward-compatible con `artifact_loader._validate_artifacts`, que actualmente requiere `{'binary_threshold', 'score_percentiles'}` en el dict de thresholds. Al mantener estas keys al nivel raíz (con los valores globales), no es necesario modificar `_validate_artifacts`.

**Implementación de `SegmentedThresholdCalibrator`:**

```python
# src/fraud_detector/calibration/segmented.py
class SegmentedThresholdCalibrator:
    MIN_N = 200  # guarda: segmentos con n<200 no reciben threshold propio

    def fit(
        self,
        scores: np.ndarray,       # shape (n,), higher = more anomalous
        facility_ids: np.ndarray,  # shape (n,), int
        currencies: np.ndarray,    # shape (n,), str
        percentile: float = 95.0,
    ) -> dict:
        """Calibra umbrales por segmento sobre val set scores."""
        result = {}
        
        # Global (siempre disponible)
        global_threshold = float(np.percentile(scores, percentile))
        global_lut = np.percentile(scores, np.linspace(0, 100, 1001)).tolist()
        result["binary_threshold"] = global_threshold  # backward compat
        result["score_percentiles"] = global_lut       # backward compat
        
        # Por facility (n>=MIN_N)
        by_facility = {}
        for fid in np.unique(facility_ids):
            mask = facility_ids == fid
            n = int(mask.sum())
            if n < self.MIN_N:
                continue
            seg_scores = scores[mask]
            by_facility[str(fid)] = {
                "binary_threshold": float(np.percentile(seg_scores, percentile)),
                "n": n,
                "fallback_level": "facility",
                "score_percentiles": np.percentile(
                    seg_scores, np.linspace(0, 100, 1001)
                ).tolist(),
            }
        result["by_facility"] = by_facility
        
        # Por currency (n>=MIN_N)
        by_currency = {}
        for cur in np.unique(currencies):
            mask = currencies == cur
            n = int(mask.sum())
            if n < self.MIN_N:
                continue
            seg_scores = scores[mask]
            by_currency[cur] = {
                "binary_threshold": float(np.percentile(seg_scores, percentile)),
                "n": n,
                "fallback_level": "currency",
                "score_percentiles": np.percentile(
                    seg_scores, np.linspace(0, 100, 1001)
                ).tolist(),
            }
        result["by_currency"] = by_currency
        
        result.update({
            "schema_version": "thresholds-segmented-v1",
            "model_version": "frame-v1",
            "feature_version": "frame-v1",
            "calibration_source": "validation_set",
            "calibration_rows": int(len(scores)),
            "min_n_threshold": self.MIN_N,
            "percentile": percentile,
            "threshold_version": "segmented-v1",
        })
        return result
```

---

### Pattern 3: SegmentedThresholdClassifier

**What:** Extiende `ThresholdClassifier` para usar cadena de fallback facility→currency→global.

**Inputs al classify():** `score: float, facility_id: int, currency: str` → devuelve `(is_anomaly, risk_level, percentile, fallback_level, calibration_segment)`.

```python
# src/fraud_detector/scoring/classifier.py
class SegmentedThresholdClassifier:
    """Clasificador con cadena de fallback: facility -> currency -> global."""

    def __init__(self, config: dict):
        self._global_threshold = config["binary_threshold"]
        self._global_lut = np.array(config["score_percentiles"], dtype=np.float32)
        self._by_facility = config.get("by_facility", {})
        self._by_currency = config.get("by_currency", {})

    def classify(
        self, score: float, facility_id: int, currency: str
    ) -> tuple[bool, str, float, str, str]:
        """
        Returns: (is_anomaly, risk_level, percentile, fallback_level, calibration_segment)
        """
        fid_key = str(facility_id)
        
        # 1. Try facility-level
        if fid_key in self._by_facility:
            seg = self._by_facility[fid_key]
            threshold = seg["binary_threshold"]
            lut = np.array(seg["score_percentiles"], dtype=np.float32)
            fallback_level = "facility"
            calibration_segment = f"facility:{facility_id}"
        # 2. Try currency-level
        elif currency in self._by_currency:
            seg = self._by_currency[currency]
            threshold = seg["binary_threshold"]
            lut = np.array(seg["score_percentiles"], dtype=np.float32)
            fallback_level = "currency"
            calibration_segment = f"currency:{currency}"
        # 3. Global fallback
        else:
            threshold = self._global_threshold
            lut = self._global_lut
            fallback_level = "global"
            calibration_segment = "global"
        
        is_anomaly = score > threshold
        percentile = self._compute_percentile(score, lut)
        risk_level = _assign_risk_level(percentile)
        return is_anomaly, risk_level, percentile, fallback_level, calibration_segment
```

---

### Pattern 4: Artifacts dataclass — extensión retrocompatible

**Hallazgo crítico:** `model_metadata_frame_v1.json` NO tiene `artifact_files` key. El `artifact_loader._load_metadata()` actual usa `artifact_files` para saber qué cargar. Para que el frame-v1 path funcione, hay dos opciones:

**Opción elegida (mínima invasión):** Añadir `artifact_files` a `model_metadata_frame_v1.json` y registrar un segundo metadata file como `model_metadata_frame_v1_full.json`, o simplemente añadir los campos faltantes al `model_metadata_frame_v1.json` existente.

**Estructura de `Artifacts` extendida:**

```python
# scorer/artifact_loader.py
@dataclass(frozen=True)
class Artifacts:
    model: Any
    scaler: Any
    feature_list: list[str]
    thresholds: dict
    metadata: dict
    # Nuevos campos opcionales — None significa legacy path
    facility_stats: dict | None = None        # facility_stats_v1.json deserialized
    thresholds_segmented: dict | None = None  # thresholds_segmented_v1.json deserialized
```

**load_artifacts() extendido:**

```python
def load_artifacts(model_dir: Path) -> Artifacts:
    ...  # código existente sin cambios
    
    # Cargar facility_stats si existe referencia en metadata
    facility_stats = None
    stats_artifact = metadata.get("stats_artifact")
    if stats_artifact:
        stats_path = model_dir / stats_artifact
        if stats_path.exists():
            facility_stats = json.loads(stats_path.read_text())
    
    # Cargar thresholds_segmented si existe
    thresholds_segmented = None
    seg_artifact = metadata.get("thresholds_segmented_artifact")
    if seg_artifact:
        seg_path = model_dir / seg_artifact
        if seg_path.exists():
            thresholds_segmented = json.loads(seg_path.read_text())
    
    _validate_artifacts(model, scaler, feature_list, thresholds, metadata)
    
    return Artifacts(
        model=model,
        scaler=scaler,
        feature_list=feature_list,
        thresholds=thresholds,
        metadata=metadata,
        facility_stats=facility_stats,
        thresholds_segmented=thresholds_segmented,
    )
```

**Retrocompatibilidad garantizada:** Si `facility_stats=None` y `thresholds_segmented=None`, el scorer usa la ruta IF-40 legacy (ThresholdClassifier global, EnrichedFeatureCalculator). Ningún cambio en el path existente.

**Nota sobre `_validate_artifacts`:** No requiere cambios. La validación existente (`{'binary_threshold', 'score_percentiles'}`) funciona porque `thresholds_segmented_v1.json` expone estas keys al nivel raíz (ver Pattern 2). Sin embargo, es conveniente añadir un warning si `thresholds_segmented` no está presente pero `facility_stats` sí.

---

### Pattern 5: API Contract — ScoreRequest y ScoreResponse

**API-01: `ScoreRequest` — campos frame-v1 sin default silencioso**

```python
# scorer/schemas.py — cambios en ScoreRequest
class ScoreRequest(BaseModel):
    # ... campos existentes sin cambios ...
    
    # Cambiar: currency: str = "USD"  ->
    currency: Optional[str] = None
    # None = ausente, scorer usa currency del stats artifact o emite frame_flags.currency_missing=True
    
    # NUEVO: timezone IANA enviado por Rails (Fase 3 lo populará)
    facility_time_zone_iana: Optional[str] = None
    # None = ausente -> frame_flags.timezone_missing=True (NUNCA default silencioso a UTC)
    
    # NUEVO: monto en moneda local (para logging de contexto, no para features)
    amount_local: Optional[float] = None
```

**API-02: `ScoreResponse` — nuevos campos estructurados**

```python
class FrameFlags(BaseModel):
    timezone_missing: bool = False     # facility_time_zone_iana ausente en request
    currency_missing: bool = False     # currency ausente en request
    currency_unknown: bool = False     # currency='EMPTY' o no reconocida

class ScoreResponse(BaseModel):
    # Campos existentes (sin cambios para backward compat)
    raw_score: float
    percentile: float
    risk_level: str
    is_anomaly: bool
    factors: List[FactorItem]
    model_version: str
    feature_version: str
    threshold_version: str
    # NUEVOS (opcionales para no romper clientes legacy)
    calibration_segment: Optional[str] = None  # "facility:1234" | "currency:USD" | "global"
    fallback_level: Optional[str] = None        # "facility" | "currency" | "global"
    frame_flags: Optional[FrameFlags] = None
```

**Importante:** Los campos nuevos en `ScoreResponse` son `Optional` con `= None` para que clientes Rails que no consuman esos campos no se rompan. Esto es distinto del `ScoreRequest` donde queremos observabilidad: ahí `= None` significa "ausente, generar flag", no "no importa".

---

### Anti-Patterns to Avoid

- **Usar `np.percentile` en segmentos con n<200:** Produce umbrales inestables. La guarda MIN_N=200 no es negociable. Verificación: todo entry en `by_facility` o `by_currency` tiene `"n" >= 200`.
- **Calibrar sobre el test set:** Los scores de val para calibración ya están disponibles (1,130,117 rows). Test set permanece intocable. El archivo `thresholds_v2.json` ya usa validation set correctamente; seguir el mismo patrón.
- **Default silencioso `= "UTC"` para timezone:** El pitfall documentado en PITFALLS.md Pitfall 9. Siempre `= None` + `frame_flags.timezone_missing=True`.
- **Copiar `_N_CURRENCY_FALLBACKS` como hardcode en tests:** El test `test_builder_min_n` actualmente verifica `MIN_N == 30`. Añadir test análogo para `_MIN_CURRENCY_N == 1000` (o el valor que se elija).
- **LUT global de 1001 puntos por cada facility:** Genera un JSON de ~50MB. Considerar reducir LUT por segmento a 101 puntos; mantener 1001 solo para global. La LUT de 1001 puntos en `thresholds_v2.json` es para el scorer IF-40; el nuevo puede usar 201 por segmento sin pérdida de resolución percibida.
- **Cargar `facility_stats` desde ClickHouse en real-time:** Anti-pattern documentado en ARCHITECTURE.md. El artefacto JSON se carga una vez al startup y se accede como dict en O(1).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Percentile threshold calibration | Código ad-hoc por segmento | `numpy.percentile(scores[mask], 95)` | numpy garantiza exactitud, IQR de muestreo correcto |
| Score percentile LUT | Interpolación custom | `np.searchsorted(lut, score) / len(lut)` | El patrón ya existe en `ThresholdClassifier._compute_percentile()` — reusar exactamente |
| Currency clustering | Algoritmo de clustering KMeans/jerárquico | Currency = su propio grupo (one-to-one) | Los datos muestran que 17/19 monedas tienen n≥200 en val, no hay necesidad de agrupar |
| Pydantic "sentinel not-provided" | `MISSING = object()` custom sentinel | `Optional[str] = None` + check `is None` | Pydantic v2 maneja correctamente: Optional sin default = required, Optional con = None = opcional |

---

## Common Pitfalls

### Pitfall 1: frame-v1 val scores no están en `output/scores/`

**What goes wrong:** Los archivos en `output/scores/` son del modelo IF-40 (enriched-40 features). `if_val_scores_final.npy` tiene p95=0.024 (IF-40). El modelo frame-v1 tiene p95=0.044 (calculado durante este research). Usar los scores de IF-40 para calibrar los umbrales del frame-v1 produce umbrales completamente incorrectos.

**How to avoid:** El script `calibrate_segmented_thresholds.py` debe computar los scores del frame-v1 desde cero: cargar `isolation_forest_frame_v1.joblib` + `scaler_frame_v1.joblib`, calcular features via `FrameV1FeatureCalculator.calculate_from_row()` sobre `val_features_enriched.parquet`, y aplicar `-model.decision_function(X_scaled)`. Mismo patrón que `scripts/retrain_frame_v1.py` líneas 499-508.

**Warning signs:** p95 del threshold segmentado global ≈ 0.024 (es el IF-40 p95, no frame-v1).

### Pitfall 2: model_metadata_frame_v1.json no tiene `artifact_files`

**What goes wrong:** `artifact_loader._load_metadata()` busca `artifact_files` en `model_metadata.json`. El archivo `model_metadata_frame_v1.json` actualmente tiene `feature_version: "frame-v1"` y `stats_artifact: "facility_stats_v1.json"` pero NO tiene `artifact_files`. Ni `model_version` ni `score_function` ni `threshold_version`.

**How to avoid:** Como parte de la tarea de artifact_loader, añadir los campos faltantes a `model_metadata_frame_v1.json`:
```json
{
  "model_version": "frame-v1",
  "score_function": "decision_function",
  "threshold_version": "segmented-v1",
  "artifact_files": {
    "model": "isolation_forest_frame_v1.joblib",
    "scaler": "scaler_frame_v1.joblib",
    "feature_list": "model_metadata_frame_v1.json",
    "thresholds": "thresholds_segmented_v1.json"
  },
  "stats_artifact": "facility_stats_v1.json",
  "thresholds_segmented_artifact": "thresholds_segmented_v1.json"
}
```
Nota: `feature_list` puede apuntar al mismo metadata file (el campo `feature_names` ya contiene la lista) o a un `feature_list_frame_v1.json` separado.

### Pitfall 3: `_validate_artifacts` falla con score_function=decision_function para frame-v1

**What goes wrong:** `_validate_artifacts` verifica que `score_function in {"decision_function", "score_samples"}`. Para frame-v1 usamos `decision_function` (negado). Pero la negación (`-model.decision_function(...)`) ocurre en el script de calibración, no en el scorer. En el scorer, si se usa `ThresholdClassifier` (o `SegmentedThresholdClassifier`) con scores ya negados (mayores = más anómalos), entonces el scorer debe llamar `-model.decision_function()` internamente.

**Clarificación del score convention:**
- `model.decision_function(X)` para IsolationForest: mayor positivo = más inlier, negativo = outlier
- Todos los artefactos de thresholds (incluyendo thresholds_v2.json) asumen: mayor score = más anómalo
- `ThresholdClassifier.classify()` hace `score > threshold` → es_anomalía
- El scorer debe calcular `score = -model.decision_function(X_scaled)` para que el signo sea coherente

**How to avoid:** Verificar que `SingleTransactionScorer.score_features()` aplica la negación cuando `score_function == "decision_function"`.

### Pitfall 4: LUT de 1001 puntos × 452 facilities → JSON de ~50MB

**What goes wrong:** 452 facilities × 1001 floats × 8 bytes ≈ 3.6MB solo en by_facility, más 17 monedas × 1001 floats. El tamaño total es manejable (~4MB para 1001 puntos). Sin embargo, si se usa 1001 puntos por segmento, la carga al startup (json.loads de 4MB) añade ~20ms, aceptable.

**How to avoid:** Opción A: usar 201 puntos por segmento (suficiente para ~0.5% de resolución). Opción B: 1001 puntos igual que el global (consistencia con el patrón existente). Recomendación: 201 puntos por segmento, 1001 para global — la resolución per-segment no requiere más.

### Pitfall 5: `currency` cambio de `str = "USD"` a `Optional[str] = None` rompe el router

**What goes wrong:** `scorer/routers/score.py` llama `scorer.score(payment)` donde `payment = request.model_dump()`. Si `currency` era siempre un string y ahora puede ser `None`, el scorer y `FrameV1FeatureCalculator.calculate()` deben manejar `currency=None` explícitamente (usar la currency del stats artifact del facility, o emitir `frame_flags.currency_missing=True`).

**How to avoid:** Esta fase (Fase 2) solo define el contrato; el wiring completo del scorer para el path frame-v1 es Fase 3. En Fase 2 se implementa el contrato Pydantic y se añaden los campos, pero el router sigue usando el scorer IF-40. El campo `currency=None` en `ScoreRequest` se serializa y llega al scorer; el scorer IF-40 existente hace `payment.get("currency", "USD")` → si None hace `None or "USD"` = "USD" — backward compatible.

### Pitfall 6: test_facility_stats_builder.py `test_currency_fallbacks_usd_present` no cubre el nuevo criterio

**What goes wrong:** El test actual verifica que USD está en `currency_fallbacks`. No verifica el nuevo criterio (n≥1000) ni las monedas mandatadas. Después de regenerar el artefacto, tests que usan `synthetic_train_df` (con solo USD y CAD en pocos rows) no tendrán las nuevas monedas — correcto para unit tests sintéticos, pero hay que añadir un test de integración que verifique que el artefacto materializado tiene AUD, ILS, GTQ, PKR, SGD.

---

## Code Examples

### Calibración segmentada (script offline)

```python
# scripts/calibrate_segmented_thresholds.py
import json, numpy as np, pandas as pd, joblib, sys
sys.path.insert(0, "src")

from fraud_detector.scoring.features_frame_v1 import FrameV1FeatureCalculator
from fraud_detector.calibration.segmented import SegmentedThresholdCalibrator

# 1. Cargar modelo frame-v1 y scaler
model = joblib.load("output/models/isolation_forest_frame_v1.joblib")
scaler = joblib.load("output/models/scaler_frame_v1.joblib")

# 2. Cargar facility_stats (regenerado con currency_fallbacks ampliado)
with open("output/models/facility_stats_v1.json") as f:
    facility_stats = json.load(f)

# 3. Calcular features frame-v1 sobre val set
val_df = pd.read_parquet("data/processed/val_features_enriched.parquet")
calc = FrameV1FeatureCalculator(
    facility_stats=facility_stats,
    feature_engineer_path="output/models/feature_engineer.joblib"
)
X_val = np.vstack(
    val_df.apply(lambda row: calc.calculate_from_row(row), axis=1).values
).astype(np.float32)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_val_scaled = np.clip(scaler.transform(X_val), -10, 10).astype(np.float32)

# 4. Scores frame-v1 val (mayor = más anómalo)
# Source: scripts/retrain_frame_v1.py líneas 503-508
scores = -model.decision_function(X_val_scaled)

# 5. Calibrar umbrales segmentados
calibrator = SegmentedThresholdCalibrator()
thresholds_dict = calibrator.fit(
    scores=scores,
    facility_ids=val_df["facility_id"].to_numpy(),
    currencies=val_df["currency"].to_numpy(dtype=str),
    percentile=95.0,
)

# 6. Guardar artefacto
with open("output/models/thresholds_segmented_v1.json", "w") as f:
    json.dump(thresholds_dict, f, indent=2)
print(f"Saved: {len(thresholds_dict['by_facility'])} facility thresholds, "
      f"{len(thresholds_dict['by_currency'])} currency thresholds")
```

### Lookup de percentil (patrón reutilizado del ThresholdClassifier existente)

```python
# src/fraud_detector/scoring/classifier.py
# Source: existing ThresholdClassifier._compute_percentile() — reusar exactamente
def _compute_percentile(score: float, lut: np.ndarray) -> float:
    if len(lut) == 0:
        return 0.5
    idx = np.searchsorted(lut, score)
    return min(idx / len(lut), 1.0)
```

### Extensión de builder para currency_fallbacks

```python
# src/fraud_detector/stats/builder.py — sección modificada
_MIN_CURRENCY_N = 1_000  # reemplaza _N_CURRENCY_FALLBACKS = 5

# Dentro de FacilityStatsBuilder.build():
eligible_currencies = (
    train_df[train_df["currency"].isin(
        [c for c in train_df["currency"].unique() if c != "EMPTY"]
    )]
    .groupby("currency")
    .size()
    .pipe(lambda s: s[s >= _MIN_CURRENCY_N])
    .index.tolist()
)
if "USD" not in eligible_currencies:
    eligible_currencies.append("USD")
currency_fallbacks = self._compute_currency_fallbacks(train_df, eligible_currencies)
```

### Artifact loader — campos opcionales con retrocompatibilidad

```python
# scorer/artifact_loader.py
@dataclass(frozen=True)
class Artifacts:
    model: Any
    scaler: Any
    feature_list: list[str]
    thresholds: dict
    metadata: dict
    facility_stats: dict | None = None        # None = legacy path
    thresholds_segmented: dict | None = None  # None = usa ThresholdClassifier global
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Top-5 currencies por volumen (`_N_CURRENCY_FALLBACKS=5`) | n≥1000 en train (14 monedas) | 3 facilities pasan de global→currency; nuevas facilities en AUD/ILS/GTQ/PKR/HKD tienen mejor fallback |
| Umbral global único (thresholds_v2.json, p95=0.024 IF-40) | Umbral segmentado (frame-v1 p95=0.044 global, variable por facility/currency) | MYR p95=0.100, AED p95=0.101 vs USD p95=0.024 — diferencias de 4x no captadas por umbral global |
| `currency: str = "USD"` (default silencioso) | `currency: Optional[str] = None` + `frame_flags` | Ausencia es observable; Rails puede omitir el campo sin contaminar el comportamiento |
| `Artifacts` inmutable de 5 campos | `Artifacts` con 2 campos opcionales adicionales | Retrocompatible: None→legacy path, no requiere redeploy de IF-40 |

**Deprecated:**
- `_N_CURRENCY_FALLBACKS = 5`: reemplazado por `_MIN_CURRENCY_N = 1_000`
- `ThresholdClassifier` para el path frame-v1: reemplazado por `SegmentedThresholdClassifier` (el existente se mantiene para IF-40 legacy)

---

## Open Questions

1. **Feature list para frame-v1 en artifact_files**
   - What we know: `model_metadata_frame_v1.json` tiene `feature_names` como lista interna; no hay `feature_list_frame_v1.json` separado
   - What's unclear: ¿Debe `artifact_files.feature_list` apuntar a un JSON separado o al propio metadata?
   - Recommendation: Crear `feature_list_frame_v1.json` con solo el array de nombres, análogo a `final_feature_list.json`. Más limpio que leer la lista del metadata.

2. **Score_percentiles LUT: 201 vs 1001 puntos por segmento**
   - What we know: El LUT de 1001 puntos existe en los artefactos legacy y en ThresholdClassifier
   - What's unclear: ¿Hay tests que hardcodeen `len(lut) == 1001`?
   - Recommendation: Verificar antes de implementar. Si hay tests que asumen 1001, usar 1001 también para segmented. Si no, usar 201 por segmento para reducir tamaño del JSON.

3. **Regenerar facility_stats: ¿se ejecuta desde un script dedicado o inline?**
   - What we know: `scripts/retrain_frame_v1.py` ya genera el modelo; `builder.py` ya existe con `FacilityStatsBuilder`
   - What's unclear: No existe un `scripts/build_facility_stats.py` standalone (ARCHITECTURE.md lo menciona como NEW pero no existe aún)
   - Recommendation: La tarea del builder debe incluir la creación de este script.

4. **`amount_local` en ScoreRequest: ¿se usa en features o solo en logging?**
   - What we know: `FrameV1FeatureCalculator` no usa `amount_local`; los features de magnitud se basan en `reservation_paid_out` normalizado a USD
   - What's unclear: El objetivo menciona `amount_local` como campo de contrato — ¿para qué exactamente?
   - Recommendation: Incluir como campo de logging/contexto en `ScoreRequest` pero no pasar a `calculate()`. Útil para Fase 4 (shadow dashboard, comparar monto local vs USD).

---

## Sources

### Primary (HIGH confidence — codebase directo)

- `scorer/artifact_loader.py` — estructura actual de `Artifacts`, `_validate_artifacts`, `_load_metadata`
- `scorer/schemas.py` — `ScoreRequest` actual (currency: str = "USD"), `ScoreResponse`
- `src/fraud_detector/stats/builder.py` — `FacilityStatsBuilder`, `_N_CURRENCY_FALLBACKS = 5`
- `src/fraud_detector/scoring/classifier.py` — `ThresholdClassifier`, patrón `_compute_percentile`
- `src/fraud_detector/scoring/features_frame_v1.py` — `FrameV1FeatureCalculator.calculate_from_row()`
- `output/models/facility_stats_v1.json` — estructura verificada: 1876 facilities, 5 currency fallbacks
- `output/models/model_metadata_frame_v1.json` — sin `artifact_files`, con `stats_artifact`
- `output/models/thresholds_v2.json` — patrón de estructura (binary_threshold, score_percentiles, threshold_version)
- `tests/test_parity_phase1.py` — contrato de parity test existente
- `tests/test_facility_stats_builder.py` — tests existentes del builder
- `.planning/phases/02-calibracion-segmentada-y-contrato-api/02-CONTEXT.md` — directiva del usuario

### Datos computados en vivo durante este research (HIGH confidence)

- Frame-v1 val scores: 1,130,117 valores via `FrameV1FeatureCalculator.calculate_from_row()` + `-model.decision_function(X_scaled)`. Global p95 = 0.0436.
- KS distances por moneda (scipy.ks_2samp vs USD): MYR=0.31, EUR=0.50, HKD=0.24, NIO=0.28, HNL=0.21, COP=0.22
- Distribución de segmentos en val: 452 facilities n≥200 (99.1% txns), 17 monedas n≥200
- Currency_fallbacks extension: train_n threshold n≥1000 → 14 monedas (5 existentes + AUD,ILS,GTQ,PKR,HKD,AED,BWP,SGD,COP)
- Facilities en fallback `global` con su dominant_currency: fid=1214 (GTQ), fid=1232 (AUD), fid=1373 (HKD) → mejoran con extensión; fid=716,1377 (MXN), fid=1069 (EUR), fid=1380 (INR) → quedan en global (n<1000)

### Secondary (MEDIUM confidence — prior research)

- `.planning/research/ARCHITECTURE.md` — build order, component responsibilities, data flow
- `.planning/research/PITFALLS.md` — Pitfall 3 (segmentos ralos), Pitfall 6 (calibración sobre test set), Pitfall 9 (default silencioso en shadow mode)
- `.planning/research/STACK.md` — stack confirmado, tzdata, JSON vs joblib para stats dict

---

## Metadata

**Confidence breakdown:**
- Currency fallbacks extension: HIGH — datos de train reales, análisis de impacto verificado
- Segment sizes / min-n=200 decision: HIGH — computado sobre val real con frame-v1 scores
- thresholds_segmented_v1.json structure: HIGH — basado en artefactos existentes y análisis de backward compat
- artifact_loader extension: HIGH — código fuente leído completo
- Pydantic v2 API contract: HIGH — verificado en runtime (Python 3.9, pydantic 2.12.5)
- currency_group simplification: HIGH — datos confirman que currency directa es suficiente
- LUT size recommendation (201 vs 1001): MEDIUM — no hay tests con hardcode verificado

**Research date:** 2026-07-06
**Valid until:** 2026-08-06 (artefactos estables; score_samples convention fija)
