# Phase 3: Wiring del Scorer e Integración Platform — Research

**Researched:** 2026-07-06
**Domain:** Python FastAPI scorer (ml-fraud-detector) + Ruby on Rails pack anomaly_detection (platform)
**Confidence:** HIGH — basado en lectura directa del código fuente de ambos repos

---

## Summary

Esta fase reconecta el scorer en vivo para despachar frame-v1 cuando los artefactos correspondientes están presentes, y extiende el pack Rails con los tres campos nuevos del contrato de respuesta (`calibration_segment`, `fallback_level`, `frame_flags`). También alinea la lógica `scorable?` entre batch y real-time, y añade una migración para persistir los nuevos campos de metadata en la tabla `anomaly_detection_alerts`.

La investigación verificó el código real de ambos repos. Los hallazgos más importantes son:

1. El `artifact_loader.py` YA carga `facility_stats` y `thresholds_segmented` opcionalmente y los expone en `Artifacts`. El `model_metadata_frame_v1.json` YA está generado y referencia ambos artefactos. El scorer NO los usa todavía (usa `ThresholdClassifier` en todos los casos y el `feature_calc` se decide por conteo de features).

2. La resolución de IANA está resuelta del lado del scorer: `FrameV1FeatureCalculator._lookup_facility()` la lee del artefacto `facility_stats_v1.json` que tiene `iana_tz` para las 1876 facilities conocidas, con fallback a `"Etc/UTC"`. Rails NO necesita enviar `facility_time_zone_iana` — el scorer resuelve la zona por sí mismo. Además, `facility.tzinfo_identifier` existe como columna de DB (safer que `time_zone_identifier`).

3. El gap `scorable?` es real: el batch scorer excluye `reversal` y `free` en SQL (`payment_method NOT IN ('reversal', 'free')`), pero el `RealTimeScoringService#scorable?` solo excluye `reversal` y pagos YA reembolsados al momento del request — no excluye `free` en tiempo real. Son asimétricos.

4. `ScoringResult` (dataclass) no tiene campos `calibration_segment`/`fallback_level`/`frame_flags`. El router `score.py` no los llena. El esquema Pydantic `ScoreResponse` ya los define como `Optional`. Toda la tubería está preparada — falta cablearla.

5. La tabla `anomaly_detection_alerts` persiste el metadata en una columna JSON. No hay columnas dedicadas para `calibration_segment`, `fallback_level`, `frame_flags`, `feature_frame_version`. Se agregan via `add_column` o incluyéndolos en el hash `metadata` existente.

**Primary recommendation:** Cablear scorer en dos pasos: (1) extender `ScoringResult` con los 3 campos nuevos + wiring de `SingleTransactionScorer` para despachar a frame-v1; (2) cambios mínimos en Rails para consumir y persistir esos campos. No enviar timezone desde Rails — el scorer lo resuelve desde el artefacto.

---

## Standard Stack

### Python (ml-fraud-detector)

| Componente | Versión/Archivo | Estado actual |
|------------|----------------|--------------|
| `SingleTransactionScorer` | `src/fraud_detector/scoring/scorer.py` | Usa `ThresholdClassifier` + decide `EnrichedFeatureCalculator` vs `SingleFeatureCalculator` por conteo (40 vs otro). NO usa `FrameV1FeatureCalculator` ni `SegmentedThresholdClassifier` |
| `FrameV1FeatureCalculator` | `src/fraud_detector/scoring/features_frame_v1.py` | Fase 1 — listo, 30 features, resuelve IANA desde artefacto |
| `SegmentedThresholdClassifier` | `src/fraud_detector/scoring/classifier.py` | Fase 2 — listo, fallback facility→currency→global, retorna 5-tuple |
| `Artifacts` dataclass | `scorer/artifact_loader.py` | Tiene `facility_stats: Optional[dict]` y `thresholds_segmented: Optional[dict]` — listo |
| `load_artifacts` | `scorer/artifact_loader.py` | Carga ambos opcionales si metadata los referencia — listo |
| `model_metadata_frame_v1.json` | `output/models/` | Generado, referencia `facility_stats_v1.json` y `thresholds_segmented_v1.json` |
| `ScoringResult` | `src/fraud_detector/scoring/classifier.py` | Solo tiene: `score`, `is_anomaly`, `risk_level`, `percentile`, `factors`, `model_version`, `feature_version`, `threshold_version`. **Falta:** `calibration_segment`, `fallback_level`, `frame_flags` |
| Router `score.py` | `scorer/routers/score.py` | Llama `scorer.score(payment)` y construye `ScoreResponse` sin los 3 nuevos campos. **Falta cablearlos.** |
| `ScoreResponse` Pydantic | `scorer/schemas.py` | YA tiene `calibration_segment: Optional[str]`, `fallback_level: Optional[str]`, `frame_flags: Optional[FrameFlags]` — contrato listo |
| `ScoreRequest` Pydantic | `scorer/schemas.py` | YA tiene `facility_time_zone_iana: Optional[str]` y `currency: Optional[str]` |

### Ruby/Rails (platform pack anomaly_detection)

| Componente | Archivo | Estado actual |
|------------|---------|--------------|
| `RealTimeScoringService` | `app/services/anomaly_detection/real_time_scoring_service.rb` | `build_payload` no envía `facility_time_zone_iana`. `scorable?` excluye `reversal` y pagos refunded — no excluye `free` |
| `AlertManager` | `app/services/anomaly_detection/alert_manager.rb` | `build_metadata` no persiste `calibration_segment`, `fallback_level`, `frame_flags`, `feature_frame_version` |
| `BatchScoringService` | `app/services/anomaly_detection/batch_scoring_service.rb` | Llama al scorer; no tiene lógica `scorable?` — la exclusión la hace el SQL de batch en Python |
| `Alert` model | `app/models/anomaly_detection/alert.rb` | Persiste en columna `metadata JSON`. Sin columnas dedicadas para los nuevos campos |
| Tabla `anomaly_detection_alerts` | `db/migrate/20260617040330_...rb` | Columns: `id`, `payment_id`, `facility_id`, `status`, `severity`, `alert_type`, `anomaly_score`, `metadata JSON`, `created_at`, `updated_at`, `assigned_to_id` |
| `Facility#tzinfo_identifier` | `app/models/facility.rb:56` | Columna DB existente, `default("America/New_York")`. Se mantiene via `before_save :update_tzinfo_identifier`. **Forma segura** de obtener IANA. |
| `Facility#time_zone_identifier` | `app/models/facility.rb:1503-1505` | Método inseguro: `Time.find_zone(time_zone).tzinfo.identifier` — si `time_zone` es nil o inválido → `NoMethodError` |

---

## Architecture Patterns

### Patrón 1: Decisión IF-40 vs frame-v1 en SingleTransactionScorer

**Lo que existe hoy:** el scorer decide el `_feature_calc` en `__init__` según `len(feature_names)`:
- 40 features → `EnrichedFeatureCalculator` (IF-40)
- otro → `SingleFeatureCalculator` (legado base-31)

**Lo que hay que hacer:** añadir una tercera rama para frame-v1. La señal correcta es la presencia de `artifacts.facility_stats` y `artifacts.thresholds_segmented` (no el conteo de features, porque 30 != 40 pero otros futuros modelos también podrían tener conteos distintos).

**Patrón recomendado** — detección por presencia de artefactos opcionales:

```python
# En SingleTransactionScorer.__init__:
if artifacts.facility_stats is not None and artifacts.thresholds_segmented is not None:
    # frame-v1 path
    self._feature_calc = FrameV1FeatureCalculator(
        facility_stats=artifacts.facility_stats,
        feature_engineer_path=feature_engineer_path,
    )
    self._classifier = SegmentedThresholdClassifier(artifacts.thresholds_segmented)
    self._is_frame_v1 = True
elif len(self._feature_names) == 40:
    # IF-40 path (existente)
    self._feature_calc = EnrichedFeatureCalculator(...)
    self._classifier = ThresholdClassifier(config=artifacts.thresholds)
    self._is_frame_v1 = False
else:
    # legacy base-31 path (existente)
    self._feature_calc = SingleFeatureCalculator(feature_engineer_path)
    self._classifier = ThresholdClassifier(thresholds_path)
    self._is_frame_v1 = False
```

**Importante:** el `FrameV1FeatureCalculator.__init__` recibe `facility_stats` dict (ya cargado) y `feature_engineer_path` string. Ambos están disponibles en `__init__` del scorer.

### Patrón 2: ScoringResult extendido (campos opcionales)

`ScoringResult` es un `@dataclass` en `classifier.py`. Hay que añadir 3 campos con defaults `None` para mantener retrocompat:

```python
@dataclass
class ScoringResult:
    score: float
    is_anomaly: bool
    risk_level: str
    percentile: float
    factors: List[dict] = field(default_factory=list)
    model_version: str = "IF-31-v1"
    feature_version: str = "base-31"
    threshold_version: str = "v1"
    # Nuevos (Fase 3) — solo poblados en frame-v1 path
    calibration_segment: Optional[str] = None
    fallback_level: Optional[str] = None
    frame_flags: Optional[dict] = None  # dict para no importar FrameFlags en classifier
```

### Patrón 3: Llamada a SegmentedThresholdClassifier en scorer.score()

`SegmentedThresholdClassifier.classify` requiere `facility_id` y `currency` adicionales. El scorer debe pasarlos:

```python
# En SingleTransactionScorer.score() — rama frame-v1
if self._is_frame_v1:
    is_anomaly, risk_level, percentile, fallback_level, calibration_segment = \
        self._classifier.classify(
            score=raw_score,
            facility_id=payment.get("facility_id", 0),
            currency=(payment.get("currency") or "USD").upper(),
        )
else:
    # ThresholdClassifier retorna 3-tuple
    is_anomaly, risk_level, percentile = self._classifier.classify(raw_score)
    fallback_level = None
    calibration_segment = None
```

### Patrón 4: Router cablea frame_flags

El router `score.py` construye `ScoreResponse`. Para el path frame-v1, debe pasar los campos adicionales:

```python
# scorer/routers/score.py — en score_single()
payment = request.model_dump()
result = scorer.score(payment)

# Construir frame_flags desde result si está poblado
frame_flags_obj = None
if result.frame_flags is not None:
    frame_flags_obj = FrameFlags(**result.frame_flags)

return ScoreResponse(
    raw_score=result.score,
    percentile=result.percentile,
    risk_level=result.risk_level,
    is_anomaly=result.is_anomaly,
    factors=[FactorItem(**f) for f in result.factors],
    model_version=result.model_version,
    feature_version=result.feature_version,
    threshold_version=result.threshold_version,
    calibration_segment=result.calibration_segment,
    fallback_level=result.fallback_level,
    frame_flags=frame_flags_obj,
)
```

### Patrón 5: Resolución de IANA — fuente única en el scorer

**Decisión:** el scorer resuelve IANA desde `facility_stats_v1.json` vía `_lookup_facility()`. Rails NO necesita enviar `facility_time_zone_iana`. Esto elimina el blocker del PLAT-01 y hace el scorer autosuficiente.

`FrameV1FeatureCalculator.calculate()` ya ignora el campo `facility_time_zone_iana` del payload — resuelve la zona internamente. La firma `calculate(payment, context)` no recibe timezone como argumento separado.

`frame_flags.timezone_missing` se genera si el artefacto no tiene entrada para una facility y cae al fallback `"Etc/UTC"`. En la práctica el artefacto tiene 1876 facilities. Una facility desconocida recibe `"Etc/UTC"` (no excepción), y el flag debe ponerse a `True` solo si se usó el fallback UTC. La lógica de generación de `frame_flags` vive en `scorer.score()`, no en `FrameV1FeatureCalculator`.

**Implicación para Rails:** `build_payload` no necesita el campo `facility_time_zone_iana`. El campo `facility_time_zone_iana: Optional[str]` en `ScoreRequest` es ignorable — el scorer lo ignora si frame-v1 resuelve desde artefacto. Los specs de `RealTimeScoringService` no deben romper.

### Patrón 6: AlertManager — persistir metadata nueva

La tabla persiste en `metadata JSON`. La forma más segura y sin migración obligatoria es extender `build_metadata` en `AlertManager`:

```ruby
def build_metadata(data)
  {
    user_id: data["user_id"],
    amount_usd: data["amount_usd"],
    percentile: data["percentile"],
    is_anomaly: data["is_anomaly"],
    factors: data["factors"],
    model_version: data["model_version"],
    feature_version: data["feature_version"],
    threshold_version: data["threshold_version"],
    # Nuevos campos Fase 3 (presentes solo en frame-v1; nil se compacta)
    calibration_segment: data["calibration_segment"],
    fallback_level: data["fallback_level"],
    frame_flags: data["frame_flags"],
    feature_frame_version: data["feature_frame_version"],
  }.compact
end
```

Y `RealTimeScoringService#create_alert` debe pasar esos campos al hash `alerts_data`.

**Alternativa:** añadir columnas dedicadas `calibration_segment VARCHAR(64)`, `fallback_level VARCHAR(32)`. Pros: consultable en SQL directo. Cons: requiere migración, cambios de modelo. Para Fase 3 (integración/test), el JSON es suficiente. Para Fase 4 (shadow/analytics ClickHouse), se evalúa.

### Patrón 7: PLAT-03 — alinear scorable?

**Brecha real verificada:**

| Exclusión | Batch (Python SQL) | Real-time (Rails) |
|-----------|-------------------|------------------|
| `reversal` | `payment_method NOT IN ('reversal', 'free')` | `@payment.payment_method == "reversal"` → `false` |
| `free` | `payment_method NOT IN ('reversal', 'free')` | **AUSENTE** — `free` payments se PUNTÚAN en real-time |
| Reembolsados | N/A (no puntúa refunded payments, el cursor va hacia adelante) | `@payment.status.in?(Payment::REFUNDED_STATUSES)` → `false` |

La corrección mínima: añadir `return false if @payment.payment_method == "free"` en `scorable?`.

Los reembolsos "post-hoc" (payment creado y luego cambiado de status) no llegan al scorer real-time porque el callback es `after_commit on: :create` solamente. No hay callback en update. Los reembolsos que sí bloquea el `scorable?` son los que llegan al callback `on: :create` ya con `status IN ('totally_refunded', 'refunded_to_credit')` — caso edge, pero correcto bloquearlo.

---

## Don't Hand-Roll

| Problema | No construir | Usar | Por qué |
|----------|--------------|------|---------|
| Resolución timezone IANA en scorer | Lógica custom Rails→scorer | `_lookup_facility()` en `FrameV1FeatureCalculator` | Ya funciona, 1876 facilities cubiertas, fallback a UTC |
| Decisión frame-v1 vs IF-40 | Feature flag externo, config en DB | Presencia de `artifacts.facility_stats` en `Artifacts` | La señal ya existe, es determinista al startup |
| Migración de columnas de metadata | ALTER TABLE en producción inmediata | Extender `metadata JSON` existente | La tabla ya tiene JSON; consultas MySQL en JSON son posibles; migración de columnas es riesgo innecesario para Fase 3 |
| Parseo de `ZoneInfo` inválido en scorer | Try/except wrapper custom | `_lookup_facility` devuelve `"Etc/UTC"` para facilities desconocidas | El fallback ya es correcto; `ZoneInfoNotFoundError` es subclase de `KeyError` y se trataría como error de scoring (fail-open) |
| Timezone en Rails | Método `time_zone_identifier` (llama `find_zone.tzinfo.identifier`) | Columna `facility.tzinfo_identifier` (persistida por `before_save`) | El método inseguro puede levantar `NoMethodError`; la columna es DB-safe |

---

## Common Pitfalls

### Pitfall 1: Condición de dispatch por conteo de features (no por artefactos)

**What goes wrong:** si se añade una nueva rama `elif len == 30` en el scorer, un futuro modelo con 30 features pero sin `facility_stats` tomaría el path frame-v1 incorrectamente.
**Why it happens:** el código existente usa `len(feature_names) == 40` como señal, que es frágil.
**How to avoid:** usar `artifacts.facility_stats is not None` como condición primaria — es la señal semántica correcta. Los tests de `test_artifact_loader.py` ya cubren esto.
**Warning signs:** `FrameV1FeatureCalculator` recibe `facility_stats=None` → `AttributeError` en `_lookup_facility`.

### Pitfall 2: ScoringResult sin los nuevos campos en el path IF-40

**What goes wrong:** si se extiende `ScoringResult` con campos obligatorios, los constructores existentes en `BatchScorer._score_all()` y `SingleTransactionScorer.score()` (path IF-40) rompen.
**Why it happens:** `ScoringResult` es construido directamente en `scorer.score()` — cualquier campo sin default rompe.
**How to avoid:** añadir los 3 campos nuevos con `default=None`. Los tests de batch (`test_if40_artifacts.py`) verificarán retrocompat.
**Warning signs:** `TypeError: __init__() missing required positional argument`.

### Pitfall 3: router.score_single no propaga calibration_segment/fallback_level

**What goes wrong:** el scorer los calcula y los guarda en `ScoringResult`, pero el router construye `ScoreResponse` sin pasarlos — Rails nunca los recibe.
**Why it happens:** el router fue escrito antes de Fase 2. El `ScoreResponse` ya tiene los campos `Optional`, pero el router nunca los asigna.
**How to avoid:** extender la construcción de `ScoreResponse` en `score.py` para incluir los 3 campos de `result`.
**Warning signs:** `score_result["calibration_segment"]` retorna `None` siempre en Rails aunque el scorer esté en mode frame-v1.

### Pitfall 4: Batch scorer no propaga calibration_segment a critical_alerts

**What goes wrong:** `BatchScorer._score_all()` construye los `critical_alerts` dicts manualmente (líneas 433-444). Aunque el scorer calcule `calibration_segment`, el dict de critical_alert no lo incluye → Rails no puede persistirlo en batch.
**Why it happens:** el dict fue escrito para IF-40 y tiene campos fijos.
**How to avoid:** extender el dict de `critical_alerts` con `calibration_segment`, `fallback_level` si el scorer los produjo. Verificar en `BatchScoringService` que `AlertManager.call` recibe estos campos.

### Pitfall 5: `free` payments puntuados en real-time

**What goes wrong:** pagos con `payment_method == "free"` llegan al scorer real-time. El FeatureCalculator los procesa sin error pero producen features sin sentido (amount 0, ratio infinito, etc.).
**Why it happens:** `scorable?` en `RealTimeScoringService` no excluye `free`.
**How to avoid:** añadir `return false if @payment.payment_method == "free"` en `scorable?`. El test `spec/services/anomaly_detection/real_time_scoring_service_spec.rb` debe tener un `context "when payment is free"`.
**Warning signs:** scores de 0.0 con `risk_level: "minimal"` para `payment_method: "free"` en ClickHouse.

### Pitfall 6: `SegmentedThresholdClassifier.classify` recibe `currency=None`

**What goes wrong:** si Rails no envía `currency` y el scorer pasa `None` a `classify`, el lookup `None in self._by_currency` siempre falla (la key sería `None`, no un ISO code), cayendo a global.
**Why it happens:** `ScoreRequest.currency` es `Optional[str]`. Si Rails envía `currency=None` y el scorer no defaultea antes de pasarlo a `classify`, `None in dict` es `False` pero no rompe.
**How to avoid:** en `scorer.score()`, normalizar currency antes de pasarla al classifier: `currency = (payment.get("currency") or "USD").upper()`. Esto ya está hecho en `FrameV1FeatureCalculator.calculate()` — replicarlo en la llamada al classifier.
**Warning signs:** todos los scores caen a segmento `"global"` incluso para currencies con calibración (flag `fallback_level: "global"` inesperado).

### Pitfall 7: `Facility#time_zone_identifier` vs `facility.tzinfo_identifier`

**What goes wrong:** si Rails necesita IANA para algún log/display futuro, usar `time_zone_identifier` puede generar `NoMethodError` si `time_zone` es nil (facilities de staging sin timezone).
**Why it happens:** `Time.find_zone(nil)` retorna `nil`; `nil.tzinfo` → `NoMethodError`.
**How to avoid:** usar la columna `facility.tzinfo_identifier` (DB, siempre string). Para el scorer en esta fase, es irrelevante porque el scorer no necesita la zona desde Rails.
**Warning signs:** `NoMethodError: undefined method 'tzinfo' for nil:NilClass` en producción.

---

## Code Examples

### Detección frame-v1 en SingleTransactionScorer.__init__

```python
# scorer/main.py ya pasa artifacts al constructor:
# scorer = SingleTransactionScorer(feature_engineer_path=..., ch_connector=..., artifacts=artifacts)

# En SingleTransactionScorer.__init__ (src/fraud_detector/scoring/scorer.py):
if artifacts is not None:
    self._model = artifacts.model
    self._scaler = artifacts.scaler
    self._feature_names = list(artifacts.feature_list)
    self._metadata = dict(artifacts.metadata)

    # Dispatch por presencia de artefactos opcionales
    if (artifacts.facility_stats is not None
            and artifacts.thresholds_segmented is not None):
        from fraud_detector.scoring.features_frame_v1 import FrameV1FeatureCalculator
        from fraud_detector.scoring.classifier import SegmentedThresholdClassifier
        self._feature_calc = FrameV1FeatureCalculator(
            facility_stats=artifacts.facility_stats,
            feature_engineer_path=feature_engineer_path,
        )
        self._classifier = SegmentedThresholdClassifier(artifacts.thresholds_segmented)
        self._is_frame_v1 = True
    elif len(self._feature_names) == 40:
        # IF-40 path existente
        self._feature_calc = EnrichedFeatureCalculator(
            feature_engineer_path=feature_engineer_path,
            feature_list=self._feature_names,
        )
        self._classifier = ThresholdClassifier(config=artifacts.thresholds)
        self._is_frame_v1 = False
    else:
        # legacy base-31
        self._feature_calc = SingleFeatureCalculator(feature_engineer_path)
        self._classifier = ThresholdClassifier(config=artifacts.thresholds)
        self._is_frame_v1 = False
```

### scorer.score() — rama frame-v1

```python
# En SingleTransactionScorer.score():
features = self._feature_calc.calculate(payment, context)
raw_score, X_scaled = self.score_features(features)

calibration_segment = None
fallback_level = None
frame_flags = None

if self._is_frame_v1:
    facility_id = int(payment.get("facility_id", 0))
    currency = (payment.get("currency") or "USD").upper()
    is_anomaly, risk_level, percentile, fallback_level, calibration_segment = \
        self._classifier.classify(raw_score, facility_id=facility_id, currency=currency)
    # Construir frame_flags: timezone_missing si facility_id no está en artefacto
    fid_str = str(facility_id)
    tz_missing = (
        self._feature_calc._stats["facilities"].get(fid_str) is None
    )
    frame_flags = {
        "timezone_missing": tz_missing,
        "currency_missing": payment.get("currency") is None,
        "currency_unknown": False,  # por ahora; puede extenderse
    }
else:
    is_anomaly, risk_level, percentile = self._classifier.classify(raw_score)
```

### Rails — scorable? corregido

```ruby
# En RealTimeScoringService (platform):
def scorable?
  return false if @payment.payment_method == "reversal"
  return false if @payment.payment_method == "free"
  return false if @payment.status.present? && @payment.status.in?(Payment::REFUNDED_STATUSES)

  Setting.get("realtime_scoring_enabled") != "false"
end
```

### Rails — AlertManager#build_metadata extendido

```ruby
def build_metadata(data)
  {
    user_id: data["user_id"],
    amount_usd: data["amount_usd"],
    percentile: data["percentile"],
    is_anomaly: data["is_anomaly"],
    factors: data["factors"],
    model_version: data["model_version"],
    feature_version: data["feature_version"],
    threshold_version: data["threshold_version"],
    calibration_segment: data["calibration_segment"],
    fallback_level: data["fallback_level"],
    frame_flags: data["frame_flags"],
    feature_frame_version: data["feature_frame_version"],
  }.compact
end
```

### Rails — RealTimeScoringService#create_alert extendido

```ruby
def create_alert(score_result)
  AlertManager.call(alerts_data: [
    {
      "payment_id" => @payment.id.to_s,
      "facility_id" => @payment.facility_id,
      "user_id" => @payment.user_id,
      "raw_score" => score_result["raw_score"],
      "risk_level" => score_result["risk_level"],
      "percentile" => score_result["percentile"],
      "is_anomaly" => score_result["is_anomaly"],
      "factors" => score_result["factors"],
      "model_version" => score_result["model_version"],
      "feature_version" => score_result["feature_version"],
      "threshold_version" => score_result["threshold_version"],
      "amount_usd" => @payment.reservation_paid_out.to_f,
      "alert_type" => "anomaly_score",
      # Nuevos campos Fase 3
      "calibration_segment" => score_result["calibration_segment"],
      "fallback_level" => score_result["fallback_level"],
      "frame_flags" => score_result["frame_flags"],
      "feature_frame_version" => score_result["feature_version"],
    }
  ])
end
```

---

## State of the Art

| Antes de Fase 3 | Después de Fase 3 |
|----------------|------------------|
| Scorer siempre usa IF-40 (ThresholdClassifier, EnrichedFeatureCalculator) | Scorer despacha frame-v1 si `artifacts.facility_stats != None` |
| Router no llena `calibration_segment`/`fallback_level`/`frame_flags` | Router los llena desde `ScoringResult` |
| Rails no persiste campos de calibración | `AlertManager` los persiste en `metadata JSON` |
| `scorable?` no excluye `free` | `scorable?` excluye `reversal` y `free` |
| IANA timezone requería coordinación Rails→scorer | Scorer resuelve IANA autónomamente desde artefacto |

---

## Open Questions

1. **¿feature_frame_version separado o reusa feature_version?**
   - Lo que sabemos: `ScoreResponse` tiene `feature_version` (actualmente `"enriched-40"` o `"frame-v1"`). `feature_frame_version` se menciona en PLAT-02 como campo adicional.
   - Lo que no está claro: si `feature_frame_version` es un alias de `feature_version` para el path frame-v1, o un campo distinto.
   - Recomendación: usar `feature_version` directamente (`"frame-v1"`). No añadir `feature_frame_version` como campo separado — agregar la key en metadata con el mismo valor de `feature_version` si se quiere el nombre explícito.

2. **¿Migración de columnas dedicadas en alerts o solo JSON?**
   - Lo que sabemos: la tabla tiene `metadata JSON`. MySQL 8 permite consultas JSON (`JSON_EXTRACT`), pero son lentas sin índices.
   - Lo que no está claro: si hay queries que necesitarán filtrar por `calibration_segment` directamente en MySQL.
   - Recomendación: para Fase 3 (test/integration), JSON es suficiente. Dejar la decisión de columnas para Fase 4 cuando se conozcan los patrones de consulta.

3. **¿BatchScorer debe propagar calibration_segment al INSERT de anomaly_scores?**
   - Lo que sabemos: `_INSERT_COLUMNS` en `scorer/batch/scorer.py` está hardcodeado. El DDL de `anomaly_scores` (ClickHouse) no está en este repo.
   - Lo que no está claro: si el DDL de `anomaly_scores` tiene columnas para los nuevos campos.
   - Recomendación: el planner debe verificar el DDL de `anomaly_scores` en ClickHouse antes de tocar `_INSERT_COLUMNS`. Si el DDL no las tiene, es riesgo de Fase 4.

4. **¿`frame_flags.timezone_invalid` o `frame_flags.timezone_missing`?**
   - El contexto menciona `timezone_invalid=true` para zona inválida. El schema Pydantic tiene `timezone_missing: bool`. Son semánticamente distintos.
   - Recomendación: en el scorer, dado que el `FrameV1FeatureCalculator` nunca lanza excepción por zona (usa fallback `"Etc/UTC"`), el flag correcto es `timezone_missing` (facility no conocida por el artefacto), no `timezone_invalid`. Un `ZoneInfoNotFoundError` sería un error de datos en el artefacto — rarísimo.

---

## Sources

### Primary (HIGH confidence)
- Código fuente directo: `ml-fraud-detector/src/fraud_detector/scoring/scorer.py`
- Código fuente directo: `ml-fraud-detector/src/fraud_detector/scoring/features_frame_v1.py`
- Código fuente directo: `ml-fraud-detector/src/fraud_detector/scoring/classifier.py`
- Código fuente directo: `ml-fraud-detector/scorer/artifact_loader.py`
- Código fuente directo: `ml-fraud-detector/scorer/schemas.py`
- Código fuente directo: `ml-fraud-detector/scorer/routers/score.py`
- Código fuente directo: `ml-fraud-detector/scorer/batch/scorer.py`
- Código fuente directo: `ml-fraud-detector/output/models/model_metadata_frame_v1.json`
- Código fuente directo: `ml-fraud-detector/output/models/model_metadata.json`
- Código fuente directo: `platform/packs/anomaly_detection/app/services/anomaly_detection/real_time_scoring_service.rb`
- Código fuente directo: `platform/packs/anomaly_detection/app/services/anomaly_detection/alert_manager.rb`
- Código fuente directo: `platform/packs/anomaly_detection/app/services/anomaly_detection/batch_scoring_service.rb`
- Código fuente directo: `platform/packs/anomaly_detection/db/migrate/20260617040330_create_anomaly_detection_alerts.rb`
- Código fuente directo: `platform/packs/anomaly_detection/app/models/anomaly_detection/alert.rb`
- Código fuente directo: `platform/app/models/facility.rb` (líneas 55-56, 531-532, 1503-1505)
- Código fuente directo: `platform/app/models/payment.rb` (líneas 117, 208, 1717-1721)
- Inspección JSON: `facility_stats_v1.json` — 1876 facilities, todas con `iana_tz`
- Inspección JSON: `thresholds_segmented_v1.json` — 452 by_facility, 17 by_currency

### Secondary (MEDIUM confidence)
- Tests existentes verificados: `tests/test_artifact_loader.py`, `spec/services/anomaly_detection/real_time_scoring_service_spec.rb`, `spec/services/anomaly_detection/alert_manager_spec.rb`

---

## Metadata

**Confidence breakdown:**
- Wiring del scorer (dispatch frame-v1): HIGH — código fuente leído, artifacts cargados, condición de dispatch clara
- Resolución IANA (scorer autónomo): HIGH — `_lookup_facility()` verificado, 1876 facilities en artefacto
- Cambios Rails (payload, alert_manager, scorable?): HIGH — código fuente leído, brechas identificadas exactamente
- Esquema de tabla alerts: HIGH — migration leída, JSON column confirmada
- Batch scorer + ClickHouse DDL anomaly_scores: MEDIUM — columnas `_INSERT_COLUMNS` leídas, pero DDL de ClickHouse no está en repo

**Research date:** 2026-07-06
**Valid until:** 2026-08-06 (30 días — código está en filesystem local, no drift de librería)
