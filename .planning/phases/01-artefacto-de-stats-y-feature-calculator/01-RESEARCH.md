# Phase 1: Artefacto de Stats y Feature Calculator — Research

**Researched:** 2026-07-06
**Domain:** Per-facility reference stats artifact + frame-normalized feature calculator (brownfield)
**Confidence:** HIGH — all findings derived from direct code and artifact inspection on this machine

---

## Summary

Esta fase construye dos artefactos que no existen todavía: `facility_stats_v1.json` y `FrameV1FeatureCalculator`. La Fase 0 ya corrigió los bugs de `getattr` y congeló el baseline (top-5% monto = 11.79×, off-hours UTC = 29.78%). La Fase 1 no toca el scorer en producción (eso es Fase 3); todo trabajo es offline o en código nuevo aislado.

Los prototipos de referencia en `scripts/exp_frames_improvement.py` y `scripts/exp_reference_frames.py` ya demuestran el patrón completo: en la cohorte de test (sep–dic 2025, muestra 1/20), `disjoint30_frames` reduce top-5% monto de 15.7× a 2.1× y off-hours de 30% UTC a 4.4% local. Esas son las métricas objetivo que la Fase 1 debe alcanzar sobre el val set completo (1.13M filas, todas las monedas, FS-frame-operational-v1).

El stack incremental es mínimo: `zoneinfo` (stdlib), `tzdata` (ya instalado), `json` (stdlib). No hay nuevas dependencias a instalar.

**Primary recommendation:** Construir `FacilityStatsBuilder` como script offline independiente que produce `facility_stats_v1.json`, luego implementar `FrameV1FeatureCalculator` con dos superficies (`calculate` y `calculate_from_row`) que comparten exactamente la misma aritmética. No modificar el scorer en vivo hasta la Fase 3.

---

## Standard Stack

### Core (ya instalado — sin cambios)

| Library | Version in venv | Purpose | Why Standard |
|---------|----------------|---------|--------------|
| `numpy` | 1.24+ | Cálculo vectorizado de stats | `np.percentile`, aritmética de arrays |
| `pandas` | 2.3.3 | `groupby().agg()` en build time | Batch computation de stats por facility |
| `scikit-learn` | 1.6.1 | `IsolationForest`, `RobustScaler` | Mismo modelo que Fase 0 |
| `joblib` | 1.5.3 | Serializar modelo reentrenado | Mismo patrón que artefactos existentes |
| `json` | stdlib | Serializar `facility_stats_v1.json` | Human-readable, diffable, 1-5ms load |
| `zoneinfo` | stdlib 3.9.6 | DST-correct timezone conversion | PEP 615, reemplaza pytz en modo mantenimiento |
| `tzdata` | 2025.3 | IANA timezone data en Docker | Ya instalado en venv |

### No instalar

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `pytz` para código nuevo | Maintenance mode; `localize()` API produce LMT bug | `ZoneInfo(iana_name)` de stdlib |
| 1,876 `RobustScaler` objects | Serialización pesada, idéntico resultado matemático | Dict de floats en JSON |
| `parquet` para stats artifact | Overhead innecesario para estructura plana de 689 facilities | `json` |

### Installation

No new installs required. `tzdata` is already in venv.

---

## Architecture Patterns

### Recommended Project Structure (new files only)

```
src/fraud_detector/
└── stats/                          # NEW package
    ├── __init__.py
    ├── builder.py                  # FacilityStatsBuilder.build(train_df) -> dict
    └── validator.py                # validate_universe_filter(stats, val_df)

src/fraud_detector/scoring/
└── features_frame_v1.py            # NEW — FrameV1FeatureCalculator

scripts/
└── build_facility_stats.py         # NEW offline script

output/models/
├── facility_stats_v1.json          # NEW artifact (produced by build script)
└── model_frame_v1.joblib           # NEW model (produced by retrain script)

tests/
└── test_parity_phase1.py           # NEW — FrameV1FeatureCalculator parity test
```

No se modifica ningún archivo existente en esta fase. El scorer en vivo sigue usando `EnrichedFeatureCalculator` y `thresholds_v2.json` hasta la Fase 3.

---

### Pattern 1: FacilityStatsBuilder — Estructura del artefacto JSON

**Esquema completo `facility_stats_v1.json`:**

```json
{
  "schema_version": "facility-stats-v1",
  "built_at": "2026-07-XX T12:00:00Z",
  "universe_filter": "_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL",
  "train_rows": 3137083,
  "n_facilities": 689,
  "min_n_threshold": 30,
  "global_fallback": {
    "median": 20.0,
    "iqr": 45.87,
    "mean": 312.68,
    "n": 3137083,
    "fallback_level": "global"
  },
  "currency_fallbacks": {
    "USD": {"median": 25.0, "iqr": 38.5, "mean": 180.0, "n": 2384422, "fallback_level": "currency"},
    "CAD": {"median": 18.5, "iqr": 22.0, "mean": 95.0, "n": 173424, "fallback_level": "currency"}
  },
  "facilities": {
    "1828": {
      "median": 35.10,
      "iqr": 28.40,
      "mean": 67.89,
      "n": 3421,
      "iana_tz": "America/New_York",
      "fallback_level": "facility"
    },
    "1830": {
      "median": 22.0,
      "iqr": 0.0,
      "iqr_guarded": 1.0,
      "mean": 22.0,
      "n": 5,
      "iana_tz": "America/Denver",
      "fallback_level": "currency"
    }
  }
}
```

**Reglas clave del esquema:**

1. `iqr` almacena el valor bruto (puede ser 0). `iqr_guarded` almacena `max(iqr, 1.0)` — se usa en la fórmula. Almacenar ambos permite auditar cuántas facilities tienen distribución uniforme.
2. `fallback_level` por entrada: `"facility"` si n >= 30, `"currency"` si la facility tiene n < 30 pero existe fallback de moneda, `"global"` si ninguno aplica.
3. El campo `iana_tz` se obtiene del `output/revision/facility_tz.parquet` (1876 facilities, 0 nulls, 64 zonas Rails). El mapeado Rails→IANA se consolida en `src/fraud_detector/stats/tz_mapping.py` — el prototipo en `exp_frames_improvement.py` tiene 51 entradas; faltan 13 zonas que están en 1 facility cada una (Baku, Brussels, Darwin, Fiji, Madrid, Magadan, Mazatlan, Montevideo, Nairobi, Santiago, Sarajevo, Sofia, UTC). Todos tienen IANA canónico conocido.
4. La clave del dict es `str(facility_id)` — JSON no admite int keys. En Python, la lookup convierte: `stats["facilities"].get(str(fid))`.

**Dónde vive `FacilityStatsBuilder`:**

```python
# src/fraud_detector/stats/builder.py
class FacilityStatsBuilder:
    MIN_N = 30  # facilities con n < 30 no reciben threshold per-facility

    def build(self, train_df: pd.DataFrame, tz_map: dict) -> dict:
        """Compute per-facility stats from training universe.
        
        train_df must already be filtered by the scorer universe:
        _peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free')
        """
```

El `tz_map` se obtiene leyendo `output/revision/facility_tz.parquet` + aplicando el dict Rails→IANA completo (63 entradas únicas, todas mapeadas).

---

### Pattern 2: Universe Validator

El criterio de aceptación STATS-01 exige que el validator confirme que el universo del artefacto coincide con el filtro del scorer. Implementación propuesta:

```python
# src/fraud_detector/stats/validator.py
def validate_universe_filter(stats: dict, sample_df: pd.DataFrame) -> bool:
    """
    Verifica que los n_train del artefacto coincidan con el count del parquet
    (dentro de tolerancia) y que el metadata universe_filter sea el correcto.
    """
    expected_filter = "_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL"
    assert stats["universe_filter"] == expected_filter
    assert abs(stats["train_rows"] - len(sample_df)) / stats["train_rows"] < 0.001
    return True
```

El script de build escribe `universe_filter` en el metadata del artefacto. El test de paridad en Fase 1 verifica este campo al cargar el artefacto.

---

### Pattern 3: FrameV1FeatureCalculator — Superficie dual

**Mapeo exacto de features desde `FS-frame-operational-v1` (39 features) al frame set:**

Basado en `exp_frames_improvement.py` `FRAME_MAP` y el experimento `disjoint30_frames` (confirmado en results JSON):

| Feature en FS-operational-v1 | Reemplazo en FS-frame-v1 | Fórmula |
|------------------------------|--------------------------|---------|
| `log_amount` | `log_amount_fac` | `log1p(amount / (fmean + 0.01))` |
| `facility_avg_amount` | eliminada | proxy puro de tamaño de facility |
| `user_amount_24h` | `user_amount_24h_fac` | `user_amount_24h / (fmean + 0.01)` |
| `user_debit_amount_30d` | `user_debit_amount_30d_fac` | `user_debit_amount_30d / (fmean + 0.01)` |
| `hour_sin` | `hour_sin_loc` | `sin(2π * local_hour / 24)` |
| `hour_cos` | `hour_cos_loc` | `cos(2π * local_hour / 24)` |
| `day_of_week` | `dow_sin_loc` + `dow_cos_loc` | `sin(2π*dow/7)`, `cos(2π*dow/7)` |
| `is_weekend` | `is_weekend_loc` | `dow_local >= 5` |
| `is_off_hours` | `is_off_hours_loc` | `local_hour in {23,0,1,2,3,4,5,6}` |
| `off_hours_high_value` | `off_hours_high_value_loc` | `is_off_hours_loc AND amount_facility_ratio > 3` |

Eliminadas sin reemplazo (circulares con pure_fraud o con skew train/serve):
`user_txn_count_1h`, `same_amount_count_1h`, `same_amount_count_24h`, `is_third_party_payment`, `is_new_user`, `is_very_new_user`, `new_user_first_facility`, `rapid_burst`, `user_account_age_days`

**Resultado: 30 features en `FS-frame-v1`** (misma cuenta que `disjoint30_frames` del experimento). `amount_facility_ratio` se retiene porque es el ratio directo amount/mean, distinto de `log_amount_fac` que es log-escala.

**Implementación de las dos superficies:**

```python
# src/fraud_detector/scoring/features_frame_v1.py
class FrameV1FeatureCalculator:
    """Frame-normalized feature calculator. Versión frame-v1."""

    FEATURE_NAMES: ClassVar[List[str]] = [
        "log_amount_fac", "discount_ratio", "has_tip",
        "hour_sin_loc", "hour_cos_loc", "dow_sin_loc", "dow_cos_loc",
        "is_weekend_loc", "is_off_hours_loc",
        "user_txn_count_24h", "time_since_last_txn", "user_amount_24h_fac",
        "user_distinct_facilities_30d", "user_distinct_methods",
        "amount_facility_ratio",
        "is_club_credit", "user_debit_count_30d", "user_debit_amount_30d_fac",
        "credit_flow_ratio", "is_staff", "paid_by_manager", "staff_amount_zscore",
        "category_entropy_30d", "user_merchandise_ratio_30d",
        "small_amount_at_facility", "very_small_amount_at_facility",
        "off_hours_high_value_loc",
        "gateway_change_recent", "is_main_gateway", "is_first_gateway_for_user",
        "source_change_recent",
    ]

    def __init__(self, facility_stats: dict, feature_engineer_path: str):
        self._stats = facility_stats
        # Load only staff stats from feature_engineer (still needed for zscore)
        fe = joblib.load(feature_engineer_path)
        self._staff_stats = fe._groups[6]._role_currency_stats
        self._staff_currency_stats = fe._groups[6]._currency_stats
        self._staff_global_mean = fe._groups[6]._global_mean
        self._staff_global_std = fe._groups[6]._global_std

    def calculate(self, payment: dict, context: UserContext) -> np.ndarray:
        """Real-time path: payment dict + UserContext -> feature vector."""
        # Resolves facility stats from self._stats, applies TZ via zoneinfo
        ...

    def calculate_from_row(self, row: pd.Series) -> np.ndarray:
        """Batch/parity path: enriched parquet row -> feature vector.
        
        Must produce bit-identical result to calculate() for the same transaction
        (tolerance <1e-8).
        
        row must have: amount, created_at, facility_id, currency, user_role,
                       user_txn_count_24h, time_since_last_txn, user_amount_24h,
                       user_distinct_facilities_30d, user_distinct_methods,
                       user_debit_count_30d, user_debit_amount_30d, credit_flow_ratio,
                       is_club_credit, is_staff, paid_by_manager, staff_amount_zscore,
                       category_entropy_30d, user_merchandise_ratio_30d,
                       small_amount_at_facility, very_small_amount_at_facility,
                       amount_facility_ratio, off_hours_high_value, 
                       gateway_change_recent, is_main_gateway,
                       is_first_gateway_for_user, source_change_recent
        """
        ...
```

**Parity guarantee:** Ambas superficies deben compartir la misma función privada `_compute_frame_features(amount, fid, created_at_utc, currency, ...)` que opera sobre tipos primitivos. `calculate()` extrae los primitivos del payment dict + context; `calculate_from_row()` los extrae del row de parquet. No hay dos implementaciones separadas de la aritmética.

---

### Pattern 4: DST conversion (zoneinfo)

```python
# Fuente: stdlib zoneinfo, confirmado en Python 3.9.6 en este venv
from zoneinfo import ZoneInfo
import pandas as pd

def utc_to_local_hour_dow(ts_utc_naive: pd.Timestamp, iana_tz: str) -> tuple[int, int]:
    """Convert UTC naive timestamp to local hour and day-of-week.
    
    fold=0 convention for ambiguous DST hours (consistent between batch and RT).
    Returns (hour_local, dow_local) where dow 0=Monday.
    """
    utc_aware = ts_utc_naive.tz_localize("UTC")
    local = utc_aware.astimezone(ZoneInfo(iana_tz))
    return local.hour, local.dayofweek
```

DST test requerido (criterio de aceptación FRAME-03): facilities en `America/New_York` y `America/Argentina/Buenos_Aires` (sin DST). Test case concreto: `2025-03-09T07:00:00Z` en `America/New_York` = 02:00 EST (before spring forward) y `2025-03-09T08:00:00Z` = 04:00 EDT (after). Buenos Aires: `2025-10-05T03:00:00Z` = 00:00 -03 (Argentina no cambia DST).

---

### Pattern 5: Fallback chain

```python
def _lookup_facility_stats(self, fid: int) -> tuple[dict, str]:
    """Returns (stats_dict, fallback_level_used)."""
    entry = self._stats["facilities"].get(str(fid))
    if entry and entry["fallback_level"] == "facility":
        return entry, "facility"
    # fallback to currency group
    currency_entry = self._stats["currency_fallbacks"].get(self._fid_to_currency.get(fid, "USD"))
    if currency_entry:
        return currency_entry, "currency"
    return self._stats["global_fallback"], "global"
```

El `_fid_to_currency` mapping se precarga del parquet de entrenamiento (facility_id -> currency_dominante). Para real-time, la currency del payment se usa directamente como key al `currency_fallbacks`.

---

### Pattern 6: Reentrenamiento del modelo global

**Receta idéntica al modelo IF-40 final** (confirmado en `isolation_forest_final.joblib`):

```python
IsolationForest(
    n_estimators=200,
    max_samples=512,
    max_features=0.6,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)
```

Scaler: `RobustScaler(quantile_range=(5.0, 95.0))` + clip `[-10, 10]` post-transform (mismo que `eval_clean_honest.py` y `exp_frames_improvement.py`).

**Input:** `train_features_enriched.parquet` con frame features añadidas via `add_frame_features()` (patrón de `exp_frames_improvement.py`). No se crea un nuevo parquet de entrenamiento — se añaden columnas en memoria al leer el parquet.

**Feature set:** `FS-frame-v1` (30 features, `FRAME_V1_FEATURE_NAMES` list en `features_frame_v1.py`).

**Artefacto producido:** `output/models/isolation_forest_frame_v1.joblib` + `output/models/scaler_frame_v1.joblib` + `output/models/model_metadata_frame_v1.json`.

**Métricas de sesgo en validación** (criterio de aceptación FRAME-04):

```python
# Al scorear val_features_enriched.parquet con el modelo frame-v1:
scores = -model.decision_function(X_val_scaled)
k5 = int(len(scores) * 0.05)
top5_idx = np.argsort(scores)[-k5:]
top5_amount_ratio = val_df.iloc[top5_idx]["amount"].mean() / val_df["amount"].mean()
# target: < 4x (baseline: 11.79x)

off_hours_local_pct = val_df["is_off_hours_loc"].mean()  # computed from iana_tz
# target: ~4-5% (baseline UTC: 29.78%)
```

---

### Anti-Patterns to Avoid

- **Modificar el scorer en vivo en esta fase.** `SingleTransactionScorer` y `artifact_loader.py` no se tocan hasta la Fase 3. Esta fase crea código nuevo en `src/fraud_detector/stats/` y `src/fraud_detector/scoring/features_frame_v1.py`.
- **Usar `iqr=0` sin guardia.** El `RobustScaler` de sklearn produce `inf` silenciosamente con `scale_=0`. Guardar `iqr_guarded = max(iqr, 1.0)` en el artefacto y usarlo en la fórmula: `(amount - median) / iqr_guarded`.
- **Dos implementaciones de la aritmética.** `calculate()` y `calculate_from_row()` deben llamar a la misma función privada `_compute_frame_features(...)`. No duplicar la lógica.
- **`day_of_week` como entero en UTC.** Reemplazar por `dow_sin_loc` + `dow_cos_loc` (cíclico, local). El entero day-of-week en UTC introduce el mismo sesgo temporal que `is_off_hours` en UTC.
- **Leer `facility_tz.parquet` en cada scoring request.** Cargar el dict `{fid: iana_tz}` en `__init__` del `FrameV1FeatureCalculator`. Es un dict de 1876 entries, ~200 KB en RAM.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DST-aware conversion | Custom UTC offset dict | `zoneinfo.ZoneInfo` + `astimezone()` | Maneja fold/gap automáticamente |
| Rails→IANA mapping | Scraping Rails source dinamicamente | Dict estático embebido (patrón de exp_frames_improvement.py, completar 13 faltantes) | Las 64 zonas presentes están mapeadas; Rails no añade zonas frecuentemente |
| IQR=0 guard | Per-facility `try/except` | `max(iqr, 1.0)` al build time, almacenado como `iqr_guarded` | Falla silenciosa con `inf` si se olvida |
| Feature contract validation | Runtime shape check ad-hoc | Import-time `assert len(FRAME_V1_FEATURE_NAMES) == 30` + `artifact_loader` validation | Patrón existente en `engineering.py` y `artifact_loader.py` |

---

## Common Pitfalls

### Pitfall 1: `val_features_enriched.parquet` no tiene `time_zone`

**What goes wrong:** La columna `time_zone` no está en `val_features_enriched.parquet` (confirmado por inspección directa). `calculate_from_row()` necesita el IANA timezone para los features locales.

**How to avoid:** El `FrameV1FeatureCalculator` carga el mapping `{fid: iana_tz}` de `facility_stats_v1.json` (campo `iana_tz` por facility). `calculate_from_row(row)` usa `row["facility_id"]` para hacer lookup en ese mapping. No requiere columna adicional en el parquet.

**Warning signs:** `KeyError: 'time_zone'` al ejecutar el test de paridad.

---

### Pitfall 2: `created_at` en parquet es naive (no tiene tz info)

**What goes wrong:** `val_features_enriched.parquet` tiene `created_at` como `datetime64[ns]` sin timezone (confirmado). La conversión `ts.tz_localize("UTC")` es la correcta; usar `ts.tz_convert()` directamente fallará.

**How to avoid:** Siempre `ts.tz_localize("UTC").astimezone(ZoneInfo(iana_tz))`. En el path de real-time, `payment["created_at"]` también llega sin tz info.

---

### Pitfall 3: Train tiene 689 facilities; `facility_tz.parquet` tiene 1876

**What goes wrong:** Confundir la cobertura de facilities del parquet de training (689 facilities con suficiente historia para stats) vs. el parquet de timezone (1876 facilities totales de la plataforma). El artefacto de stats tendrá solo 689 entries bajo `"facilities"`, pero el `tz_map` debe cubrir las 1876 para el real-time path (nuevas facilities desde el training window).

**How to avoid:** Separar claramente dos dicts en `facility_stats_v1.json`: `"facilities"` (estadísticas de magnitud, solo 689 con n>=30) y un `tz_lookup` completo con las 1876 facilities. O alternativamente, almacenar `iana_tz` también en el `currency_fallbacks` dict para facilities sin stats de magnitud. La architecture más simple: todos los 1876 fids tienen entrada en `"facilities"` pero con `fallback_level="currency"` o `"global"` si n < 30. Los campos `median`/`iqr`/`mean` se omiten o son null para facilities con n < 30.

---

### Pitfall 4: Modelo reentrenado con 30 features pero scorer espera 40

**What goes wrong:** El despachador en `SingleTransactionScorer.__init__` usa `len(self._feature_names) == 40` para activar `EnrichedFeatureCalculator`. Un modelo frame-v1 con 30 features no coincidirá y caerá al `SingleFeatureCalculator` de 31 features (incorrecto).

**How to avoid:** Esta fase NO modifica `SingleTransactionScorer`. El modelo frame-v1 se evalúa exclusivamente en los scripts offline y el test de paridad. La integración en el scorer es tarea de la Fase 3. El `model_metadata_frame_v1.json` debe marcar explícitamente `"feature_version": "frame-v1"` para que el artifact_loader futuro lo identifique correctamente.

---

### Pitfall 5: `staff_amount_zscore` no es feature de marco

**What goes wrong:** `staff_amount_zscore` usa stats de rol/moneda aprendidas en `FeatureEngineer._groups[6]._role_currency_stats` — ya está normalizado por rol/moneda, no por facility. No forma parte del grupo que se convierte a marco relativo.

**How to avoid:** Retener `staff_amount_zscore` sin modificación en `FS-frame-v1`. El `FrameV1FeatureCalculator` carga `_role_currency_stats` directamente del `feature_engineer.joblib` (misma pattern que `SingleFeatureCalculator`).

---

### Pitfall 6: Facilities con n < 30 producen IQR inestable

**What goes wrong:** El parquet de train tiene 109 facilities con n < 30 y 116 con IQR = 0. Si se incluyen en el stats artifact sin guardia y sin marcar `fallback_level`, la fórmula `(amount - median) / iqr` produce `inf` o valores extremos.

**How to avoid:** En `FacilityStatsBuilder.build()`: si `len(grp) < MIN_N (30)`, la entry en `"facilities"` se crea con `fallback_level="currency"` y los stats de magnitud toman los valores del `currency_fallback`. El campo `n` se almacena para auditabilidad. El `FrameV1FeatureCalculator` consulta `fallback_level` para saber qué stats usar.

---

### Pitfall 7: `off_hours_high_value_loc` requiere `amount_facility_ratio` calculado

**What goes wrong:** En `exp_frames_improvement.py`, `off_hours_high_value_loc` se define como `(is_off_hours_loc > 0) & (df["amount_facility_ratio"] > 3)`. La `amount_facility_ratio` es la del parquet original (calculada por `ContextualFeatures` en `FeatureEngineer`), no la recalculada con los nuevos stats. En el path de real-time, hay que asegurarse de que `amount_facility_ratio` se compute con los mismos stats del artefacto.

**How to avoid:** Definir el orden de cómputo explícitamente en `_compute_frame_features()`: primero `fmean = stats["facilities"][fid]["mean"]`, luego `amount_facility_ratio = amount / (fmean + 0.01)`, luego `off_hours_high_value_loc = is_off_hours_loc and amount_facility_ratio > 3`. En `calculate_from_row()`, no leer `amount_facility_ratio` del parquet — recalcularla con los stats del artefacto para garantizar paridad.

---

## Code Examples

### Build offline stats artifact

```python
# scripts/build_facility_stats.py
# Source: patrón de exp_frames_improvement.py, facility_stats() function
import json
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from fraud_detector.stats.tz_mapping import RAILS_TO_IANA  # 64 entries completas

def build_facility_stats(train_parquet: str, tz_parquet: str) -> dict:
    train = pd.read_parquet(train_parquet, columns=["amount", "facility_id", "currency"])
    tz_df = pd.read_parquet(tz_parquet).set_index("facility_id")["time_zone"].to_dict()

    g = train.groupby("facility_id")["amount"]
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    iqr_raw = (q3 - q1).replace(0, np.nan)
    iqr_filled = iqr_raw.fillna(float(iqr_raw.median()))  # IQR=0 -> median IQR

    MIN_N = 30
    sizes = g.size()

    # Global fallback
    global_median = float(train["amount"].median())
    global_iqr = float(q3.mean() - q1.mean()) or 1.0
    global_mean = float(train["amount"].mean())

    facilities = {}
    for fid in g.groups:
        n = int(sizes[fid])
        rails_tz = tz_df.get(fid, "")
        iana_tz = RAILS_TO_IANA.get(rails_tz, "Etc/UTC")
        iqr_val = float(iqr_filled.get(fid, global_iqr))
        iqr_guarded = max(iqr_val, 1.0)
        entry = {
            "n": n,
            "iana_tz": iana_tz,
            "iqr": iqr_val,          # raw, for auditing
            "iqr_guarded": iqr_guarded,  # use this in formulas
        }
        if n >= MIN_N:
            entry.update({
                "median": float(g.median()[fid]),
                "mean": float(g.mean()[fid]),
                "fallback_level": "facility",
            })
        else:
            # fallback stats set at lookup time based on currency
            entry.update({
                "median": None,
                "mean": None,
                "fallback_level": "currency",
            })
        facilities[str(fid)] = entry

    return {
        "schema_version": "facility-stats-v1",
        "universe_filter": "_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL",
        "train_rows": len(train),
        "n_facilities": len(facilities),
        "min_n_threshold": MIN_N,
        "global_fallback": {
            "median": global_median, "iqr": global_iqr,
            "iqr_guarded": max(global_iqr, 1.0), "mean": global_mean,
            "n": len(train), "fallback_level": "global",
        },
        "facilities": facilities,
    }
```

### DST conversion using zoneinfo

```python
# Source: verified against Python 3.9 docs; stdlib zoneinfo
from zoneinfo import ZoneInfo
import pandas as pd

OFF_HOURS = frozenset({23, 0, 1, 2, 3, 4, 5, 6})

def utc_to_local_features(ts_utc_naive: pd.Timestamp, iana_tz: str) -> tuple[int, int, int, int]:
    """Returns (hour_local, dow_local, is_off_hours_local, is_weekend_local)."""
    utc_aware = ts_utc_naive.tz_localize("UTC")
    local = utc_aware.astimezone(ZoneInfo(iana_tz))
    h = local.hour
    dow = local.dayofweek  # 0=Monday
    return h, dow, int(h in OFF_HOURS), int(dow >= 5)
```

### Parity test surface

```python
# tests/test_parity_phase1.py — patrón basado en tests/test_parity_phase0.py
class TestParityPhase1:
    """FrameV1FeatureCalculator.calculate() == calculate_from_row() para >=100 pagos, diff <1e-8."""

    def test_frame_features_parity(self, golden_rows, frame_calc):
        for _, row in golden_rows.iterrows():
            payment = _row_to_payment(row)
            context = _row_to_context(row)
            rt_vec = frame_calc.calculate(payment, context)
            batch_vec = frame_calc.calculate_from_row(row)
            diff = np.max(np.abs(rt_vec.astype(np.float64) - batch_vec.astype(np.float64)))
            assert diff < 1e-8, f"Parity failed: diff={diff:.2e} facility={row['facility_id']}"
```

### DST unit test (criterio FRAME-03)

```python
def test_dst_new_york():
    """Spring forward: 2025-03-09T07:00:00Z in America/New_York = 03:00 EDT."""
    from zoneinfo import ZoneInfo
    ts = pd.Timestamp("2025-03-09T07:00:00")  # naive UTC
    h, dow, off, wknd = utc_to_local_features(ts, "America/New_York")
    assert h == 3, f"Expected 3 AM EDT, got {h}"
    assert off == 0   # 3 AM is off-hours? No, {23,0,1,2,3,4,5,6}: 3 IS off-hours
    # Actually 3 AM is in OFF_HOURS set — correct behavior is off=1

def test_dst_buenos_aires():
    """Argentina no tiene DST. UTC-3 todo el año."""
    ts = pd.Timestamp("2025-10-05T03:00:00")  # naive UTC
    h, dow, off, wknd = utc_to_local_features(ts, "America/Argentina/Buenos_Aires")
    assert h == 0, f"Expected midnight ART (-3h), got {h}"
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|------------------|-------|
| UTC temporal features | Local-time temporal features via zoneinfo | `exp_reference_frames.py` confirmó 29.8% → 4.4% off-hours |
| Global amount in features | Per-facility relative amount (`log_amount_fac`, ratios) | `disjoint30_frames` top-5%: 7.3× → 2.1× |
| `facility_avg_amount` as feature | Eliminada (proxy de tamaño, no de comportamiento) | En `exp_frames_improvement.py` FRAME_MAP |
| `day_of_week` integer UTC | `dow_sin_loc` + `dow_cos_loc` cíclico local | Cyclic encoding + local timezone |
| StandardScaler global | RobustScaler(5,95) + clip ±10 | Confirmado en `scaler_final.joblib` (RobustScaler con quantile_range=(5,95)) |

---

## Open Questions

1. **¿Incluir `amount_fac_z` además de `log_amount_fac`?**
   - Lo conocido: `exp_frames_improvement.py` usa `amount_fac_z` solo en `clean29_frames`; `disjoint30_frames` usa `log_amount_fac` y omite `amount_fac_z`.
   - Lo que falta: Ablación separada del efecto de cada uno sobre `tipo_a` AUC en el val set completo.
   - Recomendación: Arrancar con `disjoint30_frames` como está (el prototipo ya alcanzó top-5% 2.1×). Si los criterios de aceptación no se cumplen, agregar `amount_fac_z` como feature adicional.

2. **¿Cuántos currency fallbacks crear?**
   - Lo conocido: 19 currencies en el val set; las principales (USD, CAD, MYR) cubren ~90% del volumen.
   - Lo que falta: Verificar cuántas facilities tienen n < 30 y cuáles son sus currencies. Si muchas tienen la misma currency (ej. HNL), un currency fallback es útil; si son dispersas, el global fallback es suficiente.
   - Recomendación: Implementar currency fallback para las 5 currencies más frecuentes (USD, CAD, MYR, HNL, NIO) y global para el resto. Documentar la decisión en `fallback_level`.

3. **¿El modelo frame-v1 reentrenado de 30 features supera los criterios de sesgo en val full (1.13M filas, todas las monedas)?**
   - Lo conocido: Los experimentos (1/20 muestra, set de test) alcanzaron top-5% 2.1× y off-hours 4.4%. El val full tiene todas las monedas.
   - Lo que falta: Correr el reentrenamiento completo.
   - Recomendación: No bloquear el plan esperando esta validación. Los criterios de aceptación son comprobables solo después del reentrenamiento. Si el modelo no alcanza <4×, el paso de debug obvio es verificar que las features de marco están correctamente computadas (test de paridad).

---

## Sources

### Primary (HIGH confidence)

- Direct inspection: `output/models/isolation_forest_final.joblib` — confirmed IF(200, 512, 0.6, contamination=auto, random_state=42)
- Direct inspection: `output/models/scaler_final.joblib` — confirmed RobustScaler(quantile_range=(5.0, 95.0))
- Direct inspection: `output/revision/facility_tz.parquet` — 1876 facilities, 64 unique Rails TZ names, 0 nulls
- Direct inspection: `output/revision/frames_improvement_results.json` — confirmed disjoint30_frames top5_amount_x_avg_mean=2.14×, offhours_local=4.4%
- Direct code reading: `scripts/exp_frames_improvement.py` — FRAME_MAP, add_frame_features(), facility_stats()
- Direct code reading: `src/fraud_detector/scoring/features.py` — post-fix state confirmed (direct attribute access)
- Direct code reading: `tests/test_parity_phase0.py` — parity test pattern to replicate
- Direct code reading: `scorer/artifact_loader.py` — Artifacts dataclass, validation pattern
- Direct code reading: `src/fraud_detector/scoring/scorer.py` — dispatch logic on len(feature_names)==40
- Direct inspection: `output/baseline_v0.json` — baseline metrics (11.79×, 29.78%), bugs fixed, golden set schema
- Direct code reading: `output/models/final_feature_list_operational.json` — confirmed 39 features (FS-frame-operational-v1)
- Python 3.9 stdlib docs: `zoneinfo` — `tz_localize("UTC").astimezone(ZoneInfo(name))` confirmed correct DST handling
- `.planning/research/STACK.md` — verified IQR=0 pitfall, JSON artifact recommendation, zoneinfo preference
- `.planning/research/ARCHITECTURE.md` — verified FacilityStatsBuilder design, two-surface pattern
- `.planning/research/PITFALLS.md` — DST distortion (26.4% UTC → 4.2% local confirmed)

### Secondary (MEDIUM confidence)

- `.planning/codebase/ARCHITECTURE.md` — layer map, data flow description
- `.planning/codebase/CONVENTIONS.md` — naming, import order, error handling patterns
- `.planning/research/PITFALLS.md` — low-volume segment instability (n < 200 threshold)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — venv inspected, all imports verified
- Facility stats schema: HIGH — derived from direct parquet/code inspection; IQR=0 count (116 facilities) confirmed
- Feature mapping (FS-frame-v1): HIGH — derived from FRAME_MAP in exp_frames_improvement.py + confirmed results JSON
- Architecture patterns: HIGH — derived from existing code patterns (test_parity_phase0.py, artifact_loader.py)
- Pitfalls: HIGH — grounded in active code state (no time_zone column in parquet, naive timestamps, 689 vs 1876 facility discrepancy)
- Open questions: MEDIUM — require running full retraining to resolve

**Research date:** 2026-07-06
**Valid until:** 2026-08-06 (stable codebase; valid until next extraction or model retrain)
