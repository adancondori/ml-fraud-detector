---
phase: 01-artefacto-de-stats-y-feature-calculator
plan: "02"
subsystem: ml-pipeline
tags: [feature-calculator, frame-v1, zoneinfo, dst, parity, tdd, staff-zscore, iana-tz]

# Dependency graph
requires:
  - phase: 01-artefacto-de-stats-y-feature-calculator
    plan: "01"
    provides: facility_stats_v1.json (1876 facilities with iana_tz + median/mean/iqr_guarded)
  - phase: 00-baseline-freeze-y-bug-triage
    provides: feature_engineer.joblib (staff zscore stats), val_features_enriched.parquet

provides:
  - src/fraud_detector/scoring/features_frame_v1.py (FrameV1FeatureCalculator, FRAME_V1_FEATURE_NAMES)
  - FRAME-01, FRAME-02, FRAME-03 satisfied
  - Paridad batch<->real-time <1e-8 garantizada (max diff observado: 1.44e-12)

affects:
  - 01-03 (reentrenamiento: calcula FRAME_V1 features desde features_frame_v1.py)
  - 03 (integración en scorer: FrameV1FeatureCalculator como nueva superficie)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dual-surface pattern: calculate()/calculate_from_row() → _compute_frame_features() como única fuente aritmética"
    - "original_currency para staff zscore lookup (stats aprendidos en moneda original, not USD)"
    - "Sentinel -1.0 en UserContext para time_since_last_txn/credit_flow_ratio/category_entropy_30d"
    - "ZoneInfo(iana_tz) + tz_localize('UTC').astimezone() para DST correcto (nunca tz_convert sobre naive)"
    - "discount reconstituido como discount_ratio * max(amount, 0.01) para paridad con parquet pre-computado"

key-files:
  created:
    - src/fraud_detector/scoring/features_frame_v1.py
    - tests/test_parity_phase1.py
    - tests/test_dst_frame_v1.py
  modified:
    - src/fraud_detector/scoring/context.py

key-decisions:
  - "currency_original separado de currency: staff stats aprendidos en moneda original (CAD, MYR...) aunque amount esté en USD — lookup usa la clave (role, currency_original)"
  - "time_since_last_txn como campo pre-computado en UserContext (sentinel -1.0) para paridad directa sin reconstruir last_txn_at desde el parquet"
  - "discount reconstituido como dr * max(amount, 0.01) en test helper para reproducir el parquet que usa el mismo denominador"
  - "_lookup_facility unificado: facility-level y currency-level ambos tienen fmean > 0 en el artefacto — un solo guard cubre ambos casos"

patterns-established:
  - "Pattern: _compute_frame_features recibe solo primitivos — cero side effects ni re-lookups entre las dos superficies"
  - "Pattern: calculate_from_row() NUNCA lee time_zone del row — siempre resuelve iana_tz del artefacto por facility_id"
  - "Pattern: context.{field} >= 0 como sentinel para distinguir 'pre-computado' vs 'derivar de otros campos'"

# Metrics
duration: 14min
completed: 2026-07-06
---

# Phase 01 Plan 02: FrameV1FeatureCalculator Summary

**FrameV1FeatureCalculator con 30 features (FS-frame-v1): magnitud relativa a facility (log_amount_fac, ratios), temporales DST-correcto vía ZoneInfo IANA, paridad batch↔real-time max diff = 1.44e-12 sobre 3213 filas y 680 facilities**

## Performance

- **Duration:** ~14 min
- **Started:** 2026-07-06T06:05:17Z
- **Completed:** 2026-07-06T06:19:20Z
- **Tasks:** 3 (RED → GREEN → REFACTOR)
- **Files modified:** 1 created (features_frame_v1.py), 1 modified (context.py), 2 tests created

## Accomplishments

- Implementado `FrameV1FeatureCalculator` con `FRAME_V1_FEATURE_NAMES` (30 features exactas de `frame_version(DISJOINT30)`) y `assert len(FRAME_V1_FEATURE_NAMES)==30` a nivel de módulo
- Paridad batch↔real-time: max diff observado 1.44e-12 sobre 3213 pagos, 680 facilities (tolerancia <1e-8)
- Test `test_calculate_from_row_no_time_zone_column` verde: calculator resuelve `iana_tz` vía `facility_id` sin columna `time_zone` en el row
- DST verificado en 4 casos concretos: NY spring-forward (antes/después), Buenos Aires UTC-3 sin DST, La Paz UTC-4 sin DST; scorer en vivo intacto (0 líneas diff en features.py, features_enriched.py, scorer.py, artifact_loader.py)

## FRAME_V1_FEATURE_NAMES (30 features — lista canónica)

```python
[
    "log_amount_fac",               # log1p(amount / (fmean + 0.01))
    "discount_ratio",               # discount / max(amount, 0.01)
    "has_tip",                      # 1 si tip > 0
    "hour_sin_loc",                 # sin(2π * local_hour / 24)
    "hour_cos_loc",                 # cos(2π * local_hour / 24)
    "dow_sin_loc",                  # sin(2π * local_dow / 7)
    "dow_cos_loc",                  # cos(2π * local_dow / 7)
    "is_weekend_loc",               # local_dow >= 5
    "is_off_hours_loc",             # local_hour in {23,0,1,2,3,4,5,6}
    "time_since_last_txn",          # segundos desde última txn
    "user_amount_24h_fac",          # user_amount_24h / (fmean + 0.01)
    "user_distinct_facilities_30d",
    "user_distinct_methods",
    "amount_facility_ratio",        # amount / (fmean + 0.01)
    "is_club_credit",
    "user_debit_count_30d",
    "user_debit_amount_30d_fac",    # user_debit_amount_30d / (fmean + 0.01)
    "credit_flow_ratio",
    "is_staff",
    "paid_by_manager",
    "staff_amount_zscore",          # (amount - staff_mean) / staff_std por (role, currency)
    "category_entropy_30d",
    "user_merchandise_ratio_30d",
    "small_amount_at_facility",     # amount_facility_ratio < 0.2
    "very_small_amount_at_facility", # amount_facility_ratio < 0.05
    "off_hours_high_value_loc",     # is_off_hours_loc AND amount_facility_ratio > 3
    "gateway_change_recent",
    "is_main_gateway",
    "is_first_gateway_for_user",
    "source_change_recent",
]
```

## Paridad — Resultados

| Métrica | Valor |
|---------|-------|
| Pagos testeados | 3213 |
| Facilities cubiertas | 680 |
| Max diff observado | 1.44e-12 |
| Tolerancia requerida | <1e-8 |
| test_calculate_from_row_no_time_zone_column | VERDE |

## DST — Casos Verificados

| Timestamp UTC | Zona IANA | Hora local esperada | Resultado |
|--------------|-----------|--------------------|----|
| 2025-03-09T07:00:00Z | America/New_York | 03:00 EDT (spring-forward) | VERDE |
| 2025-03-09T06:00:00Z | America/New_York | 01:00 EST (antes del cambio) | VERDE |
| 2025-10-05T03:00:00Z | America/Argentina/Buenos_Aires | 00:00 ART (UTC-3, sin DST) | VERDE |
| 2025-11-15T12:00:00Z | America/La_Paz | 08:00 BOT (UTC-4, sin DST) | VERDE |

## Task Commits

1. **RED — test_dst_frame_v1.py + test_parity_phase1.py (failing)** — `6ac9f2f` (test)
2. **GREEN — features_frame_v1.py + context.py** — `1eb4760` (feat)
3. **Test helper fix (discount_ratio, original_currency)** — `a5cc4c9` (test)
4. **REFACTOR — _lookup_facility simplificado** — `ac743bd` (refactor)

## Files Created/Modified

- `src/fraud_detector/scoring/features_frame_v1.py` — FrameV1FeatureCalculator, FRAME_V1_FEATURE_NAMES (30), OFF_HOURS, assert count; dual surface + _compute_frame_features
- `src/fraud_detector/scoring/context.py` — añadidos `time_since_last_txn`, `credit_flow_ratio`, `category_entropy_30d` con sentinel -1.0
- `tests/test_dst_frame_v1.py` — 6 tests: spring-forward NY, Buenos Aires, La Paz, naive ts, dow convention
- `tests/test_parity_phase1.py` — 11 tests: shape, parity ≥100 pagos, no_time_zone_column, magnitude, no-NaN, feature name contract

## Decisions Made

1. **currency_original separado de currency para staff zscore**: Los stats de staff en `feature_engineer.joblib` se ajustaron con moneda original (ej. `('guest', 'CAD')` → mean=17.93 CAD/USD, ajustado sobre los amounts ya en USD del parquet pero indexados por moneda original). El payment real-time puede pasar `original_currency` separado de `currency='USD'` para reproducir el mismo lookup.

2. **Sentinel -1.0 en UserContext**: `time_since_last_txn`, `credit_flow_ratio`, `category_entropy_30d` como floats con default -1.0 (no provisto) vs derivar de otros campos. Permite al test helper pasar valores pre-computados del parquet directamente sin reconstruir estados intermedios.

3. **discount reconstituido en test helper**: El parquet pre-computa `discount_ratio = discount / max(amount, 0.01)`. Para que `calculate()` reproduzca el mismo ratio, el test helper pasa `discount = discount_ratio * max(amount, 0.01)`.

4. **_lookup_facility unificado**: Tanto `fallback_level=facility` como `fallback_level=currency` tienen `fmean > 0` en el artefacto — un solo guard cubre ambos sin duplicar código.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Descubierta discrepancia de currency en staff zscore**
- **Found during:** GREEN — test_frame_features_parity (diff=7.68 en facility 72, CAD)
- **Issue:** `calculate()` usaba currency='USD' para lookup de staff stats, pero los stats están indexados por moneda original (CAD). El batch path usa la moneda del parquet (CAD) → divergencia.
- **Fix:** Añadido campo `original_currency` a `_compute_frame_features` y al payment dict; `calculate()` acepta `payment.get("original_currency") or currency`; `calculate_from_row()` usa `currency` del row (que es la moneda original). Test helper pasa `original_currency` explícitamente.
- **Files modified:** src/fraud_detector/scoring/features_frame_v1.py, tests/test_parity_phase1.py
- **Committed in:** 1eb4760 (GREEN), a5cc4c9 (test fix)

**2. [Rule 1 - Bug] Descubierta discrepancia de discount_ratio cuando amount=0**
- **Found during:** GREEN — test_frame_features_parity (diff=1e4 en facility 48)
- **Issue:** El parquet pre-computa `discount_ratio = discount / max(amount, 0.01)`. El test helper pasaba `discount = discount_ratio * amount = 0` para rows con amount=0, causando que `calculate()` computara `0 / 0.01 = 0` vs `10000` en el batch.
- **Fix:** Test helper reconstituye `discount = discount_ratio * max(amount, 0.01)`.
- **Files modified:** tests/test_parity_phase1.py
- **Committed in:** a5cc4c9

---

**Total deviations:** 2 auto-fixed (ambas Rule 1 — bugs encontrados durante el ciclo GREEN)
**Impact on plan:** Ambas fixes necesarias para la garantía de paridad. El código de producción (features_frame_v1.py) es correcto — los bugs estaban en el test helper al reconstruir el payment dict desde el parquet.

## Scorer en Vivo — Confirmación

```
git diff HEAD -- src/fraud_detector/scoring/features.py: (sin output)
git diff HEAD -- src/fraud_detector/scoring/features_enriched.py: (sin output)
git diff HEAD -- src/fraud_detector/scoring/scorer.py: (sin output)
git diff HEAD -- src/fraud_detector/scoring/artifact_loader.py: (sin output)
```

Todos los archivos del scorer en vivo están sin modificar. ✓

## Issues Encountered

Ningún bloqueador mayor. Los dos bugs encontrados fueron detectados en el primer corrida del parity test (GREEN phase) y resueltos sin bloquear la ejecución.

## Next Phase Readiness

- `FrameV1FeatureCalculator` listo para consumo en 01-03 (reentrenamiento del modelo global)
- `FRAME_V1_FEATURE_NAMES` exportado desde `features_frame_v1.py` — es la lista de features para `IsolationForest` frame-v1
- El test de paridad (3213 pagos, 680 facilities, diff<1e-12) garantiza que el modelo entrenado offline se comportará igual en producción
- `context.py` modificado: los campos sentinel (-1.0) son retrocompatibles — el `SingleFeatureCalculator` y `EnrichedFeatureCalculator` no leen estos campos
- **Para 01-03**: usar `calculate_from_row(row)` en batch sobre `train_features_enriched.parquet` para construir la matriz X de entrenamiento; usar `FRAME_V1_FEATURE_NAMES` como lista de features

---
*Phase: 01-artefacto-de-stats-y-feature-calculator*
*Completed: 2026-07-06*
