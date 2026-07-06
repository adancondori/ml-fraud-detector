---
phase: 00-baseline-freeze-y-bug-triage
plan: 01
subsystem: scoring
tags: [python, pandas, numpy, scikit-learn, isolation-forest, feature-engineering, train-serve-skew]

# Dependency graph
requires: []
provides:
  - SingleFeatureCalculator con acceso directo _facility_avg_amount y _role_currency_stats (no getattr)
  - Assertions post-carga en SingleFeatureCalculator (falla ruidosa si artefacto corrupto)
  - Fallback chain staff_amount_zscore idéntico al de StaffRoleFeatures.transform() (3 pasos)
  - _capture_delay_seconds robusto contra NaT y strings malformados (pd.isnull + try/except)
  - test_parity_phase0.py: guardrail de regresión batch<->real-time (3 tests, 680 facilities)
affects:
  - 00-03 (baseline congelado debe medirse post-fix; métricas de score delta documentadas aquí)
  - cualquier plan que use SingleFeatureCalculator o evalúe IF-40 en real-time

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Acceso directo con assert post-carga (no getattr) para atributos de artefactos entrenados"
    - "Test de paridad batch<->real-time como guardrail de regresión permanente"
    - "Fallback chain 3 pasos para z-score: (role, currency) -> currency -> global"

key-files:
  created:
    - tests/test_parity_phase0.py
  modified:
    - src/fraud_detector/scoring/features.py
    - src/fraud_detector/scoring/features_enriched.py

key-decisions:
  - "Usar actual_role (no forzar a player) en lookup de staff_amount_zscore — StaffRoleFeatures.transform() itera rol raw"
  - "Eliminar fallback (role, USD) del plan original — batch no tiene ese paso; agrega divergencia"
  - "Fallback chain real: (role, currency) -> _currency_stats[currency] -> global (3 pasos exactos del batch)"

patterns-established:
  - "getattr con default={} es antipatrón para artefactos entrenados: usar acceso directo + assert len>0"
  - "pd.isnull() cubre todos los paths de construcción de NaT; 'is pd.NaT' no es confiable"

# Metrics
duration: 9min
completed: 2026-07-06
---

# Phase 00 Plan 01: Bug Triage getattr + NaT Summary

**Train/serve skew crítico corregido: scorer real-time ahora usa 689 facility means y 81 combinaciones (role, currency) vs 0 dicts vacíos pre-fix; delta facility_avg_amount y staff_amount_zscore = 0.00e+00 sobre 14831 filas, 680 facilities**

## Performance

- **Duration:** 9 min
- **Started:** 2026-07-06T05:06:03Z
- **Completed:** 2026-07-06T05:14:32Z
- **Tasks:** 3
- **Files modified:** 3 (2 src + 1 test)

## Accomplishments

- Corregidos dos `getattr` con nombres incorrectos en `SingleFeatureCalculator.__init__`: `_facility_avg` → `_facility_avg_amount` y `_staff_stats` → `_role_currency_stats`, con assertions post-carga que fallan ruidosamente si el artefacto es corrupto
- Reescrito lookup de `staff_amount_zscore` para usar (a) rol raw del contexto sin forzar a "player", y (b) fallback chain 3 pasos idéntico al de `StaffRoleFeatures.transform()`: `(role, currency)` → `_currency_stats[currency]` → global
- Robustecido `_capture_delay_seconds` con `pd.isnull()` y `try/except (ValueError, TypeError)` — strings malformados, `None` y `NaT` devuelven `0.0` sin excepción
- Creado guardrail de regresión `tests/test_parity_phase0.py` con 3 tests que verifican paridad <1e-6 sobre 14831 filas estratificadas por facility_id (680 facilities)
- 192 tests del suite completo pasan sin regresiones

## Task Commits

Cada tarea commiteada atómicamente:

1. **Task 1: Corregir getattr + lookup de staff** - `480e5af` (fix)
2. **Task 1 additional: Corregir fallback role_key para paridad exacta** - `3b745da` (fix)
3. **Task 2: Robustecer _capture_delay_seconds** - `ab8b1d6` (fix)
4. **Task 3: Test de paridad batch<->real-time** - `862239a` (test)

## Files Created/Modified

- `src/fraud_detector/scoring/features.py` — acceso directo `_facility_avg_amount` y `_role_currency_stats`, assertions post-carga, fallback chain 3 pasos con `actual_role`
- `src/fraud_detector/scoring/features_enriched.py` — `_capture_delay_seconds` con `pd.isnull()` + `try/except`
- `tests/test_parity_phase0.py` — 3 tests de paridad batch<->real-time sobre 14831 filas, 680 facilities

## Delta de Scores Pre/Post Fix (evidencia de bug activo)

### Bug activo pre-fix
- `_facility_avgs` (scorer RT) = `{}` (dict vacío) — toda lookup caía al global fallback
- `_staff_stats` (scorer RT) = `{}` (dict vacío) — toda lookup caía al global fallback
- Pre-fix: `facilities=0, staff_roles=0` en log de carga

### Post-fix
- `facilities=689, staff_role_currency=81` en log de carga
- `_facility_avg_amount` range: min=0.00, max=259375.00, median=35.78 (USD-normalized)
- Delta facility_avg vs global_mean (368.61 USD): mean=1498.41, max=259006.39

### Magnitud del train/serve skew pre-fix
Las facilities que más se alejaban del global (ej: facilities en AED/ARS con medias >>368 USD) tenían `facility_avg_amount` en RT siempre = 368.61 (global). El batch usaba el valor real (ej: 190790 para facilidades de alto monto en moneda local). Esto significa que `amount_facility_ratio` (F23) estaba completamente errado en RT para ~680 facilities.

Para `staff_amount_zscore`: roles como `guest`, `rental_user`, `court_manager` en monedas no-USD tenían z-scores basados en media global (368.61, std=121834) en lugar de sus stats reales por (role, currency). El skew era de hasta 1030 unidades para casos extremos como `guest` con montos grandes en COP.

### Post-fix parity (resultado final)
- `facility_avg_amount` max_delta: 0.00e+00 (14831 filas, 680 facilities)
- `staff_amount_zscore` max_delta: 0.00e+00 (14831 filas, 680 facilities)

**El baseline congelado (plan 00-03) DEBE medirse con el scorer post-fix. Las métricas pre-fix no son comparables.**

## Decisions Made

- **Usar `actual_role` en lugar de `role_key`**: El plan original especificaba `role_key = context.user_role if is_staff else "player"`. Esta remapeación no existe en `StaffRoleFeatures.transform()` — el batch usa el rol raw para todos los usuarios. La corrección es usar `actual_role = context.user_role or "player"` directamente, sin remapear no-staff a "player".

- **Eliminar fallback `(role, "USD")`**: El plan del research (RESEARCH.md) incluía un paso intermedio `elif (role_key, "USD") in self._staff_stats` que no existe en `StaffRoleFeatures.transform()`. Este paso causaba divergencia para roles con monedas no-USD (ej: `teacher` en JPY usaba stats de `teacher/USD` en RT pero stats de `currency_stats[JPY]` en batch). La cadena correcta es 3 pasos: `(role, currency)` → `_currency_stats[currency]` → global.

- **Paridad exacta como criterio**: La tolerancia <1e-6 especificada en el plan se verificó sobre las 14831 filas del set estratificado, no solo sobre las 500 mínimas. El delta final es exactamente 0.0 para ambas features porque los stats almacenados son float32 y el path de cálculo es idéntico.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] role_key fuerza a "player" para roles no-staff (guest, rental_user)**

- **Found during:** Task 3 (test_staff_zscore_parity fallaba con max_delta=1030)
- **Issue:** `role_key = context.user_role if is_staff else "player"` mapea `guest`, `rental_user` etc. a `"player"`, pero `StaffRoleFeatures.transform()` usa el rol raw para todos. Resultado: z-scores completamente distintos para ~2087/14831 filas (14%).
- **Fix:** `actual_role = context.user_role or "player"` — sin remapear a "player" implícitamente.
- **Files modified:** `src/fraud_detector/scoring/features.py`
- **Verification:** max_delta=0.00e+00 sobre 14831 filas después del fix
- **Committed in:** `3b745da`

**2. [Rule 1 - Bug] Fallback `(role, "USD")` extra diverge del batch**

- **Found during:** Task 3 (6 filas residuales con delta>1e-4 después de fix anterior)
- **Issue:** Paso `elif (role_key, "USD") in self._staff_stats` introducía una divergencia: `teacher` en JPY usaba `teacher/USD` stats (mean=85.13, std=2854.59) en RT, pero el batch usaba `_currency_stats['JPY']` (mean=87.88, std=121.54). Este paso no existe en el batch.
- **Fix:** Eliminar el paso intermedio; cadena queda en 3 pasos exactos del batch.
- **Files modified:** `src/fraud_detector/scoring/features.py`
- **Verification:** max_delta=0.00e+00 después del fix
- **Committed in:** `3b745da` (mismo commit que fix anterior)

---

**Total deviations:** 2 auto-fixed (ambas Rule 1 - Bug)
**Impact on plan:** Correcciones necesarias para alcanzar paridad exacta. El plan del research describía el fallback chain con 4 pasos, pero el batch real tiene 3. El test de paridad reveló la discrepancia. Sin estas correcciones, el test habría fallado y el guardrail no protegería realmente el scorer.

## Issues Encountered

Ninguno de carácter bloqueante. El test de paridad falló en primera ejecución (max_delta=1030) por los bugs de role_key y fallback chain no identificados en el research. El diagnóstico fue directo: comparar la cadena de fallback de `calculate()` con `StaffRoleFeatures.transform()` línea a línea.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 00-02**: Listo para ejecutar (sanitización de currency EMPTY, freeze de feature set, deprecación de thresholds legacy). No depende de este plan.
- **Plan 00-03 (Baseline Freeze)**: El baseline debe medirse POST-FIX. El scorer ahora usa stats per-facility reales. El delta de scores vs pre-fix es potencialmente sustancial para facilities con monedas locales y altos montos. Documentar distribución del delta en 00-03.
- **Guardrail activo**: `tests/test_parity_phase0.py` corre en CI. Cualquier regresión de paridad batch<->real-time fallará inmediatamente.

---
*Phase: 00-baseline-freeze-y-bug-triage*
*Completed: 2026-07-06*
