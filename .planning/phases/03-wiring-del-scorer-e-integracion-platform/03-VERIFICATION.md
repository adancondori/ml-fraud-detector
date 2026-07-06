---
phase: 03-wiring-del-scorer-e-integracion-platform
verified: 2026-07-06T17:16:19Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 3: Wiring del Scorer e Integración Platform — Verification Report

**Phase Goal:** El scorer en vivo despacha a frame-v1 por presencia de artefactos (retrocompat IF-40); IANA resuelta autónomamente en el scorer; `AlertManager` persiste metadata ampliada; `scorable?` alineado batch↔real-time.
**Verified:** 2026-07-06T17:16:19Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `SingleTransactionScorer` despacha a `FrameV1FeatureCalculator` + `SegmentedThresholdClassifier` cuando `facility_stats is not None AND thresholds_segmented is not None` (no `len==30`) | ✓ VERIFIED | `scorer.py` líneas 55-67: `getattr(artifacts, "facility_stats", None) is not None and getattr(artifacts, "thresholds_segmented", None) is not None` |
| 2 | IANA resuelta autónomamente (`timezone_missing` observable, sin excepción) | ✓ VERIFIED | `scorer.py` líneas 153-158: `tz_missing = self._feature_calc._stats["facilities"].get(fid_str) is None`; tests `test_frame_flags_timezone_missing_for_unknown_facility` PASS |
| 3 | `AlertManager#build_metadata` persiste `calibration_segment`/`fallback_level`/`frame_flags`/`feature_frame_version` con `.compact` nil-safe | ✓ VERIFIED | `alert_manager.rb` líneas 70-75: 4 campos presentes, `.compact` al final; retrocompat IF-40 por claves nil eliminadas |
| 4 | Router `/score` propaga los 3 campos a `ScoreResponse` | ✓ VERIFIED | `score.py` líneas 43-58: `FrameFlags(**result.frame_flags)` construido en boundary; `calibration_segment`, `fallback_level`, `frame_flags` pasados a `ScoreResponse` |
| 5 | `scorable?` excluye `payment_method IN ('reversal','free')`; `scorer/batch/scorer.py._INSERT_COLUMNS` no tocado en Fase 3 | ✓ VERIFIED | `real_time_scoring_service.rb` líneas 31-34; `_INSERT_COLUMNS` último commit es `a112eca` (Fase 2) — ningún commit de Fase 3 toca `scorer/batch/scorer.py` |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/fraud_detector/scoring/scorer.py` | Dispatch frame-v1 por presencia de artefactos | ✓ VERIFIED | 220 líneas; rama `if (facility_stats is not None and thresholds_segmented is not None)` antes del `elif len==40`; `_is_frame_v1` flag; score() bifurcación 5-tuple/3-tuple |
| `src/fraud_detector/scoring/classifier.py` | `ScoringResult` con 3 campos opcionales trailing | ✓ VERIFIED | `calibration_segment: Optional[str] = None`, `fallback_level: Optional[str] = None`, `frame_flags: Optional[Dict] = None` líneas 61-63; constructores IF-40 sin `TypeError` |
| `scorer/routers/score.py` | Propaga 3 campos al `ScoreResponse` | ✓ VERIFIED | Importa `FrameFlags`; construye `FrameFlags(**result.frame_flags)` si no None; pasa los 3 campos a `ScoreResponse` |
| `scorer/schemas.py` | `ScoreResponse` declara 3 campos opcionales | ✓ VERIFIED | Líneas 76-78: `calibration_segment`, `fallback_level`, `frame_flags` declarados como `Optional` con default `None` |
| `platform/packs/anomaly_detection/app/services/anomaly_detection/alert_manager.rb` | `build_metadata` con 4 campos frame-v1 + `.compact` | ✓ VERIFIED | 86 líneas; 4 campos frame-v1 en hash líneas 70-74; `.compact` en línea 75 |
| `platform/packs/anomaly_detection/app/services/anomaly_detection/real_time_scoring_service.rb` | `scorable?` excluye `free`; `create_alert` propaga 4 campos | ✓ VERIFIED | 103 líneas; `scorable?` líneas 31-34 con guarda `free`; `create_alert` líneas 95-98 propaga los 4 campos |
| `tests/test_scorer_frame_v1_dispatch.py` | Suite de tests frame-v1 dispatch | ✓ VERIFIED | Existe; 11 tests PASS (verificado con pytest) incluyendo `timezone_missing`, `currency_missing`, latency budget, router propagation |
| `tests/test_if40_artifacts.py` | Tests retrocompat IF-40 | ✓ VERIFIED | Existe; 2 tests PASS |
| `platform/packs/anomaly_detection/spec/services/anomaly_detection/alert_manager_spec.rb` | Specs TDD frame-v1 en metadata | ✓ VERIFIED | Existe; contiene contexto con `calibration_segment`/`fallback_level`/`frame_flags`/`feature_frame_version` y prueba retrocompat IF-40 con `.compact` |
| `platform/packs/anomaly_detection/spec/services/anomaly_detection/real_time_scoring_service_spec.rb` | Spec `scorable?` excluye `free` | ✓ VERIFIED | Existe; `context "when payment is free"` presente en línea 180 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `SingleTransactionScorer.__init__` | `FrameV1FeatureCalculator` + `SegmentedThresholdClassifier` | presencia de artefactos (`getattr(...) is not None`) | ✓ WIRED | Dispatch correcto — NOT por `len==30`; `_is_frame_v1=True` seteado |
| `SingleTransactionScorer.score()` | `frame_flags` dict con `timezone_missing` | `_feature_calc._stats["facilities"].get(fid_str) is None` | ✓ WIRED | IANA lookup intacta; `timezone_missing=True` cuando facility ausente; nunca excepción |
| `score.py router` | `ScoreResponse` | `FrameFlags(**result.frame_flags)` en boundary Pydantic | ✓ WIRED | Construcción nil-safe: solo ejecuta si `result.frame_flags is not None` |
| `RealTimeScoringService#create_alert` | `AlertManager` | hash con 4 campos frame-v1 | ✓ WIRED | `calibration_segment/fallback_level/frame_flags` desde `score_result[key]`; `feature_frame_version` mapeado desde `score_result["feature_version"]` |
| `AlertManager#build_metadata` | `Alert.metadata` | hash con `.compact` | ✓ WIRED | 4 campos frame-v1 en hash; `.compact` elimina nil en modo IF-40 |
| `scorer/batch/scorer.py._INSERT_COLUMNS` | DDL ClickHouse `anomaly_scores` | sin columnas frame-v1 | ✓ WIRED (untouched) | Último commit en `scorer/batch/scorer.py` es `a112eca` (Fase 2); ningún commit de Fase 3 lo toca — `_INSERT_COLUMNS` tiene 23 columnas, sin `calibration_segment`/`fallback_level`/`frame_flags` |

---

### Test Execution Results

**Python (ejecutado en entorno local):**

```
tests/test_scorer_frame_v1_dispatch.py ...........   [11/11 PASS]
tests/test_if40_artifacts.py ..                      [2/2  PASS]
================================ 13 passed in 46.32s ==============================
```

**Rails (lectura de código — RSpec no ejecutable sin Docker):**
- `alert_manager_spec.rb`: spec existente verifica persistencia de 4 campos frame-v1 en metadata y retrocompat IF-40 vía `.compact`
- `real_time_scoring_service_spec.rb`: spec existente verifica `scorable?` con `when payment is free` → no puntúa

---

### Anti-Patterns Found

Ninguno. No se detectaron TODOs, stubs, `return null`, ni implementaciones vacías en los archivos modificados.

### Human Verification Required

**1. Rails RSpec suite completa**

**Test:** Correr `bundle exec rspec packs/anomaly_detection/spec/` desde `platform/` con Docker activo
**Expected:** Suite verde, incluyendo los nuevos contextos de frame-v1 en `alert_manager_spec.rb` y `real_time_scoring_service_spec.rb`
**Why human:** RSpec requiere Docker (MySQL, Redis) y el entorno Rails completo — no disponible en esta sesión de verificación automatizada. El análisis estático confirma correctitud estructural pero no ejecución de specs Rails.

---

### Gaps Summary

No hay gaps. Los 5 criterios de éxito verificados contra el código real:

1. **PLAT-01 reconciliado** — Dispatch por presencia de artefactos (`is not None`) confirmado en código; condición `len==30` ausente (pitfall 1 evitado). IF-40 legacy intacto en rama `elif len==40`.
2. **IANA autónoma** — `timezone_missing` derivado de lookup en `_stats["facilities"]`; nunca lanza excepción para facility desconocida.
3. **PLAT-02** — `build_metadata` con 4 campos frame-v1 + `.compact`; `create_alert` los propaga desde `score_result`.
4. **PLAT-03** — `scorable?` excluye `free` (además de `reversal`), alineado con SQL batch `payment_method NOT IN ('reversal', 'free')`.
5. **_INSERT_COLUMNS intocado** — Confirmado por git log: ningún commit de Fase 3 modifica `scorer/batch/scorer.py`.

---

*Verified: 2026-07-06T17:16:19Z*
*Verifier: Claude (so-verifier)*
