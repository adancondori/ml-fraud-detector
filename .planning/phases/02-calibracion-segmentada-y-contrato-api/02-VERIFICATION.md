---
phase: 02-calibracion-segmentada-y-contrato-api
verified: 2026-07-06T14:33:18Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 2: Calibración Segmentada y Contrato API — Verification Report

**Phase Goal:** Umbrales calibrados por segmento (facility → currency → global) sobre val con guarda de n mínimo; `artifact_loader` carga stats + thresholds segmentados retrocompatible; el contrato `frame-v1` falla de forma observable ante campos ausentes.
**Verified:** 2026-07-06T14:33:18Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `thresholds_segmented_v1.json` existe con cadena de fallback, guarda n≥200, `fallback_level` registrado, global p95≈0.0436 | VERIFIED | 452 facility + 17 currency entries; min n=203; all n≥200; binary_threshold=0.04358835 (≈0.0436, NOT 0.024); `fallback_level` presente en todos los segmentos |
| 2 | `artifact_loader`: `Artifacts` tiene campos opcionales; legacy IF-40 no rompe | VERIFIED | `Artifacts` dataclass tiene `facility_stats: Optional[dict]=None` y `thresholds_segmented: Optional[dict]=None`; `load_artifacts` carga via metadata keys opcionales; `_validate_artifacts` no toca estos campos; 10/10 tests pasan |
| 3 | `schemas.py`: `currency` y `facility_time_zone_iana` son `Optional[str]=None`; `ScoreResponse` tiene `calibration_segment`, `fallback_level`, `frame_flags` | VERIFIED | Ambos campos con `Optional[str] = None` (comentarios explícitos anti-silenciado); `ScoreResponse` tiene los tres campos opcionales; 15/15 tests pasan |
| 4 | `facility_stats_v1.json` tiene ≥14 monedas incluyendo AUD, ILS, PKR, GTQ, SGD; paridad Fase 1 verde | VERIFIED | 14 monedas exactas: AED, AUD, BWP, CAD, COP, GTQ, HKD, HNL, ILS, MYR, NIO, PKR, SGD, USD; 11/11 tests parity_phase1 pasan |
| 5 | `model_metadata_frame_v1.json` tiene `artifact_files`; `feature_list_frame_v1.json` tiene 30 features | VERIFIED | `artifact_files` presente con 4 entradas; `thresholds_segmented_artifact` y `stats_artifact` también referenciados; feature list = 30 features |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `output/models/thresholds_segmented_v1.json` | Fallback chain + n≥200 + p95≈0.0436 | VERIFIED | 452 facilities, 17 currencies, min n=203, binary_threshold=0.04358835 |
| `output/models/facility_stats_v1.json` | ≥14 monedas incl. AUD/ILS/PKR/GTQ/SGD | VERIFIED | 14 monedas exactas, todos los requeridos presentes |
| `output/models/model_metadata_frame_v1.json` | `artifact_files` key presente | VERIFIED | Tiene `artifact_files`, `thresholds_segmented_artifact`, `stats_artifact` |
| `output/models/feature_list_frame_v1.json` | 30 features | VERIFIED | Lista de 30 features; `log_amount_fac` como primer elemento |
| `scorer/artifact_loader.py` | `Artifacts` con campos opcionales; legacy compat | VERIFIED | 142 líneas; dataclass frozen con dos campos `Optional`; load path con guards |
| `scorer/schemas.py` | `currency`/`facility_time_zone_iana` = `Optional[str]=None`; `ScoreResponse` extendido | VERIFIED | 138 líneas; campos con comentarios anti-silenciado; `ScoreResponse` con 3 nuevos campos opcionales |
| `src/fraud_detector/scoring/classifier.py` | `SegmentedThresholdClassifier` existe pero NO cableado al path en vivo | VERIFIED | Clase existe en líneas 117-193; comentario explícito "NOT connected to SingleTransactionScorer (that is Phase 3)" |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `artifact_loader.load_artifacts` | `facility_stats` | `metadata.get("stats_artifact")` | WIRED | Carga condicional si key presente; None si ausente |
| `artifact_loader.load_artifacts` | `thresholds_segmented` | `metadata.get("thresholds_segmented_artifact")` | WIRED | Carga condicional; warn si stats loaded pero thresholds_segmented=None |
| `model_metadata_frame_v1.json` | `thresholds_segmented_v1.json` | `thresholds_segmented_artifact` key | WIRED | `"thresholds_segmented_artifact": "thresholds_segmented_v1.json"` |
| `scorer/routers/score.py` | `SingleTransactionScorer` + `ThresholdClassifier` | `get_scorer` dependency | WIRED (intacto) | Usa `ThresholdClassifier`; sin referencia a `SegmentedThresholdClassifier` |
| `SegmentedThresholdClassifier` | live scorer path | (ninguna) | CORRECTLY ORPHANED | No importado en `scorer/` ni en `scorer.py`; aislamiento Fase 3 correcto |

---

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| Thresholds calibrados por segmento con guarda n≥200 | SATISFIED | — |
| Cadena fallback facility→currency→global con `fallback_level` | SATISFIED | — |
| `artifact_loader` retrocompatible con IF-40 | SATISFIED | — |
| Contrato `frame-v1`: ausencia de `currency`/`timezone` observable (no silenciada) | SATISFIED | — |
| `ScoreResponse` extendido con campos segmentación | SATISFIED | — |
| `currency_fallbacks` con ≥14 monedas incl. AUD, ILS, PKR, GTQ, SGD | SATISFIED | — |
| Scorer en vivo NO re-cableado (Fase 3) | SATISFIED | — |

---

### Anti-Patterns Found

None. No TODOs/FIXMEs bloqueantes, no placeholders, no handlers vacíos en paths críticos. Comentarios `# NOT connected to SingleTransactionScorer (that is Phase 3)` son documentación explícita de aislamiento, no stubs.

---

### Human Verification Required

None. Todos los criterios de éxito son verificables estructuralmente:
- JSON artifacts leídos y validados numéricamente
- Tests ejecutados (86/86 pasan)
- Imports verificados con grep
- Tipos de campos verificados contra código fuente

---

### Summary

La Fase 2 alcanza completamente su objetivo. Los cinco criterios de éxito están verificados contra código y artefactos reales:

1. `thresholds_segmented_v1.json` contiene 452 segmentos de facility + 17 de currency, todos con n≥200 (mínimo n=203 en facility 1376), `fallback_level` registrado en cada entrada, y global `binary_threshold=0.04358835` (≈0.0436, correctamente alejado del 0.024 que habría indicado recalibración sobre train).

2. `artifact_loader.py` implementa carga condicional real con guards: si `stats_artifact` o `thresholds_segmented_artifact` no están en metadata, ambos campos quedan `None` sin excepción. `_validate_artifacts` no toca esos campos. Legacy IF-40 path verificado en 10 tests.

3. `schemas.py` tiene `currency: Optional[str] = None` y `facility_time_zone_iana: Optional[str] = None` con comentarios explícitos que documentan la decisión de no silenciar la ausencia. `ScoreResponse` extendido con `calibration_segment`, `fallback_level`, `frame_flags` opcionales. 15 tests de contrato pasan.

4. `facility_stats_v1.json` tiene exactamente 14 monedas en `currency_fallbacks` incluyendo los 5 requeridos (AUD, ILS, PKR, GTQ, SGD). Paridad de Fase 1: 11/11 tests verdes.

5. `model_metadata_frame_v1.json` referencia `artifact_files` (4 entradas), `thresholds_segmented_artifact`, y `stats_artifact`. `feature_list_frame_v1.json` tiene 30 features.

El scorer en vivo (`SingleTransactionScorer` + `ThresholdClassifier` + `scorer/routers/score.py`) está intacto. `SegmentedThresholdClassifier` existe en `classifier.py` con documentación explícita de que el cableado al path en vivo es Fase 3.

---

_Verified: 2026-07-06T14:33:18Z_
_Verifier: Claude (so-verifier)_
