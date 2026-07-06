---
phase: 03-wiring-del-scorer-e-integracion-platform
plan: 01
subsystem: scoring
tags: [isolation-forest, fastapi, pydantic, scikit-learn, frame-v1, segmented-classifier]

# Dependency graph
requires:
  - phase: 02-calibracion-segmentada-y-contrato-api
    provides: SegmentedThresholdClassifier, thresholds_segmented_v1.json, facility_stats_v1.json, FrameFlags/ScoreResponse schema
  - phase: 01-features-frame-v1
    provides: FrameV1FeatureCalculator (30 features, IANA autónoma desde artefacto)
provides:
  - SingleTransactionScorer con dispatch frame-v1 activado por presencia de artefactos
  - ScoringResult extendido con calibration_segment/fallback_level/frame_flags (default=None)
  - Router /score propaga los 3 campos nuevos al ScoreResponse
  - timezone_missing observable para facilities desconocidas; nunca lanza excepción
  - Retrocompatibilidad IF-40 byte-a-byte garantizada por tests
affects:
  - 03-02 (integración platform Rails — consume calibration_segment/fallback_level/frame_flags)
  - 04-shadow-dual-run (activa shadow/dual-run sobre scorer ya cableado)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Dispatch por presencia de artefactos opcionales (is not None), no por conteo de features
    - Bifurcación 5-tuple / 3-tuple en classify() gobernada por _is_frame_v1
    - frame_flags como dict en ScoringResult (no Pydantic) para no acoplar scorer a FastAPI
    - FrameFlags(**dict) construido en el router (boundary Pydantic-free → Pydantic)

key-files:
  created:
    - tests/test_scorer_frame_v1_dispatch.py
  modified:
    - src/fraud_detector/scoring/classifier.py
    - src/fraud_detector/scoring/scorer.py
    - scorer/routers/score.py

key-decisions:
  - "Dispatch por presencia de artefactos (facility_stats is not None AND thresholds_segmented is not None), no por len(feature_names)==30"
  - "frame_flags es dict en ScoringResult; FrameFlags Pydantic se construye solo en el router boundary"
  - "timezone_missing=True sii facility_id ausente del artefacto (fallback Etc/UTC); nunca timezone_invalid"
  - "_INSERT_COLUMNS del batch scorer no tocado — persistencia batch de campos frame-v1 diferida a Fase 4"
  - "currency normalizada antes de classify: (payment.get('currency') or 'USD').upper()"

patterns-established:
  - "Dispatch por presencia de artefactos opcionales (is not None), nunca por conteo de features"
  - "Bifurcación 5-tuple / 3-tuple: _is_frame_v1 flag; rama else usa ThresholdClassifier 3-tuple"
  - "Router: FrameFlags(**result.frame_flags) solo si result.frame_flags is not None"

# Metrics
duration: 8min
completed: 2026-07-06
---

# Phase 3 Plan 01: Wiring del Scorer e Integración Platform Summary

**SingleTransactionScorer cableado para despachar a FrameV1FeatureCalculator + SegmentedThresholdClassifier por presencia de artefactos; ScoringResult extendido y router propagando calibration_segment/fallback_level/frame_flags al ScoreResponse, con retrocompat IF-40 garantizada**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-06T16:55:53Z
- **Completed:** 2026-07-06T17:03:14Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- `ScoringResult` extendido con 3 campos trailing opcionales (`calibration_segment`, `fallback_level`, `frame_flags=None`) — constructores IF-40 sin cambio, sin `TypeError`
- `SingleTransactionScorer.__init__` tiene tercera rama de dispatch activada por `artifacts.facility_stats is not None AND artifacts.thresholds_segmented is not None` (antes del `elif len==40`), instancia `FrameV1FeatureCalculator` + `SegmentedThresholdClassifier`, pone `_is_frame_v1=True`
- `score()` bifurca: rama frame-v1 llama `classify(raw_score, facility_id=..., currency=...)` (5-tuple) y construye `frame_flags` con `timezone_missing/currency_missing/currency_unknown`; rama IF-40 llama `classify(raw_score)` (3-tuple) sin cambio
- Router `/score` importa `FrameFlags`, construye `FrameFlags(**result.frame_flags)` y pasa los 3 campos a `ScoreResponse`; 52 tests PASS

## Task Commits

1. **Task 1: Extender ScoringResult + RED/GREEN test** — `e786607` (test)
2. **Task 2: Dispatch frame-v1 en __init__ + score() bifurcación** — `fd61456` (feat)
3. **Task 3: Router propaga campos frame-v1** — `62dbd3d` (feat)

## Files Created/Modified

- `tests/test_scorer_frame_v1_dispatch.py` — Suite de 11 tests: ScoringResult defaults, dispatch por artefactos, score() populates fields, timezone_missing, currency_missing, latency budget, router propagation (IF-40 y frame-v1)
- `src/fraud_detector/scoring/classifier.py` — `ScoringResult` extendido con `calibration_segment: Optional[str]`, `fallback_level: Optional[str]`, `frame_flags: Optional[Dict]`; `Optional/Dict` importados
- `src/fraud_detector/scoring/scorer.py` — `__init__` con rama frame-v1 por presencia de artefactos + `_is_frame_v1` flag; `score()` con bifurcación 5-tuple/3-tuple y `frame_flags` construction
- `scorer/routers/score.py` — Importa `FrameFlags`; construye `FrameFlags(**result.frame_flags)` y pasa los 3 campos nuevos al `ScoreResponse`

## Decisions Made

- **Dispatch por presencia de artefactos, no por conteo:** condición `facility_stats is not None AND thresholds_segmented is not None`; el `elif len==40` permanece para el path IF-40, pero NO se añade `elif len==30` (pitfall 1 del research)
- **frame_flags como dict en ScoringResult:** `classifier.py` no importa Pydantic; la conversión a `FrameFlags` ocurre en el router (boundary layer). Permite usar `ScoringResult` en contextos sin FastAPI
- **timezone_missing (no timezone_invalid):** el scorer nunca lanza excepción de zona — `_lookup_facility` retorna `Etc/UTC` para facilities desconocidas. El flag correcto es `missing` (facility no en artefacto), no `invalid`
- **_INSERT_COLUMNS intocado:** el DDL de ClickHouse `anomaly_scores` no tiene columnas frame-v1; extender el INSERT ahora rompería el batch. Fase 4 lo aborda

## Deviations from Plan

None — plan ejecutado exactamente como especificado.

## Issues Encountered

- `UserContext.__init__` no acepta `user_id`/`facility_id`/`account_age_days` (no son campos del dataclass). El helper `_make_minimal_context()` en el test file fue corregido para usar solo los campos reales del dataclass. Corregido inline, sin impacto en el plan.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- El scorer en vivo está cableado y probado en modo frame-v1 (test) y IF-40 (artifacts default)
- Plan 03-02 (integración Rails) puede proceder: `calibration_segment`/`fallback_level`/`frame_flags` disponibles en `/score` response
- `facility_time_zone_iana` de Rails es ignorable — IANA resuelta autónomamente desde `facility_stats_v1.json` (`_lookup_facility`)
- Blocker confirmado resuelto: PLAT-01 (dependencia de Rails para IANA) eliminado en esta fase

---
*Phase: 03-wiring-del-scorer-e-integracion-platform*
*Completed: 2026-07-06*
