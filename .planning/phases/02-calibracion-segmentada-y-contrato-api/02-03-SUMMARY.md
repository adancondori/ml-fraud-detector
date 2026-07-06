---
phase: 02-calibracion-segmentada-y-contrato-api
plan: "03"
subsystem: api
tags: [pydantic, fastapi, artifact-loader, dataclass, schemas, frame-v1, optional-fields, retrocompat]

# Dependency graph
requires:
  - phase: 02-02
    provides: thresholds_segmented_v1.json (452 facilities, 17 currencies) and facility_stats_v1.json (Wave 1+2)
  - phase: 01-03
    provides: model_metadata_frame_v1.json base (feature_names, parity_check, bias_metrics)
provides:
  - model_metadata_frame_v1.json complete with artifact_files, model_version, score_function, threshold_version, thresholds_segmented_artifact
  - feature_list_frame_v1.json (30 feature names, programmatically derived from metadata)
  - Artifacts dataclass with optional facility_stats and thresholds_segmented fields (None for IF-40 legacy)
  - load_artifacts() loads both new artifacts conditionally; _validate_artifacts unchanged
  - ScoreRequest with currency/facility_time_zone_iana/amount_local as Optional=None (no silent defaults)
  - FrameFlags(BaseModel) for observability of missing context
  - ScoreResponse enriched with calibration_segment/fallback_level/frame_flags (all Optional=None)
affects:
  - 03-scorer-frame-v1 (will re-cable SingleTransactionScorer to use new Artifacts fields)
  - 04-api-observability (will consume calibration_segment/fallback_level/frame_flags in responses)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Optional=None without silent defaults: absence of context (timezone, currency) is observable as None, not silenced by a fallback value"
    - "Additive Pydantic extension: new ScoreResponse fields are Optional=None so existing Rails clients are unaffected"
    - "Conditional artifact loading: Artifacts fields default to None for legacy paths; loaded only when metadata references them"

key-files:
  created:
    - output/models/feature_list_frame_v1.json
    - tests/test_artifact_loader.py
    - tests/test_schemas_frame_v1.py
  modified:
    - output/models/model_metadata_frame_v1.json
    - scorer/artifact_loader.py
    - scorer/schemas.py

key-decisions:
  - "currency: Optional[str]=None replaces currency: str='USD' — absence of currency must be observable (None), not silenced to USD"
  - "facility_time_zone_iana: Optional[str]=None, never ='UTC' — UTC default would reintroduce the off-hours bias that the entire project corrects"
  - "Artifacts.facility_stats and .thresholds_segmented as trailing optional fields = None (frozen dataclass, backward compat with positional construction)"
  - "_validate_artifacts not modified: thresholds_segmented_v1.json exposes binary_threshold + score_percentiles at root level, so validation passes natively"
  - "frame-v1 test in test_artifact_loader.py uses tmp_path with model_metadata.json renamed from frame-v1 metadata — avoids conflict with IF-40 model_metadata.json in output/models"

patterns-established:
  - "Pattern: frame-v1 tests always use tmp_path isolation to avoid IF-40 vs frame-v1 metadata conflict"
  - "Pattern: FrameFlags booleans all default to False — existing consumers never receive unexpected truthy flags"

# Metrics
duration: 3min
completed: 2026-07-06
---

# Phase 2 Plan 03: Artifacts Extension and frame-v1 API Contract Summary

**Artifacts dataclass extended with Optional facility_stats/thresholds_segmented (None for IF-40 legacy); model_metadata_frame_v1.json completed with artifact_files; ScoreRequest gains Optional timezone/currency without UTC/USD defaults; ScoreResponse gains calibration_segment/fallback_level/FrameFlags**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-07-06T14:26:05Z
- **Completed:** 2026-07-06T14:29:00Z
- **Tasks:** 3
- **Files modified/created:** 7

## Accomplishments

- `model_metadata_frame_v1.json` now complete: adds `artifact_files` (model/scaler/feature_list/thresholds), `model_version=frame-v1`, `score_function=decision_function`, `threshold_version=segmented-v1`, `thresholds_segmented_artifact` — all existing keys (parity_check, bias_metrics, stats_artifact, etc.) preserved
- `feature_list_frame_v1.json` generated programmatically from `feature_names` (30 features, byte-identical to metadata list)
- `Artifacts` frozen dataclass extended with `facility_stats: Optional[dict] = None` and `thresholds_segmented: Optional[dict] = None`; `load_artifacts()` loads both conditionally from metadata keys; `_validate_artifacts` left intact; IF-40 legacy path unaffected (both fields remain None)
- `ScoreRequest`: `currency` and `facility_time_zone_iana` changed to `Optional[str] = None` (no silent defaults); `amount_local: Optional[float] = None` added for logging context
- New `FrameFlags(BaseModel)` with `timezone_missing`, `currency_missing`, `currency_unknown` booleans (all default False)
- `ScoreResponse` enriched with `calibration_segment`, `fallback_level`, `frame_flags` (all `Optional = None`), additive for backward compat

## Task Commits

1. **Task 1: Completar model_metadata_frame_v1.json y crear feature_list_frame_v1.json** - `4526869` (feat)
2. **Task 2: Extender Artifacts y load_artifacts (retrocompatible)** - `aa7a6c5` (feat)
3. **Task 3: Contrato Pydantic frame-v1 (ScoreRequest/ScoreResponse)** - `2bf2223` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `output/models/model_metadata_frame_v1.json` — Added artifact_files, model_version, score_function, threshold_version, thresholds_segmented_artifact
- `output/models/feature_list_frame_v1.json` — New: 30 frame-v1 feature names, programmatically generated
- `scorer/artifact_loader.py` — Artifacts dataclass + load_artifacts extended; _validate_artifacts unchanged
- `scorer/schemas.py` — ScoreRequest optional fields; FrameFlags class; ScoreResponse enriched
- `tests/test_artifact_loader.py` — New: 10 tests (IF-40 legacy, frame-v1 both fields, warning behavior)
- `tests/test_schemas_frame_v1.py` — New: 15 tests (ScoreRequest Optional invariants, FrameFlags, ScoreResponse backward compat + roundtrip)

## Decisions Made

- `currency: Optional[str]=None` — The IF-40 scorer uses `payment.get("currency", "USD")` so None is backward-compat at the scorer level; the absence is observable at the Pydantic level where it matters
- `facility_time_zone_iana: Optional[str]=None`, never `="UTC"` — UTC default is the exact bias being corrected by the project; must be observable
- Trailing optional fields in frozen `Artifacts` dataclass avoids breaking existing positional construction
- `_validate_artifacts` not modified: `thresholds_segmented_v1.json` already exposes `binary_threshold` and `score_percentiles` at root level
- `test_artifact_loader.py` uses `tmp_path` with `model_metadata.json` renamed from frame-v1 content to avoid the IF-40 priority in `_load_metadata`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None — all three tasks executed cleanly on first attempt. The potential issue of `_load_metadata` picking up IF-40 `model_metadata.json` before the frame-v1 metadata was anticipated in the plan and handled via tmp_path isolation in tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Ready:** `load_artifacts()` can now load frame-v1 artifacts including `facility_stats` and `thresholds_segmented`; `ScoreRequest` will propagate absent timezone/currency as observable `None`; `ScoreResponse` can carry `calibration_segment`/`fallback_level`/`frame_flags`
- **Fase 3 pre-condition:** `SingleTransactionScorer` re-cabling will consume `artifacts.facility_stats` and `artifacts.thresholds_segmented`; `facility_time_zone_iana` from `ScoreRequest` will drive `FrameV1FeatureCalculator`
- **Blocker (pre-Fase 4):** Rails payload extension (`facility_time_zone_iana`) has its own lead time — confirm availability before activating Fase 4

---
*Phase: 02-calibracion-segmentada-y-contrato-api*
*Completed: 2026-07-06*
