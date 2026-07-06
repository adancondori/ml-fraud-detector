---
phase: 01-artefacto-de-stats-y-feature-calculator
plan: "01"
subsystem: ml-pipeline
tags: [facility-stats, iqr-guard, timezone, zoneinfo, json-artifact, fallback-chain, pandas]

# Dependency graph
requires:
  - phase: 00-baseline-freeze-y-bug-triage
    provides: train_features_enriched.parquet (scorer universe), facility_tz.parquet (1876 facilities)
provides:
  - src/fraud_detector/stats/ package (tz_mapping, builder, validator)
  - output/models/facility_stats_v1.json (facility-stats-v1 artifact)
  - scripts/build_facility_stats.py (offline build script)
  - STATS-01 and STATS-02 satisfied
affects:
  - 01-02 (FrameV1FeatureCalculator consumes facility_stats_v1.json for iana_tz + magnitude stats)
  - 01-03 (reentrenamiento carga el mismo artifact)
  - 03 (integración en scorer: artifact_loader cargar facility_stats_v1.json)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FacilityStatsBuilder iterates tz_map (not train_df.groupby) to guarantee 1876-facility coverage"
    - "iqr_guarded = max(iqr, 1.0) — not iqr+1e-6 — stored in artifact for formula use"
    - "Fallback chain: facility (n>=30) -> currency (top-5 currencies + USD) -> global"
    - "resolve_iana() safe fallback to Etc/UTC for unknown Rails names (no KeyError)"
    - "validate_universe_filter() as pre-write gate: aborts build if coverage incomplete"

key-files:
  created:
    - src/fraud_detector/stats/__init__.py
    - src/fraud_detector/stats/tz_mapping.py
    - src/fraud_detector/stats/builder.py
    - src/fraud_detector/stats/validator.py
    - scripts/build_facility_stats.py
    - tests/test_facility_stats_builder.py
  modified: []

key-decisions:
  - "iqr_guarded = max(iqr, 1.0) not max(iqr, 1e-6): production spec to avoid amplifying noise in near-uniform facilities"
  - "Base iteration over tz_map (1876) not train_df.groupby (~689): closes Pitfall 3, ensures iana_tz on all facilities for real-time path"
  - "currency_fallbacks for USD, CAD, MYR, HNL, NIO (top-5 by frequency); 7 facilities fall to global"
  - "validate_universe_filter() receives tz_df as third arg to assert len(facilities)==tz_df.facility_id.nunique()"
  - "JSON artifact (not parquet): human-readable, diffable, 1-5ms load time, suitable for 1876 flat entries"

patterns-established:
  - "Pattern: build script runs validator before writing — abort-on-fail gate prevents corrupt artifact"
  - "Pattern: str(facility_id) as JSON keys (JSON does not support int keys); lookup converts with str(fid)"
  - "Pattern: all numpy scalars converted to native Python float/int before json.dump()"

# Metrics
duration: 5min
completed: 2026-07-06
---

# Phase 01 Plan 01: Facility Stats Artifact Summary

**facility_stats_v1.json artifact with median/IQR/mean per facility for all 1876 facilities, 64 Rails zones fully mapped to IANA, fallback chain (580 facility / 1289 currency / 7 global), 116 IQR=0 cases guarded with iqr_guarded=1.0**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-07-06T05:57:17Z
- **Completed:** 2026-07-06T06:01:46Z
- **Tasks:** 3 of 3
- **Files modified:** 6 created, 0 modified

## Accomplishments

- Created `src/fraud_detector/stats/` package with `RAILS_TO_IANA` (64 entries), `FacilityStatsBuilder`, and `validate_universe_filter` — all new isolated code, scorer untouched
- Materialized `output/models/facility_stats_v1.json`: 1876 facilities (not just the 689 with train history), all with `iana_tz`, `iqr_guarded`, and `fallback_level`
- 22 tests green including integration test asserting `n_facilities == 1876` and `validate_universe_filter` passing against real parquets

## Task Commits

1. **Task 1: RAILS_TO_IANA mapping + tz_mapping module** — `660db3c` (feat)
2. **Task 2: FacilityStatsBuilder with fallback chain + IQR guard** — `4acbd3b` (feat)
3. **Task 3: validator + build script + materialized artifact** — `afdc674` (feat)

## Files Created/Modified

- `src/fraud_detector/stats/__init__.py` — package marker
- `src/fraud_detector/stats/tz_mapping.py` — RAILS_TO_IANA (64 entries), resolve_iana(), import-time ZoneInfo validation
- `src/fraud_detector/stats/builder.py` — FacilityStatsBuilder.build(train_df, tz_map, fid_currency), MIN_N=30, full fallback chain
- `src/fraud_detector/stats/validator.py` — validate_universe_filter(stats, sample_df, tz_df) -> bool
- `scripts/build_facility_stats.py` — offline build script with summary output
- `tests/test_facility_stats_builder.py` — 22 tests (unit + integration)

## Artifact Counts (Real Values)

| Metric | Value | Expected |
|--------|-------|----------|
| n_facilities | 1876 | 1876 |
| fallback_level=facility | 580 | ~689 (range depends on min_n cutoff) |
| fallback_level=currency | 1289 | remainder |
| fallback_level=global | 7 | small (currencies not in top-5) |
| iqr=0.0 | 116 | ~116 |
| iqr_guarded=1.0 | 121 | >=116 (includes cold-start facilities) |
| currency_fallbacks | USD, CAD, MYR, HNL, NIO | USD + top-4 |

Note: `fallback=facility` is 580 (not 689) because the research estimate was based on distinct facility_id count in train, while 580 reflects facilities with n>=30 actual rows after universe filter application.

## Decisions Made

1. **iqr_guarded = max(iqr, 1.0)** (not iqr+1e-6): The prototype used `iqr+1e-6` which amplifies noise for near-uniform distributions. The production spec specifies `max(iqr, 1.0)` — a meaningful floor that avoids inf without introducing instability.

2. **Base iteration over tz_map (not train_df.groupby)**: Crucial to get 1876 entries vs 689. Facilities without train history receive fallback stats but retain `iana_tz` — essential for the real-time scoring path where any of the 1876 facilities can submit a payment.

3. **currency_fallbacks for top-5 + USD always**: Top-5 by frequency in train are USD, CAD, MYR, HNL, NIO. Only 7 facilities (mostly uncommon currencies) fall to global. USD is always included even if it appears in top-5 organically.

4. **validate_universe_filter receives tz_df as 3rd arg**: The plan initially specified a 2-arg signature in the research but the plan tasks specify a 3-arg signature for the facility coverage check. Implemented the 3-arg form as specified in the tasks (the authoritative spec).

## Deviations from Plan

None — plan executed exactly as written. The 51 Rails zones from the prototype + 13 missing zones were assembled as specified. The IQR guard, fallback chain, coverage invariant, and validator all match the plan spec.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `facility_stats_v1.json` is ready for consumption by 01-02 (`FrameV1FeatureCalculator`)
- The `iana_tz` field on all 1876 facilities resolves Pitfall 1 (no `time_zone` column in val parquet) — calculator looks up iana_tz from the artifact via `facility_id`
- `validate_universe_filter` can be called from any test to verify artifact integrity
- Scorer in vivo untouched: `features.py`, `features_enriched.py`, `scorer.py`, `artifact_loader.py` have 0 diff lines

---
*Phase: 01-artefacto-de-stats-y-feature-calculator*
*Completed: 2026-07-06*
