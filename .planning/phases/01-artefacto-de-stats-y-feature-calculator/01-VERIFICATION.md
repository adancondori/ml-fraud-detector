---
phase: 01-artefacto-de-stats-y-feature-calculator
verified: 2026-07-06T17:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 01: Artefacto de Stats y Feature Calculator — Verification Report

**Phase Goal:** Existe `facility_stats_v1.json` versionado (mediana/IQR/zona-IANA/fallback_level por facility, universo del scorer) y un `FrameV1FeatureCalculator` que produce features de marco IDÉNTICAS en batch y real-time.
**Verified:** 2026-07-06T17:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `facility_stats_v1.json` versionado existe con schema_version, universe_filter correcto, y 1876 facilities | VERIFIED | Artifact present; `schema_version="facility-stats-v1"`, `universe_filter="_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL"`, `n_facilities=1876`, `len(facilities)=1876` |
| 2 | Cadena de fallback completa (facility/currency/global) con `fallback_level` por entrada; 0 facilities sin `iana_tz` | VERIFIED | Distribution: 580 facility / 1289 currency / 7 global; `Facilities without iana_tz: 0`; all required fields present on all 1876 entries |
| 3 | `FrameV1FeatureCalculator` con `FRAME_V1_FEATURE_NAMES` (assert==30 at module level), features relativas a facility, hora local IANA+DST, sin USD absoluto ni UTC | VERIFIED | `assert len(FRAME_V1_FEATURE_NAMES) == 30` at line 65; `_local_hour_dow` uses `ZoneInfo`; no `amount_usd_ratio`/`facility_avg_amount`/`log_amount`/`hour_sin`/`hour_cos`/`is_off_hours` (UTC variants) in feature list; test_eliminated_features_absent passes |
| 4 | Modelo frame-v1 reentrenado; gate de sesgo pasa: top-5% robusto (winsorizado p99.9) <4× y off-hours local en banda 3-7% | VERIFIED | `frame_v1_bias_report.json`: `top5_amount_ratio_winsorized_p999=1.485155 < 4.0` (gate1_pass=true); `off_hours_local_pct=0.064626` = 6.46% dentro de [0.03, 0.07] (gate2_pass=true); ratio_raw=7.06× documentado como artefacto de 2 filas corruptas |
| 5 | Test de paridad ≥100 pagos: `calculate` vs `calculate_from_row` diff <1e-8; incluye `test_calculate_from_row_no_time_zone_column` | VERIFIED | `39 passed` — `test_frame_features_parity` corre sobre N_ROWS=100 pagos estratificados (≥20 facilities); `test_calculate_from_row_no_time_zone_column` presente y verde |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `output/models/facility_stats_v1.json` | Artefacto versionado, 1876 facilities | VERIFIED | Exists, 9 top-level keys, 1876 facility entries, schema_version="facility-stats-v1", self-versioned via schema_version field (*.json excluded from git by .gitignore; versioning is the schema_version field per plan spec line 37) |
| `src/fraud_detector/stats/tz_mapping.py` | RAILS_TO_IANA dict 64 zonas | VERIFIED | Exists, exports `RAILS_TO_IANA`, 100% coverage in tests |
| `src/fraud_detector/stats/builder.py` | FacilityStatsBuilder con fallback chain | VERIFIED | 245 lines, `class FacilityStatsBuilder`, `MIN_N=30`, full fallback chain implemented, 94% coverage |
| `src/fraud_detector/stats/validator.py` | validate_universe_filter(stats, sample_df, tz_df) | VERIFIED | 99 lines, 100% test coverage, validates all 6 invariants including `n_facilities==tz_df.facility_id.nunique()` |
| `scripts/build_facility_stats.py` | Script offline de construcción | VERIFIED | Exists, references parquets via `read_parquet` |
| `src/fraud_detector/scoring/features_frame_v1.py` | FrameV1FeatureCalculator, FRAME_V1_FEATURE_NAMES | VERIFIED | 458 lines, 91% coverage; `calculate()` and `calculate_from_row()` both delegate to `_compute_frame_features`; module-level assert at line 65 |
| `tests/test_parity_phase1.py` | Parity tests ≥100 pagos | VERIFIED | 11 tests, N_ROWS=100, MIN_FACILITIES=20, all pass |
| `tests/test_dst_frame_v1.py` | DST tests ≥2 zonas latinoamericanas | VERIFIED | 6 tests; `America/Argentina/Buenos_Aires` (Argentina) + `America/La_Paz` (Bolivia) |
| `tests/test_facility_stats_builder.py` | Integration + unit tests | VERIFIED | 22 tests, all pass |
| `output/models/isolation_forest_frame_v1.joblib` | Modelo reentrenado frame-v1 | VERIFIED | Exists in `output/models/`; `model_metadata_frame_v1.json` confirms `n_features=30`, `feature_version="frame-v1"` |
| `output/frame_v1_bias_report.json` | Gate de sesgo con gate1_pass + gate2_pass | VERIFIED | Both gates pass; `n_val_rows=1130117` (full val set); `built_at=2026-07-06T13:08:46Z` |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `FrameV1FeatureCalculator.__init__` | `facility_stats_v1.json` (loaded externally) | dict passed to `__init__` | WIRED | Constructor accepts `facility_stats: dict`; consumed via `self._stats["facilities"]` and `self._stats["global_fallback"]` |
| `_compute_frame_features` | `_lookup_facility` | `fmean, fmedian, iqr_guarded, iana_tz` | WIRED | All 4 values used in frame arithmetic; iana_tz passed to `_local_hour_dow` |
| `_local_hour_dow` | `ZoneInfo(iana_tz)` | `ts_utc_naive.tz_localize("UTC").astimezone(ZoneInfo(...))` | WIRED | Correct UTC→local conversion; no raw UTC hour used in output |
| `calculate()` → `calculate_from_row()` | `_compute_frame_features` | both call same private method | WIRED | Paridad guaranteed by single arithmetic source; verified by 100-payment test |
| `FacilityStatsBuilder.build()` | `tz_mapping.resolve_iana()` | `from fraud_detector.stats.tz_mapping import resolve_iana` | WIRED | Called at line 84 for every facility in tz_map loop |
| `frame_v1_bias_report.json` | `isolation_forest_frame_v1.joblib` | retrain script | WIRED | `model_metadata_frame_v1.json` confirms same feature set as bias report; `built_at` timestamps consistent |

---

### Live Scorer Isolation

| File | Modified in Phase 01? | Evidence |
|------|-----------------------|---------|
| `src/fraud_detector/scoring/features.py` | No (phase 01 commits only) | Last phase-01-tagged commit: fix(00-01) — cross-phase Phase 0 bugfix (staff z-score fallback correction); no phase-01 feature commits touch this file |
| `src/fraud_detector/scoring/features_enriched.py` | No (phase 01 commits only) | Same as above — only fix(00-01) robustness patch for `_capture_delay_seconds` |
| `src/fraud_detector/scoring/scorer.py` | No | `git log a6ba60d..HEAD -- scorer.py` returns empty |
| `scorer/artifact_loader.py` | No | `git log a6ba60d..HEAD -- scorer/artifact_loader.py` returns empty |
| `output/models/isolation_forest_final.joblib` | No | `git log a6ba60d..HEAD -- isolation_forest_final.joblib` returns empty; file dated Jun 13 |
| `output/models/thresholds_v2.json` | No | Last commit: `5e9ddf6` (Jul 1, pre-phase-01) |

Note: The fix(00-01) commits are Phase 0 backports applied during Phase 1 work; they corrected bugs in `SingleFeatureCalculator` (features.py) and `EnrichedFeatureCalculator` (features_enriched.py) — the old scorer path — not Phase 1 artifacts. The new frame-v1 path (`features_frame_v1.py`) is separate code.

---

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None found | — | — | — |

No TODO/FIXME/placeholder patterns in phase-01 files. No empty handlers or stub returns. `_compute_frame_features` returns a fully-populated `np.ndarray(30,)`.

---

### Nota sobre `currency_group` per-facility

The phase goal description mentions "currency_group" as a per-facility field. The artifact does NOT store a `currency_group` field inside each facility entry. Instead, the builder stores `currency_fallbacks` at the top level (USD/CAD/MYR/HNL/NIO) and selects the appropriate fallback at build time. `FrameV1FeatureCalculator._lookup_facility` does not consume `currency_group`. The `must_haves.truths` in `01-01-PLAN.md` do not list `currency_group` as a per-facility field requirement — the plan spec enumerates `median/iqr/iqr_guarded/mean/n/iana_tz/fallback_level`. No test asserts `currency_group`. The omission is a wording artifact in the goal description; the fallback mechanism is implemented correctly and tested.

---

### Human Verification Required

None. All success criteria are structurally verifiable and confirmed via automated tests and JSON inspection.

---

## Gaps Summary

No gaps. All 5 must-have truths are verified:

1. `facility_stats_v1.json` exists, is self-versioned (`schema_version="facility-stats-v1"`), universe_filter matches scorer exactly, covers all 1876 facilities with all required fields.
2. Fallback chain (580 facility / 1289 currency / 7 global) fully implemented; 0 facilities without `iana_tz`.
3. `FrameV1FeatureCalculator` with 30-feature contract (assert at module level), relative magnitude features, IANA+DST local time, no UTC or USD-absolute features.
4. Bias gate report: winsorized top-5% ratio = 1.49× (<4×) and off-hours local = 6.46% (within 3-7% band); both gates pass.
5. 39 tests green including 100-payment parity test, 2 Latin American DST zones, and `test_calculate_from_row_no_time_zone_column`.

---

_Verified: 2026-07-06T17:00:00Z_
_Verifier: Claude (so-verifier)_
