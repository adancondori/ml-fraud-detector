# Project Research Summary

**Project:** ml-fraud-detector — Reference-Frame Normalization Milestone
**Domain:** Brownfield payment anomaly detection — per-facility normalization, local-time temporal features, segmented threshold calibration, shadow dual-run, human-in-the-loop
**Researched:** 2026-07-06
**Confidence:** HIGH

## Executive Summary

This is a brownfield milestone, not a greenfield build. An IsolationForest scorer is already in production; the goal is to eliminate two known bias sources — nominal USD magnitude (amount top-5% ratio 15.7x, target <4x) and UTC temporal features (off-hours rate ~26–30%, target ~4–5% local) — without breaking the live system. The architectural strategy is straightforward: build a versioned per-facility stats artifact offline at training time, load it into the scorer's in-memory state at startup, and use it uniformly across both the real-time and batch scoring paths. This single artifact is the dependency anchor for every downstream capability: frame-v1 features, segmented thresholds, shadow comparison metrics, and the HITL review queue.

The recommended approach is a strict 9-step build order derived from the dependency graph: fix two active production bugs first (wrong `getattr` attribute names causing silent fallback to global averages), then build the stats artifact, then the normalized feature calculator, then segmented thresholds, then wire it all into the scorer, and finally activate shadow dual-run. Success is measured by bias reduction on shadow data, not AUC. AUC against `pure_fraud` is explicitly a diagnostic anti-metric here: four of the model's own features define the proxy label, making it circular and unsuitable as a gate.

The critical cross-cutting risk is silent train/serve skew: the existing codebase already has two confirmed instances of features computing differently in training vs. real-time due to `getattr` with wrong attribute names. Frame normalization adds more per-facility state; any reuse of the same fragile loading pattern will silently degrade the new features too. Every phase must enforce a parity test — scoring the same transaction via batch pipeline and via real-time scorer and asserting identical feature values — before moving to the next phase.

## Key Findings

### Recommended Stack

The base stack (scikit-learn 1.6.1, pandas 2.3.3, numpy 1.24+, FastAPI, joblib 1.5.3, loguru, ClickHouse) is pinned and unchanged. The only incremental dependency is `tzdata>=2025.1` added to `requirements-scorer.txt` for IANA timezone data availability in Docker containers. All other needs are covered by the existing venv.

For timezone conversion, use stdlib `zoneinfo` (Python 3.9+, confirmed working on this system) rather than `pytz` (maintenance mode). The Rails `facilities.time_zone` field stores ActiveSupport names (~134 entries); a static `RAILS_TO_IANA` mapping dict embedded in the scorer handles the translation — no third-party package exists for this, and the mapping is stable across Rails versions.

**Core technologies (incremental):**
- `zoneinfo` (stdlib): DST-aware UTC-to-local conversion — replaces pytz for all new code; no install needed on Python 3.9+
- `tzdata>=2025.1`: IANA timezone data for Docker scorer container — add to `requirements-scorer.txt`
- JSON (stdlib): serialization for the facility stats artifact — consistent with existing `thresholds.json`/`model_metadata.json`
- `numpy.percentile` (installed): per-segment threshold calibration on `score_samples()` output
- `joblib` + `lz4` (installed): model object serialization only, not for the stats dict

**What NOT to use:**
- `pytz.timezone()` for new code: `localize()` footgun, maintenance mode
- `joblib.dump` for the stats dict: opaque, version-dependent, ~50x larger than JSON for flat float dicts
- `decision_function` for per-segment calibration: bakes in train-time `offset_`; use `score_samples()`
- 1,876 separate `RobustScaler` objects: serialization overhead dominates; identical math via direct numpy formula

### Expected Features

The feature dependency tree has a single root: the Stats Artifact v1 (`facility_stats_v1.json`). Everything else is downstream of it.

**Must have (P1 — required for shadow run to produce valid bias-reduction evidence):**
- Stats Artifact v1: per-facility median/IQR/IANA-tz/currency_group, from training universe, versioned, loaded in-memory at scorer startup
- FS-frame-operational-v1 feature set: `amount_facility_ratio`, robust z-score, `hour_local` via IANA tz, `day_of_week_sin/cos`; excludes `capture_delay_seconds` and raw `log_amount`
- Frame-v1 model retrained on FS-frame-operational-v1, gated on bias reduction (top-5% amount ratio <4x, off-hours ~30%→~5%)
- Per-segment threshold calibration: facility → currency_group → global fallback, val set, min n=100–200 per segment, `thresholds_segmented_v1.json`
- Fallback handling: missing timezone → `fallback_level=utc_only`; currency `"EMPTY"` → sanitize to `"USD"` with `frame_flags.currency_unknown=true`
- Shadow dual-run: champion and frame-v1 score every payment; both rows persisted with distinct `model_version`
- Shadow monitoring queries: alert rate per segment, bias metric (top-5% amount ratio), Jaccard@100
- Human review queue: top-k export sorted by risk_level/percentile for frame-v1, `review_status` writable
- Label capture schema: `reviewer_label`, `reviewed_at`, `reviewer_id`, `score_at_label_time`, `model_version`, `reviewer_saw_factors`

**Should have (P2 — after 2+ weeks of shadow data):**
- SHAP per-alert explanation for `risk_level IN ('high','critical')` only
- Drift monitoring of stats artifact: PSI per facility, rolling 30d, alert on PSI > 0.2
- Reviewer label independence assessment (kappa proxy); needs ~300 reviewed cases
- Alert rate deviation trigger: flag when 7-day rolling rate deviates >20% from target

**Defer (v2+):** Full SHAP for all risk levels; per-segment model variants; external escalation routing; supervised fine-tuning on reviewer labels

**Anti-features to avoid:**
- AUC against `pure_fraud` as a gate metric: circular (four model features define the proxy label)
- UTC-based `is_off_hours`: inflates off-hours 4–6x for Latin American facilities
- Global threshold lowering without segment calibration: amplifies facility-size bias
- Real-time SHAP for every transaction: violates 200ms scoring budget

### Architecture Approach

A frozen train-time artifact flows into a stateless inference path. `FacilityStatsBuilder` computes per-facility stats from the training DataFrame and writes `facility_stats_v1.json`; `artifact_loader.py` loads it at API startup and adds it to the `Artifacts` frozen dataclass; `FrameV1FeatureCalculator` receives the frozen stats via constructor injection for O(1) lookups at score time. The same calculator instance and stats artifact serve both the real-time `/score` endpoint and the batch scorer — this shared-instance pattern guarantees parity.

**Major components (new or modified):**
1. `src/fraud_detector/stats/builder.py` (NEW) — `FacilityStatsBuilder.build(train_df)`; min n=30 per facility before currency/global fallback
2. `src/fraud_detector/stats/schema.py` (NEW) — `FacilityStats` frozen dataclass with `metadata.feature_version`
3. `src/fraud_detector/scoring/features_frame_v1.py` (NEW) — `FrameV1FeatureCalculator`; both `calculate(payment, context)` and `calculate_from_row(row, facility_stats)` parity surfaces
4. `src/fraud_detector/calibration/segmented.py` (NEW) — `SegmentedThresholdCalibrator` + `SegmentedThresholdClassifier` with fallback chain
5. `scorer/artifact_loader.py` (MODIFY) — add `facility_stats` field; validate `feature_version`; backward-compatible (`None` → legacy path)
6. `scorer/shadow/dual_runner.py` (NEW) — persist `scoring_mode='shadow_old'` and `'shadow_new'` rows per payment
7. `src/fraud_detector/scoring/features.py` (MODIFY) — fix `_facility_avg`→`_facility_avg_amount`, `_staff_stats`→`_role_currency_stats` (active bug)

### Critical Pitfalls

1. **Silent train/serve skew from wrong `getattr` names (active in production)** — `scoring/features.py:27-28`; both silently return `{}`, forcing global averages. Fix: direct attribute access + post-load assertion + parity test. Fase 0.
2. **DST gap/fold → silent wrong local-time (6x off-hours inflation confirmed)** — Rails names raise `UnknownTimeZoneError` in pytz. Fix: 134-entry `RAILS_TO_IANA` + `zoneinfo`; store IANA in artifact. Fase 1.
3. **Low-volume segments → unstable percentile thresholds** — at n<100, p95 from <5 obs swings ±10–20pts. Fix: min-n guard (≥200) + fallback hierarchy; store `n`/`fallback_level`. Fase 2.
4. **Proxy circularity — `pure_fraud` AUC is partial autovalidation** — four IF-40 features define the proxy. Fix: gate = bias reduction; label any pure_fraud AUC "diagnostic only". Fase 0.
5. **Shadow comparison invalidated by silent field defaults** — missing `facility_time_zone_iana` → UTC default hides the correction. Fix: fail observably (`frame_flags`); gate shadow on platform integration. Fase 2 + Fase 4.

## Implications for Roadmap

### Fase 0 — Baseline Freeze and Bug Triage
**Rationale:** Four active bugs contaminate any pre-fix measurement (getattr names, test-set threshold, capture_delay=0 real-time, "EMPTY" currency). Fixing first ensures a valid baseline; also locks the gate (bias reduction, not AUC) before anyone optimizes the circular metric.
**Delivers:** Clean baseline scorer; val-calibrated thresholds; locked gate metric; "EMPTY" data-quality report; parity integration test as regression guard.
**Avoids:** Pitfalls 1, 4, 6 (test-set calibration), 7 (capture_delay), 8 (EMPTY currency).

### Fase 1 — Stats Artifact and Feature Calculator
**Rationale:** The stats artifact is the dependency root. Building the calculator in the same phase validates offline build and inference lookup for parity immediately.
**Delivers:** `facility_stats_v1.json`; `FrameV1FeatureCalculator` with parity surfaces; `scripts/build_facility_stats.py`; parity test (batch==real-time for 100+ val rows).
**Avoids:** Pitfalls 2 (DST/timezone), 5 (artifact universe documented in metadata).

### Fase 2 — Segmented Threshold Calibration + Artifact Loader Extension
**Rationale:** Thresholds require val-set scores in the frame-v1 space. Loader extended here so Fase 3 can load stats+thresholds together. API contract locked to prevent silent shadow defaults.
**Delivers:** `thresholds_segmented_v1.json` with fallback hierarchy; `SegmentedThresholdCalibrator` with min-n guard; extended `Artifacts`; backward-compatible loader; API contract with observable failure modes.
**Avoids:** Pitfall 3 (segment instability), Pitfall 9 (silent shadow defaults).

### Fase 3 — Scorer Wiring and Platform Integration
**Rationale:** Assemble tested components into the live scorer. Requires Rails to send `facility_time_zone_iana` before shadow can be meaningful.
**Delivers:** `SingleTransactionScorer` dispatching to frame-v1; `SegmentedThresholdClassifier` receiving facility_id+currency; `frame_flags` persisted; Rails payload extended.
**Avoids:** Pitfall 9 (missing payload fields hide the correction).

### Fase 4 — Shadow Dual-Run and Bias Validation
**Rationale:** Last piece before any promotion. Both paths fully implemented. Shadow run produces the primary evidence — bias reduction on live data.
**Delivers:** `ShadowDualRunner` (both rows per payment); shadow monitoring queries (top-5% ratio, off-hours, per-segment alert rate, Jaccard@100); go/no-go gate (top-5% <4x, off-hours ~4–5%, Spearman ≥0.90, alert-rate delta ≤2pp).
**Avoids:** Pitfall 9.

### Fase 5 — Human Review Queue and Label Capture
**Rationale:** HITL requires active shadow data and the label schema before the first review. Queue opens with z-score `top_factors`; SHAP is a P2 enhancement.
**Delivers:** Review queue; label columns with provenance; HITL sampling including ≥20% non-alerted (below p50) for false-negative estimation; SHAP for high/critical as first v1.x add.
**Avoids:** Pitfall 10 (HITL selection bias).

### Phase Ordering Rationale
- Fases 0–1 strictly sequential: Fase 0 bugs corrupt measurement; Fase 1 artifact is the dependency root.
- Fase 2 depends on Fase 1 (val-set scores in frame-v1 space).
- Fase 3 scorer-internal changes can begin in parallel with Fase 2, but live update needs Fase 2 complete.
- Fase 4 shadow gates on Fase 3 platform integration.
- Fase 5 gates on shadow being active.
- `capture_delay_seconds` exclusion and `"EMPTY"` sanitization must precede Fase 1 stats computation.

### Research Flags
- **Fase 2 (thresholds):** min-n (100 vs 200) needs validation against actual val-set segment size distribution for 1,876 facilities; tabulate sizes first.
- **Fase 5 (HITL):** 80/20 top-k vs random split needs operational validation with review-team capacity; governance sign-off on whether reviewer decisions feed future model decisions.
- Well-documented (no extra research): Fase 0 (localized bug fixes), Fase 1 (numpy/pandas + parity test), Fase 3 (FastAPI lifespan already in use), Fase 4 (standard pre-promotion practice).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified against official docs + installed venv; single new dep (`tzdata`) |
| Features | HIGH | Dependency graph from direct codebase audit; matches `anomaly_scores` schema |
| Architecture | HIGH | Build order from source + experiment scripts; confirmed vs `artifact_loader.py` |
| Pitfalls | HIGH | 8/10 grounded in verified production bugs with line refs; experiment quantifies DST + amount bias |

**Overall confidence:** HIGH

### Gaps to Address
- Minimum-n for segmented calibration: depends on TechSport's val-set segment size distribution; tabulate first in Fase 2.
- Rails payload extension (`facility_time_zone_iana`): parallel work item with its own lead time; confirm before Fase 4.
- HITL review team capacity: 80/20 sampling is best-practice guidance, not observed throughput; confirm before Fase 5.
- Stats artifact refresh cadence: monthly scheduled vs event-triggered; decide during Fase 1.

## Sources

### Primary (HIGH)
- `.planning/codebase/CONCERNS.md` — 12 confirmed concerns incl. active `getattr` bug at `scoring/features.py:27-28`
- `docs/analisis-marcos-referencia.md` — off-hours 26.4% UTC vs 4.2% local; top-5% amount ratio 16.9x vs 1.8x normalized
- `.planning/PROJECT.md` — active requirements and success criteria
- scikit-learn IsolationForest / RobustScaler docs; Python 3.9 `zoneinfo` docs; pytz PyPI (maintenance); FastAPI lifespan docs; joblib compressors comparison; ActiveSupport::TimeZone Rails API

### Secondary (MEDIUM)
- Shadow deployment best practices; threshold calibration in transaction monitoring; SHAP TreeExplainer for IsolationForest

### Tertiary (LOW)
- Min-n for stable percentile thresholds: inferred from order-statistic theory; needs empirical tabulation in Fase 2

---
*Research completed: 2026-07-06*
*Ready for roadmap: yes*
