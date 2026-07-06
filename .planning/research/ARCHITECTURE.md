# Architecture Research

**Domain:** Reference-frame-normalized payment anomaly scoring (brownfield milestone)
**Researched:** 2026-07-06
**Confidence:** HIGH — derived from direct codebase audit; supplemented by industry pattern verification

---

## Standard Architecture

### System Overview: As-Is vs. Target

```
AS-IS (current)
───────────────────────────────────────────────────────────────────────
 Rails platform                FastAPI scorer/
 ┌──────────────────┐          ┌─────────────────────────────────────┐
 │  after_commit    │ HTTP     │  lifespan startup                   │
 │  POST /score  ──────────▶  │  ┌──────────────────────────────┐   │
 │  POST /score/batch │       │  │  Artifacts (frozen dataclass) │   │
 └──────────────────┘         │  │  ├─ model (IF-40 .joblib)     │   │
                               │  │  ├─ scaler (.joblib)         │   │
 ClickHouse (prod, RO)         │  │  ├─ feature_list (JSON)      │   │
 ┌──────────────────┐          │  │  ├─ thresholds (JSON)        │   │
 │  payments FINAL  │◀─────── │  │  └─ metadata (JSON)          │   │
 │  users FINAL     │ 5-6 q   │  └──────────────────────────────┘   │
 │  facilities_users│  per    │                                     │
 └──────────────────┘  txn    │  SingleTransactionScorer            │
                               │  ┌──────────────────────────────┐   │
 ClickHouse (local, RW)        │  │  UserContextProvider  (5-6q)  │   │
 ┌──────────────────┐          │  │  EnrichedFeatureCalculator    │   │
 │  anomaly_scores  │◀─────── │  │  │ SingleFeatureCalculator    │   │
 └──────────────────┘  INSERT │  │  │  ├─ global_avg_amount ✓   │   │
                               │  │  │  ├─ _facility_avg     ✗   │   │
                               │  │  │  │  (BUG: returns {})     │   │
                               │  │  │  └─ _staff_stats      ✗   │   │
                               │  │  ThresholdClassifier          │   │
                               │  │  (global percentile LUT)      │   │
                               │  └──────────────────────────────┘   │
                               └─────────────────────────────────────┘

KNOWN SKEW POINTS:
  ① _facility_avg_amount: getattr uses wrong attr name → always {}
  ② _staff_stats: getattr uses wrong attr name → always {}
  ③ temporal features in UTC instead of facility local time
  ④ capture_delay_seconds = 0 at real-time (≠ historical distribution)
  ⑤ thresholds: global percentile LUT only, no per-segment calibration
```

```
TARGET (this milestone adds)
───────────────────────────────────────────────────────────────────────
 OFFLINE BUILD (train time)
 ┌────────────────────────────────────────────────────────────────────┐
 │  src/fraud_detector/stats/                                         │
 │  ┌───────────────────────────────────────────────────────────┐    │
 │  │  FacilityStatsBuilder.build(train_df)                      │    │
 │  │  → per_facility: {fid: {mean, median, iqr, p25, p75, n}}  │    │
 │  │  → per_currency: {cur: {...}}                              │    │
 │  │  → global: {mean, median, iqr}                             │    │
 │  │  → metadata: {built_at, train_rows, model_version, ...}   │    │
 │  └───────────────────────────────────────────────────────────┘    │
 │             │                                                       │
 │             ▼                                                       │
 │  output/models/facility_stats_v1.joblib   (or .json)               │
 └────────────────────────────────────────────────────────────────────┘
          │
          │  (bundled alongside model, scaler, thresholds)
          ▼
 OFFLINE CALIBRATION (val set only)
 ┌────────────────────────────────────────────────────────────────────┐
 │  src/fraud_detector/calibration/                                   │
 │  ┌───────────────────────────────────────────────────────────┐    │
 │  │  SegmentedThresholdCalibrator.fit(scores, segments, val_df)│    │
 │  │  Fallback chain:                                           │    │
 │  │    1. facility-specific threshold (if n_val ≥ min_n)      │    │
 │  │    2. currency-group threshold (if n_val ≥ min_n)         │    │
 │  │    3. global threshold (always available)                  │    │
 │  │  Output: thresholds_segmented_v1.json                      │    │
 │  └───────────────────────────────────────────────────────────┘    │
 └────────────────────────────────────────────────────────────────────┘
          │
          ▼
 API STARTUP (scorer/main.py lifespan)
 ┌────────────────────────────────────────────────────────────────────┐
 │  load_artifacts(model_dir) now also loads facility_stats_v1.joblib │
 │  Artifacts frozen dataclass: (model, scaler, feature_list,         │
 │                               thresholds, facility_stats, metadata) │
 │                                                                     │
 │  SingleTransactionScorer(artifacts=...)                             │
 │  ├─ FrameV1FeatureCalculator  ← uses facility_stats in-memory      │
 │  │  (replaces EnrichedFeatureCalculator for frame-v1 features)      │
 │  └─ SegmentedThresholdClassifier  ← fallback: facility→global      │
 └────────────────────────────────────────────────────────────────────┘

 REAL-TIME PATH
 Rails → POST /score → FrameV1FeatureCalculator
                         ├─ lookup facility_stats[fid] (in-memory dict, O(1))
                         ├─ fallback to currency_stats[cur] if missing
                         └─ fallback to global_stats if missing
                        → model.score_features()
                        → SegmentedThresholdClassifier.classify(score, fid, cur)
                           fallback: facility_threshold → currency_threshold → global

 BATCH PATH (BatchScorer)
 cursor → fetch payments → BatchContextProvider (6 queries)
        → FrameV1FeatureCalculator (SAME stats artifact, same code path)
        → model.score_features()
        → SegmentedThresholdClassifier (SAME instance)
        → INSERT anomaly_scores
```

---

## Component Responsibilities

| Component | Responsibility | Location | Communicates With |
|-----------|---------------|----------|-------------------|
| `FacilityStatsBuilder` | Compute per-facility, per-currency, global reference stats from train_df | `src/fraud_detector/stats/builder.py` (NEW) | `FeatureEngineer` train output |
| `FacilityStats` (dataclass) | Frozen container for all reference stats + metadata | `src/fraud_detector/stats/schema.py` (NEW) | `artifact_loader`, `FrameV1FeatureCalculator` |
| `artifact_loader.load_artifacts()` | Atomic load: model + scaler + feature_list + thresholds + facility_stats | `scorer/artifact_loader.py` (MODIFY) | `scorer/main.py` lifespan |
| `Artifacts` (dataclass) | Frozen bundle: adds `facility_stats` field | `scorer/artifact_loader.py` (MODIFY) | `SingleTransactionScorer` |
| `FrameV1FeatureCalculator` | Compute frame-v1 normalized features using in-memory facility_stats (no CH per-request) | `src/fraud_detector/scoring/features_frame_v1.py` (NEW) | `FacilityStats`, `UserContext` |
| `SegmentedThresholdCalibrator` | Fit per-facility/currency/global thresholds on val set scores | `src/fraud_detector/calibration/segmented.py` (NEW) | val parquet, scores |
| `SegmentedThresholdClassifier` | Classify score with fallback chain: facility→currency→global | `src/fraud_detector/scoring/classifier.py` (MODIFY) | `thresholds_segmented_v1.json` |
| `SingleTransactionScorer` | Facade: context → features → score → classify | `src/fraud_detector/scoring/scorer.py` (MODIFY) | all scoring components |
| `BatchScorer` | Bulk fetch → context → score → insert | `scorer/batch/scorer.py` (no structural change) | same scorer instance |
| `ShadowDualRunner` | Score same payload through old + new path, compare, log | `scorer/shadow/dual_runner.py` (NEW) | both classifiers |

---

## Recommended Project Structure

The new code slots into the existing tree without restructuring it. New directories are marked `(NEW)`.

```
src/fraud_detector/
├── data/
├── features/
│   └── engineering.py          # FeatureEngineer (unchanged for training)
├── models/
├── stats/                      # (NEW) reference-stats artifact
│   ├── __init__.py
│   ├── builder.py              # FacilityStatsBuilder.build(df) → FacilityStats
│   └── schema.py               # FacilityStats frozen dataclass
├── calibration/                # (NEW) per-segment threshold calibration
│   ├── __init__.py
│   └── segmented.py            # SegmentedThresholdCalibrator + JSON serde
├── scoring/
│   ├── classifier.py           # MODIFY: add SegmentedThresholdClassifier
│   ├── context.py              # unchanged
│   ├── features.py             # MODIFY: fix _facility_avg_amount bug (#8)
│   ├── features_enriched.py    # unchanged (IF-40 path remains)
│   ├── features_frame_v1.py    # (NEW) FrameV1FeatureCalculator
│   └── scorer.py               # MODIFY: accept FacilityStats, dispatch to FrameV1
├── evaluation/
├── reporting/
└── utils/

scorer/
├── artifact_loader.py          # MODIFY: load facility_stats, expand Artifacts
├── dependencies.py             # unchanged
├── main.py                     # MODIFY: pass facility_stats to scorer init
├── schemas.py                  # unchanged (Rails contract unchanged)
├── shadow/                     # (NEW) shadow dual-run infrastructure
│   ├── __init__.py
│   └── dual_runner.py          # ShadowDualRunner: old path vs new path
├── routers/
│   ├── model.py                # unchanged
│   └── score.py                # MODIFY: shadow mode flag in request (optional)
└── batch/
    └── scorer.py               # unchanged (uses scorer instance)

output/models/
├── isolation_forest_final.joblib   (existing)
├── scaler_final.joblib             (existing)
├── feature_engineer.joblib         (existing)
├── model_metadata.json             (existing)
├── thresholds_v2.json              (existing — global, used as fallback)
├── facility_stats_v1.joblib        (NEW — built offline, loaded at startup)
└── thresholds_segmented_v1.json    (NEW — per-facility/currency/global)

scripts/
├── build_facility_stats.py         (NEW — offline build step)
└── calibrate_segmented_thresholds.py (NEW — offline calibration step)
```

---

## Architectural Patterns

### Pattern 1: Frozen Stats Artifact (Train-Time Build, Serve-Time Load)

**What:** Reference statistics (per-facility mean/median/IQR, per-currency fallbacks, global fallbacks) are computed once from the training set, serialized to `facility_stats_v1.joblib`, and loaded into memory at API startup alongside the model. No per-request ClickHouse lookups for facility stats.

**When to use:** Any feature that depends on group-level aggregates (facility average amount, staff z-score, currency normalizer), where computing per-request would violate the 0.2s budget or introduce non-determinism between batch and real-time paths.

**Why it eliminates skew:** The exact same dict is used in `FeatureEngineer.transform()` during batch training (learned from `fit()`) and in `FrameV1FeatureCalculator.calculate()` at inference. Both paths share the same frozen object rather than each computing it independently.

**Example (Python):**
```python
# builder.py — called once during offline build
@dataclass(frozen=True)
class FacilityStats:
    per_facility: dict[int, dict]    # {fid: {mean, median, iqr, p25, p75, n}}
    per_currency: dict[str, dict]    # {cur: {mean, median, iqr}}
    global_stats: dict               # {mean, median, iqr}
    metadata: dict                   # {built_at, train_rows, feature_version}

class FacilityStatsBuilder:
    MIN_FACILITY_N = 30  # below this, fall back to currency/global

    def build(self, train_df: pd.DataFrame) -> FacilityStats:
        grouped = train_df.groupby("facility_id")["amount_usd"]
        per_facility = {}
        for fid, grp in grouped:
            if len(grp) >= self.MIN_FACILITY_N:
                q1, q3 = grp.quantile(0.25), grp.quantile(0.75)
                per_facility[int(fid)] = {
                    "mean": float(grp.mean()),
                    "median": float(grp.median()),
                    "iqr": float(q3 - q1) or 1.0,
                    "n": int(len(grp)),
                }
        # ... currency and global stats similarly
        return FacilityStats(per_facility=per_facility, ...)

# artifact_loader.py — called at API startup
@dataclass(frozen=True)
class Artifacts:
    model: Any
    scaler: Any
    feature_list: list[str]
    thresholds: dict
    facility_stats: FacilityStats   # NEW
    metadata: dict
```

**Trade-offs:**
- Pro: O(1) lookup at inference; exact parity with training; no runtime CH dependency for stats.
- Con: Stats are frozen to the training distribution. Stale if facility mix changes significantly (re-train required).
- Pro: Versioning is automatic — the stats artifact carries a `feature_version` and `built_at` field, and `artifact_loader` validates it matches `model_metadata`.

---

### Pattern 2: Fallback Chain for Segmented Thresholds

**What:** `SegmentedThresholdClassifier.classify(score, facility_id, currency)` resolves the threshold by trying the most specific segment first, falling back to the next if insufficient calibration data exists.

**Why:** Facilities with fewer than ~100 val-set observations produce unreliable per-segment percentile estimates. The fallback chain avoids a proliferation of noisy per-facility thresholds that overfit to small samples.

**Fallback resolution order:**
```
1. facility_thresholds[facility_id]    → if n_val ≥ MIN_CALIBRATION_N (e.g., 100)
2. currency_thresholds[currency]       → if n_val ≥ MIN_CALIBRATION_N
3. global_threshold                    → always available (current v2 threshold)
```

**Example:**
```python
class SegmentedThresholdClassifier:
    MIN_N = 100

    def classify(self, score: float, facility_id: int, currency: str) -> tuple:
        threshold = (
            self._facility_thresholds.get(facility_id)
            or self._currency_thresholds.get(currency)
            or self._global_threshold
        )
        percentile_lut = (
            self._facility_percentiles.get(facility_id)
            or self._currency_percentiles.get(currency)
            or self._global_percentiles
        )
        is_anomaly = score > threshold
        percentile = self._lookup_percentile(score, percentile_lut)
        risk_level = self._assign_risk_level(percentile)
        return is_anomaly, risk_level, percentile
```

**Trade-offs:**
- Pro: High-volume facilities (courts with thousands of daily transactions) get calibrated thresholds tuned to their actual score distribution, reducing false positives from currency/facility size confounders.
- Con: Requires `facility_id` and `currency` to flow from the scorer into the classifier — already present in `ScoringResult` but classifier needs these two extra args.
- Risk: Threshold proliferation — monitor per-segment anomaly_rate in `anomaly_scores` to detect calibration drift.

---

### Pattern 3: Feature Contract (frame-v1)

**What:** `FrameV1FeatureCalculator` replaces group-absolute features with group-relative equivalents, using the frozen `FacilityStats` artifact. The contract is declared as a versioned named list (`FRAME_V1_FEATURE_NAMES`), analogous to the existing `FEATURE_NAMES` / `FEATURE_NAMES_30`.

**Why it prevents skew:** The current `SingleFeatureCalculator` accesses `fe._groups[4]._facility_avg` (wrong attribute name — see CONCERNS #8) and falls back to `{}`, silently substituting the global mean everywhere. `FrameV1FeatureCalculator` injects the stats artifact explicitly, eliminating the fragile `getattr` chain.

**New features in frame-v1 contract (from `exp_frame_feature_small.py` evidence):**
- `amount_facility_z`: `(amount_usd - facility_median) / facility_iqr` — robust z-score relative to facility
- `user_amount_24h_facility_ratio`: `user_amount_24h / facility_mean` — velocity normalized to facility scale
- `user_debit_amount_30d_facility_ratio`: `user_debit_amount_30d / facility_mean`
- `day_of_week_sin` / `day_of_week_cos`: cyclic encoding replaces `day_of_week` integer

**Features removed from frame-v1 (absolute scale removed per experiment findings):**
- `amount` (raw USD)
- `log_amount`
- `amount_usd_ratio` (global ratio — replaced by facility-relative)
- `facility_avg_amount` (now used only as denominator, not a feature)
- `user_amount_24h` (replaced by ratio)
- `user_debit_amount_30d` (replaced by ratio)

**Parity rule:** Any computation in `FrameV1FeatureCalculator.calculate(payment, context)` must be reproducible as a pandas vectorized expression during offline evaluation. Add a `calculate_from_row(row, facility_stats)` method that takes a parquet row and returns the same vector — this is the parity test surface.

---

### Pattern 4: Shadow Dual-Run for Batch↔Real-Time Parity Validation

**What:** A `ShadowDualRunner` scores the same payload through two code paths simultaneously (old `EnrichedFeatureCalculator` + global threshold vs. new `FrameV1FeatureCalculator` + segmented threshold), logs both outputs to `anomaly_scores` with `scoring_mode='shadow_old'` and `scoring_mode='shadow_new'`, and records score deltas. No separate branch or feature flag required: use the existing `scoring_mode` column in `anomaly_scores`.

**Build order dependency:** Shadow dual-run requires both paths to be fully implemented. It should be the last piece added before retiring the old path.

**Example:**
```python
class ShadowDualRunner:
    def run(self, payment: dict, context: UserContext) -> tuple[ScoringResult, ScoringResult]:
        old_result = self._old_scorer.score(payment, context)
        new_result = self._new_scorer.score(payment, context)
        delta = abs(old_result.score - new_result.score)
        logger.info(f"shadow delta={delta:.4f} fid={payment['facility_id']}")
        return old_result, new_result
```

**Parity acceptance criterion (suggested):** Spearman correlation of old vs. new scores across a batch ≥ 0.90; alert_rate difference ≤ 2pp.

---

## Data Flow

### Stats Artifact: Build → Bundle → Load

```
OFFLINE (run once per model version)
────────────────────────────────────
train_features.parquet
        │
        ▼
FacilityStatsBuilder.build(train_df)
        │
        ▼
facility_stats_v1.joblib   ─── written to output/models/
                                alongside isolation_forest_final.joblib


API STARTUP (scorer/main.py lifespan)
──────────────────────────────────────
load_artifacts(model_dir)
    reads: model_metadata.json  →  determines artifact filenames
    loads: isolation_forest_final.joblib
    loads: scaler_final.joblib
    loads: final_feature_list.json
    loads: thresholds_segmented_v1.json   (NEW)
    loads: facility_stats_v1.joblib       (NEW)
    validates: metadata.feature_version == facility_stats.metadata.feature_version
    returns: Artifacts(model, scaler, feature_list, thresholds, facility_stats, metadata)

SingleTransactionScorer.__init__(artifacts)
    self._feature_calc = FrameV1FeatureCalculator(
        feature_engineer_path=...,
        facility_stats=artifacts.facility_stats,   # injected, no CH
        feature_list=artifacts.feature_list,
    )
    self._classifier = SegmentedThresholdClassifier(config=artifacts.thresholds)


REAL-TIME REQUEST (per payment)
────────────────────────────────
POST /score  ←  Rails payload (factual fields only, no stats)
    │
    ▼
UserContextProvider.get_context(uid, fid, ts, payment)
    → 5-6 ClickHouse queries (rolling aggregates only)
    → returns UserContext
    │
    ▼
FrameV1FeatureCalculator.calculate(payment, context)
    → facility_stats.lookup(fid)      # in-memory dict, O(1)
    → no ClickHouse call for stats    # ← parity guaranteed
    → returns feature vector
    │
    ▼
scorer.score_features(features)
    → scaler.transform(X) → model.decision_function(X_scaled)
    │
    ▼
SegmentedThresholdClassifier.classify(score, fid, currency)
    → fallback chain: facility → currency → global
    → returns (is_anomaly, risk_level, percentile)
    │
    ▼
ScoreResponse → Rails


BATCH REQUEST (per cursor window)
───────────────────────────────────
BatchScorer.score_batch(cursor)
    → fetch payments (CH READ)
    → BatchContextProvider.get_batch_context (6 queries total, not 6×N)
    → for each payment:
        FrameV1FeatureCalculator.calculate(payment, context)  # SAME code, SAME stats
        scorer.score_features()
        SegmentedThresholdClassifier.classify(score, fid, currency)  # SAME instance
    → INSERT anomaly_scores (CH WRITE local)
```

The key parity guarantee is that `FrameV1FeatureCalculator` and `SegmentedThresholdClassifier` are shared instances — the exact same Python objects with the same in-memory stats are used for both paths.

---

## Build Order and Dependencies

The milestone has a strict dependency graph. Phases that appear parallel are safe to implement in parallel.

```
Step 1: Fix existing bugs (unblocks everything)
──────────────────────────────────────────────────
Fix CONCERNS #8 (wrong getattr attribute names in SingleFeatureCalculator):
  - _facility_avg → _facility_avg_amount
  - _staff_stats  → _role_currency_stats
This is a prerequisite for FrameV1FeatureCalculator, which inherits from the same
base. Without fixing #8, the frame-v1 baseline comparison is measuring a broken scorer.

Step 2: FacilityStatsBuilder + FacilityStats schema  [depends on: Step 1]
──────────────────────────────────────────────────────
New module: src/fraud_detector/stats/
Tests: verify per-facility stats match what FeatureEngineer._groups[4] already learns.
Offline script: scripts/build_facility_stats.py
Output: output/models/facility_stats_v1.joblib

Step 3: FrameV1FeatureCalculator  [depends on: Step 2]
──────────────────────────────────────────────────────
New module: src/fraud_detector/scoring/features_frame_v1.py
Must implement calculate() AND calculate_from_row() (parity test surface).
Tests: verify calculate(payment, context) == calculate_from_row(enriched_parquet_row)
       for a sample of 100+ rows from val_features_enriched.parquet.

Step 4: Artifact loader expansion  [depends on: Step 2]
──────────────────────────────────────────────────────
Modify scorer/artifact_loader.py: add facility_stats field to Artifacts.
Add version-match validation: facility_stats.metadata.feature_version
must match model_metadata.feature_version.
Backward-compat: if facility_stats_v1.joblib absent, load with facility_stats=None
and fall back to legacy EnrichedFeatureCalculator path.

Step 5: Segmented threshold calibration  [depends on: Step 2, Step 3]
──────────────────────────────────────────────────────
New module: src/fraud_detector/calibration/segmented.py
Offline script: scripts/calibrate_segmented_thresholds.py
  - loads val_features_enriched.parquet
  - scores with FrameV1FeatureCalculator (requires facility_stats from Step 2)
  - fits per-facility / per-currency / global thresholds at p95
  - applies MIN_N=100 guard per segment
  - writes output/models/thresholds_segmented_v1.json
Output: thresholds_segmented_v1.json with three top-level keys:
  {"global": {...}, "by_facility": {fid: {...}}, "by_currency": {cur: {...}}}

Step 6: SegmentedThresholdClassifier  [depends on: Step 5]
──────────────────────────────────────────────────────
Extend src/fraud_detector/scoring/classifier.py.
classify() signature gains facility_id and currency params.
Tests: verify fallback chain — unknown facility falls to global;
       low-n facility falls to currency then global.

Step 7: SingleTransactionScorer wiring  [depends on: Step 3, Step 4, Step 6]
──────────────────────────────────────────────────────
Modify scorer.py: when artifacts.facility_stats is not None, instantiate
FrameV1FeatureCalculator instead of EnrichedFeatureCalculator.
Pass facility_id and currency through to SegmentedThresholdClassifier.classify().
Update BatchScorer._score_all() to pass fid + currency to classify().

Step 8: Shadow dual-run  [depends on: Step 7]
──────────────────────────────────────────────────────
New scorer/shadow/dual_runner.py
Route flag or env var SCORING_MODE=shadow enables both paths on each request.
Both results stored in anomaly_scores with distinct scoring_mode values.
Parity report: Spearman correlation + alert_rate delta across a batch window.

Step 9: Retire old path  [depends on: shadow validation in Step 8]
──────────────────────────────────────────────────────
Once shadow shows Spearman ≥ 0.90 and alert_rate delta ≤ 2pp, set new path
as default (SCORING_MODE=active). Remove EnrichedFeatureCalculator from
SingleTransactionScorer dispatch logic (keep EnrichedFeatureCalculator module
for offline eval compatibility — do not delete).
```

---

## Anti-Patterns

### Anti-Pattern 1: Per-Request Stats Lookup in ClickHouse

**What people do:** Query `SELECT avg(amount) FROM payments WHERE facility_id = ?` for each real-time score request.

**Why it's wrong:** Violates the 0.2s budget (150-300ms overhead per facility stat query), and — critically — returns current stats, not training-time stats. A facility that grew 5x since the model was trained would produce different normalization denominators, introducing dynamic skew.

**Do this instead:** Build stats from `train_df` at model-build time. Freeze them in `facility_stats_v1.joblib`. Load at startup. All paths share the frozen dict.

---

### Anti-Pattern 2: Separate Stats Computation in Training vs. Serving

**What people do:** `FeatureEngineer.fit()` computes facility stats one way; `SingleFeatureCalculator` recomputes them independently via `getattr(fe._groups[4], ...)`. The two computations diverge silently (as documented in CONCERNS #8).

**Why it's wrong:** Training uses `_facility_avg_amount` with n_samples from the full training set; serving falls back to `{}` (returning `{}.get(fid, global_avg)` = always global_avg). The model learned facility-relative features; the scorer applies global-relative features. This is the known silent skew reported in the milestone context.

**Do this instead:** `FacilityStatsBuilder` computes from the same train DataFrame that `FeatureEngineer` was fit on. The stats are injected into `FrameV1FeatureCalculator` at construction. There is exactly one code path for the computation.

---

### Anti-Pattern 3: Threshold Calibration on the Test Set

**What people do:** `run_fase7_evaluation.py` uses `np.percentile(test_scores, 95)` as the operational threshold (documented in CONCERNS #6).

**Why it's wrong:** The threshold is optimized for the exact distribution of the test set, introducing circularity. If the test AUC is also reported against this threshold, both numbers are inflated by the same sample.

**Do this instead:** Calibrate thresholds exclusively on the val set (already done correctly in `calibrate_threshold_v2.py`). Apply the same discipline to segmented thresholds.

---

### Anti-Pattern 4: `getattr` with a Default as a Safety Net for Schema Drift

**What people do:** `getattr(fe._groups[4], "_facility_avg", {})` — using `{}` as default to avoid `AttributeError`.

**Why it's wrong:** The `{}` default silently masks the wrong attribute name. The correct attribute (`_facility_avg_amount`) exists and has data; the wrong attribute (`_facility_avg`) does not. The `getattr` default hides a bug that degrades production scoring quality with no log warning.

**Do this instead:** Access attributes directly — `fe._groups[4]._facility_avg_amount`. Let `AttributeError` surface at startup, not silently at scoring time. Add an explicit assertion in `load_artifacts()` that validates the feature engineer's learned attributes before serving begins.

---

### Anti-Pattern 5: Feature Contract Drift Between Frame Versions

**What people do:** Add new features to `FrameV1FeatureCalculator` without updating the corresponding training-time feature set, or train the model with features A-Z but deploy a calculator that only computes A-W.

**Why it's wrong:** Silent dimension mismatch (caught at startup by artifact validation) or silent wrong-column mismatch (not caught if feature count happens to match).

**Do this instead:** Declare `FRAME_V1_FEATURE_NAMES` as the single source of truth for the feature contract. Both the offline training script and `FrameV1FeatureCalculator.calculate()` index into the same list. Add an import-time assertion: `assert len(FRAME_V1_FEATURE_NAMES) == EXPECTED_N`.

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| ClickHouse (prod READ) | `UserContextProvider` — 5-6 queries per real-time request for rolling aggregates | Rolling aggregates (velocity, behavior, credit, diversity, user info, role) must remain live per-request; cannot be frozen |
| ClickHouse (local WRITE) | `BatchScorer._insert_chunks()` — chunked INSERT with dedup tokens | Guarded by `assert_write_target_is_safe`; never points to READ host |
| Rails platform | HTTP POST payloads (factual fields only) — no stats, no pre-computed features | Rails contract must not change: `ScoreRequest` schema is stable |
| MLflow (optional) | Artifact registry can track `facility_stats_v1.joblib` as a child artifact of the run that produced the model | LOW confidence — not currently used for artifact tracking |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `FacilityStatsBuilder` ↔ `FrameV1FeatureCalculator` | `FacilityStats` dataclass (frozen) | Never share a mutable dict — frozen dataclass prevents accidental mutation |
| `artifact_loader` ↔ `scorer/main.py` | `Artifacts` dataclass — extended with `facility_stats` | Backward-compatible: `facility_stats=None` triggers legacy path |
| `FrameV1FeatureCalculator` ↔ `SegmentedThresholdClassifier` | Shared via `SingleTransactionScorer` | Neither calls the other directly; scorer mediates |
| `BatchScorer` ↔ `SingleTransactionScorer` | Calls `scorer._feature_calc.calculate()` and `scorer.score_features()` directly | BatchScorer must forward `facility_id` and `currency` to classifier |
| `ShadowDualRunner` ↔ both scorer paths | Old scorer instance + new scorer instance, both sharing the same CH client | Use separate instances constructed from the same `Artifacts` |

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Current (~100 real-time req/day, batch nightly) | Single FastAPI instance, in-memory dict, no cache layer needed |
| 10K real-time req/day | Facility stats dict fits in ~10MB RAM for thousands of facilities; no change |
| 100K+ real-time req/day | `UserContextProvider` becomes the bottleneck (5-6 CH queries/request at ~150ms each); consider async CH client or a Redis pre-populated with rolling aggregates; stats artifact stays the same |

### Scaling Priorities

1. **First bottleneck:** `UserContextProvider` CH query latency — already ~150ms for 5-6 queries. Stats artifact load eliminates any facility-stats query. The only remaining CH dependency at real-time is rolling aggregates, which cannot be frozen.
2. **Second bottleneck:** `SegmentedThresholdClassifier` — pure in-memory dict lookup, negligible. No scaling concern.

---

## Sources

- Codebase audit: `src/fraud_detector/scoring/features.py` (CONCERNS #8 — wrong `getattr` attribute names), `scorer/artifact_loader.py`, `scorer/main.py` (lifespan pattern), `scorer/batch/scorer.py` (batch flow), `.planning/codebase/CONCERNS.md` (train/serve skew inventory), `.planning/codebase/ARCHITECTURE.md` (existing layer map) — HIGH confidence (direct code reading)
- Experiment evidence: `scripts/exp_reference_frames.py` (V0 vs V1 frame comparison, facility stats computed from `fit_stats()`), `scripts/exp_frame_feature_small.py` (`add_frame_features()` showing `amount_facility_z`, ratio features, cyclic DOW) — HIGH confidence (direct code reading)
- Threshold calibration: `scripts/calibrate_threshold_v2.py` (uses val set, correct), `scripts/run_fase7_evaluation.py` (uses test set, incorrect per CONCERNS #6) — HIGH confidence
- MLOps patterns — artifact versioning, lifespan loading: FastAPI official docs + community practice confirmed by multiple 2025 sources — MEDIUM confidence
- Train-serve skew patterns: Nubank engineering blog, dev.to synapcores, systemoverflow.com — MEDIUM confidence (web search, consistent with codebase evidence)
- Per-segment threshold calibration: academic literature (ACM 2025, ScienceDirect 2025) confirms viability of segment-specific thresholds with fallback — LOW confidence for exact parameters (min_n, segment definitions need domain validation)

---

*Architecture research for: reference-frame-normalized payment anomaly scoring milestone*
*Researched: 2026-07-06*
