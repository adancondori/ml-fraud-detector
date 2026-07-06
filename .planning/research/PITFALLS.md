# Pitfalls Research

**Domain:** Reference-frame-normalized payment anomaly detection — brownfield addition of per-facility stats, local-time DST conversion, segmented threshold calibration, shadow dual-run, and HITL label capture to an existing Isolation Forest scorer.
**Researched:** 2026-07-06
**Confidence:** HIGH — all pitfalls grounded in this codebase's CONCERNS.md, verified source code, and the frames-analysis experiment in `docs/analisis-marcos-referencia.md`. General ML pitfalls cross-checked against published literature.

---

## Critical Pitfalls

### Pitfall 1: Silent train/serve skew from wrong attribute names in per-facility lookup

**What goes wrong:**
`SingleFeatureCalculator` loads facility means via `getattr(fe._groups[4], "_facility_avg", {})` and staff stats via `getattr(fe._groups[6], "_staff_stats", {})`. The actual attribute names in `FeatureEngineer` are `_facility_avg_amount` and `_role_currency_stats`. Both `getattr` calls silently return `{}`, so every real-time score uses the global mean for `facility_avg_amount` and global stats for `staff_amount_zscore`. The model was trained with per-facility values, so the feature distribution the model sees at serve time differs from training — but no exception fires and the scorer returns a number.

This exact bug is active in production today (CONCERNS.md §8). Frame normalization adds *more* per-facility attributes (IQR, median, currency_group, time_zone). If the same loading pattern is reused, those will silently fall back to global values too.

**Why it happens:**
`getattr(obj, name, default)` swallows missing attributes. During refactoring, the attribute was renamed in `FeatureEngineer` but the name string in `SingleFeatureCalculator` was not updated. The bug is invisible at startup and only detectable by comparing score distributions.

**How to avoid:**
- In Fase 1, fix the existing names before adding new ones (`_facility_avg` → `_facility_avg_amount`, `_staff_stats` → `_role_currency_stats`).
- Replace all `getattr(obj, name, {})` patterns used for trained artifacts with direct attribute access (`obj._facility_avg_amount`) inside the artifact-loading path. Reserve `getattr` with a default only when the field genuinely may be absent (e.g., version-upgrade path where old artifacts lack new fields).
- Add a post-load validation step in `SingleFeatureCalculator.__init__` that asserts `len(self._facility_avgs) > 0` and `len(self._staff_stats) > 0` — an empty dict is always a signal of a wrong name.
- Add a parity integration test: score the same transaction via batch pipeline and via `SingleTransactionScorer`; assert that `facility_avg_amount` is identical in both paths.

**Warning signs:**
- `facility_avg_amount` in scorer output equals `self._global_avg_amount` for every transaction regardless of facility.
- `staff_amount_zscore` distributions are implausibly flat or narrow in real-time logs compared to batch evaluation.
- `SingleFeatureCalculator loaded: facilities=0` in startup logs (already logged on line 29 of `scoring/features.py`).

**Phase to address:** Fase 0 (baseline freeze) — fix before Fase 1 computes new per-facility stats, so the artifact loading pattern is correct before being extended.

---

### Pitfall 2: DST gap/fold produces silent wrong local-time for temporal features

**What goes wrong:**
`created_at` is stored as UTC `DateTime` (no timezone). Converting to local time requires the facility's IANA zone (e.g., `America/New_York`) and DST-aware arithmetic. A naive offset conversion (e.g., UTC−5 hardcoded) produces the right result in winter but is off by one hour during DST — affecting March/November transitions. Worse, during the DST "fall back" hour (e.g., 01:00–02:00 local), the same UTC instant maps to two possible local times (fold), which makes `is_off_hours` indeterminate if not handled.

Beyond fold/gap: `facilities.time_zone` in the Rails platform stores Rails timezone names (e.g., `"Eastern Time (US & Canada)"`) which are ActiveSupport names, not IANA names. Python's `pytz` and `zoneinfo` use IANA names (`"America/New_York"`). Without explicit mapping, `pytz.timezone("Eastern Time (US & Canada)")` raises `UnknownTimeZoneError`.

The experiment in `docs/analisis-marcos-referencia.md` measured that off-hours inflates from 4.6% (correct local) to 18.5% for Eastern facilities when using UTC — a 4× overcount. At full scale the UTC artifact is 26.4% vs 4.2% local (a 6× distortion).

**Why it happens:**
`pd.Timestamp(payment["created_at"])` constructs a tz-naive timestamp. `ts.hour` returns UTC hour. The facility time zone exists in the Rails DB but is never fetched or applied in the current scorer.

**How to avoid:**
- In Fase 1, build a `facility_tz.parquet` artifact (already partially done at `output/revision/facility_tz.parquet`) that stores the IANA name, not the Rails name. Maintain an explicit Rails→IANA mapping table. Apply conversion with `zoneinfo.ZoneInfo` (Python 3.9+ stdlib) or `pytz`, using `ts.astimezone(ZoneInfo(iana_name))` so DST is handled automatically.
- For fold ambiguity, choose the `fold=0` interpretation (first occurrence) consistently in both batch and real-time and document it. The difference is 1 hour in 1 year, which is acceptable.
- Verify the IANA zone for the 577 facilities with empty/null `time_zone` (CONCERNS.md §4) — assign a fallback of `"UTC"` with `fallback_level=global` flag so reviewers can filter those transactions.
- In Fase 2, ensure `SingleFeatureCalculator.calculate()` receives the pre-resolved IANA zone from the artifact, not the raw Rails string.

**Warning signs:**
- `is_off_hours` rate in production real-time scoring is materially higher than 5% for a facility operating in UTC−5 or later.
- `pytz.exceptions.UnknownTimeZoneError` in scorer logs during Fase 2 integration testing.
- Off-hours rate in shadow-mode logs (Fase 4) diverges significantly between old and new scorer for the same transactions.

**Phase to address:** Fase 1 (offline frame artifact) — the IANA mapping and DST conversion must be correct in the artifact before Fase 2 replicates it in the scorer.

---

### Pitfall 3: Low-volume segments produce unstable percentile thresholds

**What goes wrong:**
Segmented calibration assigns a threshold percentile (e.g., p95) per segment (facility, currency_group, or global). For segments with fewer than ~200 scored transactions, the p95 is computed from fewer than 10 observations and is therefore highly sensitive to individual outliers. A single unusual transaction can shift the threshold by 10–20 percentile points, causing the alert rate to swing between 0% and 30% in consecutive weeks. This is especially acute for new facilities (< 30 days old) and minor currencies with few transactions.

Additionally, if calibration is done on the validation set and the segment is sparse in validation, the threshold will overfit to those few samples. In production, the sparse segment may have a different score distribution simply due to sampling noise.

**Why it happens:**
Teams compute one threshold per segment without checking n. They trust the percentile function to be stable without knowing the sampling distribution of order statistics at low n.

**How to avoid:**
- In Fase 2, define a minimum n threshold (e.g., n ≥ 200 scored transactions in the calibration window) for a per-facility threshold to be used. Below this, fall back to the currency-group threshold; below the group minimum, fall back to global.
- Version the fallback hierarchy in the threshold artifact as `{facility_id: {threshold, n, fallback_level}}` so the scorer knows which level it used and the shadow monitor can stratify by `fallback_level`.
- Use bootstrap confidence intervals for thresholds in sparse segments to surface instability rather than hiding it.
- Apply a rolling calibration window (e.g., last 60 days of scored transactions) rather than a static calibration set to reduce the impact of concept drift in low-volume segments.
- Enforce that calibration is done on the validation set, never the test set (see Pitfall 6).

**Warning signs:**
- Alert rate for a specific facility varies >5× week over week without a known event.
- `n` in the threshold artifact is below 200 for segments that are being given per-segment thresholds.
- Any threshold artifact entry where `fallback_level = "facility"` but `n < 100`.

**Phase to address:** Fase 2 (scorer API) — the fallback hierarchy and minimum-n logic must be built into the calibration step before the threshold artifact is first produced.

---

### Pitfall 4: Proxy circularity — AUC against pure_fraud measures self-reference, not detection quality

**What goes wrong:**
`pure_fraud` is defined as `same_amount_count_1h >= 3 OR (user_account_age_days < 14 AND user_txn_count_1h >= 3) OR (is_third_party_payment == 1 AND user_txn_count_1h >= 2)`. These four variables (`same_amount_count_1h`, `user_account_age_days`, `user_txn_count_1h`, `is_third_party_payment`) are features of IF-40. The model learns to isolate the rule boundaries that define the label it is then evaluated against. The reported AUC of 0.841 is partial autovalidation — the model recognizes its own input patterns, not fraud.

This pitfall extends to frame normalization: if any of the frame-normalized features are correlated with the proxy definition variables (e.g., a facility-relative velocity feature correlates with `user_txn_count_1h`), AUC against `pure_fraud` will again partially reward constructing the proxy, not detecting anomalies.

**Why it happens:**
`pure_fraud` was designed as a more precise proxy precisely by encoding known fraud patterns — but those patterns were operationalized using the same feature vocabulary as the model. The circularity is structural, not accidental.

**How to avoid:**
- In Fase 0, establish that the primary evaluation criterion for all frame-normalization work is bias reduction (top-5% amount ratio, off-hours rate, segment-level alert rate) and enrichment factor, not AUC against `pure_fraud`.
- Treat AUC vs `pure_fraud` as a diagnostic metric only — never present it as a validation of detection quality and never surface it to reviewers in the shadow dashboard (Fase 4) or HITL interface (Fase 5).
- In Fase 1, evaluate the new `FS-frame-operational-v1` model against Tipo A (structurally independent proxy: `status IN ('totally_refunded','refunded_to_credit')`) with the `FS-disjoint` variant that removes the proxy variables from features.
- Document in `SUMMARY.md` and any thesis-adjacent artifacts that the gate metric for Fase 1 approval is bias reduction, not AUC lift.

**Warning signs:**
- Anyone citing AUC 0.841 (or any AUC vs `pure_fraud`) as evidence the model works.
- A new feature that happens to correlate with `user_txn_count_1h` or `same_amount_count_1h` showing large AUC lift vs `pure_fraud` — investigate circularity before celebrating.
- Evaluation scripts being run with `proxy = "pure_fraud"` without `FS-disjoint` variant active.

**Phase to address:** Fase 0 (baseline freeze) — lock the evaluation criterion and success gate before any feature work begins, so there is no temptation to optimize toward the circular metric.

---

### Pitfall 5: Facility stats artifact drift — training stats diverge from scorer's runtime population

**What goes wrong:**
Facility stats (mean, median, IQR) are computed once from the training universe and saved in an artifact. Over time, a facility's transaction volume, typical amount, or user mix changes. The scorer continues using the stale stats. This is a form of concept drift specific to the per-facility normalization: the denominator of `amount_facility_ratio` drifts, causing the score to shift even for constant behavior.

A related failure: if the artifact is computed from a different universe than the scorer's live query universe (e.g., including `payment_method = 'free'` in stats but excluding it in scoring), the facility mean encodes a population the scorer never sees.

**Why it happens:**
Static artifacts are easy to build once and forget. Per-facility stats have a smaller n than global stats, so they drift faster. The universe mismatch is caused by not enforcing the same filter clause in both artifact computation and context queries.

**How to avoid:**
- In Fase 1, compute the facility stats artifact from the exact universe the scorer uses: `_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL` — the same filter as all `UserContextProvider` queries. Document this constraint in the artifact metadata.
- Include `n_transactions`, `computed_at`, and `facility_universe_filter` in the artifact metadata so drift can be detected.
- In Fase 4 (shadow mode), monitor the distribution of `amount_facility_ratio` over time per segment. If the median of this ratio drifts from 1.0, the stats artifact is stale.
- Plan a scheduled re-computation cadence (e.g., monthly or triggered when a facility's transaction count grows >50% since last computation).

**Warning signs:**
- Mean `amount_facility_ratio` for a specific facility drifts away from ~1.0 in shadow logs over a 30-day window.
- A facility that opened or significantly changed operations has stats computed from < 90 days of history (high variance).
- Artifact's `computed_at` is more than 90 days old in production.

**Phase to address:** Fase 1 (offline frame artifact) — the artifact design must include metadata and the universe constraint must be explicit before Fase 2 uses the artifact.

---

### Pitfall 6: Threshold calibrated on the test set contaminates both evaluation and production

**What goes wrong:**
`scripts/run_fase7_evaluation.py` computes `threshold = np.percentile(scores_if, 95)` from the test set scores, then reports HE1–HE4 metrics on the same test set. The same test set is used for threshold selection and for reporting — the threshold is optimized for that specific distribution. The JSON explicitly records `"threshold_source": "percentile_95_test_set"`. Additionally, the operational scorer uses this threshold, meaning production alerts are tuned to a held-out set that should be untouched.

**Why it happens:**
Practitioners conflate "set aside for final evaluation" with "available for any post-training decision." Threshold selection feels like a deployment decision, not a modeling decision, so it does not feel like contamination.

**How to avoid:**
- In Fase 0 (or as a prerequisite fix), create a separate calibration step that uses the validation set only. Split `run_fase7_evaluation.py` into `calibrate_threshold.py` (val set) and `evaluate_model.py` (test set, read-only).
- For segmented calibration in Fase 2, use the validation set for per-segment threshold fitting. The test set is used only once, for final bias/enrichment/AUC reporting.
- Store the threshold source in the artifact: `{"source": "validation_set_p95", "n": 1130118}` — the scorer can expose this on the `/health` endpoint.

**Warning signs:**
- Any threshold artifact with `"threshold_source": "percentile_95_test_set"` in production.
- Evaluation metrics (AUC, enrichment) reported on the same set used to set the threshold.
- Validation-set and test-set alert rates differing by more than 2 percentage points for the same threshold (sign that the threshold was tuned to test).

**Phase to address:** Fase 0 (baseline freeze) — fix before frame work begins to ensure the baseline is measured correctly. The corrected process carries forward into Fase 2's segmented calibration.

---

### Pitfall 7: `capture_delay_seconds = 0` in real-time — constant feature creates train/serve skew

**What goes wrong:**
In the `after_commit on:create` Rails hook, the payment has just been created; `captured_at` is null because capture happens asynchronously. `features_enriched.py` returns `0.0` when `captured_at` is `None`. In batch evaluation on historical parquets, `captured_at` is populated (payments were already captured), so `capture_delay_seconds` carries real signal (seconds to minutes). IF-40 learned to discriminate using this feature, but in production it always receives 0.

Additionally, the NaT check `if captured is pd.NaT` is unreliable: `pd.Timestamp(None)` returns `NaT` but the identity comparison `is pd.NaT` can be False depending on construction path. The correct check is `pd.isnull(captured)`.

**Why it happens:**
The batch pipeline processes historical data where the full payment lifecycle is complete. The real-time path processes payments at the instant of creation. These are structurally different populations, but the same feature vector is used.

**How to avoid:**
- In Fase 1 (when redefining `FS-frame-operational-v1`), exclude `capture_delay_seconds` from the feature set entirely. The IF-40 variant `FS-disjoint-30` already does this.
- Fix the NaT check to `pd.isnull(captured)` as part of the Fase 0 bugfix pass.
- If the feature is retained for a different reason, replace it with a binary flag `is_captured` (0 at creation, 1 after capture) that has stable, honest semantics in both paths.
- Add a parity test: run the scorer against a transaction with `captured_at=None` and assert `capture_delay_seconds = 0`, then run it with a historical transaction and assert the value is non-zero — fail if both return 0 when historical data is expected.

**Warning signs:**
- Score distribution for real-time `POST /api/v1/score` is systematically lower (less anomalous) than for the same transaction scored via batch endpoint.
- `capture_delay_seconds` shows as constant 0 in `factors` for all real-time alerts.
- `pd.isnull` check removed or absent from `features_enriched.py` after any refactor.

**Phase to address:** Fase 0/Fase 1 — exclude the feature from the new feature set before retraining, and fix the NaT bug before Fase 2 runs any live scoring.

---

### Pitfall 8: Currency "EMPTY" treated as USD — silent amount distortion

**What goes wrong:**
`fallback_rate()` calls `_FALLBACK_RATES.get(currency, 1.0)`. If `currency` is `"EMPTY"`, `""`, or any unrecognized string, it returns 1.0 (USD identity). A transaction in a high-exchange-rate currency that was recorded with `currency = "EMPTY"` will have its amount treated as USD, making it appear at USD face value. For currencies like PKR (rate 0.00356) this inflates the apparent amount by ~280×, directly contaminating `amount_facility_ratio` and `staff_amount_zscore`.

The loader sanitizes `""` → `"USD"` but not `"EMPTY"`. This affects 577 facilities and 117K records (documented in `docs/analisis-marcos-referencia.md`).

**Why it happens:**
`"EMPTY"` is a sentinel value from the ETL pipeline. The normalization utility was written expecting `None` or empty string, not a non-empty sentinel. The silent default-to-1.0 swallows the problem without logging.

**How to avoid:**
- In Fase 0 (bugfix pass), add `"EMPTY"` to the sanitization step in `_postprocess_extraction()` alongside `""`: `currency.replace({"EMPTY": "USD", "": "USD"})`.
- Add logging when a non-null, non-empty, non-recognized currency is encountered: `logger.warning(f"Unrecognized currency '{currency}' — defaulting to USD")`.
- In the facility stats artifact (Fase 1), tag facilities with a high `"EMPTY"` rate as `currency_group = "unknown"` with `fallback_level = "global"` so their per-facility stats are not trusted for normalization.
- Include a data quality report in the Fase 0 baseline that counts `"EMPTY"` rows per facility so the scope is known before frame normalization begins.

**Warning signs:**
- Facilities in non-USD markets appearing at the top of the anomaly ranking with implausibly high `amount_facility_ratio` values (>10×).
- `exchange_rate_source = "lookup_or_fallback"` but `exchange_rate_applied = 1.0` for a non-USD facility.
- More than 1% of transactions in any month have `currency = "EMPTY"` — indicates an upstream ETL issue that may grow.

**Phase to address:** Fase 0 (baseline freeze / bugfix) — clean data quality before computing facility stats artifacts in Fase 1.

---

### Pitfall 9: Shadow mode comparison invalidated by scorer path divergence

**What goes wrong:**
Shadow mode runs both the old scorer and the new scorer on each transaction and compares results. The comparison is only meaningful if both scorers receive exactly the same input payload and query the same data at the same logical timestamp. Two common failures: (1) the new scorer introduces a new required field (e.g., `facility_time_zone_iana`) that the Rails platform does not send yet — the new scorer silently receives `None` and falls back to UTC, making it appear to perform identically to the old scorer on the DST-corrected feature (hiding the improvement); (2) the two scorers are called at different wall-clock times, so their ClickHouse context queries return different rolling aggregates for velocity features.

**Why it happens:**
Shadow mode is treated as "run both and log results" rather than as a controlled experiment. The payload contract between platform and scorer is extended incrementally, and missing fields default silently.

**How to avoid:**
- In Fase 2, define the `frame-v1` API contract explicitly: list all required fields and all optional fields with their explicit `NO_DEFAULT_SENTINEL` behavior (missing optional field → raise, or → explicit `frame_flag = "missing_tz"`). Never default a new field to a value that makes it indistinguishable from the baseline behavior — make the degraded case observable.
- In Fase 4, the shadow dual-run must serialize both scorer calls to the same timestamp before executing either. Alternatively, pass context as an input parameter (the `context: Optional[UserContext]` parameter already exists in `SingleTransactionScorer.score()`) to ensure both scorers consume the same pre-fetched context.
- Log `frame_flags` in every shadow result so the shadow monitor can distinguish "new scorer ran with full frame data" from "new scorer ran with fallback because tz was missing."
- Gate Fase 4 launch on Fase 3 (platform integration) completing the payload extension first.

**Warning signs:**
- Shadow divergence rate (Jaccard disagreement) is consistently < 1% — suspiciously low; likely means the new scorer is defaulting to the old behavior silently.
- `frame_flags` contains `"missing_tz"` or `"missing_currency_group"` in more than 5% of shadow logs — platform is not sending required fields.
- Shadow logs show identical `is_off_hours` distributions for old and new scorer — DST fix is not active.

**Phase to address:** Fase 2 (API contract) and Fase 4 (shadow run) — the API must enforce observable failures before Fase 4 can produce meaningful comparisons.

---

### Pitfall 10: HITL label capture biased by alert pre-selection — labels don't cover non-alerted transactions

**What goes wrong:**
Human reviewers only see the top-k anomalies surfaced by the model. Their labels (confirmed anomaly / not anomaly) are collected only on transactions the model already scored high. This creates a systematic gap: the label set has zero coverage of transactions the model scores low. Any evaluation using HITL labels will overestimate precision (reviewed items tend to be anomalies) and provide no information about recall (how many real anomalies were scored low and never reviewed). Training or re-calibrating on these labels without correction replicates and amplifies the model's existing blind spots.

An additional risk in Fase 5: if the HITL interface presents the model's explanation (`factors`) to the reviewer, the reviewer's judgment is anchored by the model's own reasoning. The label captures not an independent human judgment but a partially model-influenced one.

**Why it happens:**
Presenting only the top-k is a practical necessity (reviewers cannot review all 6.7M transactions). The gap is accepted implicitly rather than designed around.

**How to avoid:**
- In Fase 5, design the label capture to include a random sample of non-alerted transactions alongside the top-k (e.g., 80% top-k anomalies + 20% random sample from bottom 80% of the score distribution). This enables estimation of false-negative rate.
- Separate the "explanation" view (with SHAP factors) from the "label" action in the HITL interface. Reviewers should label first, then optionally view factors, so anchoring bias is reduced.
- Store alongside each HITL label: `{transaction_id, score_at_label_time, model_version, reviewer_saw_factors: bool, label, label_timestamp}` — all fields needed to detect and correct for selection bias in future analyses.
- In Fase 5's evaluation, use Tipo A (proxy-derived, independent of reviewer) as the control group to validate whether HITL labels are structurally different from proxy labels — do not assume they are ground truth.

**Warning signs:**
- HITL label set has zero transactions with `score < p50` — reviewers are seeing only high-score items.
- Human precision on labeled set is > 60% — suspiciously high, likely due to pre-selection, not model quality.
- Reviewer agreement rate is abnormally high for items where model explanation was shown — anchor effect.
- HITL labels correlate 0.95+ with `pure_fraud` proxy — if so, HITL captured only the circular proxy behavior, not independent human judgment.

**Phase to address:** Fase 5 (HITL) — capture design must include random sampling before the first reviewer sees any transaction.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `getattr(obj, name, {})` for artifact attributes | Survives artifact version mismatches | Silent wrong values in production; no error to alert the team | Never for trained-artifact attributes. Use direct access with an explicit version-check guard. |
| UTC timestamps for temporal features | No timezone infrastructure needed | Off-hours inflated 4–6×; model measures geography not behavior | Never for behavioral features. |
| Global threshold (single percentile, all segments) | One number to calibrate | Over-marks large facilities, under-marks small ones; alert fatigue | Acceptable only as the fallback in the hierarchy, not as the primary threshold. |
| Test-set threshold calibration | No need for a separate calibration split | Threshold is optimized to the evaluation set; AUC and alert rate are overly optimistic | Never. Use validation set for calibration. |
| Evaluating against `pure_fraud` without disjoint variant | Higher AUC numbers for reports | Circular: the model learns its own proxy rules; misleads stakeholders | Never as a primary metric. Diagnostic use only, always labeled as circular. |
| Shadow mode with sequential (not simultaneous) scorer calls | Simpler implementation | Rolling aggregates differ between calls; velocity features cause phantom divergence | Only acceptable if the time between calls is < 50ms and velocity windows are > 1 hour. |
| Static facility stats artifact (no refresh cadence) | Single computation, easy | Stats drift; new facilities get global fallback indefinitely | Acceptable for MVP if artifact age is logged and an alert fires at 90 days. |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Rails `facilities.time_zone` → Python timezone | Use the Rails timezone string directly in `pytz.timezone()` — raises `UnknownTimeZoneError` | Maintain an explicit Rails→IANA mapping table; store only IANA names in the facility stats artifact |
| ClickHouse `FINAL` modifier | Omitting `FINAL` on `SharedReplacingMergeTree` tables returns duplicate rows for replaced/deleted payments | Every query against `payments`, `users`, `facilities_users` must include `FINAL` and `_peerdb_is_deleted = 0` |
| ClickHouse hardcoded database name | `pbp_productionDB_optimized` hardcoded in `context.py` and `batch/scorer.py` prevents testing against staging | Parametrize via `settings.clickhouse_database` interpolated into SQL templates at init time |
| Rails `after_commit on:create` → scorer | `captured_at` is null at creation time; any feature using it returns 0 silently | Either exclude `capture_delay_seconds` from the feature set or add `is_captured: bool` as a separate field |
| Frame-v1 API new fields | New optional fields default silently to baseline behavior — shadow mode cannot detect if tz fix is active | Use explicit sentinel: missing `facility_time_zone_iana` → `frame_flags = ["missing_tz"]`, logged, threshold falls back to global |
| MLflow artifact versioning | Loading `isolation_forest.joblib` directly bypasses version validation in `artifact_loader.py` | Always load through `artifact_loader.py` which validates feature count; direct joblib load only in tests |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `BehavioralFeatures._distinct_facilities_30d()` — O(n²) Python loop | Feature engineering on 3M-row train takes hours | Replace with vectorized `_rolling_shifted_stat` using `nunique`; add a benchmark test | Already slow at 3M rows; will double when frame variant adds more facilities |
| `OperationalDiversityFeatures._category_entropy_30d()` — same O(n²) | 30–60 min for full train set | Same fix as above — rolling apply with `apply(lambda: entropy)` | Already at the limit; adding 40 features will push it over |
| 8 sequential ClickHouse queries per real-time score | Latency 50–200ms in happy path; 1–2s under load | Merge queries where possible; pre-load facility stats from artifact rather than querying per request | At > 50 req/s concurrent scoring |
| Rolling calibration on full transaction history per segment | Recalibration takes hours for 6.7M rows | Use a 60-day rolling window with pre-aggregated score deciles stored in ClickHouse | When facility count grows beyond 3,000 and monthly recalibration is triggered |

---

## "Looks Done But Isn't" Checklist

- [ ] **Per-facility stats artifact**: Often missing universe constraint documentation — verify that the SQL used to compute stats exactly matches `UserContextProvider` filters (`FINAL`, `_peerdb_is_deleted=0`, `payment_method NOT IN ('reversal','free')`).
- [ ] **DST conversion**: Looks done when `ts.hour` returns a local-looking number — verify by checking a known Eastern facility transaction at 01:30 UTC during DST transition; local should be 21:30 EST, not 01:30 UTC.
- [ ] **Segmented threshold calibration**: Looks done when the artifact has per-facility rows — verify that every row includes `n`, `fallback_level`, and that facilities with `n < 200` have `fallback_level != "facility"`.
- [ ] **Shadow dual-run**: Looks done when both scorers return a result — verify that `frame_flags` is non-empty for transactions where tz or currency_group is missing, and that the divergence rate is not suspiciously low (< 1%).
- [ ] **HITL label capture**: Looks done when the UI saves reviewer decisions — verify that the stored schema includes `score_at_label_time`, `model_version`, `reviewer_saw_factors`, and that at least 20% of presented transactions are from below the p50 score.
- [ ] **Train/serve parity**: Looks done when scorer starts without errors — verify via integration test that `facility_avg_amount` in a real-time score matches the value from the batch pipeline for the same transaction.
- [ ] **`capture_delay_seconds` exclusion**: Looks done when removed from `FEATURE_NAMES` — verify that `features_enriched.py` no longer computes it and that the retrained model artifact's feature list does not contain it.
- [ ] **`"EMPTY"` currency sanitization**: Looks done when `currency.py` handles empty string — verify with a test that passes `currency="EMPTY"` and asserts the warning is logged and the rate is not silently 1.0.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Silent `getattr` wrong attribute names (active in prod) | LOW | Fix two attribute name strings in `scoring/features.py` lines 27–28; redeploy scorer; verify via integration test that `facility_avg_amount` changes |
| DST wrong timezone (deployed to prod) | MEDIUM | Rebuild facility stats artifact with IANA zones; retrain model with corrected temporal features; roll out new scorer artifact; shadow validate before switching traffic |
| Threshold calibrated on test set | LOW | Re-run calibration on val set; replace `thresholds.json` artifact; no retraining needed |
| Facility stats artifact stale (> 90 days) | LOW | Re-run stats computation script against train+val window; replace artifact; no retraining needed |
| HITL labels with no non-alerted samples | HIGH | Cannot recover historical labels; must restart label collection with correct sampling design; existing labels can be used for top-k precision only |
| `pure_fraud` AUC cited as primary metric in reports | MEDIUM | Issue correction memo; re-run evaluation with Tipo A + FS-disjoint as primary; update dashboards and any thesis-adjacent documents |
| `"EMPTY"` currency in production data | LOW | One-line fix in `_postprocess_extraction()`; re-extract or patch affected parquets; recompute facility stats for affected 577 facilities |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Silent getattr wrong attribute names | Fase 0 (bugfix) | Integration test: batch score == real-time score for `facility_avg_amount` |
| DST / Rails→IANA timezone conversion | Fase 1 (offline frame) | Unit test: Eastern facility at 01:30 UTC yields `hour_local = 21` not `1` |
| Low-volume segment unstable percentiles | Fase 2 (scorer API) | Artifact audit: every per-facility threshold row has `n >= 200` or `fallback_level != "facility"` |
| Proxy circularity (`pure_fraud` AUC) | Fase 0 (gate definition) | Fase 1 success gate document uses bias metrics, not AUC vs `pure_fraud` |
| Facility stats artifact drift | Fase 1 (design) | Artifact includes `computed_at`; shadow monitor tracks median `amount_facility_ratio` per facility |
| Threshold calibrated on test set | Fase 0 (bugfix) | `thresholds.json` metadata shows `"source": "validation_set"` |
| `capture_delay_seconds` = 0 in real-time | Fase 0/Fase 1 | Feature excluded from `FS-frame-operational-v1` feature list; NaT bug fixed |
| Currency "EMPTY" treated as USD | Fase 0 (bugfix) | Data quality report shows 0 rows with `currency="EMPTY"` after sanitization |
| Shadow mode payload/timestamp divergence | Fase 2 (API contract) | Shadow logs show `frame_flags` field present; divergence rate > 1% |
| HITL selection bias (no non-alerted samples) | Fase 5 (design) | HITL label set contains >= 20% transactions with `score < p50` |

---

## Sources

- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/.planning/codebase/CONCERNS.md` — direct codebase audit (concerns #1–#12)
- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/docs/analisis-marcos-referencia.md` — experimental confirmation of UTC/DST distortion (26.4% → 4.2% off-hours) and amount bias (16.9× → 1.8× top-5% ratio)
- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/.planning/PROJECT.md` — confirmed train/serve skew bugs, active concerns, and project constraints
- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/src/fraud_detector/scoring/features.py` — verified `getattr` attribute name mismatch at lines 27–28
- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/src/fraud_detector/scoring/context.py` — verified UTC timestamp usage and hardcoded DB name
- `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/src/fraud_detector/utils/currency.py` — verified `"EMPTY"` silent fallback to 1.0 at line 52

---
*Pitfalls research for: reference-frame-normalized payment anomaly detection (brownfield)*
*Researched: 2026-07-06*
