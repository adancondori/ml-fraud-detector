# Feature Research

**Domain:** Operational anomaly-detection + human-review system for payments (brownfield — IsolationForest scorer already in production)
**Researched:** 2026-07-06
**Confidence:** HIGH (system context from codebase audit + MEDIUM from cross-referenced web sources on shadow mode, HITL, and threshold calibration patterns)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features the review team and operations assume exist. Missing these = the system feels unreliable or untrustworthy.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Anomaly ranking with normalized score (percentile) | Reviewers need to know where a payment sits in the distribution, not just a raw IF score | LOW | Already in `ThresholdClassifier`; real value is that percentile must be computed relative to a stable reference distribution, not recalculated per-batch |
| Risk-level labels (`minimal / low / medium / high / critical`) | Without buckets, reviewers cannot triage; every alert looks equally urgent | LOW | Already persisted in `anomaly_scores.risk_level`; must remain stable across model versions during shadow run |
| Alert explanation: top-N contributing features per flagged payment | Without "why flagged", reviewers cannot decide faster than random and cannot build intuition | MEDIUM | `top_factors` already stored in `anomaly_scores`; current implementation uses z-score magnitude, not SHAP; SHAP TreeExplainer is the production standard but adds latency — compute for top-k only, not all payments |
| Per-segment threshold: facility-level fallback chain | A global 95th-percentile threshold produces alert rates wildly uneven across large vs. small, domestic vs. international facilities | MEDIUM | Threshold hierarchy: facility → currency_group → global; each tier falls back to the next when segment N is too small for reliable calibration; must be versioned as `threshold_version` |
| Batch scoring completeness and cursor tracking | Operations needs to know no payment window was skipped; cursor-based batch must be observable | LOW | `next_cursor` already in `BatchScoreResponse`; needs a monitoring surface showing last-scored timestamp and gap detection |
| Fallback handling for missing timezone | `is_off_hours` computed in UTC today; when facility timezone is absent or malformed, scoring must degrade gracefully with a `fallback_level` flag, not silently use UTC as ground truth | LOW | Hierarchy: IANA tz from facility → currency-group default → UTC with flag `fallback_level=utc_only`; flag stored on alert so reviewers know the confidence |
| Fallback handling for missing/unknown currency | `amount_usd_ratio` and relative-magnitude features break if currency is `"EMPTY"` or not in the 21-currency lookup | LOW | Sanitize `"EMPTY"` → `"USD"` in preprocessing; emit `frame_flags.currency_unknown=true` on the alert; log warning per payment in `error` column |
| Shadow mode: dual scoring without affecting live alerts | New frame-v1 model must run silently in parallel before any promotion; production alert queue must be driven only by the current champion | MEDIUM | `SCORING_MODE=shadow` already in env schema and `scoring_mode` column in `anomaly_scores`; the new requirement is persisting *both* champion and challenger rows for the same `payment_id` with distinct `model_version` tags |
| Human review queue: ranked list of top-k pending alerts | Without a queue UI or export, reviewers work from raw ClickHouse queries — not sustainable | MEDIUM | Top-k from `anomaly_scores` filtered to `risk_level IN ('high','critical') AND review_status IS NULL`; complexity comes from the Rails-side queue surface, not the scorer |
| Label capture: reviewer verdict stored per alert | Without capturing reviewer decisions, the shadow run produces no feedback signal; the entire HITL investment is lost | MEDIUM | Add `review_status`, `reviewer_id`, `reviewed_at`, `reviewer_label` columns to `anomaly_scores` or a linked `alert_reviews` table; labels must be captured independently of the proxy |
| Stats artifact versioning | The facility stats artifact (medians, IQRs, currency, timezone) must be versioned alongside model and threshold artifacts so that any scoring run is fully reproducible | LOW | Extend `artifact_loader.py` to load a `stats_artifact_version` field; store version tag in `anomaly_scores.feature_version` |

### Differentiators (Competitive Advantage)

Features that make the review system meaningfully more trustworthy than the status quo (nominal-amount ranking + UTC artifacts).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Reference-frame normalization: amount relative to facility | Replaces nominal USD magnitude with "how unusual is this amount *for this facility*" — reduces amount bias from ~15.7× toward <4× in top-5% | HIGH | Core of the milestone; requires facility stats artifact (median, IQR per facility/currency_group); `amount_facility_ratio` and robust z-score replace raw `log_amount` as the primary magnitude signal |
| Local-hour temporal features (IANA timezone) | Replaces UTC `is_off_hours` (~30% of payments flagged as off-hours) with local-time classification (~4-5% true off-hours); eliminates the dominant false-positive driver for facilities in UTC-4 to UTC-8 | MEDIUM | `facility_time_zone_iana` from Rails facilities table; already available in `output/revision/facility_tz.parquet`; must be identical in batch and real-time scorer to achieve paridad |
| Shadow-mode bias comparison: amount distribution of champion vs. challenger top-5% | Quantifies whether frame-v1 actually reduces the amount-concentration bias vs. current model; this is the primary go/no-go gate, not AUC | MEDIUM | Compare `PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount_usd)` / mean for flagged payments under each `model_version`; target: top-5% mean drops from 15.7× to <4× global mean |
| Shadow-mode Jaccard top-k overlap | Measures rank agreement between champion and challenger; if overlap is high, frame-v1 is a conservative improvement; if low, warrants deeper review before promotion | MEDIUM | Jaccard@k = |top_k_champion ∩ top_k_challenger| / |top_k_champion ∪ top_k_challenger|; track daily over shadow window |
| Shadow-mode per-segment alert rate comparison | Detects whether frame-v1 shifts alert load from high-value facilities to behaviorally unusual ones; the operational value is visible here, not in aggregate alert count | MEDIUM | Group by `currency_group` and `risk_tier` (facility size buckets); compare alert rate champion vs. challenger; surface facilities that exit or enter the alert queue |
| Explainability via SHAP (top-3 features per alert) | Reviewer can immediately see "this payment was flagged because its amount is 8.2 IQRs above this facility's median AND it occurred at 2 AM local time" — context, not just a score | HIGH | `shap.TreeExplainer` on IsolationForest; compute only for `risk_level IN ('high','critical')` to stay within latency budget; cache SHAP background dataset derived from training set; store `shap_factors` JSON on alert row |
| Drift monitoring of stats artifact | Detects when facility transaction distributions shift enough that the frozen stats artifact no longer reflects current behavior; triggers a re-computation cycle before bias creeps back | MEDIUM | Track PSI per facility for `log_amount` and `hour_local` rolling 30d vs. stats artifact baseline; alert when PSI > 0.2 (standard threshold for significant drift); store per-facility drift scores in a monitoring table |
| Reviewer label independence assessment | After collecting N reviews, compute correlation between proxy label (refund) and human label (fraud/legit/admin-error) to quantify how well the proxy approximates real fraud in the top-k | HIGH | Requires ~300 reviewed cases to reach statistical power; output is a confusion matrix + Cohen's kappa between proxy and reviewer labels; documents proxy validity for thesis governance |
| Threshold recalibration trigger based on alert rate | If the 7-day rolling alert rate (flagged/scored) deviates >20% from the calibrated target, automatically flag for threshold review; prevents silent alert-rate collapse after model promotion | LOW | Single ClickHouse query: `COUNT(*) WHERE is_anomaly=true / COUNT(*) WHERE is_anomaly IS NOT NULL` over rolling 7d per segment; compare to `threshold_version` metadata `{target_alert_rate}` |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem useful but create correctness, governance, or scope problems in this specific system.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Reporting circular-proxy AUC as validation | AUC against `pure_fraud` looks impressive (0.841) and stakeholders expect a numeric validation metric | `pure_fraud` is defined using four features that are also model inputs (`user_account_age_days`, `user_txn_count_1h`, `same_amount_count_1h`, `is_third_party_payment`) — the model learns exactly the rules that define the evaluation criterion; this is autovalidation, not external validation | Report AUC only against Tipo A (structurally independent); lead with bias reduction metric (top-5% amount ratio); reserve `pure_fraud` AUC as a diagnostic footnote with explicit circularity disclaimer |
| Per-facility model (one IsolationForest per facility) | Sounds like better personalization | Most facilities have <500 payments/month — far below the minimum for stable IsolationForest training; produces high-variance models that are impossible to monitor and recalibrate consistently | Global model + per-facility stats artifact + per-segment threshold calibration achieves personalization without per-facility model instability |
| Real-time SHAP for every transaction | Providing explanation on every scored payment feels thorough | TreeExplainer adds ~50-100ms per call at scale; the real-time budget is ~200ms total including ClickHouse context queries; computing SHAP on minimal/low-risk payments wastes budget with near-zero reviewer value | Compute SHAP only for `risk_level IN ('high','critical')`; for lower tiers, return z-score-based top factors (already implemented) |
| AUC on val/test as the threshold gate for frame-v1 promotion | Easy to compute and familiar | The proxy ceiling constrains AUC regardless of model quality; frame-v1 may show equal or lower AUC vs. current model while being a strictly better detector of behavioral anomalies; using AUC as the gate would reject a valid improvement | Gate promotion on bias reduction (amount top-5% ratio) + off-hours rate correction + alert rate stability, not AUC |
| Automated alert routing to payment-provider risk team | Reduces review workload | Sending automated alerts to an external party based on an unsupervised model with no confirmed ground truth creates regulatory and reputational risk; the proxy is refunds, not confirmed fraud | Keep alerts internal; route only cases that pass human review with explicit fraud/abuse label to any external escalation |
| Training the model on reviewer labels once collected | Seems like a natural progression to supervised learning | This project's model is explicitly unsupervised by design (thesis constraint and operational constraint: no confirmed fraud labels); using reviewer labels to retrain without careful governance blurs the unsupervised framing and the proxy-independence claim | Use reviewer labels only for evaluation (confusion matrix, kappa vs. proxy); document as a future milestone if label volume reaches sufficient scale |
| Global alert threshold lowering to "catch more fraud" | Stakeholder pressure to increase sensitivity | Lowering from percentile-95 without segment-specific calibration amplifies the facility-size bias — large high-volume facilities dominate the alert queue even more; recall increases but precision drops and reviewer fatigue rises | Calibrate segment-specific thresholds; then tune sensitivity per segment independently based on alert-rate targets per currency group |
| UTC-based `is_off_hours` as the off-hours signal | Already implemented, so "just use it" | For UTC-5 to UTC-8 facilities (most of TechSport's Latin American network), 23:00 UTC = 6 PM local time — peak business hours classified as off-hours; this is the single largest source of false positives in the current model | Replace with IANA-timezone local-hour computation; flag `fallback_level=utc_only` when timezone is missing rather than silently using UTC |
| Storing raw feature vectors in `features_json` for all payments | Useful for debugging and retraining | At 6.78M transactions/year, storing 31-float JSON per row adds ~2GB/month to the `anomaly_scores` table; most rows are never reviewed | Store `features_json` only for `risk_level IN ('high','critical')`; for others, store only `top_factors` (already 5-element array) |

---

## Feature Dependencies

```
[Stats Artifact v1]
    └──requires──> [Facility timezone lookup (IANA)]
    └──requires──> [Facility transaction history (train window)]

[FS-frame-operational-v1 features]
    └──requires──> [Stats Artifact v1]  (amount_facility_ratio, robust z-score, local hour)

[Frame-v1 model (retrained)]
    └──requires──> [FS-frame-operational-v1 features]

[Per-segment threshold calibration]
    └──requires──> [Frame-v1 model scores on val set]
    └──requires──> [Stats Artifact v1]  (segment labels: currency_group, facility tier)

[Shadow mode dual-run]
    └──requires──> [Frame-v1 model (retrained)]
    └──requires──> [Per-segment threshold calibration]
    └──requires──> [Champion model still running]

[Bias comparison metric (amount top-5% ratio)]
    └──requires──> [Shadow mode dual-run]  (both model versions scoring same payments)

[Jaccard top-k overlap]
    └──requires──> [Shadow mode dual-run]

[Per-segment alert rate comparison]
    └──requires──> [Shadow mode dual-run]
    └──requires──> [Per-segment threshold calibration]

[Human review queue]
    └──requires──> [Shadow mode dual-run]  (reviewer should evaluate both models' output)
    └──requires──> [Alert explanation: top factors or SHAP]

[Label capture]
    └──requires──> [Human review queue]

[Reviewer label independence assessment]
    └──requires──> [Label capture]  (minimum ~300 reviewed cases)

[Drift monitoring of stats artifact]
    └──requires──> [Stats Artifact v1]  (baseline)
    └──enhances──> [Per-segment threshold calibration]  (triggers recalibration)

[SHAP per-alert explanation]
    └──requires──> [Frame-v1 model (retrained)]
    └──enhances──> [Human review queue]  (context for reviewer)

[Threshold recalibration trigger]
    └──requires──> [Per-segment threshold calibration]
    └──requires──> [Shadow mode dual-run data]

[Fallback: timezone]
    └──requires──> [Stats Artifact v1]  (has fallback_level field)
    └──enhances──> [FS-frame-operational-v1 features]

[Fallback: currency]
    └──requires──> [Stats Artifact v1]
    └──enhances──> [FS-frame-operational-v1 features]
```

### Dependency Notes

- **Stats Artifact v1 is the foundational dependency**: everything downstream of frame normalization — new features, per-segment thresholds, shadow comparisons — requires it. It must be built and versioned first.
- **Shadow mode requires both models running simultaneously**: this means the scorer's `SHADOW_MODEL_DIR` infrastructure (already partially present) must be fully exercised before the review queue is activated.
- **Label capture requires the review queue**: labels cannot be captured independently; the queue is the instrument.
- **SHAP enhances the review queue but does not block it**: the queue can open with z-score-based `top_factors`; SHAP can be layered in without changing queue structure.
- **Drift monitoring is independent of shadow mode**: it monitors the stats artifact, not the models; it can run in parallel with any phase.

---

## MVP Definition

### Launch With (v1)

Minimum viable for the shadow + review milestone to produce the intended bias-reduction evidence.

- [ ] Stats artifact v1 — facility median/IQR/timezone/currency_group computed on train window, versioned, loaded in memory at scorer startup (without this, nothing else works)
- [ ] FS-frame-operational-v1 feature set — amount_facility_ratio, robust z-score, local-hour IANA, no reversal features, no capture_delay_seconds; paridad batch↔real-time
- [ ] Frame-v1 model retrained on FS-frame-operational-v1, approved on bias-reduction gate (top-5% amount ratio <4×, off-hours correction from ~30% UTC to ~4-5% local)
- [ ] Per-segment threshold calibration on val set (facility → currency_group → global fallback hierarchy), replacing the current test-set-derived threshold
- [ ] Fallback handling for missing timezone and unknown currency, with `fallback_level` and `frame_flags` persisted on alert rows
- [ ] Shadow mode dual-run: both champion and frame-v1 score every payment, both rows persisted in `anomaly_scores` with distinct `model_version`; no live alert queue change
- [ ] Shadow monitoring dashboard (minimal): alert rate per segment, bias metric (amount top-5% ratio), Jaccard@100 — queryable from `anomaly_scores` via ClickHouse
- [ ] Human review queue (minimal): top-k export sorted by `risk_level` and `percentile` for frame-v1 model version, with `review_status` column writable
- [ ] Label capture: `reviewer_label`, `reviewed_at`, `reviewer_id` columns on `anomaly_scores` (or linked table); schema change is a prerequisite for any review session

### Add After Validation (v1.x)

Add once shadow run has produced at least 2 weeks of dual data and initial reviews are complete.

- [ ] SHAP per-alert explanation for high/critical — trigger: shadow data confirms frame-v1 bias reduction gate passes and promotion is being considered
- [ ] Drift monitoring of stats artifact (PSI per facility, alert on PSI > 0.2) — trigger: frame-v1 promoted to champion; need to know when artifact goes stale
- [ ] Reviewer label independence assessment (proxy vs. human label confusion matrix + kappa) — trigger: ~300 reviewed cases accumulated
- [ ] Threshold recalibration trigger (alert-rate deviation >20% from target flags for review) — trigger: frame-v1 promoted and segment thresholds are live

### Future Consideration (v2+)

Defer until product-market fit of the review system is established (sustained reviewer engagement, stable alert quality).

- [ ] Full automated SHAP for all risk levels — why defer: latency budget + low reviewer value for minimal/low tier
- [ ] Per-segment model variants (e.g., per currency_group model) — why defer: stats artifact + threshold calibration likely resolves most segment variance without model proliferation
- [ ] External escalation routing — why defer: requires human-verified labels at scale and explicit governance policy; cannot launch on proxy alone
- [ ] Supervised fine-tuning using reviewer labels — why defer: requires confirmed label volume (>1,000 verified cases) + thesis scope constraint; future milestone

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Stats artifact v1 (facility median/IQR/tz/currency) | HIGH | MEDIUM | P1 |
| FS-frame-operational-v1 features (amount_ratio, local hour) | HIGH | MEDIUM | P1 |
| Frame-v1 model retrain + bias gate | HIGH | MEDIUM | P1 |
| Per-segment threshold calibration (val set, not test) | HIGH | MEDIUM | P1 |
| Fallback: timezone + currency with flags | MEDIUM | LOW | P1 |
| Shadow mode dual-run (both model versions persisted) | HIGH | MEDIUM | P1 |
| Shadow monitoring: alert rate, bias metric, Jaccard | HIGH | LOW | P1 |
| Human review queue (top-k export + review_status column) | HIGH | LOW | P1 |
| Label capture schema (reviewer_label, reviewed_at) | HIGH | LOW | P1 |
| SHAP per high/critical alert | MEDIUM | MEDIUM | P2 |
| Drift monitoring of stats artifact (PSI per facility) | MEDIUM | MEDIUM | P2 |
| Reviewer label independence assessment (kappa vs. proxy) | MEDIUM | LOW | P2 |
| Alert rate deviation trigger for threshold recalibration | LOW | LOW | P2 |
| Full SHAP for all risk levels | LOW | HIGH | P3 |
| Per-segment model variants | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for the shadow + human-review milestone to produce valid evidence
- P2: Should have — adds depth to shadow analysis and review feedback loop
- P3: Nice to have — future milestone after review system is operating sustainably

---

## Competitor / Reference Feature Analysis

This is an internal operational system, not a commercial product. The reference frame is best practices in production payment-fraud anomaly detection systems (Stripe Radar, PayPal risk, internal bank AML systems) and research literature.

| Feature | Industry Standard | This System (current) | This Milestone |
|---------|-------------------|-----------------------|----------------|
| Per-segment thresholds | Segment-specific (by merchant tier, channel, geography) | Single global percentile-95 (test-set derived) | Facility → currency_group → global fallback, calibrated on val set |
| Alert explanation | Top-N SHAP features per alert | Top-5 z-score magnitude factors | Top-5 z-score for low tiers; SHAP for high/critical |
| Temporal normalization | Local time (business-hours by facility) | UTC for all facilities | IANA timezone per facility with fallback chain |
| Amount normalization | Relative to peer group / merchant baseline | Absolute log_amount in USD | Relative to facility median/IQR (robust z-score) |
| Shadow mode | Standard pre-promotion practice | Partially present (SCORING_MODE env var + column) | Fully exercised: dual rows per payment, monitoring queries |
| Label capture | Human review verdict stored per alert | Not implemented | reviewer_label, reviewed_at, reviewer_id on alert row |
| Drift monitoring | PSI / KS test on feature distributions vs. training baseline | Not implemented | PSI on amount and hour_local per facility, rolling 30d |
| Proxy validation | External ground truth (chargebacks, confirmed fraud) | Proxy only (refund status Tipo A) | Human review label as independent proxy validation (N~300) |

---

## Sources

- Codebase audit: `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector/.planning/codebase/` (ARCHITECTURE.md, CONCERNS.md, INTEGRATIONS.md) — HIGH confidence
- PROJECT.md active requirements — HIGH confidence (primary spec)
- [Shadow Deployment: Risk-free performance comparison — Statsig](https://www.statsig.com/perspectives/shadow-deployment-comparison) — MEDIUM confidence
- [Shadow Mode Deployment for ML Model Testing — ML Journey](https://mljourney.com/shadow-mode-deployment-for-ml-model-testing/) — MEDIUM confidence
- [Threshold Tuning and Risk-Based Calibration in Transaction Monitoring — Sanction Scanner](https://www.sanctionscanner.com/blog/threshold-tuning-and-risk-based-calibration-in-transaction-monitoring-1372) — MEDIUM confidence
- [Combining Threshold Monitoring and Anomaly Detection — FraudNet](https://www.fraud.net/resources/combining-threshold-monitoring-and-anomaly-detection-for-superior-merchant-risk-management) — MEDIUM confidence
- [Isolation Forest Anomaly Detection with SHAP — EliteDev](https://python.elitedev.in/machine_learning/isolation-forest-anomaly-detection-complete-guide-with-shap-explainability-for-robust-ml-systems-619f569e/) — MEDIUM confidence
- [Dynamic Calibration of Decision Thresholds for Financial Anomaly Detection — ScienceDirect](https://www.sciencedirect.com/org/science/article/pii/S1062737525001118) — MEDIUM confidence (title/abstract only)
- [Alert Triage Process — CyberDefenders](https://cyberdefenders.org/blog/alert-triage-process/) — LOW confidence (security SOC context, not payment-specific)
- [Explaining Outliers using Isolation Forest and Shapley Interactions — ESANN 2025](https://www.esann.org/sites/default/files/proceedings/2025/ES2025-163.pdf) — MEDIUM confidence (peer-reviewed)

---

*Feature research for: Reference-frame-normalized payment anomaly detection + operational review system*
*Researched: 2026-07-06*
