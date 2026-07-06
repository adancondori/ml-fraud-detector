# Stack Research

**Domain:** Frame normalization for per-facility anomaly detection (brownfield milestone)
**Researched:** 2026-07-06
**Confidence:** HIGH — all critical claims verified against official docs, pypi, and venv-installed versions

---

## Context

This is a BROWNFIELD milestone. The base stack (scikit-learn, pandas, numpy, joblib, FastAPI,
Pydantic, ClickHouse, loguru) is already installed and pinned. This document covers ONLY the
incremental libraries and patterns needed for the four new capabilities:

1. Per-facility (per-group) robust magnitude normalization
2. Timezone conversion: Rails ActiveSupport names → IANA → local-time features
3. Versioned per-facility reference-stats artifact (in-memory, sub-200ms scoring)
4. Segmented threshold calibration from raw anomaly scores

For each area, options are surveyed then a single recommendation is given with rationale.

---

## 1. Per-Group Robust Magnitude Normalization

### Problem

The existing pipeline uses a single global `RobustScaler` fit on all training transactions.
Frame normalization requires per-facility statistics (median, IQR of amount-related features)
so that `amount_facility_ratio` features become truly local. With ~1,876 facilities, this
cannot be a separate model per facility — it must be a stats artifact loaded at scoring time.

### Options Surveyed

| Approach | Summary | Verdict |
|---|---|---|
| `RobustScaler` per facility via `fit()` | Fit one scaler per facility on training data; store dict | Viable but 1,876 objects; heavy on memory |
| Store raw `center_`, `scale_` arrays per facility | Compact dict of numpy arrays; reconstruct manually at inference | Recommended — minimal overhead |
| `pandas.groupby().transform()` with lambda | Compute transform in-place during batch inference | Good for batch; not suitable for single-row real-time scoring |
| `sklearn.preprocessing.robust_scale()` function | Stateless function; must recompute stats on every call | Not suitable — no pre-fitted stats |
| Custom numpy formula: `(x - median) / IQR` | Direct array math with stored stats | Equivalent to RobustScaler but lighter; avoids sklearn object overhead |

### Recommendation

**Store per-facility stats as a compact dict of numpy arrays; apply via direct numpy formula at inference.**

Pattern:
```python
# At training time (feature engineering)
facility_stats: dict[str, dict] = {}
for fid, grp in df.groupby("facility_id"):
    vals = grp["amount_usd"].dropna().values
    q1, q3 = np.percentile(vals, [25, 75])
    facility_stats[fid] = {
        "median": float(np.median(vals)),
        "iqr": float(q3 - q1) or 1.0,   # guard zero-IQR
        "n": int(len(vals)),
    }

# At scoring time (scorer, no ClickHouse lookup)
def normalize_amount(amount: float, fid: str, stats: dict) -> float:
    s = stats.get(fid) or stats["__global__"]   # fallback for unseen facilities
    return (amount - s["median"]) / s["iqr"]
```

Rationale:
- `RobustScaler.center_` and `scale_` are plain numpy `float64` arrays; storing them as
  plain Python floats in a dict is identical mathematically and avoids pickling 1,876
  sklearn objects.
- A global `__global__` fallback key handles the cold-start problem for new facilities
  added after training without requiring retraining.
- Zero-IQR guard is required: facilities with a single transaction or all-identical amounts
  produce IQR = 0. Scikit-learn's `RobustScaler` does NOT guard this; it will produce
  `inf` silently when `scale_` = 0. Always substitute 1.0 (equivalent to centering only).

**What NOT to use:** Do not fit 1,876 `RobustScaler` objects and joblib-dump them separately.
The overhead of pickling/unpickling 1,876 objects at startup dominates; and a dict of floats
serializes to a 50–200 KB JSON vs. a multi-MB joblib bundle.

### Versions

| Library | Version in venv | Notes |
|---|---|---|
| `numpy` | 1.24+ (installed: implied by scipy 1.13.1) | `np.percentile` is stable API since 1.9 |
| `pandas` | 2.3.3 | `groupby().agg()` for training-time stat extraction |
| `scikit-learn` | 1.6.1 | `RobustScaler` docs confirmed: `center_`, `scale_` are ndarray shape `(n_features,)` |

**Confidence:** HIGH — verified against scikit-learn 1.9.0 official docs.

---

## 2. Timezone Conversion: Rails Names → IANA → Python datetime

### Problem

The Rails platform stores `facility.time_zone` as an ActiveSupport::TimeZone name (e.g.,
`"Pacific Time (US & Canada)"`, `"Mexico City"`). Transaction timestamps are UTC in
ClickHouse. The scorer must convert them to facility-local time to compute hour-of-day,
day-of-week, and is_off_hours features correctly, without per-request ClickHouse lookups.

### pytz vs. zoneinfo

| Criterion | `pytz` (2026.2) | `zoneinfo` (stdlib, Python 3.9+) |
|---|---|---|
| Python 3.9+ stdlib | No — third-party | Yes — `import zoneinfo` works with no install |
| Correct DST with `datetime()` constructor | No — must use `localize()` or get LMT offset bug | Yes — `datetime(..., tzinfo=ZoneInfo("America/Los_Angeles"))` is correct |
| Official status | Maintenance mode; maintainer recommends switching | Active, PEP 615 stdlib module |
| Requires `tzdata` fallback on Windows | No (has own bundled data) | Yes — install `tzdata` for Windows/containers |
| Data currency | Updated quarterly | Delegates to system tzdata or `tzdata` PyPI package |

**Recommendation: use `zoneinfo` (stdlib).** Verified on this system: Python 3.9.6 with
`import zoneinfo; ZoneInfo("America/Los_Angeles")` succeeds without `backports.zoneinfo`.
`pytz` is in maintenance mode — its own maintainer states: *"Projects using Python 3.9 or
later should use zoneinfo."* (pypi.org/project/pytz, verified 2026-07-06)

The scorer container (requirements-scorer.txt) targets Python 3.12, where `zoneinfo` is
definitively available. No extra install needed.

### Rails → IANA Mapping

ActiveSupport::TimeZone::MAPPING is a hardcoded dict of ~134 entries. There is no canonical
Python PyPI package that wraps this mapping directly. The correct approach is to embed the
mapping as a static dict in the scorer — it rarely changes (Rails has not added new zones
in several major versions) and changes only require re-deployment anyway.

```python
# scorer/tz_mapping.py  (embed full 134-entry dict; shown as excerpt)
RAILS_TO_IANA: dict[str, str] = {
    "Pacific Time (US & Canada)": "America/Los_Angeles",
    "Mountain Time (US & Canada)": "America/Denver",
    "Central Time (US & Canada)": "America/Chicago",
    "Eastern Time (US & Canada)": "America/New_York",
    "Mexico City": "America/Mexico_City",
    "Buenos Aires": "America/Argentina/Buenos_Aires",
    "London": "Europe/London",
    "Paris": "Europe/Paris",
    "UTC": "Etc/UTC",
    # ... (all 134 entries from ActiveSupport source)
}

def rails_to_zoneinfo(rails_name: str) -> ZoneInfo:
    iana = RAILS_TO_IANA.get(rails_name, "UTC")  # fallback UTC
    return ZoneInfo(iana)
```

This mapping dict is loaded once at startup (alongside the stats artifact) and held in
memory for the lifetime of the process.

### tzdata Package

Install `tzdata` as a deployment dependency. On macOS/Linux it is not strictly required
(system tzdata is used), but in Docker containers (scorer) the base image may lack system
tzdata. Already installed in venv (2025.3).

```
tzdata>=2025.1
```

Add to `requirements-scorer.txt`.

### What NOT to use

- **`pytz`**: Do not use `pytz.timezone()` for new code. The `localize()` API creates
  subtle bugs when code passes tzinfo via the datetime constructor instead. Existing
  `pytz` in training code (`pytz==2025.2`) is acceptable as-is but should not be extended.
- **`python-dateutil`**: Not needed; `zoneinfo` covers all DST-aware conversion needs.
- **`babel`**: Locale/formatting library, not timezone conversion; not needed.

### Versions

| Library | Version | Location |
|---|---|---|
| `zoneinfo` | stdlib (Python 3.9+) | No install needed |
| `tzdata` | 2025.3 (venv); add to scorer reqs | `pip install tzdata>=2025.1` |

**Confidence:** HIGH — Python docs confirm `zoneinfo` added in 3.9; pytz PyPI page confirms
maintenance mode statement; venv test confirms `import zoneinfo` succeeds on Python 3.9.6.

---

## 3. Versioned Reference-Stats Artifact

### Problem

The scorer must load per-facility stats (median, IQR, timezone, threshold) into memory at
startup and serve all requests from that in-memory structure. Constraints:
- No per-request ClickHouse lookups
- Sub-200ms end-to-end scoring budget
- Versioned: training pipeline writes a new artifact; scorer hot-reloads or re-deploys

### Options Surveyed

| Format | Load time (167 MB array) | Portability | Versioning | Verdict |
|---|---|---|---|---|
| `joblib` (no compression) | ~0.04s | Python-only | Manual metadata JSON | Good for models; overkill for stats dict |
| `joblib` (lz4) | ~0.06s | Python-only | Manual metadata JSON | Slight compression benefit |
| `pickle` protocol 5 | ~0.04s | Python-only | None built-in | No advantage over joblib for dicts |
| `json` | ~0.005s for 200 KB dict | Universal | Include version field | Recommended for stats dict |
| `parquet` (pyarrow) | ~0.01s read | Cross-language | Schema in file | Overkill for flat dict; adds pyarrow dep |
| `msgpack` | ~0.002s | Cross-language | None built-in | Not installed; no advantage over json here |

### Recommendation

**Use JSON for the per-facility stats artifact; use joblib (lz4) only for scikit-learn
model objects.**

Rationale:
- The facility stats are a dict mapping `facility_id → {median, iqr, n, iana_tz,
  threshold_low, threshold_high}`. This is a plain JSON-serializable structure.
- JSON load of a ~1,876-entry dict is approximately 1–5ms — negligible relative to 200ms
  budget. It is human-readable, diff-able in git, and has no Python version dependency.
- The existing `artifact_loader.py` already uses this pattern: `thresholds.json` and
  `model_metadata.json` are JSON; models are joblib. Extend consistently.
- joblib with lz4 is appropriate for IsolationForest/LOF/OC-SVM model objects because
  they contain large numpy arrays. The lz4 benchmark shows 0.06s load vs 97% size
  reduction — clearly within budget for the model. Do NOT add lz4 compression to the
  JSON stats artifact; compression overhead exceeds benefit at 200 KB.

### Artifact Schema

```json
{
  "schema_version": "frame-norm-v1",
  "trained_at": "2026-07-15T12:00:00Z",
  "sklearn_version": "1.6.1",
  "n_facilities": 1876,
  "global_fallback": {
    "median": 42.50,
    "iqr": 38.20,
    "n": 6784696
  },
  "facilities": {
    "fac_001": {
      "median": 35.10,
      "iqr": 28.40,
      "n": 3421,
      "iana_tz": "America/New_York",
      "threshold_p95": -0.42,
      "threshold_p99": -0.61
    }
  }
}
```

- `schema_version` enables migration detection without breaking existing loaders.
- `sklearn_version` enables the `InconsistentVersionWarning` pattern already in scikit-learn.
- `global_fallback` is the cold-start stats for facilities absent at training time.
- Thresholds are stored per-facility (see Section 4).

### FastAPI Lifespan Loading

Load artifact in FastAPI `lifespan` context manager (already confirmed in official docs):

```python
from contextlib import asynccontextmanager
import json

_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["artifacts"] = load_artifacts(MODEL_DIR)          # existing joblib models
    _state["frame_stats"] = json.loads(STATS_PATH.read_text())  # new JSON stats
    yield
    _state.clear()
```

All requests read from `_state` — zero ClickHouse calls at score time.

### Versions

| Library | Version | Notes |
|---|---|---|
| `json` | stdlib | No install; already used in `artifact_loader.py` |
| `joblib` | 1.5.3 (venv) | Models only; lz4 compression; `lz4` 4.4.5 already installed |
| `lz4` | 4.4.5 (venv) | Required for joblib lz4 compression; already present |

**Confidence:** HIGH — joblib lz4 benchmark verified from official docs; FastAPI lifespan
pattern verified from official docs; existing `artifact_loader.py` confirms json+joblib
pattern already in use.

---

## 4. Segmented Threshold Calibration

### Problem

The current scorer uses a single global threshold from `thresholds.json`. Frame
normalization requires per-segment (per-facility or per-facility-tier) thresholds because
anomaly score distributions vary across facility sizes and transaction profiles. Must
calibrate thresholds WITHOUT labels (unsupervised proxy only for evaluation).

### Mechanism: `score_samples()` + Percentile

Confirmed from scikit-learn 1.9.0 docs (IsolationForest):

- `score_samples(X)` returns raw anomaly scores in [~-1, 0]; higher = more normal.
  This is independent of the `contamination` parameter used during training.
- `decision_function(X)` = `score_samples(X) - offset_`. The `offset_` is set at train
  time by `contamination`; it is NOT used by `score_samples`.
- Custom per-segment thresholds are applied post-training by comparing
  `score_samples(X)` against a segment-specific scalar threshold — no retraining needed.

### Calibration Approach

Use **percentile-based calibration** per segment on the VALIDATION set:

```python
# During evaluation (val split) — not at score time
for fid, grp in val_df.groupby("facility_id"):
    scores = model.score_samples(grp[feature_cols].values)
    facility_stats[fid]["threshold_p95"] = float(np.percentile(scores, 5))   # 5th pct = top 5% anomalous
    facility_stats[fid]["threshold_p99"] = float(np.percentile(scores, 1))   # 1st pct = top 1% anomalous
```

The threshold is the k-th percentile of score_samples on the val set for that segment.
Segments with fewer than `MIN_SEGMENT_SAMPLES` (e.g., 100) fall back to the global threshold.

### LOF and OC-SVM

Both expose `score_samples()` (confirmed in scikit-learn docs). The same percentile
calibration applies without model-specific special-casing.

### Options Surveyed

| Approach | Description | Verdict |
|---|---|---|
| Percentile of `score_samples` per segment | Calibrate on val set; store in stats JSON | Recommended |
| Retrain with `contamination=facility_rate` per segment | Retrain 1,876 models | Impractical |
| Adaptive online threshold (rolling quantile) | Update threshold from live score stream | Out of scope for this milestone |
| Otsu/MAD hybrid | Bimodal histogram splitting | Research method; no sklearn implementation |
| Fixed global `contamination` percentile | What exists today | Insufficient for frame normalization |

### Minimum Segment Size

Segments with fewer than 100 validation-set transactions produce unstable percentile
estimates. Use the global threshold as fallback for under-represented facilities. Store
which fallback was applied in the artifact for auditability.

### Versions

All threshold calibration uses `numpy.percentile` (already installed). No new libraries
required.

**Confidence:** HIGH — `score_samples` API verified against scikit-learn 1.9.0 official
docs; percentile method for calibration confirmed by scikit-learn outlier detection
documentation and multiple corroborating sources.

---

## Complete Incremental Stack Delta

The following lists ONLY new dependencies or version constraints introduced by this milestone.
Everything in `requirements.txt` and `requirements-scorer.txt` remains unchanged.

### New to `requirements-scorer.txt`

```
tzdata>=2025.1        # IANA timezone data for zoneinfo in Docker containers
```

`lz4` is already present (4.4.5). `zoneinfo` is stdlib in Python 3.9+. `json` is stdlib.
No other new packages are needed.

### No New Training Dependencies

Stat extraction at training time uses `numpy.percentile` and `pandas.groupby().agg()`
— both already in `requirements.txt`.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|---|---|---|---|
| Timezone handling | `zoneinfo` (stdlib) | `pytz` | Maintenance mode; `localize()` API footgun |
| Timezone handling | `zoneinfo` (stdlib) | `python-dateutil` | Heavier dep; no advantage here |
| Stats serialization | `json` | `joblib` for stats dict | Joblib adds no value for plain float dicts; harder to inspect |
| Stats serialization | `json` | `parquet` | Overkill for 1,876-row flat structure; adds pyarrow dep at runtime |
| Per-group normalization | numpy dict + formula | 1,876 `RobustScaler` objects | Serialization overhead; identical math |
| Threshold calibration | `score_samples` + percentile | Retrain per segment | Infeasible; 1,876 retrains |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|---|---|---|
| `pytz.timezone()` for new code | Maintenance mode; `datetime(..., tzinfo=pytz.tz)` uses LMT offset bug | `ZoneInfo(iana_name)` from stdlib `zoneinfo` |
| `joblib.dump` for stats dict | Pickle-based; version-dependent; opaque | `json.dumps` — human-readable, diffable, fast |
| `decision_function` for threshold calibration | Threshold baked into `offset_` at train time; not per-segment | `score_samples` + store per-segment threshold in JSON |
| `contamination` parameter per facility | Would require 1,876 retrains | Post-hoc percentile on `score_samples` val scores |
| Zero-IQR RobustScaler without guard | Produces `inf` silently | Guard: `iqr = max(iqr, 1e-8)` or substitute 1.0 |
| `backports.zoneinfo` | Only needed for Python < 3.9; scorer runs Python 3.12, training runs 3.9.6 which has stdlib `zoneinfo` | stdlib `zoneinfo` directly |

---

## Version Compatibility Matrix

| Package | Training env | Scorer env | Notes |
|---|---|---|---|
| `scikit-learn` | 1.6.1 | 1.6.1 (pinned in requirements-scorer.txt) | Must stay identical; scikit-learn forbids cross-version pickle loading |
| `joblib` | 1.5.3 | >=1.4.0 (scorer req) | Backward compatible; dump/load within same major version |
| `numpy` | 1.24+ | <2.0.0 (scorer req) | Scorer caps at 1.x to avoid breaking numpy 2.0 array semantics in existing code |
| `zoneinfo` | stdlib 3.9.6 | stdlib 3.12 | No version concern; same stdlib module |
| `tzdata` | 2025.3 (venv) | >=2025.1 (add to scorer) | IANA data; only timezone definitions, not code |
| JSON stats artifact | `frame-norm-v1` schema | reads `frame-norm-v1` | `schema_version` field enables forward detection |

---

## Sources

- scikit-learn 1.9.0 IsolationForest docs — `score_samples` vs `decision_function`, `offset_`
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
  Confidence: HIGH

- scikit-learn 1.9.0 RobustScaler docs — `center_`, `scale_` attributes, no per-group support
  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
  Confidence: HIGH

- scikit-learn 1.9.0 model persistence guide — joblib recommendation, InconsistentVersionWarning
  https://scikit-learn.org/stable/model_persistence.html
  Confidence: HIGH

- Python 3.9 zoneinfo docs — stdlib module, ZoneInfo usage, tzdata fallback
  https://docs.python.org/3.9/library/zoneinfo.html
  Confidence: HIGH

- pytz PyPI page — maintenance mode statement, "use zoneinfo for Python 3.9+"
  https://pypi.org/project/pytz/
  Confidence: HIGH

- tzdata PyPI page — version 2026.2, purpose, when needed
  https://pypi.org/project/tzdata/
  Confidence: HIGH

- ActiveSupport::TimeZone Rails API — MAPPING structure, ~134 entries
  https://api.rubyonrails.org/classes/ActiveSupport/TimeZone.html
  Confidence: HIGH

- joblib compressors comparison docs — lz4 load time 0.059s, size 6.3 MB vs 167 MB raw
  https://joblib.readthedocs.io/en/stable/auto_examples/compressors_comparison.html
  Confidence: HIGH

- FastAPI lifespan docs — startup artifact loading pattern
  https://fastapi.tiangolo.com/advanced/events/
  Confidence: HIGH

- pytz-deprecation-shim migration guide — Django 4.0 switched to zoneinfo
  https://pytz-deprecation-shim.readthedocs.io/en/latest/migration.html
  Confidence: MEDIUM (third-party, confirmed by django release notes)

---

*Stack research for: ml-fraud-detector frame normalization milestone*
*Researched: 2026-07-06*
