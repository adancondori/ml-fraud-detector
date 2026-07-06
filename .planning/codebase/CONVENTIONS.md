# Coding Conventions

**Analysis Date:** 2026-07-06

## Language

All source code, docstrings, inline comments, variable names, function names, and test descriptions are written in **English**. Spanish appears only in configuration comments (e.g., proxy taxonomy descriptions in `config/config.py`) and in commit messages for the thesis track.

## Naming Patterns

**Files:**
- `snake_case.py` throughout: `engineering.py`, `artifact_loader.py`, `context_provider.py`
- Test files prefixed with `test_`: `test_features.py`, `test_trainer.py`
- Scripts are standalone named scripts in `scripts/`: `eval_clean_honest.py`, `calibrate_threshold_v2.py`

**Classes:**
- `PascalCase` always: `FeatureEngineer`, `ModelTrainer`, `UnsupervisedPreprocessor`, `BatchContextProvider`, `SingleTransactionScorer`
- Abstract base classes explicitly named: `FeatureGroup(ABC)` in `src/fraud_detector/features/engineering.py`

**Functions and methods:**
- `snake_case` for all public functions: `fit_transform`, `score_samples`, `run_mann_whitney`, `assign_proxy_labels`
- Private/module-level helpers prefixed with `_`: `_coerce_timestamp`, `_rolling_shifted_stat`, `_series_group_shift`, `_as_arrays`
- Private static methods also use `_` prefix: `VelocityFeatures._rolling_count`, `BehavioralFeatures._distinct_facilities_30d`

**Variables:**
- `snake_case` throughout
- Numpy arrays for training data use short conventional names: `X_train`, `X_val`, `y_val`, `X_sub`
- Result dicts use descriptive names: `metrics`, `results`, `manifest`
- Proxy labels stored as `np.int8`, scores as `np.float32` or `np.float64` (then cast to `float32` at boundary)

**Constants:**
- `UPPER_SNAKE_CASE` for module-level constants: `FEATURE_NAMES`, `FEATURE_NAMES_30`, `FEATURE_NAMES_21`, `REQUIRED_COLUMNS`, `METADATA_COLS`, `CANONICAL_SQL`

**Type aliases:**
- `Literal` types used for constrained string parameters: `FeatureVariant = Literal["full", "sensitivity", "core"]` in `src/fraud_detector/features/preprocessor.py`

## Feature Set Conventions

The canonical feature sets are defined as module-level lists in `src/fraud_detector/features/engineering.py`:

- `FEATURE_NAMES` — 31 features, the primary feature set (IF-31 variant)
- `FEATURE_NAMES_30` — 30 features, sensitivity variant (excludes `user_reversal_ratio_30d`)
- `FEATURE_NAMES_21` — 21 features, core/ablation variant (first 21 of `FEATURE_NAMES`)

These lists are validated at import time with `raise ValueError` if their lengths are wrong. All code that references feature sets must import from this module; never hardcode feature name lists elsewhere.

The `UnsupervisedPreprocessor` accepts `variant: Literal["full", "sensitivity", "core"]` to select the appropriate set automatically.

## Scaler Convention

The pipeline scaler is **`StandardScaler`** (scikit-learn), fit exclusively on training data. Defined in `src/fraud_detector/features/preprocessor.py`.

- `RobustScaler(quantile_range=(5.0, 95.0))` is used in experimental scripts under `scripts/` (e.g., `eval_clean_honest.py`, `ensemble_if_ecod_copod.py`) but is **not** the canonical preprocessor for the main pipeline.
- After scaling, the scorer clips values to `[-10, 10]` at inference time: `X_scaled = np.clip(X_scaled, -10, 10)` in `src/fraud_detector/scoring/scorer.py`.

## Scoring Convention (Critical)

**Higher score = more anomalous.** All three models (IF, LOF, OC-SVM) expose scores via negation of their native sklearn output:

```python
# IF-specific: -score_samples()
raw_scores = self.model.score_samples(X)
return (-np.asarray(raw_scores, dtype=np.float64)).astype(np.float32)

# Canonical for all 3 models: -decision_function()
raw = self.model.decision_function(X)
return (-np.asarray(raw, dtype=np.float64)).astype(np.float32)
```

Both methods live in `src/fraud_detector/models/trainer.py` (`score_samples` and `decision_function_scores`). Grid search uses `-score_samples(X)` for IF (invariant to contamination). Evaluation uses `decision_function_scores` for consistent cross-model comparison. Any new model added must follow this same negation convention.

## Anti-Leakage Patterns

These rules are enforced in code and validated in tests (`tests/test_features.py`, class `TestAntiLeakage`):

1. **Rolling windows exclude the current row**: implemented via `groupby(...).shift(1)` in `_series_group_shift` and `_rolling_shifted_stat` in `src/fraud_detector/features/engineering.py`. First transaction for any user always gets zero counts.
2. **Fit only on train split**: `FeatureEngineer.fit()` learns statistics (facility averages, method history) exclusively from training data. `transform()` rejects calls before `fit()` with `RuntimeError("fit")`.
3. **Preprocessor fit only on train**: `UnsupervisedPreprocessor.fit()` sets scaler parameters from training data only. Calling `transform()` without `fit()` raises `ValueError`.
4. **Proxy labels not used in training**: `DataManager.assign_proxy_labels()` is called only for evaluation; the `status` column is never passed as a feature.
5. **Cross-split state**: `transform_with_warm_history()` carries method history from train to val without re-fitting.

## `from __future__ import annotations`

Used consistently across all source modules (15/15) and most test files (11/17). Must be the first import in any new module that uses type annotations.

## Type Hints

All public function signatures include type annotations:

```python
def fit(self, df: pd.DataFrame) -> "UnsupervisedPreprocessor":
def score_samples(self, X: np.ndarray) -> np.ndarray:
def assign_proxy_labels(df: pd.DataFrame, proxy_type: str) -> pd.Series:
```

Return types always annotated. `Optional[T]` used for nullable parameters. `Dict`, `List`, `Tuple` from `typing` (not bare `dict`/`list`) because `from __future__ import annotations` is used but backward compatibility with Python 3.9 is maintained.

## Random Seeds

- Primary seed: `42`, read from `settings.random_seed` (`config/config.py`)
- Multi-seed reproducibility list: `[42, 52, 62]` from `settings.multi_seeds_list`
- Tests use `np.random.default_rng(42)` for new-style generator or `np.random.RandomState(42)` when backward compatibility is needed
- All IsolationForest instantiations pass `random_state=settings.random_seed` (training) or `random_state=42` (grid search)

## Logging

Uses **Loguru** via a singleton configured in `src/fraud_detector/utils/logger.py`. The `logger` object is imported and used throughout the codebase:

```python
from fraud_detector.utils.logger import logger
# or, in loader.py:
from loguru import logger
```

Log format: `"{time} | {level} | {name}:{function}:{line} - {message}"`. Development adds colorized console output; production uses JSON format (configurable via `settings.log_format`). File rotation at 10 MB with 30-day retention; errors in a separate `logs/errors.log` with 60-day retention.

Logging patterns:
- Use `logger.info(f"Fitted {model_type} on {len(X_train):,} rows")` for pipeline steps
- Use `logger.warning(...)` for non-fatal issues (e.g., dispersion in multi-seed)
- Never `print()` in library code — only scripts may use print

## Docstrings

Module-level docstrings use triple-quoted strings describing the module's purpose and key contracts. Example from `src/fraud_detector/features/engineering.py`:

```python
"""
Feature engineering for the payment anomaly-detection study.

The implementation follows the 31-feature contract:
- F06 and F21 are intentionally removed because the universe excludes `free`.
- fit() learns only training-set statistics.
- transform() is leakage-safe inside the provided frame.
"""
```

Class and method docstrings are single-sentence or brief paragraphs. No reStructuredText or NumPy-style parameter docs — plain prose only.

## Error Handling

- `ValueError` for invalid input (wrong shapes, missing columns, invalid proxy type, NaN inputs)
- `RuntimeError` for state violations (calling `transform()` before `fit()`, no model to save)
- `FileNotFoundError` for missing required files (missing data splits)
- `raise ValueError(f"Unknown model type '{model_type}'. Available: {list(self.MODEL_REGISTRY)}")` — always include the invalid value and valid options in the message
- `pytest.raises(ValueError, match="...")` in tests always uses a `match=` regex to verify the error message content

## Import Organization

Order (enforced by isort with `profile = "black"`):
1. `from __future__ import annotations`
2. Standard library
3. Third-party (numpy, pandas, sklearn, loguru, pydantic)
4. Internal project (`from fraud_detector.xxx import ...`, `from config.config import settings`)

## Code Style

- **Black** with `line-length = 100`
- **isort** with `profile = "black"`, `multi_line_output = 3`, trailing comma
- **Flake8** with `--max-line-length=100 --extend-ignore=E203,W503`
- **mypy** with `disallow_untyped_defs = false`, `ignore_missing_imports = true`
- Run via: `make format` (black + isort), `make lint` (flake8), `make type-check` (mypy)

## Data Types

Numeric outputs follow strict dtype conventions to minimize memory:
- Feature matrices: `np.float32`
- Anomaly scores: `np.float32` (cast at output boundary from `float64` computation)
- Proxy labels: `np.int8`
- Boolean flags (is_staff, has_tip, is_weekend): `np.int8`
- Large IDs (payment IDs): `np.int64` (preserved to avoid overflow on 3B+ IDs)
- User/facility IDs: `np.int32`

## Module Exports

Public APIs are collected in `__init__.py` files at package root:
- `src/fraud_detector/__init__.py` — package version
- `src/fraud_detector/features/__init__.py`, etc. — typically minimal or empty, no barrel re-exports

Classes and functions are imported directly from their defining module in tests and consuming code.

---

*Convention analysis: 2026-07-06*
