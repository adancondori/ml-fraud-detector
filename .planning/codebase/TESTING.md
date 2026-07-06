# Testing Patterns

**Analysis Date:** 2026-07-06

## Test Framework

**Runner:**
- pytest 7.4.0+
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`

**Coverage:**
- pytest-cov 4.1.0+
- Coverage source: `src/fraud_detector`
- HTML report in `htmlcov/`

**Assertion Library:**
- pytest built-in assertions
- `numpy.testing.assert_array_equal`, `numpy.testing.assert_allclose`, `numpy.testing.assert_array_almost_equal`
- `pandas.testing.assert_frame_equal`

**Run Commands:**
```bash
make test              # pytest --cov=src/fraud_detector --cov-report=html --cov-report=term-missing -v
make lint              # flake8 src/ tests/ --max-line-length=100
pytest -m "not slow"   # skip slow integration tests
pytest tests/test_features.py  # single file
```

## Test File Organization

**Location:** All tests in `tests/` at project root (not co-located with source).

**Naming:** `test_{module_name}.py`, one file per source module or concern:

```
tests/
├── conftest.py                  # shared fixtures (sample_scores, sample_proxy_labels)
├── __init__.py
├── test_api.py                  # FastAPI HTTP endpoints (TestClient)
├── test_batch_scorer.py         # BatchScorer + BatchContextProvider + guardrails
├── test_config.py               # Settings TDD contract
├── test_evaluation.py           # HE1-HE4 hypothesis tests + bootstrap CI
├── test_features.py             # FeatureEngineer 31-feature contract + anti-leakage
├── test_if40_artifacts.py       # IF-40 artifact loading (requires model files)
├── test_integration.py          # End-to-end pipeline smoke tests (marked @slow)
├── test_loader.py               # DataManager SQL + proxy labels + validation
├── test_pipeline.py             # run_pipeline.py orchestrator logic
├── test_pipeline_components.py  # Multi-model preprocessing + metrics integration
├── test_preprocessor.py         # UnsupervisedPreprocessor contracts
├── test_proxy_labels.py         # Proxy taxonomy (Tipos A-E + unified)
├── test_reporting.py            # LaTeX table generation smoke tests
├── test_scoring.py              # SingleTransactionScorer + ThresholdClassifier
└── test_trainer.py              # ModelTrainer contracts + grid search
```

## Test Structure

**Class-based grouping:** Tests are grouped in `class Test{Concern}:` when multiple tests share setup or belong to the same contract. This is the dominant pattern in `test_features.py`, `test_loader.py`, `test_config.py`, `test_proxy_labels.py`, `test_preprocessor.py`, `test_pipeline_components.py`:

```python
class TestAntiLeakage:
    def test_first_transaction_counts_zero(self, engineered_features):
        ...
    def test_cold_start_user_in_val_has_neutral_history(self, engineered_features):
        ...

class TestFitTransformContract:
    def test_transform_requires_fit(self, sample_data):
        engineer = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fit"):
            engineer.transform(sample_data)
```

**Function-based:** Used in `test_trainer.py` and `test_integration.py` where each test is independent and self-describing:

```python
def test_score_convention_higher_is_more_anomalous(synthetic_data):
    ...
def test_lof_has_novelty_true(synthetic_data):
    ...
```

**Contract comments:** Tests are grouped by contract number using inline comments:

```python
# --- Contract 1: Score convention (higher = more anomalous) ---
# --- HE1: Mann-Whitney ---
# --- HE4: Model comparison ---
```

## Fixtures

**conftest.py** (`tests/conftest.py`) provides two minimal shared fixtures:

```python
@pytest.fixture
def sample_scores():
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=200).astype(np.float32)

@pytest.fixture
def sample_proxy_labels():
    rng = np.random.default_rng(42)
    return rng.choice([0, 1], size=200, p=[0.94, 0.06]).astype(np.int8)
```

**Module-scoped fixtures** are used when fixture creation is expensive (FeatureEngineer fit/transform, multi-transaction DataFrames):

```python
@pytest.fixture(scope="module")
def sample_data():
    rows = [_make_txn(...), ...]
    return pd.DataFrame(rows).sort_values(["user_id", "created_at", "id"]).reset_index(drop=True)

@pytest.fixture(scope="module")
def engineered_features(train_val_split):
    train, val = train_val_split
    engineer = FeatureEngineer()
    train_features = engineer.fit_transform(train)
    val_features, state = engineer.transform_with_warm_history(...)
    return engineer, train_features, val_features, state
```

**Inline helper functions** build test data instead of additional fixtures. Named with `_make_` or `_build_` prefix:

```python
def _make_txn(txn_id, user_id, created_at, *, facility_id=1, amount=100.0, ...) -> dict:
    ...
def _make_mock_ch_client(result_rows=None):
    ...
def _make_synthetic_data(n=500, seed=42):
    ...
```

## Mocking

**Framework:** `unittest.mock` — `MagicMock` and `patch`.

**ClickHouse mocking** (used in `test_batch_scorer.py` and `test_api.py`): The real ClickHouse client is never instantiated in tests. Mock clients expose `.query()` that returns objects with `.result_rows`, and `.insert()` for write verification:

```python
def _make_mock_ch_client(result_rows=None):
    mock = MagicMock()
    query_result = MagicMock()
    query_result.result_rows = result_rows or []
    mock.query.return_value = query_result
    return mock
```

**FastAPI dependency injection** (`test_api.py`): Uses `app.dependency_overrides` to replace all DI providers for the full test module:

```python
app.dependency_overrides[get_scorer] = lambda: MOCK_SCORER
app.dependency_overrides[get_read_ch_client] = lambda: MOCK_READ_CLIENT
app.dependency_overrides[get_write_ch_client] = lambda: MOCK_WRITE_CLIENT
```

An `autouse=True` fixture populates the app's `_state` dict before each test and clears it after:

```python
@pytest.fixture(autouse=True)
def mock_lifespan():
    scorer_state["scorer"] = MOCK_SCORER
    scorer_state["model_loaded"] = True
    ...
    yield
    scorer_state.clear()
```

**`patch.object`** for method-level patching (e.g., `_explain_top_factors` in batch scorer tests):

```python
with patch.object(SingleTransactionScorer, "_explain_top_factors", return_value=[...]):
    result = batch_scorer.score_batch(cursor)
```

**What to mock:**
- ClickHouse clients (all DB interactions)
- Scorer at HTTP layer (replace with `MagicMock` returning a `ScoringResult`)
- Methods with filesystem side effects in batch tests (`_explain_top_factors`)

**What NOT to mock:**
- NumPy/pandas operations
- scikit-learn model fitting (use small synthetic data instead)
- `FeatureEngineer`, `UnsupervisedPreprocessor`, `ModelTrainer` logic (tested directly with synthetic DataFrames)

## Synthetic Data Patterns

All tests that require transaction data build deterministic in-memory DataFrames. The standard pattern uses `np.random.default_rng(42)` for new-style generation or `np.random.RandomState(42)` for older code:

```python
rng = np.random.default_rng(42)
X_inliers = rng.standard_normal((500, 10)).astype(np.float32)
X_outliers = (rng.standard_normal((30, 10)) + 8).astype(np.float32)
```

For transaction DataFrames, a helper function `_make_txn()` builds row dicts with keyword-only overrides (`*,` separator), ensuring all required columns are present:

```python
def _make_txn(txn_id, user_id, created_at, *, amount=100.0, status="paid", ...) -> dict:
```

Proxy label distributions mirror production rates: `p=[0.94, 0.06]` (6% anomaly rate for Tipo A).

## Coverage

**Requirements:** No minimum enforced by configuration; coverage is reported but not gated.

**Excluded lines** (`pyproject.toml`):
- `if __name__ == "__main__":`
- `raise NotImplementedError`
- `if TYPE_CHECKING:`
- `def __repr__`
- `pragma: no cover` annotations

**View Coverage:**
```bash
make test              # generates htmlcov/ and term-missing report
open htmlcov/index.html
```

## Test Marks

Two custom marks defined in `pyproject.toml`:

- `@pytest.mark.slow` — integration tests that run the full pipeline; 2 tests in `test_integration.py`. Skip with `-m "not slow"`.
- `@pytest.mark.integration` — defined but not actively used.
- `@pytest.mark.parametrize` — used in `test_pipeline_components.py` for testing all 3 model types in one test.

## Test Types

**Unit tests (majority):** Test one class or function in isolation with synthetic inputs. Examples: `test_features.py` (FeatureEngineer contracts), `test_preprocessor.py` (UnsupervisedPreprocessor), `test_trainer.py` (ModelTrainer score convention).

**Contract/TDD tests:** `test_config.py` and `test_proxy_labels.py` were written as TDD contracts before the implementation. They assert exact values (split dates, threshold values, proxy lists) to lock the implementation to the thesis specification.

**HTTP integration tests:** `test_api.py` uses FastAPI `TestClient` with full DI override. No real network calls; all external dependencies mocked.

**Batch/flow tests:** `test_batch_scorer.py` tests the BatchScorer including the READ/WRITE client split and guardrail logic that prevents accidental writes to production.

**End-to-end smoke tests:** `test_integration.py` (2 tests, `@pytest.mark.slow`) run the complete pipeline: synthetic data → FeatureEngineer → UnsupervisedPreprocessor → ModelTrainer → evaluate_scores.

**Artifact-dependent tests:** `test_if40_artifacts.py` requires `output/models/isolation_forest.joblib` and related files to exist. These tests fail in CI without pre-built artifacts and are not marked with any skip condition.

## Async Testing

Not used. All code is synchronous; FastAPI routes are tested via `TestClient` (synchronous wrapper).

## Error Testing Pattern

```python
def test_transform_requires_fit(self, sample_data):
    engineer = FeatureEngineer()
    with pytest.raises(RuntimeError, match="fit"):
        engineer.transform(sample_data)

def test_guardrail_blocks_identical_fingerprint():
    with pytest.raises(ValueError, match="same ClickHouse"):
        assert_write_target_is_safe(read_fingerprint=READ_FP, write_fingerprint=READ_FP, ...)
```

Always use `match=` to verify error message content. The pattern `pytest.raises((ValueError, RuntimeError))` (tuple of types) is used when the exact exception type is not critical to the contract.

## Save/Load Roundtrip Pattern

Used in `test_features.py`, `test_preprocessor.py`, and `test_trainer.py` to verify serialization:

```python
def test_save_load_produces_same_scores(synthetic_data, tmp_path):
    trainer.fit(X_train)
    scores_before = trainer.score_samples(X_val)
    path = str(Path(tmp_path) / "model.joblib")
    trainer.save_model(path)
    trainer2 = ModelTrainer(model_type="isolation_forest")
    trainer2.load_model(path)
    scores_after = trainer2.score_samples(X_val)
    np.testing.assert_array_equal(scores_before, scores_after)
```

Uses `tmp_path` pytest built-in fixture for temporary directories (never hardcoded paths).

## Coverage Gaps

- `test_if40_artifacts.py` — requires actual trained model artifacts at `output/models/`; skips silently only if files are absent (will error on import, not gracefully skip).
- `scripts/` directory — no test coverage; scripts are run manually for analysis.
- `notebooks/` — not tested.
- `src/fraud_detector/reporting/figures.py` — only `latex_tables.py` has tests; figure generation is untested.
- ClickHouse connector (`src/fraud_detector/data/clickhouse_connector.py`) — only tested indirectly via mocks.
- `scorer/` package (FastAPI app) — tested via HTTP layer in `test_api.py` but internal routing logic coverage is limited.

---

*Testing analysis: 2026-07-06*
