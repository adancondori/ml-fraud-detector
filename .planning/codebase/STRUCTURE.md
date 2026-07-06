# Codebase Structure

**Analysis Date:** 2026-07-06

## Directory Layout

```
ml-fraud-detector/
├── run_pipeline.py              # Main 8-step pipeline orchestrator
├── smoke_test.py                # End-to-end smoke test
├── Makefile                     # Common dev commands
├── config/
│   └── config.py                # Pydantic Settings — single config source of truth
├── src/
│   └── fraud_detector/          # Installable library (pip install -e .)
│       ├── data/                # ClickHouse extraction and split I/O
│       ├── features/            # Feature engineering + preprocessing
│       ├── models/              # Unsupervised model training
│       ├── evaluation/          # Hypothesis testing (HE1-HE4) + metrics
│       ├── scoring/             # Single-transaction scoring facade
│       ├── reporting/           # LaTeX tables and figures
│       └── utils/               # Logger, currency normalizer
├── scorer/                      # FastAPI real-time + batch scoring service
│   ├── main.py                  # App entry point, lifespan, routes
│   ├── schemas.py               # All Pydantic request/response models
│   ├── artifact_loader.py       # Versioned model artifact loading
│   ├── dependencies.py          # Shared _state dict + FastAPI Depends
│   ├── routers/
│   │   ├── score.py             # POST /score, POST /score/batch
│   │   └── model.py             # GET /health, GET /model/info, POST /model/reload
│   └── batch/
│       ├── scorer.py            # BatchScorer (bulk fetch-score-insert)
│       └── context_provider.py  # BatchContextProvider (6 queries per chunk)
├── scripts/                     # Standalone execution scripts
│   ├── run_fase6_modeling.py    # Grid search + model training
│   ├── run_fase7_evaluation.py  # HE1-HE4 + thresholds
│   ├── run_fase8_sensitivity.py # Sensitivity analysis + SHAP + post-hoc
│   ├── run_fase9_reporting.py   # LaTeX tables + figures generation
│   ├── run_fase2_eda.py         # Exploratory data analysis
│   ├── run_fase4_features.py    # Feature EDA
│   ├── run_fase5_preprocessing.py
│   ├── eval_*.py                # Ad-hoc evaluation experiments
│   ├── exp_*.py                 # Experimental / exploratory scripts
│   ├── validate_*.py            # Model and data validation scripts
│   ├── benchmark_realtime.py    # API latency benchmarks
│   ├── hitl_export_alerts.py    # Human-in-the-loop: export alerts for labeling
│   ├── hitl_ingest_labels.py    # Human-in-the-loop: ingest reviewed labels
│   └── score_payment.py         # CLI single-transaction scorer
├── data/
│   ├── processed/               # Parquet splits (train/val/test/warm raw + features)
│   └── external/                # exchange_rates.csv
├── output/                      # All generated artifacts (not committed to git)
│   ├── models/                  # Serialized model + scaler + threshold artifacts
│   ├── scores/                  # Numpy arrays (X_train/val/test.npy) + score parquets
│   ├── tables/                  # Generated LaTeX .tex files
│   ├── figures/                 # Generated .pdf and .png figures
│   ├── results.json             # Primary HE1-HE4 results
│   ├── results_sensitivity.json
│   ├── results_posthoc.json
│   └── [other results_*.json]   # Experimental result snapshots
├── tests/                       # pytest test suite
├── docker/
│   └── clickhouse/              # Local ClickHouse docker config + migrations
├── docker-compose.yml           # Local infrastructure (ClickHouse + scorer)
├── docs/                        # Additional documentation
├── notebooks/                   # Jupyter notebooks (EDA)
├── .planning/                   # SpecOps planning documents
│   ├── codebase/                # This directory
│   ├── PLAN-FINAL/
│   ├── PLAN-FINAL-v3/
│   └── PLAN-FINAL-v4/
├── pyproject.toml               # Package metadata + tool config
├── requirements.txt             # Core runtime deps
├── requirements-dev.txt         # Dev/test deps
├── requirements-scorer.txt      # FastAPI scorer deps
└── setup.py                     # Editable install entry
```

## Directory Purposes

**`src/fraud_detector/data/`:**
- Purpose: Raw data extraction from ClickHouse and temporal split management.
- Contains: `loader.py` (`DataManager`, `CANONICAL_SQL`, `REQUIRED_COLUMNS`, `assign_proxy_labels`), `clickhouse_connector.py` (`ClickHouseConnector`).
- Key files: `src/fraud_detector/data/loader.py`, `src/fraud_detector/data/clickhouse_connector.py`

**`src/fraud_detector/features/`:**
- Purpose: All feature engineering logic. Stateful: fit on train, serialize, restore for scoring.
- Contains: `engineering.py` (`FeatureEngineer`, `FEATURE_NAMES` 31-item canonical list, variant lists `FEATURE_NAMES_30` / `FEATURE_NAMES_21`, `transform_with_warm_history`), `preprocessor.py` (`UnsupervisedPreprocessor`, `FeatureVariant` literal: `full`/`sensitivity`/`core`).
- Key files: `src/fraud_detector/features/engineering.py`, `src/fraud_detector/features/preprocessor.py`

**`src/fraud_detector/models/`:**
- Purpose: Train unsupervised models with grid search, multi-seed analysis, and joblib serialization.
- Contains: `trainer.py` (`ModelTrainer`, `MODEL_REGISTRY = {isolation_forest, lof, ocsvm}`).
- Key file: `src/fraud_detector/models/trainer.py`

**`src/fraud_detector/evaluation/`:**
- Purpose: Stateless hypothesis testing functions. Inputs are arrays; outputs are structured dicts.
- Contains: `hypothesis.py` (HE1 `run_mann_whitney`, HE2 `compute_discrimination`, HE3 `compute_enrichment`, HE4 `compare_models`, `full_evaluation`, `temporal_stability`, `apply_holm_bonferroni`), `metrics.py` (`evaluate_scores`, `bootstrap_ci`, `precision_at_k`, `enrichment_factor`).
- Key files: `src/fraud_detector/evaluation/hypothesis.py`, `src/fraud_detector/evaluation/metrics.py`

**`src/fraud_detector/scoring/`:**
- Purpose: All single-transaction scoring logic consumed by the FastAPI scorer and direct CLI usage.
- Contains: `scorer.py` (`SingleTransactionScorer` facade), `context.py` (`UserContext` dataclass, `UserContextProvider`), `features.py` (`SingleFeatureCalculator`), `features_enriched.py` (`EnrichedFeatureCalculator` for IF-40 variant), `classifier.py` (`ThresholdClassifier`, `ScoringResult`, `RISK_LEVELS`).
- Key file: `src/fraud_detector/scoring/scorer.py`

**`src/fraud_detector/reporting/`:**
- Purpose: Generate thesis-ready output (LaTeX tables for `output/tables/`, matplotlib figures for `output/figures/`).
- Contains: `latex_tables.py`, `figures.py`.
- Key files: `src/fraud_detector/reporting/latex_tables.py`, `src/fraud_detector/reporting/figures.py`

**`src/fraud_detector/utils/`:**
- Purpose: Shared utilities used across all layers.
- Contains: `logger.py` (Loguru singleton `logger`), `currency.py` (`CurrencyNormalizer`, `normalize_amount_value`, `clickhouse_rate_case` — generates inline SQL CASE for currency normalization at query time).
- Key files: `src/fraud_detector/utils/logger.py`, `src/fraud_detector/utils/currency.py`

**`scorer/`:**
- Purpose: Production-adjacent FastAPI service. Consumes `SingleTransactionScorer` from the library.
- Key files: `scorer/main.py`, `scorer/schemas.py`, `scorer/artifact_loader.py`, `scorer/batch/scorer.py`, `scorer/batch/context_provider.py`

**`scripts/`:**
- Purpose: Standalone runnable scripts for each pipeline phase. Called by `run_pipeline.py` via subprocess for steps 4, 5, 7, 8. Others are utility/experimental.
- Naming: `run_faseN_*.py` = pipeline phase scripts; `eval_*.py` = evaluation experiments; `exp_*.py` = exploratory experiments; `validate_*.py` = validation tools; `hitl_*.py` = human-in-the-loop labeling tools.

**`data/processed/`:**
- Purpose: Parquet files for all temporal splits. These are large (~100-500MB each) and are not committed to git.
- Key files: `data/processed/warm_raw.parquet`, `data/processed/train_raw.parquet`, `data/processed/val_raw.parquet`, `data/processed/test_raw.parquet`, `data/processed/train_features.parquet`, `data/processed/val_features.parquet`, `data/processed/test_features.parquet`.
- Enriched variants (IF-40 experiment): `data/processed/train_features_enriched.parquet`, `data/processed/test_features_enriched.parquet`.

**`output/models/`:**
- Purpose: All serialized model artifacts read by the API and pipeline.
- Key files: `output/models/isolation_forest.joblib` (IF-31, primary), `output/models/isolation_forest_final.joblib` (IF-40 experiment), `output/models/lof.joblib`, `output/models/ocsvm.joblib`, `output/models/scaler.joblib`, `output/models/scaler_final.joblib`, `output/models/feature_engineer.joblib`, `output/models/thresholds.json` (v1), `output/models/thresholds_v2.json` (IF-40), `output/models/model_metadata.json` (drives artifact selection at API startup), `output/models/best_params_if.json`, `output/models/best_params_lof.json`, `output/models/best_params_ocsvm.json`.

**`output/scores/`:**
- Purpose: Scaled numpy matrices and raw score arrays for all splits. Intermediate pipeline artifacts.
- Key files: `output/scores/X_train.npy`, `output/scores/X_val.npy`, `output/scores/X_test.npy`, `output/scores/test_scores.parquet` (model scores + metadata for all test transactions), `output/scores/if_test_scores_final.npy`, `output/scores/lof_test_scores_final.npy`, `output/scores/ocsvm_test_scores_final.npy`.

**`output/` (root):**
- Purpose: JSON result files from evaluation, sensitivity, and experimental runs.
- Key files: `output/results.json` (canonical HE1-HE4), `output/results_sensitivity.json`, `output/results_posthoc.json`, `output/user_risk_profiles.parquet`.
- Experimental snapshots: `output/results_fs_disjoint.json`, `output/results_validation_final.json`, `output/pivot_if40_disjoint_validation.json`.

**`output/tables/` and `output/figures/`:**
- Purpose: Final thesis outputs. Tables are `.tex` files with naming `table_3_NN_*.tex` (thesis chapter 3). Figures are `.pdf` + `.png` pairs with naming `cap2_*.{pdf,png}` (EDA) and descriptive names for model results.

**`tests/`:**
- Purpose: All pytest tests. Flat structure (no subdirectories). Each test file corresponds to a library module.
- Key files: `tests/test_features.py`, `tests/test_evaluation.py`, `tests/test_scoring.py`, `tests/test_trainer.py`, `tests/test_api.py`, `tests/test_batch_scorer.py`, `tests/test_loader.py`, `tests/conftest.py`.

**`config/`:**
- Purpose: Single configuration module with all tunable parameters.
- Key file: `config/config.py` (`Settings` class with `ensure_directories()`, computed path properties).

## Key File Locations

**Entry Points:**
- `run_pipeline.py`: 8-step offline pipeline orchestrator
- `scorer/main.py`: FastAPI application entry point
- `scripts/run_fase6_modeling.py`: Grid search + model training
- `scripts/run_fase7_evaluation.py`: HE1-HE4 + threshold calibration
- `scripts/run_fase8_sensitivity.py`: Sensitivity analysis + SHAP
- `scripts/run_fase9_reporting.py`: LaTeX + figure generation

**Configuration:**
- `config/config.py`: All settings (`Settings` class)
- `.env`: Credentials and environment overrides (not committed)
- `.env.example`: Template for required env vars

**Core Logic:**
- `src/fraud_detector/features/engineering.py`: `FeatureEngineer`, `FEATURE_NAMES`
- `src/fraud_detector/models/trainer.py`: `ModelTrainer`
- `src/fraud_detector/evaluation/hypothesis.py`: HE1-HE4 functions
- `src/fraud_detector/scoring/scorer.py`: `SingleTransactionScorer`
- `scorer/artifact_loader.py`: Artifact versioning contract

**Testing:**
- `tests/`: All tests in flat structure
- `tests/conftest.py`: Shared fixtures

## Naming Conventions

**Files:**
- Library modules: lowercase with underscores (`feature_engineer.py`, `clickhouse_connector.py`)
- Pipeline scripts: `run_faseN_description.py` (N matches pipeline phase number)
- Evaluation experiments: `eval_description.py`
- Exploratory scripts: `exp_description.py`
- Output JSON: `results_description.json` (primary is `results.json`)
- Parquet splits: `{split}_{variant}.parquet` where split ∈ `{warm, train, val, test}`, variant ∈ `{raw, features, features_enriched}`
- Scaled arrays: `X_{split}[_suffix].npy`
- Model artifacts: `{model_name}[_variant].joblib` where variant ∈ `{21, 30, final}`
- LaTeX tables: `table_3_{NN}_{description}.tex` (chapter 3 tables)
- Figures: `cap2_*.{pdf,png}` (chapter 2 EDA), descriptive names for model results

**Feature Set Variants (FS-):**
- `IF-31`: Full 31-feature set (`FEATURE_NAMES` in `engineering.py`) — thesis primary
- `IF-30`: Without F18 (`user_reversal_ratio_30d`) — sensitivity ablation (`FEATURE_NAMES_30`)
- `IF-21`: Core features only (`FEATURE_NAMES_21`, first 21) — ablation groups F/G/H
- `IF-40`: Enriched 40-feature set (`output/models/final_feature_list.json`) — experimental

**Proxy Labels:**
- `strict`: Tipo A only (`status IN ('totally_refunded','refunded_to_credit')`, 6.33%)
- `unified`: OR of all 5 types (A-E, 10.23%) — primary evaluation proxy

## Where to Add New Code

**New feature group (additional features):**
- Add feature names to `FEATURE_NAMES` list at the top of `src/fraud_detector/features/engineering.py`
- Add computation logic inside `FeatureEngineer._compute_*()` following the `_compute_group_A()` pattern
- Add corresponding context fields to `UserContext` in `src/fraud_detector/scoring/context.py`
- Add to `SingleFeatureCalculator.calculate()` in `src/fraud_detector/scoring/features.py`
- Add tests in `tests/test_features.py`

**New hypothesis test (HE5+):**
- Add function to `src/fraud_detector/evaluation/hypothesis.py` following `run_mann_whitney` signature
- Add threshold parameters to `config/config.py` `Settings`
- Extend `full_evaluation()` to call the new test
- Add tests in `tests/test_evaluation.py`

**New evaluation script (standalone experiment):**
- Place in `scripts/eval_{description}.py`
- Add `PROJECT_ROOT` path setup preamble (see any existing `eval_*.py`)
- Write results to `output/results_{description}.json`

**New API endpoint:**
- Add Pydantic schemas to `scorer/schemas.py`
- Add route to appropriate router (`scorer/routers/score.py` or `scorer/routers/model.py`)
- No new routers unless separating a major new concern

**New model type:**
- Add to `ModelTrainer.MODEL_REGISTRY` in `src/fraud_detector/models/trainer.py`
- Add default params in `ModelTrainer._default_params()`

**New reporting table or figure:**
- LaTeX tables: add function to `src/fraud_detector/reporting/latex_tables.py`, output to `output/tables/table_3_{NN}_{name}.tex`
- Figures: add function to `src/fraud_detector/reporting/figures.py`, output to `output/figures/{name}.{pdf,png}`

**Utilities:**
- Shared helpers: `src/fraud_detector/utils/` (add new file or extend `currency.py`/`logger.py`)

## Special Directories

**`data/processed/`:**
- Purpose: Temporal split parquets (warm/train/val/test × raw/features variants).
- Generated: Yes, by `run_pipeline.py` steps 1 and 2.
- Committed: No (in `.gitignore`).

**`output/`:**
- Purpose: All pipeline artifacts (models, scores, results, tables, figures).
- Generated: Yes, by pipeline steps 3-8.
- Committed: No (in `.gitignore`), except some result JSON files may be tracked.

**`output/models/`:**
- Purpose: Serialized sklearn models + scaler + threshold configs consumed by both pipeline and API.
- Generated: Yes, by step 3 (scaler), step 4 (models), step 5 (thresholds).
- Committed: No (binary files, up to 1.6GB for LOF).

**`output/extended/`:**
- Purpose: Extended evaluation outputs (auxiliary tables, figures, frozen baseline artifacts).
- Generated: Yes, by sensitivity and extended evaluation scripts.
- Committed: No.

**`output/revision/`:**
- Purpose: Revision-cycle outputs (alternative models, figures, scores for reviewer responses).
- Generated: Yes.
- Committed: No.

**`output/v4/` and `output/feasibility/`:**
- Purpose: Versioned experiment snapshots (feasibility study, v4 benchmarks).
- Generated: Yes.
- Committed: No.

**`.planning/`:**
- Purpose: SpecOps planning documents (change proposals, implementation plans, codebase maps).
- Generated: Manually via OpenSpec workflow.
- Committed: Yes.

**`htmlcov/`:**
- Purpose: pytest coverage HTML report.
- Generated: Yes, by `make test`.
- Committed: No.

---

*Structure analysis: 2026-07-06*
