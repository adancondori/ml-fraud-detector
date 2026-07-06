# Architecture

**Analysis Date:** 2026-07-06

## Pattern Overview

**Overall:** Sequential ML Pipeline with two distinct runtime modes (offline batch training vs. online real-time API scoring).

**Key Characteristics:**
- Eight-step linear pipeline orchestrated by a single entry point (`run_pipeline.py`). Each step validates file-based prerequisites before executing.
- `src/fraud_detector/` is the core library; it has no awareness of the API layer. The `scorer/` package is an independent FastAPI application that consumes library artifacts.
- Labels (proxy) are computed at evaluation time only. The library enforces a hard separation: `fit()` / `transform()` never receive or return proxy columns; hypothesis functions receive pre-computed scores plus proxy arrays as separate arguments.
- All model artifacts are serialized to `output/models/` as `.joblib` files and loaded atomically by both the pipeline and the API.

## Layers

**Data Layer:**
- Purpose: Extract raw transactions from ClickHouse and serialize temporal splits to disk.
- Location: `src/fraud_detector/data/`
- Contains: `loader.py` (`DataManager` class, canonical SQL, temporal split logic, proxy assignment), `clickhouse_connector.py` (HTTP connector via `clickhouse-connect`).
- Depends on: `config/config.py`, `utils/currency.py`.
- Used by: `run_pipeline.py` step 1, `scripts/run_fase7_evaluation.py`.

**Feature Engineering Layer:**
- Purpose: Compute 31 temporal and behavioral features per transaction. Fit state learned on training data only; applied to val/test via warm-history carry-over.
- Location: `src/fraud_detector/features/`
- Contains: `engineering.py` (`FeatureEngineer`, `FEATURE_NAMES` list of 31, `FEATURE_NAMES_30`, `FEATURE_NAMES_21`), `preprocessor.py` (`UnsupervisedPreprocessor` wrapping `StandardScaler`).
- Depends on: `utils/currency.py`, `utils/logger.py`.
- Used by: `run_pipeline.py` steps 2 and 3, `scripts/run_fase6_modeling.py`, `scoring/scorer.py`.

**Model Training Layer:**
- Purpose: Unsupervised training (IsolationForest, LOF, OC-SVM) with grid search on val set; multi-seed variability; serialization to `output/models/`.
- Location: `src/fraud_detector/models/`
- Contains: `trainer.py` (`ModelTrainer`, `MODEL_REGISTRY` with three sklearn estimators, grid search with checkpoint/resume).
- Depends on: `evaluation/metrics.py`, `config/config.py`.
- Used by: `scripts/run_fase6_modeling.py` (invoked by `run_pipeline.py` step 4 via subprocess).

**Evaluation Layer:**
- Purpose: Compute hypothesis test results (HE1-HE4) and statistical metrics against proxy labels. Never touches training or model fitting.
- Location: `src/fraud_detector/evaluation/`
- Contains: `hypothesis.py` (HE1 Mann-Whitney U + rank-biserial, HE2 AUC-ROC/AP, HE3 enrichment factor, HE4 model comparison, `apply_holm_bonferroni`, `temporal_stability`), `metrics.py` (`evaluate_scores`, `bootstrap_ci`, `precision_at_k`, `enrichment_factor`).
- Depends on: `config/config.py`, `utils/logger.py`.
- Used by: `scripts/run_fase7_evaluation.py`, `scripts/run_fase8_sensitivity.py`.

**Scoring Layer (library):**
- Purpose: Real-time single-transaction scoring. Reuses the fitted `FeatureEngineer` artifacts to compute features from a raw payment dict, then scores with the loaded model.
- Location: `src/fraud_detector/scoring/`
- Contains: `scorer.py` (`SingleTransactionScorer` — main facade), `context.py` (`UserContext` dataclass, `UserContextProvider` — fetches rolling aggregates from ClickHouse per-user), `features.py` (`SingleFeatureCalculator`), `features_enriched.py` (`EnrichedFeatureCalculator` for IF-40 variant), `classifier.py` (`ThresholdClassifier`, `ScoringResult`, `RISK_LEVELS` percentile bands).
- Depends on: `features/engineering.py`, `utils/currency.py`, `utils/logger.py`.
- Used by: `scorer/` FastAPI application.

**Reporting Layer:**
- Purpose: Generate LaTeX tables and figures for the thesis manuscript.
- Location: `src/fraud_detector/reporting/`
- Contains: `latex_tables.py`, `figures.py`.
- Depends on: `output/results*.json`, `output/scores/`.
- Used by: `scripts/run_fase9_reporting.py` (invoked by `run_pipeline.py` step 8 via subprocess).

**API Layer (scorer/):**
- Purpose: FastAPI service for real-time and batch scoring in production-adjacent environments.
- Location: `scorer/`
- Contains: `main.py` (FastAPI app, lifespan startup, two ClickHouse clients: READ=production, WRITE=local), `schemas.py` (Pydantic models: `ScoreRequest`, `ScoreResponse`, `BatchScoreRequest`, `BatchScoreResponse`, `CriticalAlert`), `artifact_loader.py` (versioned artifact loading with feature-count validation), `dependencies.py` (shared `_state` dict, FastAPI `Depends` accessors), `routers/score.py` (`POST /api/v1/score`, `POST /api/v1/score/batch`), `routers/model.py` (`GET /api/v1/health`, `GET /api/v1/model/info`, `POST /api/v1/model/reload`), `batch/scorer.py` (`BatchScorer` — bulk fetch+score+insert), `batch/context_provider.py` (`BatchContextProvider` — 6 ClickHouse queries per chunk regardless of batch size).

**Configuration Layer:**
- Purpose: Single source of truth for all pipeline parameters, dataset boundaries, hypothesis thresholds, and ClickHouse credentials.
- Location: `config/config.py`
- Contains: `Settings` (Pydantic `BaseSettings`, loaded from `.env`). Provides computed path properties (`processed_dir`, `scores_dir`, `models_output_dir`).

**Utilities:**
- Location: `src/fraud_detector/utils/`
- Contains: `logger.py` (Loguru singleton), `currency.py` (`CurrencyNormalizer`, `normalize_amount_value`, `clickhouse_rate_case` — inline SQL CASE expression for 21 currencies across 13 gateways).

## Data Flow

**Offline Training Pipeline (8 steps):**

1. `run_pipeline.py` step 1 → `DataManager.extract_from_clickhouse()` fetches 4 temporal splits (`warm_raw`, `train_raw`, `val_raw`, `test_raw`) from ClickHouse using `CANONICAL_SQL` with `FINAL` modifier → writes to `data/processed/`
2. Step 2 → `FeatureEngineer.fit(warm+train)` then `transform()` with warm-history carry-over for val and test → writes `train_features.parquet`, `val_features.parquet`, `test_features.parquet` to `data/processed/`; saves `feature_engineer.joblib` to `output/models/`
3. Step 3 → `UnsupervisedPreprocessor.fit_transform(train)` + `transform(val, test)` → writes `X_train.npy`, `X_val.npy`, `X_test.npy` to `output/scores/`; saves `scaler.joblib`
4. Step 4 → `scripts/run_fase6_modeling.py` grid search on val, trains IF-31/IF-30/IF-21/LOF/OC-SVM → writes `isolation_forest.joblib`, `lof.joblib`, `ocsvm.joblib`, `best_params_*.json` to `output/models/`
5. Step 5 → `scripts/run_fase7_evaluation.py` scores test set with all models; assigns proxy labels; runs HE1-HE4 with bootstrap CI → writes `output/scores/test_scores.parquet`, `output/results.json`, `output/models/thresholds.json`
6. Step 6 → Summary display only (reads `output/results.json`)
7. Step 7 → `scripts/run_fase8_sensitivity.py` runs sensitivity analysis (F18 ablation, proxy-type sensitivity, post-hoc) → writes `output/results_sensitivity.json`, `output/results_posthoc.json`
8. Step 8 → `scripts/run_fase9_reporting.py` generates LaTeX tables and figures → writes to `output/tables/`, `output/figures/`

**Real-Time Single Scoring (API):**

1. `POST /api/v1/score` receives `ScoreRequest` (raw payment fields)
2. `SingleTransactionScorer.score(payment)` queries `UserContextProvider` → 5-6 ClickHouse queries on production READ client for rolling aggregates → builds `UserContext`
3. `SingleFeatureCalculator.calculate(payment, context)` assembles 31-dim feature vector
4. `SingleTransactionScorer.score_features()` → scales with `scaler.joblib`, calls `score_samples()` or `decision_function()` on `isolation_forest.joblib` (negated, so higher = more anomalous)
5. `ThresholdClassifier.classify(raw_score)` → `is_anomaly`, `risk_level` (minimal/low/medium/high/critical), `percentile`
6. Top-5 features by absolute z-score returned as explanation factors
7. Returns `ScoreResponse`

**Batch Scoring (API):**

1. `POST /api/v1/score/batch` receives cursor `datetime`
2. `BatchScorer.score_batch(cursor)` fetches payments from READ (production) ClickHouse in a closed window `[cursor, cursor+window]`
3. `BatchContextProvider.get_batch_context(payments)` executes exactly 6 ClickHouse queries per chunk via VALUES JOIN (never 6×N)
4. Each payment scored via `SingleTransactionScorer` internals (reusing loaded artifacts)
5. Scored rows INSERTed into `anomaly_scores` table on WRITE (local) ClickHouse in 10K chunks with dedup tokens
6. Guardrail (`assert_write_target_is_safe`) aborts INSERT if WRITE fingerprint matches READ fingerprint or WRITE host is non-local without explicit bypass flag
7. Returns `BatchScoreResponse` with counts, critical alerts, and `next_cursor`

**State Management:**
- Pipeline state is file-based: each step reads input files and writes output files. No in-memory state between steps.
- API state is managed via `scorer/dependencies.py`'s `_state` dict, populated at lifespan startup. Model reload (`POST /model/reload`) atomically replaces the scorer in `_state`.

## Key Abstractions

**FeatureEngineer:**
- Purpose: Stateful transformer that computes all 31 features. Persists per-user rolling state across temporal splits via `get_feature_state()` / `transform_with_warm_history()`.
- File: `src/fraud_detector/features/engineering.py`
- Pattern: `fit(df) → transform(df) → save(path)` / `load(path)`.

**ModelTrainer:**
- Purpose: Wraps any of three sklearn estimators behind a consistent training API with higher-always-anomalous score convention.
- File: `src/fraud_detector/models/trainer.py`
- Pattern: Registry-based instantiation; negated `decision_function`/`score_samples` used uniformly.

**SingleTransactionScorer:**
- Purpose: Facade that chains context fetch → feature calculation → model scoring → threshold classification → factor explanation for a single payment dict.
- File: `src/fraud_detector/scoring/scorer.py`
- Pattern: Constructed once per API lifecycle; shared across requests via FastAPI DI.

**Artifacts:**
- Purpose: Frozen dataclass carrying `(model, scaler, feature_list, thresholds, metadata)` loaded atomically at API startup. Feature-count validation prevents mismatched model/scaler/feature-list combinations.
- File: `scorer/artifact_loader.py`

**HE1-HE4 Functions:**
- Purpose: Stateless functions that receive `(scores: np.ndarray, proxy: np.ndarray)` and return structured result dicts. Hypothesis pass/fail thresholds come from `config/config.py`.
- File: `src/fraud_detector/evaluation/hypothesis.py`

## Entry Points

**run_pipeline.py:**
- Location: `run_pipeline.py` (project root)
- Triggers: CLI (`python run_pipeline.py [--step N] [--from-step N] [--fast] [--dry-run]`)
- Responsibilities: Validates file-based prerequisites for each step; runs steps 1-8 in sequence; prints hypothesis pass/fail summary. Steps 4, 5, 7, 8 invoke sub-scripts via `subprocess.run()`.

**FastAPI Application:**
- Location: `scorer/main.py`
- Triggers: `uvicorn scorer.main:app` (or Docker)
- Responsibilities: Loads artifacts + ClickHouse clients at startup; registers two routers; exposes `GET /api/v1/health`, `GET /api/v1/model/info`, `POST /api/v1/model/reload`, `POST /api/v1/score`, `POST /api/v1/score/batch`.

**Modeling Sub-script:**
- Location: `scripts/run_fase6_modeling.py`
- Triggers: Subprocess call from `run_pipeline.py` step 4, or direct CLI with `--step` flag.
- Responsibilities: Grid search on val set, final training of all 3 models in 3 variants (IF-31, IF-30, IF-21), multi-seed variability.

**Evaluation Sub-script:**
- Location: `scripts/run_fase7_evaluation.py`
- Triggers: Subprocess call from `run_pipeline.py` step 5, or direct CLI.
- Responsibilities: Scores test set, computes HE1-HE4 with Holm-Bonferroni, bootstrap CI, temporal stability, model comparison; calibrates and writes `thresholds.json`.

## Error Handling

**Strategy:** Fail fast with explicit exceptions. Prerequisites are validated at the start of each pipeline step by checking file existence; missing files raise `FileNotFoundError` with a clear message.

**Patterns:**
- `UnsupervisedPreprocessor` raises `ValueError` if called before `fit()` or if input contains NaN/inf.
- `FeatureEngineer.FEATURE_NAMES` length is validated at import time (module-level assert).
- `artifact_loader.load_artifacts()` validates feature-count consistency between model, scaler, and feature list; raises `ValueError` on mismatch.
- `BatchScorer.assert_write_target_is_safe()` raises `ValueError` before any INSERT if the write guardrail fails — prevents accidental writes to the production ClickHouse.

## Cross-Cutting Concerns

**Logging:** Loguru singleton from `src/fraud_detector/utils/logger.py`. All library and script code imports `from fraud_detector.utils.logger import logger`. Log format (text/json) and level are controlled by `config/config.py`.

**Validation:** Pydantic `BaseSettings` for config; Pydantic models for all API request/response schemas. Feature-set integrity validated at import time and at artifact load time.

**Temporal Leakage Prevention:** `fit()` only on training data; rolling windows exclude the current row; warm-history carry-over uses tail of the previous split to pre-populate rolling state for val/test. `FINAL` modifier on all ClickHouse queries prevents reading uncommitted replicated rows.

**Score Convention:** Higher anomaly score = more anomalous, enforced uniformly by negating `decision_function` or `score_samples` in both `ModelTrainer` and `SingleTransactionScorer.score_features()`.

**READ/WRITE ClickHouse Separation:** The API maintains two separate clients — READ (production, read-only) and WRITE (local docker) — with a fingerprint-based guardrail that aborts any INSERT unless WRITE != READ and WRITE host is local.

---

*Architecture analysis: 2026-07-06*
