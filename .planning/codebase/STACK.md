# Technology Stack

**Analysis Date:** 2026-07-06

## Languages

**Primary:**
- Python 3.9+ — entire pipeline (training, feature engineering, evaluation, scorer API)

**Secondary:**
- SQL (ClickHouse dialect) — embedded in Python source files as string constants
- Shell (bash) — Makefile targets and Docker init scripts

## Runtime

**Environment:**
- CPython 3.9.6 — `venv/` (active development venv created against Xcode Python)
- CPython 3.10.19 — `.venv/` (uv-managed secondary venv, `uv = 0.9.27`)
- CPython 3.12-slim — Docker image used by `ml-scorer` container in production (`docker/scorer/Dockerfile`)

**Package Manager:**
- pip — primary, via `requirements*.txt` files
- uv 0.9.27 — secondary venv at `.venv/` managed by uv; lockfile at `uv.lock`
- Lockfile: `uv.lock` present (committed)

## Frameworks

**Serving (scorer):**
- FastAPI `0.124.4` — REST API for real-time and batch scoring (`scorer/main.py`)
- uvicorn `0.38.0` — ASGI server; `CMD ["uvicorn", "scorer.main:app", "--host", "0.0.0.0", "--port", "8000"]`
- httpx `0.28.1` — HTTP client (listed in `requirements-scorer.txt`, available for health checks)

**ML / Modeling:**
- scikit-learn `>=1.3.0` (training), pinned `==1.6.1` in scorer container — `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `StandardScaler`
- numpy `>=1.24.0,<2.0.0` — feature arrays, score computation
- pandas `>=2.0.0` — DataFrame operations, parquet I/O

**Configuration:**
- pydantic `>=2.5.0` (training), pinned `==2.12.5` in scorer — all Settings classes
- pydantic-settings `>=2.1.0` (training), pinned `==2.11.0` in scorer — env-file loading via `BaseSettings`

**Data:**
- clickhouse-connect `>=0.7.0` (training), pinned `==0.10.0` in scorer — HTTP(S) client for ClickHouse Cloud

**Serialization:**
- joblib `>=1.3.0` — model artifact serialization (`.joblib` files)
- pyarrow (via pandas) — parquet read/write with snappy compression

**Explainability:**
- shap `>=0.42.0` — SHAP value computation (`scripts/compute_shap_importance.py`)

**Visualization / Reporting:**
- matplotlib `>=3.7.0` — figures
- seaborn `>=0.12.0` — statistical plots
- LaTeX tables generated via `src/fraud_detector/reporting/latex_tables.py`

**Logging:**
- loguru `>=0.7.0` — structured JSON logging across all modules; format controlled via `LOG_FORMAT` env var

**Progress:**
- tqdm `>=4.65.0` — progress bars in batch operations

**Testing:**
- pytest `>=7.4.0`
- pytest-cov `>=4.1.0`

**Code Quality:**
- black `>=23.0.0` — line-length 100
- isort `>=5.12.0`
- flake8 `>=6.0.0`
- mypy `>=1.5.0`
- pre-commit `>=3.5.0` — config at `.pre-commit-config.yaml`

**MySQL (one-off extraction script only):**
- sqlalchemy + pymysql — used only in `scripts/extract_token_features.py` to read `user_tokens` from the Rails MySQL replica; NOT a runtime dependency of the scorer

## Key Dependencies

**Critical (pickle compatibility constraint):**
- scikit-learn version is pinned to `==1.6.1` in scorer and must match training environment to load `.joblib` artifacts; declared explicitly in `requirements-scorer.txt`

**Infrastructure:**
- clickhouse-connect — sole database client; used via both `ClickHouseConnector` wrapper (`src/fraud_detector/data/clickhouse_connector.py`) and raw `clickhouse_connect.get_client()` in scorer startup
- joblib — model serialization; artifacts at `output/models/isolation_forest.joblib`, `scaler.joblib`, `feature_engineer.joblib`
- pydantic-settings — single source of truth for all configuration (`config/config.py`)

## Configuration

**Environment:**
- `.env` file loaded by `pydantic_settings.BaseSettings` (`env_file=".env"`)
- `.env.example` documents all required vars (copy to `.env` via `make setup`)
- `extra="ignore"` — unknown env vars are silently dropped, preventing errors from legacy vars

**Critical env vars:**
```
CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD
CLICKHOUSE_DATABASE, CLICKHOUSE_SECURE
ANOMALY_SCORES_CH_HOST, ANOMALY_SCORES_CH_PORT (WRITE target — local)
ANOMALY_SCORES_TABLE
MODEL_DIR
ALLOW_NONLOCAL_ANOMALY_SCORE_WRITES   # off by default; safeguard
SHADOW_MODEL_DIR                       # optional shadow model
SCORING_MODE                           # 'active' or 'shadow'
```

**Settings singleton:**
- `from config.config import settings` — importable singleton used across all pipeline modules
- `settings.ensure_directories()` creates all output dirs at runtime

**Build:**
- `pyproject.toml` — build system (`setuptools>=68.0`), black/isort/mypy/pytest configuration
- `setup.py` — editable install: `pip install -e .`; package root at `src/`

## Platform Requirements

**Development:**
- Python 3.9+ (venv or uv)
- ClickHouse Cloud or local Docker ClickHouse (see `docker-compose.yml`)
- `make install-dev` + `make setup` to bootstrap

**Production (scorer container):**
- Docker with Python 3.12-slim
- Port 8000 inside container, mapped to 8765 on host via `docker-compose.yml`
- Two ClickHouse connections at startup: READ (prod cloud) and WRITE (local Docker container)
- Model artifacts mounted at `MODEL_DIR=/app/output/models`

---

*Stack analysis: 2026-07-06*
