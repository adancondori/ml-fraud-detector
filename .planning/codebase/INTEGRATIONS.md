# External Integrations

**Analysis Date:** 2026-07-06

## APIs & External Services

**ML Scorer (outgoing — consumed by Rails platform):**
- FastAPI microservice running at port 8000 (container) / 8765 (host)
- Consumed by Rails `packs/anomaly_detection/app/clients/anomaly_detection/fraud_scorer_client.rb`
- Rails uses Faraday with retry (`max: 2, backoff_factor: 2`) and `FRAUD_SCORER_URL` env var
- No authentication between Rails and scorer (trusted internal network via Docker bridge `pbc-network`)

## Data Storage

**Databases — ClickHouse (primary):**

Two separate connections at scorer runtime (READ vs. WRITE separation):

| Connection | Role | Env vars | Notes |
|------------|------|----------|-------|
| READ | Production ClickHouse Cloud (read-only) | `CLICKHOUSE_HOST/PORT/USER/PASSWORD/DATABASE/SECURE` | All queries; `CLICKHOUSE_SECURE=true` for Cloud; `autogenerate_session_id=False` |
| WRITE | Local Docker ClickHouse | `ANOMALY_SCORES_CH_HOST/PORT/USER/PASSWORD/DATABASE/SECURE` | Only for INSERT to `anomaly_scores`; guarded by `assert_write_target_is_safe` |

**ClickHouse database:** `pbp_productionDB_optimized`

**Tables read (production, READ client):**
- `pbp_productionDB_optimized.payments FINAL` — primary source; 6.78M+ rows; `SharedReplacingMergeTree`; `FINAL` keyword mandatory to deduplicate
- `pbp_productionDB_optimized.facilities_users FINAL` — user role lookup (staff/player)
- `pbp_productionDB_optimized.users FINAL` — user account age (`created_at`)
- `default.exchange_rates` — currency conversion rates (optional; fallback table of 20 currencies hardcoded in `src/fraud_detector/utils/currency.py`)

**Table written (local WRITE client):**
- `pbp_productionDB_optimized.anomaly_scores` — DDL at `docker/clickhouse/init/02_anomaly_scores.sql`
  - Engine: `MergeTree()` (append-only), partitioned by `toYYYYMM(payment_created_at)`, 12-month TTL
  - Idempotency via `insert_deduplication_token` (deterministic token: `batch-{cursor_start}-{cursor_end}-{model_version}-chunk-{N}`)
  - INSERT columns: `payment_id`, `facility_id`, `facility_name`, `user_id`, `scored_at`, `payment_created_at`, `amount_usd`, `raw_score`, `percentile`, `risk_level`, `is_anomaly`, `model_version`, `top_factors`, `features_json`, `scoring_mode`, `feature_version`, `threshold_version`, `latency_ms`, `error`, `gateway`, `payment_method`, `currency`, `source_enum`
  - Chunks of 10,000 rows per INSERT call

**ClickHouse connection pattern:**
- Training pipeline: `src/fraud_detector/data/clickhouse_connector.py` — `ClickHouseConnector` class wrapping `clickhouse_connect.get_client()`; supports `query_to_dataframe()` and `query_to_dataframe_chunked(chunk_size=100_000)`
- Scorer (single): `src/fraud_detector/scoring/context.py` — `UserContextProvider` runs 5–8 queries per transaction (velocity, behavior, credit, diversity, user info, role, same-amount, gateway patterns)
- Scorer (batch): `scorer/batch/context_provider.py` — `BatchContextProvider` uses VALUES JOIN strategy to run exactly 6 queries per chunk of up to 2000 `(user_id, facility_id)` pairs regardless of batch size

**SQL safety constraint — FINAL keyword:**
All queries against `payments`, `users`, `facilities_users` use `FINAL` modifier. The `payments` table is `SharedReplacingMergeTree`; omitting `FINAL` returns stale/duplicate rows. This is enforced in:
- `CANONICAL_SQL` in `src/fraud_detector/data/loader.py`
- `_FETCH_SQL` and `_CURSOR_END_SQL` in `scorer/batch/scorer.py`
- All inline SQL in `scorer/batch/context_provider.py`
- All `VELOCITY_SQL`, `BEHAVIOR_SQL`, etc. in `src/fraud_detector/scoring/context.py`

**Databases — MySQL (one-off extraction only):**
- Rails production MySQL replica (MySQL 8.0)
- Used only by `scripts/extract_token_features.py` to read `user_tokens(id, user_id, created_at, is_default)` from database `paybycourtDB`
- Client: SQLAlchemy + PyMySQL (`mysql+pymysql://`)
- Connection: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` env vars
- NOT a runtime dependency of the scorer or main pipeline; one-off script for token-feature experiments

**File Storage:**
- Local filesystem only: parquet files at `data/processed/{split}_raw.parquet` (warm, train, val, test splits); model artifacts at `output/models/`; figures at `output/figures/`; LaTeX tables at `output/tables/`

**Caching:**
- None. All context queries hit ClickHouse live. No Redis, Memcached, or in-process cache layer.

## Authentication & Identity

**Auth Provider:**
- None for the scorer API itself (internal-only service, no auth middleware)
- ClickHouse connections use username/password credentials via env vars (`CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`)

## Monitoring & Observability

**Error Tracking:**
- Not configured in this service (no Sentry, Honeybadger, or New Relic SDK)
- Rails platform uses Honeybadger + New Relic but that is outside this service

**Logs:**
- loguru with JSON format when `LOG_FORMAT=json` (default in `.env.example`)
- Level controlled via `LOG_LEVEL` env var (default: `INFO`)
- Scored-transaction errors logged as `WARNING` and persisted in the `error` column of `anomaly_scores`

## CI/CD & Deployment

**Hosting:**
- Development/research: local venv on macOS (Python 3.9/3.10)
- Integration run: Docker Compose (`docker-compose.yml`) with two services: `clickhouse` and `ml-scorer`
- `ml-scorer` container: Python 3.12-slim, built from `docker/scorer/Dockerfile` (multi-stage builder/runtime)
- Exposed port: `127.0.0.1:8765:8000` (scorer), `127.0.0.1:8123:8123` and `127.0.0.1:9000:9000` (ClickHouse)
- Docker network: `pbc-network` (external, shared with Rails platform stack)

**CI Pipeline:**
- `.github/` directory present; workflows not inspected in detail
- Pre-commit hooks: `.pre-commit-config.yaml`

## Scorer API Contract

**Base URL:** `http://ml-scorer:8000` (Docker) or `http://localhost:8765` (host)
**Router prefix:** `/api/v1`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | `HealthResponse` — model_loaded, clickhouse_connected (READ+WRITE), model_version, last_batch_at |
| GET | `/api/v1/model/info` | `ModelInfoResponse` — feature_count, threshold, risk_levels |
| POST | `/api/v1/model/reload` | Hot-reload model artifacts from `MODEL_DIR` |
| POST | `/api/v1/score` | `ScoreRequest` → `ScoreResponse` (single transaction) |
| POST | `/api/v1/score/batch` | `BatchScoreRequest{cursor: datetime}` → `BatchScoreResponse{processed, scored, critical_alerts, next_cursor}` |

**Schema definitions:** `scorer/schemas.py`

**Risk levels returned:** `minimal`, `low`, `medium`, `high`, `critical`

**Shadow model:** When `SHADOW_MODEL_DIR` is set and `SCORING_MODE=shadow`, scoring runs against an alternate model. `SCORING_MODE` is written to the `scoring_mode` column in `anomaly_scores`.

## Rails Integration (Inbound Caller)

**Client code:** `platform/packs/anomaly_detection/app/clients/anomaly_detection/fraud_scorer_client.rb`
- Uses Faraday gem with retry middleware
- `FRAUD_SCORER_URL` env var (default: `http://ml-scorer:8000`)
- `FRAUD_SCORER_TIMEOUT` env var for batch calls (default: 5s)
- `open_timeout`: 2s

**Trigger — real-time single score:**
- `Payment#after_commit :score_payment_realtime, on: :create` in `platform/app/models/payment.rb`
- Calls `AnomalyDetection::RealTimeScoringService.call(payment: self)`

**Trigger — batch score:**
- `AnomalyDetection::BatchScoringService` (in `platform/packs/anomaly_detection/`)
- Passes `cursor` (ISO8601 datetime) to `POST /api/v1/score/batch`
- Uses `next_cursor` from response to advance the cursor on next call

## Webhooks & Callbacks

**Incoming:** None — the scorer has no webhook receivers
**Outgoing:** None — the scorer returns critical alerts synchronously in `BatchScoreResponse`; no push notifications

## Environment Configuration

**Required env vars (training pipeline):**
```
CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD
CLICKHOUSE_DATABASE=pbp_productionDB_optimized
CLICKHOUSE_SECURE=true
RANDOM_SEED=42
N_JOBS=-1
```

**Required env vars (scorer container, additional):**
```
MODEL_DIR=/app/output/models
ANOMALY_SCORES_CH_HOST=clickhouse
ANOMALY_SCORES_CH_PORT=8123
ANOMALY_SCORES_CH_USER=default
ANOMALY_SCORES_CH_DATABASE=pbp_productionDB_optimized
ANOMALY_SCORES_TABLE=pbp_productionDB_optimized.anomaly_scores
ALLOW_NONLOCAL_ANOMALY_SCORE_WRITES=false
SCORING_MODE=shadow   # or 'active'
```

**Optional env vars:**
```
SHADOW_MODEL_DIR=     # directory with alternate model_metadata.json
MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE   # extract_token_features.py only
```

**Secrets location:**
- `.env` file (gitignored in development, never committed)
- `.env.example` is committed and documents all vars without values

---

*Integration audit: 2026-07-06*
