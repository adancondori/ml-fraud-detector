-- anomaly_scores: Stores per-transaction anomaly scores from the Isolation Forest model.
-- Engine: MergeTree (append-only). Idempotency handled via insert_deduplication_token at INSERT time.
-- Partitioned monthly by payment_created_at with 12-month TTL for automatic cleanup.

CREATE TABLE IF NOT EXISTS pbp_productionDB_optimized.anomaly_scores
(
    payment_id         UInt64,
    facility_id        UInt32,
    facility_name      LowCardinality(String) DEFAULT '',
    user_id            UInt64,
    scored_at          DateTime,
    payment_created_at DateTime,
    amount_usd         Float32,
    raw_score          Float32,
    percentile         Float32,
    risk_level         LowCardinality(String),
    is_anomaly         UInt8,
    model_version      LowCardinality(String),
    top_factors        String,
    features_json      String,
    scoring_mode       LowCardinality(String) DEFAULT 'active',
    feature_version    LowCardinality(String) DEFAULT 'base-31',
    threshold_version  LowCardinality(String) DEFAULT 'v1',
    latency_ms         Float32 DEFAULT 0,
    error              String DEFAULT '',
    gateway            LowCardinality(String) DEFAULT '',
    payment_method     LowCardinality(String) DEFAULT '',
    currency           LowCardinality(String) DEFAULT '',
    source_enum        LowCardinality(String) DEFAULT ''
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(payment_created_at)
ORDER BY (facility_id, payment_created_at, payment_id)
TTL payment_created_at + INTERVAL 12 MONTH
SETTINGS index_granularity = 8192,
         non_replicated_deduplication_window = 1000;
