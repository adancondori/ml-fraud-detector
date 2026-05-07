-- anomaly_scores: Stores per-transaction anomaly scores from the Isolation Forest model.
-- Engine: MergeTree (append-only). Idempotency handled via insert_deduplication_token at INSERT time.
-- Partitioned monthly by payment_created_at with 12-month TTL for automatic cleanup.

CREATE TABLE IF NOT EXISTS pbp_productionDB_optimized.anomaly_scores
(
    payment_id         UInt64,
    facility_id        UInt32,
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
    features_json      String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(payment_created_at)
ORDER BY (facility_id, payment_created_at, payment_id)
TTL payment_created_at + INTERVAL 12 MONTH
SETTINGS index_granularity = 8192,
         non_replicated_deduplication_window = 1000;
