ALTER TABLE pbp_productionDB_optimized.anomaly_scores
    ADD COLUMN IF NOT EXISTS scoring_mode LowCardinality(String) DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS feature_version LowCardinality(String) DEFAULT 'base-31',
    ADD COLUMN IF NOT EXISTS threshold_version LowCardinality(String) DEFAULT 'v1',
    ADD COLUMN IF NOT EXISTS latency_ms Float32 DEFAULT 0,
    ADD COLUMN IF NOT EXISTS error String DEFAULT '';
