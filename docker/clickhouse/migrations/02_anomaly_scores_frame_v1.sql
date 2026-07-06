ALTER TABLE pbp_productionDB_optimized.anomaly_scores
    ADD COLUMN IF NOT EXISTS calibration_segment LowCardinality(String) DEFAULT '',
    ADD COLUMN IF NOT EXISTS fallback_level       LowCardinality(String) DEFAULT '',
    ADD COLUMN IF NOT EXISTS frame_flags          String DEFAULT '';
