-- Frame-v1 columns (Fase 4 / SHAD-01) applied to a FRESH volume.
-- Mirrors docker/clickhouse/migrations/02_anomaly_scores_frame_v1.sql so a
-- clean `docker compose up` produces the 26-column schema that the scorer's
-- _INSERT_COLUMNS (scorer/batch/scorer.py) requires. Idempotent: IF NOT EXISTS
-- makes it a no-op on volumes that already have these columns.
--
-- Runs after 02_anomaly_scores.sql (init scripts execute in filename order).
-- Keep this file in sync with the migration; migrations/ remains the canonical
-- record of the delta for pre-existing volumes.

ALTER TABLE pbp_productionDB_optimized.anomaly_scores
    ADD COLUMN IF NOT EXISTS calibration_segment LowCardinality(String) DEFAULT '',
    ADD COLUMN IF NOT EXISTS fallback_level       LowCardinality(String) DEFAULT '',
    ADD COLUMN IF NOT EXISTS frame_flags          String DEFAULT '';
