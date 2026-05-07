#!/bin/bash
set -e

echo "=== Verifying anomaly_scores deduplication via insert_deduplication_token SETTINGS ==="

# Insert a test row with a dedup token via SETTINGS
# Note: In ClickHouse, SETTINGS must appear after table name and before VALUES/FORMAT/SELECT.
# Column list cannot follow SETTINGS — use positional VALUES in table column order.
clickhouse-client --query "
  INSERT INTO pbp_productionDB_optimized.anomaly_scores
  SETTINGS insert_deduplication_token='dedup-test-token-phase1'
  VALUES (99999999, 1, 1, '2025-06-15 10:00:00', '2025-06-15 10:00:00',
          25.50, 0.85, 0.97, 'critical', 1, 'if-31-v1-test', '[]', '{}')
"

# Insert a DIFFERENT row with the SAME dedup token — this should be deduplicated
clickhouse-client --query "
  INSERT INTO pbp_productionDB_optimized.anomaly_scores
  SETTINGS insert_deduplication_token='dedup-test-token-phase1'
  VALUES (99999999, 1, 1, '2025-06-15 11:00:00', '2025-06-15 10:00:00',
          99.99, 0.50, 0.60, 'low', 0, 'if-31-v1-test', '[]', '{}')
"

# Count — should be 1 because insert_deduplication_token prevents the second insert
COUNT=$(clickhouse-client --query "
  SELECT count() FROM pbp_productionDB_optimized.anomaly_scores
  WHERE payment_id = 99999999
")

if [ "$COUNT" -eq 1 ]; then
  echo "DEDUP OK: insert_deduplication_token SETTINGS prevented duplicate — $COUNT row (expected 1)"
else
  echo "DEDUP FAIL: Got $COUNT rows — insert_deduplication_token SETTINGS did not prevent duplicate insert"
  # Clean up before failing
  clickhouse-client --query "
    ALTER TABLE pbp_productionDB_optimized.anomaly_scores
    DELETE WHERE payment_id = 99999999
  "
  exit 1
fi

# Clean up test data
clickhouse-client --query "
  ALTER TABLE pbp_productionDB_optimized.anomaly_scores
  DELETE WHERE payment_id = 99999999
"

echo "=== Deduplication verification complete ==="
