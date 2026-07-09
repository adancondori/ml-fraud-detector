# Reporte comparativo de facility stats (old vs new)

- Generado: 2026-07-09T20:37:06Z
- **material_change: false**
- Umbral: |Δmedian|/median_old > 0.1 (estrictamente mayor) en facilities con n >= 1000, o cambio de fallback_level

## Resumen

| Facilities comunes | Nuevas | Removidas | Cambios fallback | Δmedian excedidas |
|---|---|---|---|---|
| 1876 | 8 | 0 | 0 | 0 |

## Evidencia de procedencia

- **snapshot**: 2026-07-09T20:34:15Z (ClickHouse now(), MCP mcp-clickhouse)
- **count_query**: SELECT count() FROM pbp_productionDB_optimized.payments FINAL WHERE _peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND user_id != 0 AND created_at >= '2025-01-01 00:00:00' AND created_at < '2025-07-01 00:00:00'
- **universe_rows**: 3137081
- **train_rows_old**: 3137083
- **train_rows_new**: 3137083
- **old_stats_path**: output/models/facility_stats_v1.json
- **old_stats_sha256**: 729bf308fe2ff4f73ce05cef38398130ec143f9dbe72e16e8d6938f29c723f4c
- **new_stats_path**: output/models/candidates/facility_stats_v1_candidate.json
- **new_stats_sha256**: d0ef97420cae8be7a6ea629f759a477e954576353fcae0a885ad61b44b39305f
- **command**: scripts/compare_facility_stats.py --old output/models/facility_stats_v1.json --new output/models/candidates/facility_stats_v1_candidate.json --old-thresholds output/models/thresholds_segmented_v1.json --new-thresholds output/models/candidates/thresholds_segmented_v1_candidate.json --snapshot '2026-07-09T20:34:15Z (ClickHouse now(), MCP mcp-clickhouse)' --count-query 'SELECT count() FROM pbp_productionDB_optimized.payments FINAL WHERE _peerdb_is_deleted=0 AND payment_method NOT IN ('"'"'reversal'"'"','"'"'free'"'"') AND user_id != 0 AND created_at >= '"'"'2025-01-01 00:00:00'"'"' AND created_at < '"'"'2025-07-01 00:00:00'"'"'' --universe-rows 3137081 --out-dir output/reports/stats_refresh_20260709
- **old_thresholds_path**: output/models/thresholds_segmented_v1.json
- **old_thresholds_sha256**: 712091a371f870f1082a7b2ccc38f8ee8476c377620e575950499eabe210c5ec
- **new_thresholds_path**: output/models/candidates/thresholds_segmented_v1_candidate.json
- **new_thresholds_sha256**: 5ea515fd89c69d49d15d671210ebf6df965563cde43475701232128ce1b7f81f
