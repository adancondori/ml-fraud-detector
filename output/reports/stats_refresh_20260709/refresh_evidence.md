# Evidencia del refresh de artefactos — frame-normalization-v1 (tasks 3.5/3.6)

Fecha de corrida: 2026-07-09 (hora local America/La_Paz), operador: sdd-apply (grupos 3-4).
Repo: ml-fraud-detector `develop` (base `62d0eaf`).

## 1. Snapshot y queries de universo (ClickHouse producción, MCP `mcp-clickhouse`, solo lectura)

**Query de conteo del universo canónico (ventana de train):**

```sql
SELECT now() AS snapshot_utc, count() AS universe_train_rows
FROM pbp_productionDB_optimized.payments FINAL
WHERE _peerdb_is_deleted = 0
  AND payment_method NOT IN ('reversal','free')
  AND user_id != 0
  AND created_at >= '2025-01-01 00:00:00'
  AND created_at < '2025-07-01 00:00:00'
```

Resultado: `snapshot_utc = 2026-07-09 20:34:15`, `universe_train_rows = 3.137.081`.

**Consistencia con el parquet de train** (`data/processed/train_features_enriched.parquet`,
extraído por el loader con los mismos 4 predicados): 3.137.083 filas.
Diferencia: 2 filas (6,4e-7), muy por debajo de la tolerancia del validator (0,1%).
La diferencia se atribuye a merges/updates tardíos del ReplacingMergeTree entre la
extracción (2026-06-13) y el snapshot de hoy.

**Query de facilities (fuente de `iana_tz`):**

```sql
SELECT count() AS live_facilities,
       countIf(tzinfo_identifier = '') AS empty_tzinfo,
       uniq(tzinfo_identifier) AS distinct_iana
FROM pbp_productionDB_optimized.facilities FINAL
WHERE _peerdb_is_deleted = 0
```

Resultado: `live_facilities = 1884`, `empty_tzinfo = 0`, `distinct_iana = 58`.

## 2. Comandos ejecutados

```bash
# Stats candidatas (fetch de facilities.tzinfo_identifier vía clickhouse_connect,
# READ prod, SELECT-only; snapshot persistido en output/revision/facility_iana.parquet)
./venv/bin/python scripts/build_facility_stats.py --fetch-iana \
    --out output/models/candidates/facility_stats_v1_candidate.json

# Recalibración candidata (siempre corrida, decisión humana 2)
./venv/bin/python scripts/calibrate_segmented_thresholds.py \
    --stats output/models/candidates/facility_stats_v1_candidate.json \
    --out output/models/candidates/thresholds_segmented_v1_candidate.json

# Reporte comparativo con evidencia
./venv/bin/python scripts/compare_facility_stats.py \
    --old output/models/facility_stats_v1.json \
    --new output/models/candidates/facility_stats_v1_candidate.json \
    --old-thresholds output/models/thresholds_segmented_v1.json \
    --new-thresholds output/models/candidates/thresholds_segmented_v1_candidate.json \
    --snapshot "2026-07-09T20:34:15Z (ClickHouse now(), MCP mcp-clickhouse)" \
    --count-query "SELECT count() FROM pbp_productionDB_optimized.payments FINAL WHERE _peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND user_id != 0 AND created_at >= '2025-01-01 00:00:00' AND created_at < '2025-07-01 00:00:00'" \
    --universe-rows 3137081 \
    --out-dir output/reports/stats_refresh_20260709
```

## 3. Rutas y hashes SHA-256

### Artefactos previos (baseline, pre-adopción)

| Artefacto | SHA-256 |
|---|---|
| `output/models/facility_stats_v1.json` (old) | `729bf308fe2ff4f73ce05cef38398130ec143f9dbe72e16e8d6938f29c723f4c` |
| `output/models/thresholds_segmented_v1.json` (old, se mantiene) | `712091a371f870f1082a7b2ccc38f8ee8476c377620e575950499eabe210c5ec` |

### Artefactos candidatos

| Artefacto | SHA-256 |
|---|---|
| `output/models/candidates/facility_stats_v1_candidate.json` | `d0ef97420cae8be7a6ea629f759a477e954576353fcae0a885ad61b44b39305f` |
| `output/models/candidates/thresholds_segmented_v1_candidate.json` | `5ea515fd89c69d49d15d671210ebf6df965563cde43475701232128ce1b7f81f` |

### Insumos

| Insumo | SHA-256 |
|---|---|
| `output/revision/facility_iana.parquet` (snapshot tzinfo, 1884 filas, 2026-07-09) | `eb110838613d857e2cf319603ce5ba3079b7833aefaf252cff988a39fe902b4c` |
| `data/processed/train_features_enriched.parquet` | `fdc04c3051111ff4469a2905be81cadad9e1c71548ab6b3a98e608476be4197b` |
| `data/processed/val_features_enriched.parquet` | `5ff5488ccbce5fe466e2330619915041063c3fb02c194813e1db6f5c487cb7e9` |
| `output/golden_set_v0.parquet` (set dorado, 14.831 filas) | `1437668ee35b306fe9c6b3ca65c740dbc403455160cbfcf7ae59534e4259c73a` |

### Artefactos congelados (verificados sin cambio, preflight 0.3)

| Artefacto | SHA-256 (idéntico a preflight) |
|---|---|
| `isolation_forest_frame_v1.joblib` | `e536644265478d87ece903f06d7ea67759adcd13c4d6226ba90455d230b2fa31` |
| `scaler_frame_v1.joblib` | `ff755ef1470dbf62a9d7826dd98cc77eaa1cc04b538f5400ad734de2917e2d01` |
| `feature_list_frame_v1.json` | `cfa5804081eb15b9323541fad47bdf4db1e1f7dea95c111adc70070c7045ed4b` |
| `final_feature_list.json` | `e283326137c127f159315cea1005b4f0e4ddf4c37a6007517d5c4eb81bf1ebb8` |
| `final_feature_list_operational.json` | `da60b6db196129d3de3febf7536a8d32089476186ef2eae516bc78cdbcef885c` |
| `model_metadata_frame_v1.json` | `453a058d955013a08bb707167504b39f10b3a7851eef792aa3170fe4069e7210` |

## 4. Resultado del reporte comparativo

**Veredicto: `material_change = false`** (umbral declarado en el reporte:
`|Δmedian|/median_old > 0.10` estricto en facilities `n >= 1000`, o cambio de
`fallback_level`). Reporte completo: `stats_compare_report.json` /
`stats_compare_summary.md` en este directorio.

- Facilities comunes: 1876 — **cero** deltas de mediana/mean/iqr/n (mismo parquet
  de train, mismo universo efectivo: el loader ya aplicaba `user_id != 0`; lo que
  estaba desalineado era la *declaración* `universe_filter` del artefacto).
- Facilities nuevas: 8 (ids 2003–2010), todas con `fallback_level = "currency"`
  (antes recibían fallback implícito, siguen en fallback → no computan como cambio).
- Cambios de `fallback_level`: 0. Δmedian excedidas: 0.
- `iana_tz` cambia en 25 facilities al pasar del diccionario Rails→IANA a la
  columna `facilities.tzinfo_identifier` (design D6): 22 son zonas de offset
  equivalente (America/Guayaquil→America/Lima, Asia/Dubai→Asia/Muscat,
  Europe/Kyiv→Europe/Kiev) y 3 son correcciones reales
  (1995 New_York→Los_Angeles, 1971 New_York→Denver, 1994 New_York→Kuala_Lumpur;
  las 3 con 0 filas en val).

## 5. Recalibración candidata de thresholds (corrida SIEMPRE, decisión humana 2)

Guardrail PASSED: `p95 = 0.043588 ∈ [0.040, 0.048]`. Resultado:
`binary_threshold = 0.04358835018747599`, `by_facility = 452`, `by_currency = 17`,
`calibration_rows = 1.130.117` — **idéntico al baseline en todos los valores**
(diff profundo por entrada: 0 diferencias; único campo distinto: `built_at`).
El path offline SÍ usa `iana_tz` para las features de hora local; el resultado
idéntico se explica porque los 22 cambios de zona son de offset equivalente y
las 3 correcciones reales corresponden a facilities sin filas en val.

**Conclusión: thresholds SIN CAMBIO** — se adopta solo `facility_stats_v1.json`
y `thresholds_segmented_v1.json` se conserva byte a byte
(`712091a3…`), con esta evidencia como respaldo (scenario acoplados/sin-cambio-material).

## 6. Set dorado y rollback (scenario rollback/restaura-scoring)

Scoring del set dorado (14.831 filas) con `scripts/score_golden_frame_v1.py`
(mismo path offline que la calibración):

| Stats usadas | SHA-256 del array de scores |
|---|---|
| baseline `facility_stats_v1.json` (`729bf308…`) | `afdf7f392f2d13a9f29f9f7bdcbca745d2e653f2eb4af88d39a2e5c4030082ec` |
| candidato (`d0ef9742…`) | `afdf7f392f2d13a9f29f9f7bdcbca745d2e653f2eb4af88d39a2e5c4030082ec` |

Scores **bit-idénticos** (p50=-0.07284725, p95=0.04163226 en ambos): la adopción
no altera el scoring del set dorado, y restaurar el artefacto previo (git)
reproduce exactamente los scores previos. La verificación post-adopción
(re-score con el artefacto restaurado desde git) se registra en la sección 7.

No existe un registro persistido de scores dorados de paridad frame-v1 que
re-registrar (la suite `test_parity_phase1.py` recomputa batch↔RT en vivo desde
el parquet de val); el hash de scores de arriba queda como registro de este refresh.

## 7. Verificación post-adopción (completada en el commit de adopción)

- [x] `facility_stats_v1.json` adoptado == candidato (`d0ef9742…`).
- [x] `thresholds_segmented_v1.json` intacto (`712091a3…`).
- [x] Rollback verificado: `git show HEAD~1:output/models/facility_stats_v1.json`
      restaurado a ruta temporal y re-scoring del set dorado reproduce
      `afdf7f39…` (== sección 6 baseline).
- [x] Artefactos congelados sin diff (sección 3).
