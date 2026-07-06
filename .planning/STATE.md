# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-06)

**Core value:** El ranking de anomalías refleja comportamiento relativo al contexto de la facility (moneda, escala, hora local), no tamaño nominal ni artefactos UTC — medido por reducción de sesgo (top-5% monto <4×, off-hours local ~4–5%), no por AUC.
**Current focus:** Fase 4 — Shadow/Dual-Run (Fase 3 completa)

## Current Position

Phase: 4 of 6 (Shadow/Dual-Run y Validación de Sesgo)
Plan: 2 of 2 en Fase 4 (04-02 código completo — checkpoint PENDING_DATA)
Status: PENDING_DATA — 04-02 código entregado (shadow_monitor.py + shadow_gate.py, 31 tests verdes); gate real diferido hasta ≥2 semanas de shadow data; checkpoint Task 3 en espera
Last activity: 2026-07-06 — Completado 04-02-PLAN.md (Tasks 1+2): shadow_monitor.py (4 métricas robustas), shadow_gate.py (guard INSUFFICIENT_DATA, Spearman scipy, exit 0/1/2), 31 tests. Checkpoint PENDING_DATA pausado.

Progress: [█████████░] ~96% de Fases 0–4 (13/14 planes — 04-02 código completo, checkpoint diferido)

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 h

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: —
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Fase 0 - 00-03]: Baseline v0 congelado: top-5% ratio=11.79x, off-hours UTC=29.78% son el punto de partida para Fase 1 (no 15.7x del research — cohorte/feature set distinto).
- [Fase 0 - 00-03]: AUC pure_fraud=0.836 etiquetado diagnostic_circular_not_a_gate_metric. 4 features: same_amount_count_1h, user_account_age_days, user_txn_count_1h, is_third_party_payment.
- [Fase 0 - 00-03]: Golden set = 14831 filas, 680 facilities, seed=42. output/golden_set_v0.parquet reproducible byte-a-byte.
- [Fase 0]: Gate de éxito = reducción de sesgo (top-5% <4×, off-hours ~4–5%); AUC vs `pure_fraud` es circularidad — diagnóstico, no criterio.
- [Fase 1]: Modelo global único + calibración segmentada (no un modelo por moneda); stats artifact cargado en memoria para presupuesto de 200ms.
- [Proyecto]: `capture_delay_seconds` excluido de `FS-frame-operational-v1` (train/serve skew: ~0 en real-time vs valor real en batch; AUC flag 0,511).
- [00-01]: `actual_role` (no `"player"` forzado) en lookup staff zscore — batch usa rol raw; fallback 3 pasos: (role,currency) → currency → global.
- [00-02]: `thresholds.json` (IF-31) marcado LEGACY_DEPRECATED; scorer operativo IF-40 usa `thresholds_v2.json` (percentile_95_validation_set, 0.024223975402714343) — no tocar.
- [00-02]: `FS-frame-operational-v1` = `output/models/final_feature_list_operational.json` (39 features, sin capture_delay_seconds). Artefacto autodescriptivo, commiteado en git.
- [00-02]: Currency EMPTY sanitized en loader.py (_sanitize_currency) y engineering.py; 0 filas afectadas en splits actuales (fix preventivo para Fase 1 facility stats).
- [01-01]: iqr_guarded = max(iqr, 1.0) no max(iqr, 1e-6) — floor significativo para distribuciones near-uniform; almacenado en artefacto para uso en fórmula z-score.
- [01-01]: facility_stats_v1.json itera tz_map (1876) no train_df.groupby (689) — garantiza iana_tz en todas las facilities para el path RT; 580 facility / 1289 currency / 7 global.
- [01-01]: currency_fallbacks para USD, CAD, MYR, HNL, NIO (top-5 frecuencia en train); 7 facilities caen a global. [SUPERSEDIDO por 02-01]
- [02-01]: _MIN_CURRENCY_N=1000 reemplaza top-5-por-volumen: 14 monedas con fallback (AED/AUD/BWP/CAD/COP/GTQ/HKD/HNL/ILS/MYR/NIO/PKR/SGD/USD); fallback=global reducido a 4; fid 1214/1232/1373 promovidos a currency.
- [02-01]: USD incluido siempre como garantía, independientemente del umbral de n en train.
- [02-01]: facility_stats_v1.json commiteado con git add -f (gitignored) para reproducibilidad; min_currency_n_threshold incluido en artefacto como campo autodescriptivo.
- [02-02]: LUT global 1001pts / segmento 201pts — consistencia con thresholds_v2.json y JSON compacto (~28MB vs ~140MB con 1001pts).
- [02-02]: SegmentedThresholdClassifier conectado a scorer.py en 03-01 (dispatch por presencia de artefactos); ThresholdClassifier IF-40 intacto.
- [02-02]: Guardrail [0.040, 0.048] en script offline para detectar mezcla IF-40/frame-v1; global p95=0.04359 PASS.
- [02-02]: by_currency=17 entries: MXN (n=88) e INR (n=2) excluidas (< MIN_N=200); todas las entries tienen n≥200.
- [02-02]: thresholds_segmented_v1.json commiteado con git add -f (gitignored); 452 facilities, 17 monedas, schema_version='thresholds-segmented-v1'.
- [02-03]: currency: Optional[str]=None (no ="USD") — ausencia de moneda es observable, no silenciada. IF-40 scorer usa payment.get("currency","USD") → None or "USD" = backward compat.
- [02-03]: facility_time_zone_iana: Optional[str]=None (nunca ="UTC") — UTC default reintroduciría el sesgo off-hours que el proyecto corrige.
- [02-03]: Artifacts.facility_stats y .thresholds_segmented como campos opcionales trailing (None=legacy IF-40); _validate_artifacts sin cambios.
- [02-03]: Tests frame-v1 en test_artifact_loader.py usan tmp_path con model_metadata.json renombrado para evitar conflicto con IF-40 en output/models.
- [01-01]: validate_universe_filter(stats, sample_df, tz_df) recibe tz_df como 3er arg para assert len(facilities)==tz_df.facility_id.nunique()==1876.
- [01-02]: currency_original separado de currency en FrameV1FeatureCalculator: staff stats aprendidos con moneda original como key (no USD) — lookup usa (role, currency_original).
- [01-02]: Sentinel -1.0 en UserContext para time_since_last_txn/credit_flow_ratio/category_entropy_30d: -1.0 = derivar de otros campos; >=0 = usar directamente.
- [01-02]: FRAME_V1_FEATURE_NAMES = frame_version(DISJOINT30) = 30 features; assert len==30 a nivel de módulo; max diff paridad = 1.44e-12 sobre 3213 pagos, 680 facilities.
- [01-03]: Gate 1 metrica robusta: top5_amount_ratio_winsorized_p999 (not raw mean) — cola pesada invalida ratio de medias; gate1_pass = winsorized < 4.0 (1.49x PASS).
- [01-03]: Saneo upstream en loader.py (no en features): DataManager.compute_amount_sanity_thresholds + sanitize_amount_df; per-currency p99.99; preserva train/serve parity 0.0.
- [01-03]: Train drop 209 filas (0.0067%, amount > p99.99 per-currency). Causa raiz: 2 filas val (USD 100M/10M, facility 1422) = 88.4% del monto top-5%. Scorer en vivo intacto.
- [01-03]: Gate 2 off-hours local: 6.46% (banda 3-7% PASS); ratio reduccion vs UTC baseline 29.78% = 78%.

### Pending Todos

None yet.

### Blockers/Concerns

- **[RESUELTO - 00-01]** Bugs de `getattr` corregidos: scorer RT ahora usa 689 facility means y 81 combinaciones (role, currency). Delta scores vs pre-fix: facility_avg_amount mean delta=1498 USD, max=259006 USD. Baseline congelado (00-03) DEBE ser post-fix.
- **[RESUELTO - 02-02]** El min-n para calibración segmentada (100 vs 200): MIN_N=200 confirmado; 452 facilities y 17 monedas lo cumplen sobre val set (1.13M filas).
- **[RESUELTO - 03-01]** PLAT-01 eliminado: scorer resuelve IANA autónomamente desde facility_stats_v1.json; Rails NO necesita enviar facility_time_zone_iana.
- **[Pre-Fase 4]** La extensión del payload Rails (`facility_time_zone_iana`) tiene su propio lead time — confirmar disponibilidad antes de activar Fase 4.
- **[Pre-Fase 5]** Capacidad de revisión del equipo HITL desconocida — confirmar antes de Fase 5 (afecta el ratio 80/20 top-k vs random).

### New Decisions (03-01)

- [03-01]: Dispatch por presencia de artefactos (facility_stats is not None AND thresholds_segmented is not None), no por len(feature_names)==30. _is_frame_v1 flag governa todo el path.
- [03-01]: frame_flags es dict en ScoringResult (no Pydantic); FrameFlags(**dict) se construye solo en el router. Permite usar ScoringResult sin FastAPI.
- [03-01]: timezone_missing=True sii facility_id ausente del artefacto (fallback Etc/UTC); nunca timezone_invalid — el scorer nunca lanza excepción de zona.
- [03-01]: _INSERT_COLUMNS del batch scorer no tocado — DDL ClickHouse anomaly_scores no tiene columnas frame-v1; persistencia batch diferida a Fase 4.

### New Decisions (03-02)

- [03-02]: AlertManager persiste frame-v1 en metadata JSON sin migración — columnas SQL dedicadas diferidas a Fase 4.
- [03-02]: feature_frame_version mapeado desde score_result['feature_version'] — reusar campo existente en lugar de añadir campo separado al scorer.
- [03-02]: .compact en build_metadata: nil-safety garantiza retrocompat IF-40 sin condicional explícito.
- [03-02]: PLAT-01 reconciliado — build_payload intocado; Rails no depende de facility_time_zone_iana; scorer resuelve IANA autónomamente.

### New Decisions (04-01)

- [04-01]: metadata_filename legacy guard: fallbacks IF-40/IF-31 solo cuando filename=='model_metadata.json'; override explícito falla con FileNotFoundError si ausente (no silencia error).
- [04-01]: _score_all_dual almacena features_json={} vacío para filas shadow — vectores son internos al scorer; re-extraerlos duplicaría lógica del feature calculator; queries SHAD-02 no necesitan vectores raw.
- [04-01]: scorer activo en _state es independiente del dual: los tres scorers (scorer, scorer_champion, scorer_new) coexisten; scorer activo responde rutas RT; dual-runner es exclusivo del path batch.
- [04-01]: _INSERT_COLUMNS 23→26; active mode pasa '' para las 3 cols frame-v1 (compatible con DEFAULT '' del DDL 02_anomaly_scores_frame_v1.sql).
- [04-01]: assert_write_target_is_safe es primera llamada tanto en _insert_chunks (active) como en _insert_chunks_dual (shadow) — contrato del guardrail preservado.

### New Decisions (04-02)

- [04-02]: INSUFFICIENT_DATA usa OR lógico: days_span<14 OR n_rows<500 — basta con que UNA condición falle para abortar (guard previo a cualquier cómputo de métricas).
- [04-02]: compute_spearman devuelve NaN (no lanza excepción) cuando hay <30 pares matched; gate marca spearman_pass=False en ese caso.
- [04-02]: off-hours aproximado con UTC (horas 0-8, 22-23) como proxy; is_off_hours_loc no persiste en anomaly_scores; tz_missing_rate desde JSON frame_flags solo para shadow_new.
- [04-02]: Import lazy de shadow_monitor en evaluate_gate() (sys.path insert) para evitar dependencia circular en tests y facilitar uso standalone.

## Session Continuity

Last session: 2026-07-06T18:27:17Z
Stopped at: Checkpoint PENDING_DATA — 04-02 código completo (shadow_monitor.py + shadow_gate.py, 31 tests, lint limpio). Gate real diferido: requiere ≥14 días de shadow data real. Reanudar cuando el operador tenga ≥2 semanas de SCORING_MODE=shadow_dual activo.
Resume file: None
