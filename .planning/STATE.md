# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-06)

**Core value:** El ranking de anomalías refleja comportamiento relativo al contexto de la facility (moneda, escala, hora local), no tamaño nominal ni artefactos UTC — medido por reducción de sesgo (top-5% monto <4×, off-hours local ~4–5%), no por AUC.
**Current focus:** Fase 2 — Calibracion Segmentada y Contrato API

## Current Position

Phase: 2 of 6 (Calibracion Segmentada y Contrato API) — En progreso
Plan: 1 of N en Fase 2 (02-01 completo)
Status: In progress
Last activity: 2026-07-06 — Completado 02-01-PLAN.md: currency_fallbacks extendido a 14 monedas (_MIN_CURRENCY_N=1000), facility_stats_v1.json regenerado, paridad batch↔real-time 0.0 confirmada (35 tests PASS)

Progress: [███████░░░] 70%

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
- **[Pre-Fase 2]** El min-n para calibración segmentada (100 vs 200) depende de la distribución de tamaño de segmentos en val set — tabular primero en Fase 2.
- **[Pre-Fase 3]** La extensión del payload Rails (`facility_time_zone_iana`) tiene su propio lead time — confirmar disponibilidad antes de activar Fase 4.
- **[Pre-Fase 5]** Capacidad de revisión del equipo HITL desconocida — confirmar antes de Fase 5 (afecta el ratio 80/20 top-k vs random).

## Session Continuity

Last session: 2026-07-06T14:16:32Z
Stopped at: Completado 02-01-PLAN.md (currency_fallbacks extendido a 14 monedas, facility_stats_v1.json regenerado, paridad 0.0). Fase 2 iniciada, plan 02-02 siguiente.
Resume file: None
