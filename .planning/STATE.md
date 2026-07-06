# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-06)

**Core value:** El ranking de anomalías refleja comportamiento relativo al contexto de la facility (moneda, escala, hora local), no tamaño nominal ni artefactos UTC — medido por reducción de sesgo (top-5% monto <4×, off-hours local ~4–5%), no por AUC.
**Current focus:** Fase 0 — Baseline Freeze y Bug Triage

## Current Position

Phase: 1 of 6 (Artefacto de Stats y Feature Calculator) — En progreso
Plan: 1 of 3 en Fase 1 (01-01 completo; 01-02, 01-03 pendientes)
Status: En progreso — 01-01 completo, iniciando 01-02
Last activity: 2026-07-06 — Completado 01-01-PLAN.md: facility_stats_v1.json materializado, 1876 facilities con iana_tz, fallback chain, IQR guard

Progress: [████░░░░░░] 40%

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
- [01-01]: currency_fallbacks para USD, CAD, MYR, HNL, NIO (top-5 frecuencia en train); 7 facilities caen a global.
- [01-01]: validate_universe_filter(stats, sample_df, tz_df) recibe tz_df como 3er arg para assert len(facilities)==tz_df.facility_id.nunique()==1876.

### Pending Todos

None yet.

### Blockers/Concerns

- **[RESUELTO - 00-01]** Bugs de `getattr` corregidos: scorer RT ahora usa 689 facility means y 81 combinaciones (role, currency). Delta scores vs pre-fix: facility_avg_amount mean delta=1498 USD, max=259006 USD. Baseline congelado (00-03) DEBE ser post-fix.
- **[Pre-Fase 2]** El min-n para calibración segmentada (100 vs 200) depende de la distribución de tamaño de segmentos en val set — tabular primero en Fase 2.
- **[Pre-Fase 3]** La extensión del payload Rails (`facility_time_zone_iana`) tiene su propio lead time — confirmar disponibilidad antes de activar Fase 4.
- **[Pre-Fase 5]** Capacidad de revisión del equipo HITL desconocida — confirmar antes de Fase 5 (afecta el ratio 80/20 top-k vs random).

## Session Continuity

Last session: 2026-07-06T06:01:46Z
Stopped at: Completado 01-01-PLAN.md (facility_stats_v1.json materializado, 1876 facilities, 22 tests verdes). Iniciando 01-02.
Resume file: None
