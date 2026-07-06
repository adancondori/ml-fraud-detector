# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-06)

**Core value:** El ranking de anomalías refleja comportamiento relativo al contexto de la facility (moneda, escala, hora local), no tamaño nominal ni artefactos UTC — medido por reducción de sesgo (top-5% monto <4×, off-hours local ~4–5%), no por AUC.
**Current focus:** Fase 0 — Baseline Freeze y Bug Triage

## Current Position

Phase: 0 of 6 (Baseline Freeze y Bug Triage) — COMPLETA
Plan: 3 of 3 en Fase 0 (00-01, 00-02, 00-03 completos)
Status: Fase 0 completa — lista para iniciar Fase 1
Last activity: 2026-07-06 — Completado 00-03-PLAN.md: golden set congelado, baseline_v0.json materializado, gate de sesgo formal, checkpoint humano aprobado

Progress: [███░░░░░░░] 30%

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

### Pending Todos

None yet.

### Blockers/Concerns

- **[RESUELTO - 00-01]** Bugs de `getattr` corregidos: scorer RT ahora usa 689 facility means y 81 combinaciones (role, currency). Delta scores vs pre-fix: facility_avg_amount mean delta=1498 USD, max=259006 USD. Baseline congelado (00-03) DEBE ser post-fix.
- **[Pre-Fase 2]** El min-n para calibración segmentada (100 vs 200) depende de la distribución de tamaño de segmentos en val set — tabular primero en Fase 2.
- **[Pre-Fase 3]** La extensión del payload Rails (`facility_time_zone_iana`) tiene su propio lead time — confirmar disponibilidad antes de activar Fase 4.
- **[Pre-Fase 5]** Capacidad de revisión del equipo HITL desconocida — confirmar antes de Fase 5 (afecta el ratio 80/20 top-k vs random).

## Session Continuity

Last session: 2026-07-06T05:24:50Z
Stopped at: Completado 00-03-PLAN.md (golden set, baseline_v0.json congelado, checkpoint humano aprobado). Fase 0 completa.
Resume file: None
