# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-06)

**Core value:** El ranking de anomalías refleja comportamiento relativo al contexto de la facility (moneda, escala, hora local), no tamaño nominal ni artefactos UTC — medido por reducción de sesgo (top-5% monto <4×, off-hours local ~4–5%), no por AUC.
**Current focus:** Fase 0 — Baseline Freeze y Bug Triage

## Current Position

Phase: 0 of 6 (Baseline Freeze y Bug Triage)
Plan: 2 of 4 in current phase (00-01 y 00-02 completos)
Status: In progress
Last activity: 2026-07-06 — Completado 00-02-PLAN.md: currency EMPTY sanitized, FS-frame-operational-v1 materializado, threshold legacy marcado

Progress: [██░░░░░░░░] 20%

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

- [Fase 0]: Gate de éxito = reducción de sesgo (top-5% <4×, off-hours ~4–5%); AUC vs `pure_fraud` es circularidad — diagnóstico, no criterio.
- [Fase 1]: Modelo global único + calibración segmentada (no un modelo por moneda); stats artifact cargado en memoria para presupuesto de 200ms.
- [Proyecto]: `capture_delay_seconds` excluido de `FS-frame-operational-v1` (train/serve skew: ~0 en real-time vs valor real en batch; AUC flag 0,511).
- [00-02]: `thresholds.json` (IF-31) marcado LEGACY_DEPRECATED; scorer operativo IF-40 usa `thresholds_v2.json` (percentile_95_validation_set, 0.024223975402714343) — no tocar.
- [00-02]: `FS-frame-operational-v1` = `output/models/final_feature_list_operational.json` (39 features, sin capture_delay_seconds). Artefacto autodescriptivo, commiteado en git.
- [00-02]: Currency EMPTY sanitized en loader.py (_sanitize_currency) y engineering.py; 0 filas afectadas en splits actuales (fix preventivo para Fase 1 facility stats).

### Pending Todos

None yet.

### Blockers/Concerns

- **[Pre-Fase 0]** Dos bugs activos en producción: `getattr` con nombres erróneos en `scoring/features.py:27-28` (train/serve skew silencioso ya activo). Estos son el primer trabajo de Fase 0.
- **[Pre-Fase 2]** El min-n para calibración segmentada (100 vs 200) depende de la distribución de tamaño de segmentos en val set — tabular primero en Fase 2.
- **[Pre-Fase 3]** La extensión del payload Rails (`facility_time_zone_iana`) tiene su propio lead time — confirmar disponibilidad antes de activar Fase 4.
- **[Pre-Fase 5]** Capacidad de revisión del equipo HITL desconocida — confirmar antes de Fase 5 (afecta el ratio 80/20 top-k vs random).

## Session Continuity

Last session: 2026-07-06T05:12:34Z
Stopped at: Completado 00-02-PLAN.md (currency sanitize, FS-frame-operational-v1, threshold legacy).
Resume file: None
