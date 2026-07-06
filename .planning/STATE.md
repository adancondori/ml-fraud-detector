# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-06)

**Core value:** El ranking de anomalías refleja comportamiento relativo al contexto de la facility (moneda, escala, hora local), no tamaño nominal ni artefactos UTC — medido por reducción de sesgo (top-5% monto <4×, off-hours local ~4–5%), no por AUC.
**Current focus:** Fase 0 — Baseline Freeze y Bug Triage

## Current Position

Phase: 0 of 6 (Baseline Freeze y Bug Triage)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-07-06 — Roadmap creado; REQUIREMENTS.md traceability confirmada (24/24 requisitos mapeados)

Progress: [░░░░░░░░░░] 0%

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

### Pending Todos

None yet.

### Blockers/Concerns

- **[Pre-Fase 0]** Dos bugs activos en producción: `getattr` con nombres erróneos en `scoring/features.py:27-28` (train/serve skew silencioso ya activo). Estos son el primer trabajo de Fase 0.
- **[Pre-Fase 2]** El min-n para calibración segmentada (100 vs 200) depende de la distribución de tamaño de segmentos en val set — tabular primero en Fase 2.
- **[Pre-Fase 3]** La extensión del payload Rails (`facility_time_zone_iana`) tiene su propio lead time — confirmar disponibilidad antes de activar Fase 4.
- **[Pre-Fase 5]** Capacidad de revisión del equipo HITL desconocida — confirmar antes de Fase 5 (afecta el ratio 80/20 top-k vs random).

## Session Continuity

Last session: 2026-07-06
Stopped at: Roadmap creado; archivos ROADMAP.md, STATE.md escritos; REQUIREMENTS.md traceability actualizada.
Resume file: None
