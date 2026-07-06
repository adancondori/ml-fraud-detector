---
phase: 03-wiring-del-scorer-e-integracion-platform
plan: 02
subsystem: platform-integration
tags: [ruby-on-rails, rspec, anomaly-detection, alert-manager, frame-v1, tdd]

# Dependency graph
requires:
  - phase: 03-01-wiring-del-scorer-e-integracion-platform
    provides: calibration_segment/fallback_level/frame_flags en ScoreResponse del scorer FastAPI
  - phase: 02-calibracion-segmentada-y-contrato-api
    provides: contrato frame-v1 con campos de calibración segmentada
provides:
  - scorable? excluye payment_method 'free' — universo real-time alineado con batch SQL
  - AlertManager#build_metadata persiste 4 campos frame-v1 en metadata JSON sin migración
  - RealTimeScoringService#create_alert propaga calibration_segment/fallback_level/frame_flags/feature_frame_version
  - Retrocompatibilidad IF-40 garantizada vía .compact (nil keys eliminadas)
  - E2E Rails → scorer → alerta con metadata completa aprobado por humano
affects:
  - 04-shadow-dual-run (activa escritura dual; encuentra Alert.metadata con campos frame-v1 ya presentes)
  - fase-5-hitl (revisión humana consume alert.metadata.calibration_segment para priorizar)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TDD Red/Green en Rails pack (RSpec spec falla antes de implementar, luego pasa)
    - .compact para nil-safety en build_metadata (sin columnas SQL dedicadas en Fase 3)
    - feature_frame_version mapeado desde feature_version del scorer (reusar campo existente)

key-files:
  created: []
  modified:
    - platform/packs/anomaly_detection/app/services/anomaly_detection/real_time_scoring_service.rb
    - platform/packs/anomaly_detection/app/services/anomaly_detection/alert_manager.rb
    - platform/packs/anomaly_detection/spec/services/anomaly_detection/real_time_scoring_service_spec.rb
    - platform/packs/anomaly_detection/spec/services/anomaly_detection/alert_manager_spec.rb

key-decisions:
  - "AlertManager persiste frame-v1 en metadata JSON sin migración — columnas SQL dedicadas diferidas a Fase 4"
  - "feature_frame_version mapeado desde score_result['feature_version'] — no añadir campo separado al scorer"
  - "PLAT-01 reconciliado: build_payload intocado — scorer resuelve IANA autónomamente desde facility_stats_v1.json (03-01)"
  - "Modo IF-40 retrocompat vía .compact: nil keys eliminadas, alerta legacy idéntica"

patterns-established:
  - ".compact en build_metadata: patron nil-safe para campos opcionales sin romper retrocompat"
  - "Mapeo feature_frame_version => score_result['feature_version']: reusar campo existente antes de añadir nuevo"

# Metrics
duration: continuación — E2E aprobado por humano
completed: 2026-07-06
---

# Phase 3 Plan 02: Platform Integration Summary

**AlertManager extendido para persistir metadata frame-v1 (calibration_segment/fallback_level/frame_flags/feature_frame_version) en JSON sin migración; scorable? alineado batch↔real-time excluyendo 'free'; E2E Rails → scorer → alerta aprobado por humano**

## Performance

- **Duration:** continuación (tasks 1-2 commiteadas; cierre post-checkpoint)
- **Started:** (ejecución original del agente)
- **Completed:** 2026-07-06T17:12:14Z
- **Tasks:** 3 (2 TDD + 1 checkpoint E2E aprobado por humano)
- **Files modified:** 4 (todos en platform pack anomaly_detection)

## Accomplishments

- `scorable?` en `RealTimeScoringService` excluye `payment_method == "free"` de forma idéntica a la exclusión SQL del batch scorer (`payment_method NOT IN ('reversal','free')`); spec TDD PASS
- `AlertManager#build_metadata` extendido con 4 campos frame-v1 (`calibration_segment`, `fallback_level`, `frame_flags`, `feature_frame_version`) persistidos en `Alert.metadata` JSON — sin migración de esquema; `.compact` garantiza retrocompatibilidad IF-40
- `RealTimeScoringService#create_alert` propaga los 4 campos desde la respuesta del scorer; `feature_frame_version` mapeado desde `score_result["feature_version"]` (open-question 1 resuelta: reusar campo existente)
- E2E Rails → scorer → alerta verificado y aprobado por humano con metadata completa y `free` excluido

## Task Commits

Todos los commits residen en el repo `platform` (los cambios son código Rails):

1. **Task 1: scorable? excluye 'free' (PLAT-03) — RED** — `8170bf5e69` (test)
2. **Task 2: Persistir metadata frame-v1 en AlertManager + create_alert (PLAT-02) — RED** — `97a8fe498f` (test)
3. **Task 2: Persistir metadata frame-v1 en AlertManager + create_alert (PLAT-02) — GREEN** — `d37e15486c` (feat)

_Nota: Task 1 GREEN fue parte del mismo ciclo TDD; la exclusión de 'free' quedó incluida en el commit feat de Task 2 como parte del GREEN unificado. TDD tasks pueden tener múltiples commits (test → feat)._

**Plan metadata:** pendiente de commit en ml-fraud-detector (este SUMMARY + STATE.md)

## Files Created/Modified

- `platform/packs/anomaly_detection/app/services/anomaly_detection/real_time_scoring_service.rb` — `scorable?` con guarda `free`; `create_alert` propaga 4 campos frame-v1
- `platform/packs/anomaly_detection/app/services/anomaly_detection/alert_manager.rb` — `build_metadata` extendido con `calibration_segment`/`fallback_level`/`frame_flags`/`feature_frame_version` + `.compact`
- `platform/packs/anomaly_detection/spec/services/anomaly_detection/real_time_scoring_service_spec.rb` — specs TDD: `when payment is free` → no puntúa; propagación de campos frame-v1 desde scorer
- `platform/packs/anomaly_detection/spec/services/anomaly_detection/alert_manager_spec.rb` — specs TDD: frame-v1 persistido en metadata; IF-40 retrocompat sin keys frame-v1

## Decisions Made

- **AlertManager persiste en JSON sin migración:** la columna `metadata JSON` ya existe en `anomaly_detection_alerts`; extender `build_metadata` es suficiente para Fase 3. Columnas SQL dedicadas (`calibration_segment`, etc.) se difieren a Fase 4 para no bloquear la integración.
- **feature_frame_version mapeado desde `score_result["feature_version"]`:** se decidió reusar el campo `feature_version` del scorer en lugar de añadir un campo separado `feature_frame_version` en el contrato Python. Rails mapea: `"feature_frame_version" => score_result["feature_version"]`.
- **PLAT-01 reconciliado — `build_payload` intocado:** el scorer resuelve la zona IANA autónomamente desde `facility_stats_v1.json` (implementado en 03-01). Rails no depende de enviar `facility_time_zone_iana`. `frame_flags.timezone_missing` es observable desde la alerta.
- **Retrocompatibilidad IF-40 vía `.compact`:** en modo IF-40 el scorer responde sin los campos frame-v1 → `score_result["calibration_segment"]` es `nil` → `.compact` los elimina → la alerta legacy no cambia. Ningún cambio en la lógica de deduplicación existente.

## Deviations from Plan

None — plan ejecutado exactamente como especificado. Los tres cambios (PLAT-03, PLAT-02, E2E) se implementaron según el diseño del plan. PLAT-01 ya estaba reconciliado en 03-01.

## Issues Encountered

None — el TDD RED/GREEN procedió sin bloqueos. Los specs existentes de `alert_manager_spec.rb` y `real_time_scoring_service_spec.rb` permanecieron verdes tras los cambios.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- La tubería real-time end-to-end está cableada y aprobada: scorer (frame-v1) → `RealTimeScoringService` → `AlertManager` → `Alert.metadata` con 4 campos frame-v1
- Fase 4 (shadow/dual-run) puede proceder: `Alert.metadata` ya tiene `calibration_segment`/`fallback_level`/`frame_flags`/`feature_frame_version` para pagos frame-v1
- Pendiente Fase 4: DDL ClickHouse `anomaly_scores` para columnas frame-v1 dedicadas + extensión del INSERT batch
- Pendiente confirmación pre-Fase 5: capacidad de revisión del equipo HITL (ratio 80/20 top-k vs random)
- No hay blockers activos para iniciar Fase 4

---
*Phase: 03-wiring-del-scorer-e-integracion-platform*
*Completed: 2026-07-06*
