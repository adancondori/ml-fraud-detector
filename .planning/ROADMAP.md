# Roadmap: Normalización de marcos para scoring de anomalías en pagos

## Overview

El scorer de anomalías de Playbypoint (Isolation Forest sobre transacciones) aísla artefactos de marco de referencia en lugar de comportamiento anómalo relativo a la facility. Este roadmap corrige dos fuentes de sesgo conocidas — monto nominal en USD y hora UTC — y las encapsula en un sistema operativo completo: artefacto de stats versionado, calibración segmentada, scorer integrado con platform Rails, shadow dual-run y cola HITL. El éxito se mide por reducción de sesgo observable (top-5% ratio <4×, off-hours local ~4–5%), nunca por AUC contra `pure_fraud`.

## Phases

**Phase Numbering:**
- Integer phases (0–5): Trabajo planificado del milestone operativo.
- Decimal phases (N.x): Inserciones urgentes entre enteros (marcadas INSERTED).

El orden es no negociable: Fase 0 desbloquea todo como gate de baseline; Fase 1 es la raíz del grafo de dependencias; Fase 2 depende de Fase 1; Fase 3 depende de Fase 2; Fase 4 depende de Fase 3; Fase 5 depende de shadow activo (Fase 4).

- [x] **Fase 0: Baseline Freeze y Bug Triage** — Corregir bugs activos en producción, fijar gate de sesgo, congelar baseline limpio antes de cualquier medición.
- [x] **Fase 1: Artefacto de Stats y Feature Calculator** — Construir la raíz de dependencias: `facility_stats_v1.json` y `FrameV1FeatureCalculator` con paridad batch↔real-time verificada.
- [x] **Fase 2: Calibración Segmentada y Contrato API** — Calibrar umbrales por segmento sobre val set con cadena de fallback; extender `artifact_loader`; bloquear contrato `frame-v1`.
- [ ] **Fase 3: Wiring del Scorer e Integración Platform** — Conectar los componentes probados al scorer en vivo; hacer que Rails envíe `facility_time_zone_iana`; persistir metadata de alerta ampliada.
- [ ] **Fase 4: Shadow Dual-Run y Validación de Sesgo** — Activar puntuación dual (champion vs frame-v1), monitoreo shadow y gate go/no-go cuantitativo sobre datos reales.
- [ ] **Fase 5: Cola HITL y Captura de Etiquetas** — Abrir cola de revisión humana top-k, capturar etiquetas con procedencia completa, incluir muestreo defensivo de falsos negativos.

## Phase Details

### Fase 0: Baseline Freeze y Bug Triage
**Goal**: El scorer opera sin bugs silenciosos conocidos, los umbrales están calibrados en val (no test), y el gate de éxito del proyecto está formalmente fijado como reducción de sesgo — antes de que cualquier medición de baseline sea tomada.
**Depends on**: Nada (primera fase).
**Requirements**: BASE-01, BASE-02, BASE-03, BASE-04, BASE-05, BASE-06
**Success Criteria** (what must be TRUE):
  1. El scorer real-time usa `_facility_avg_amount` y `_role_currency_stats` correctamente — una aserción post-carga y un test de paridad batch↔real-time pasan en CI.
  2. Los umbrales operativos en `thresholds.json` están derivados del val set; `threshold_source` no es `"percentile_95_test_set"`.
  3. `capture_delay_seconds` está excluido del feature set operativo y el bug de `is pd.NaT` está corregido.
  4. La moneda `"EMPTY"` se sanea a `"USD"` con flag en la extracción; el reporte de calidad de datos informa el conteo de registros afectados.
  5. Existe un documento de baseline congelado (métricas actuales con scorer bugfix, set dorado de ≥500 pagos, lista cerrada `FS-frame-operational-v1`, gate: top-5% monto <4×, off-hours local ~4–5%) y el AUC contra `pure_fraud` está marcado formalmente como "diagnóstico circular, no criterio".
**Plans**: 3 plans

Plans:
- [x] 00-01-PLAN.md — Fix getattr en `scoring/features.py` (acceso directo + lookup por tupla (role,currency)) + fix `pd.NaT` + test de paridad batch↔real-time [Wave 1]
- [x] 00-02-PLAN.md — Sanear moneda `"EMPTY"`→USD, materializar `FS-frame-operational-v1` (sin capture_delay), resolver threshold legacy (IF-40 usa val) [Wave 1]
- [x] 00-03-PLAN.md — Congelar baseline: set dorado ≥500 pagos, `baseline_v0.json` post-fix, gate de sesgo formal, AUC pure_fraud marcado circular [Wave 2]

### Fase 1: Artefacto de Stats y Feature Calculator
**Goal**: Existe un artefacto versionado `facility_stats_v1.json` con mediana/IQR/zona-IANA/currency_group/fallback_level por facility, computado sobre el universo exacto del scorer; y un `FrameV1FeatureCalculator` que produce features de marco idénticas en batch y en real-time para el mismo pago.
**Depends on**: Fase 0 (baseline limpio; bugs de paridad corregidos antes de medir).
**Requirements**: STATS-01, STATS-02, FRAME-01, FRAME-02, FRAME-03, FRAME-04
**Success Criteria** (what must be TRUE):
  1. `facility_stats_v1.json` existe, está versionado, y un validator verificable confirma que su universo de cómputo coincide con el filtro del scorer (`_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free')`).
  2. La cadena de fallback (facility con n<30 → currency → global) está implementada y registra `fallback_level` por entrada; ninguna facility queda sin stats.
  3. Las features de magnitud (`amount_facility_ratio`, z-score robusto) y de hora local (IANA con DST) se calculan sin usar USD absoluto ni UTC respectivamente; un test unitario verifica DST correctamente para al menos dos zonas latinoamericanas.
  4. Un model global reentrenado sobre `FS-frame-operational-v1` en validación produce: sesgo de monto top-5% <4× (vs ~15,7× actual) y off-hours local ~4–5% (vs ~30% UTC actual).
  5. Un test de paridad sobre ≥100 pagos del val set afirma que `FrameV1FeatureCalculator.calculate(payment, context)` y `calculate_from_row(row, facility_stats)` producen vectores de features idénticos (diferencia máxima <1e-8).
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — `FacilityStatsBuilder`: artefacto `facility_stats_v1.json` con fallback chain, IQR guarded y mapa Rails→IANA (64 zonas) [Wave 1]
- [x] 01-02-PLAN.md — `FrameV1FeatureCalculator` (TDD): 30 features de marco, superficie dual con paridad <1e-8 y hora local IANA con DST [Wave 2]
- [x] 01-03-PLAN.md — Reentrenamiento global sobre `FS-frame-v1` y gate de sesgo en val (top-5% <4×, off-hours local ~4-5%) [Wave 3]

### Fase 2: Calibración Segmentada y Contrato API
**Goal**: Los umbrales se calibran por segmento (facility → currency_group → global) sobre el val set con guardia de n mínimo; el `artifact_loader` carga el nuevo artefacto de stats y los thresholds segmentados de forma retrocompatible; el contrato `frame-v1` falla de forma observable ante campos ausentes.
**Depends on**: Fase 1 (requiere scores en el espacio `FS-frame-operational-v1` sobre val set).
**Requirements**: CAL-01, CAL-02, API-01, API-02
**Success Criteria** (what must be TRUE):
  1. `thresholds_segmented_v1.json` existe con cadena de fallback (facility → currency_group → global), guarda de n mínimo (≥200 por segmento), y `fallback_level` registrado por segmento.
  2. El `artifact_loader` carga `facility_stats_v1.json` + `thresholds_segmented_v1.json` en el dataclass `Artifacts` sin romper el scorer legacy (campo `None` → ruta legacy activa).
  3. Una request al scorer con `facility_time_zone_iana` ausente activa `frame_flags.timezone_missing=true` en la respuesta — nunca silencia el campo con un default UTC.
  4. La respuesta del scorer incluye `calibration_segment`, `fallback_level` y `frame_flags` como campos estructurados.
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Extender currency_fallbacks (n>=1000, 14 monedas) + regenerar `facility_stats_v1.json` + re-correr paridad Fase 1 [Wave 1]
- [x] 02-02-PLAN.md — `SegmentedThresholdCalibrator` + `SegmentedThresholdClassifier` + `thresholds_segmented_v1.json` (scores frame-v1, guarda n>=200) [Wave 2]
- [x] 02-03-PLAN.md — Loader retrocompatible (+ fix metadata frame-v1 con artifact_files) + contrato API frame-v1 (schemas sin default silencioso, respuesta enriquecida) [Wave 3]

### Fase 3: Wiring del Scorer e Integración Platform
**Goal**: El scorer en vivo despacha al `FrameV1FeatureCalculator` + `SegmentedThresholdClassifier` cuando los artefactos frame-v1 están presentes (retrocompat IF-40 preservada); la zona IANA se resuelve de forma autónoma en el scorer desde `facility_stats_v1.json` (Rails NO depende de enviarla); `AlertManager` persiste la metadata de alerta ampliada; el universo `scorable?` está alineado entre batch y real-time.
**Depends on**: Fase 2 (artefactos completos y contrato API bloqueado).
**Requirements**: PLAT-01, PLAT-02, PLAT-03
**Success Criteria** (what must be TRUE):
  1. El scorer produce features de hora local correctas resolviendo la zona IANA autónomamente desde el artefacto (`_lookup_facility`, fallback `Etc/UTC`); `frame_flags.timezone_missing=true` es observable cuando la facility no está en el artefacto, sin excepción. Rails puede enviar `facility_time_zone_iana` como metadata OPCIONAL (vía `facility.tzinfo_identifier`, DB-safe), pero el scorer no depende de ello (PLAT-01 reconciliado: fuente única de verdad en el scorer).
  2. `AlertManager` persiste `calibration_segment`, `fallback_level`, `frame_flags` y `feature_frame_version` en la metadata JSON de la alerta (sin migración; columnas dedicadas y persistencia batch en ClickHouse diferidas a Fase 4).
  3. El universo `scorable?` excluye `payment_method IN ('reversal','free')` de forma idéntica en batch y real-time; los reembolsos se marcan post-hoc sin afectar el scoring.
  4. Una transacción de prueba puntúa de extremo a extremo (Rails → scorer → alerta) en entorno de test sin errores y con todos los campos de metadata presentes (sin shadow/dual-run — eso es Fase 4).
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md — Wiring del scorer: `SingleTransactionScorer` despacha a `FrameV1FeatureCalculator` + `SegmentedThresholdClassifier` por presencia de artefactos; `ScoringResult` + router propagan `calibration_segment`/`fallback_level`/`frame_flags`; resolución IANA autónoma; retrocompat IF-40 [Wave 1]
- [ ] 03-02-PLAN.md — Integración platform Rails: `AlertManager` persiste metadata frame-v1 (JSON, sin migración); `scorable?` excluye `free`; E2E de prueba Rails→scorer→alerta [Wave 2]
### Fase 4: Shadow Dual-Run y Validación de Sesgo
**Goal**: Cada pago scorable recibe dos puntuaciones simultáneas (champion actual y frame-v1), ambas persistidas; las queries de monitoreo shadow reportan reducción de sesgo sobre datos reales; un gate go/no-go cuantitativo determina si frame-v1 puede promoverse.
**Depends on**: Fase 3 (Rails enviando `facility_time_zone_iana`; scorer wired; metadata persistida).
**Requirements**: SHAD-01, SHAD-02, SHAD-03
**Success Criteria** (what must be TRUE):
  1. Cada pago scorable produce dos filas en `anomaly_scores` con `model_version` distinto (`shadow_old` y `shadow_new`); la tasa de fallo de escritura dual es 0% durante el primer día de shadow.
  2. Las queries de monitoreo shadow reportan: alert rate por segmento, sesgo de monto top-5% (ratio champion vs frame-v1), off-hours local vs UTC, y Jaccard@100 entre los top-100 de cada modelo.
  3. Tras ≥2 semanas de shadow data, el gate go/no-go evalúa: top-5% monto ratio <4×, off-hours local ~4–5%, correlación de Spearman del ranking ≥0,90, delta de alert rate ≤2pp entre modelos por segmento. El resultado (go/no-go) queda documentado con evidencia.
  4. El sesgo de monto medido en shadow (top-5% ratio de frame-v1) es materialmente menor que el del champion — confirmando en datos reales la reducción observada en el experimento offline (15,7× → 3,3×).
**Plans**: TBD

Plans:
- [ ] 04-01: `ShadowDualRunner` — dual-score por pago, persistencia de ambas filas, manejo de fallos parciales
- [ ] 04-02: Queries de monitoreo shadow y gate go/no-go documentado

### Fase 5: Cola HITL y Captura de Etiquetas
**Goal**: Existe una cola de revisión que exporta el top-k de frame-v1 con `top_factors`; los revisores pueden capturar etiquetas con procedencia completa; el muestreo incluye ≥20% de transacciones no alertadas (bajo p50) para estimar falsos negativos.
**Depends on**: Fase 4 (shadow activo con al menos días de datos reales de frame-v1).
**Requirements**: HITL-01, HITL-02, HITL-03
**Success Criteria** (what must be TRUE):
  1. La cola de revisión exporta el top-k ordenado por `risk_level`/`percentile` del modelo frame-v1 con `top_factors` (features con mayor z-score absoluto) en formato consultable.
  2. Un revisor puede registrar una etiqueta con todos los campos de procedencia requeridos: `reviewer_label`, `reviewed_at`, `reviewer_id`, `score_at_label_time`, `model_version`, `reviewer_saw_factors`.
  3. El muestreo HITL incluye ≥20% de transacciones por debajo del p50 de score (no alertadas), documentando la estrategia para estimación de falsos negativos.
  4. La distribución de etiquetas capturadas en la primera semana de operación puede consultarse sin ambigüedad sobre qué modelo versión generó el score al momento de revisión.
**Plans**: TBD

Plans:
- [ ] 05-01: Cola de revisión top-k con `top_factors` y schema de etiquetas con procedencia
- [ ] 05-02: Muestreo HITL defensivo (80% alertados + ≥20% no alertados) y documentación de metodología

## Progress

**Execution Order:**
Fases ejecutan en orden estricto: 0 → 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 0. Baseline Freeze y Bug Triage | 3/3 | ✓ Complete | 2026-07-06 |
| 1. Artefacto de Stats y Feature Calculator | 3/3 | ✓ Complete | 2026-07-06 |
| 2. Calibración Segmentada y Contrato API | 3/3 | ✓ Complete | 2026-07-06 |
| 3. Wiring del Scorer e Integración Platform | 0/TBD | Not started | - |
| 4. Shadow Dual-Run y Validación de Sesgo | 0/TBD | Not started | - |
| 5. Cola HITL y Captura de Etiquetas | 0/TBD | Not started | - |
