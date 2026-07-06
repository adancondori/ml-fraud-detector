# Requirements: Normalización de marcos para scoring de anomalías

**Defined:** 2026-07-06
**Core Value:** El ranking de anomalías refleja comportamiento relativo al contexto de la facility (moneda, escala, hora local), no tamaño nominal ni artefactos UTC — medido por reducción de sesgo, no por AUC.

## v1 Requirements

Requisitos para el milestone operativo. Cada uno mapea a una fase del roadmap.

### Fundación / corrección de bugs (Fase 0)

- [x] **BASE-01**: El scorer real-time usa medias/estadísticas per-facility (no globales) — corregir los `getattr` con nombres erróneos en `scoring/features.py:27-28` y añadir aserción post-carga
- [x] **BASE-02**: El scorer real-time y el batch producen features idénticas para el mismo pago — test de paridad batch↔real-time como guardrail de regresión
- [x] **BASE-03**: Los umbrales operativos se calibran sobre el conjunto de validación, no sobre test
- [x] **BASE-04**: `capture_delay_seconds` se excluye del feature set operativo y su bug de check `pd.NaT` queda corregido
- [x] **BASE-05**: La moneda `"EMPTY"` se sanea explícitamente (a `"USD"` con flag) en la extracción, con reporte de calidad de datos
- [x] **BASE-06**: El gate de éxito del proyecto está fijado como reducción de sesgo (top-5% monto <4×, off-hours local ~4-5%); el AUC contra `pure_fraud` queda marcado como diagnóstico circular, no criterio

### Artefacto de stats y features de marco (Fase 1)

- [x] **STATS-01**: Existe un artefacto versionado `facility_stats_v1.json` con mediana/IQR/n, moneda primaria, currency_group y zona IANA por facility, computado sobre el universo exacto del scorer
- [x] **STATS-02**: El artefacto usa cadena de fallback (facility con n<30 → currency → global) y registra `fallback_level` por entrada
- [x] **FRAME-01**: Las features de magnitud se calculan relativas a la facility (`amount_facility_ratio`, z-score robusto contra mediana/IQR) sin USD absoluto
- [x] **FRAME-02**: Las features temporales se calculan en hora local de la facility vía zona IANA con DST (mapeo Rails→IANA), no en UTC
- [x] **FRAME-03**: El feature set cerrado `FS-frame-operational-v1` está definido, versionado y documentado (conserva las features de velocidad invariantes al marco)
- [x] **FRAME-04**: Un modelo global reentrenado sobre `FS-frame-operational-v1` pasa los gates de reducción de sesgo en validación

### Calibración segmentada y contrato API (Fase 2)

- [x] **CAL-01**: Los umbrales se calibran por segmento con cadena de fallback (facility → currency_group → global) y guarda de n mínimo, en `thresholds_segmented_v1.json`
- [x] **CAL-02**: El `artifact_loader` carga stats + thresholds segmentados de forma retrocompatible, validando `feature_version`
- [x] **API-01**: El contrato `frame-v1` acepta `amount_local`, `currency` y `facility_time_zone_iana` como opcionales sin defaults silenciosos; ausencia activa flags observables
- [x] **API-02**: La respuesta del scorer incluye `calibration_segment`, `fallback_level` y `frame_flags`

### Integración platform (Fase 3)

- [x] **PLAT-01**: El payload real-time de `RealTimeScoringService` envía `facility_time_zone_iana` (resuelto de forma segura contra zonas inválidas)
- [x] **PLAT-02**: `AlertManager` persiste `calibration_segment`, `fallback_level`, `frame_flags` y `feature_frame_version` en la metadata de alerta
- [x] **PLAT-03**: El universo `scorable?` está alineado entre batch y real-time (no puntuar `reversal`/`free`; reembolsados marcados post-hoc)

### Shadow mode y validación (Fase 4)

- [x] **SHAD-01**: El scorer puntúa cada pago con ambos modelos (actual y frame-v1) y persiste ambas filas con `model_version` distinto
- [x] **SHAD-02**: Existen queries de monitoreo shadow: alert rate por segmento, sesgo de monto (top-5%), off-hours local vs UTC, Jaccard@100
- [x] **SHAD-03**: Se evalúa un gate go/no-go de promoción contra objetivos de reducción de sesgo sobre datos shadow

### Human-in-the-loop (Fase 5)

- [x] **HITL-01**: Existe una cola de revisión que exporta el top-k ordenado por risk_level/percentile del modelo frame-v1 con `top_factors`
- [x] **HITL-02**: Se capturan etiquetas de revisor con procedencia (`reviewer_label`, `reviewed_at`, `reviewer_id`, `score_at_label_time`, `model_version`, `reviewer_saw_factors`)
- [x] **HITL-03**: El muestreo HITL incluye ≥20% de transacciones no alertadas (bajo p50) para estimar falsos negativos

## v2 Requirements

Diferidos; reconocidos pero fuera del roadmap actual.

### Explicabilidad y monitoreo avanzado

- **V2-01**: Explicación SHAP por alerta para `risk_level IN ('high','critical')`
- **V2-02**: Monitoreo de drift del artefacto de stats (PSI por facility, rolling 30d, alerta PSI > 0.2)
- **V2-03**: Evaluación de independencia de etiquetas del revisor (proxy de kappa; requiere ~300 casos)
- **V2-04**: Trigger de desviación de alert rate (>20% vs objetivo por segmento en 7 días)

## Out of Scope

Excluidos explícitamente para prevenir scope creep.

| Feature | Reason |
|---------|--------|
| Reportar AUC contra `pure_fraud` como validación | Circularidad parcial declarada (4 features del modelo definen el proxy) — solo diagnóstico |
| Reabrir el confirmatorio académico HE1–HE4 | Trabajo operativo, no de tesis; el cierre académico se mantiene |
| Modelo por moneda | Muchas monedas sin volumen; validar calibración segmentada sobre modelo global primero |
| Features de user_tokens | Ya probadas; degradaron el AUC (44% sin token → cluster artificial) |
| USD absoluto como predictor central | Reintroduce sesgo de escala; USD solo para display |
| SHAP real-time para todas las transacciones | Viola el presupuesto de 200ms |
| `is_off_hours` en UTC | Infla off-hours 4–6× para facilities de LatAm |

## Traceability

Confirmado por el roadmapper — 24/24 requisitos v1 mapeados.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Fase 0 — Baseline Freeze y Bug Triage | Complete |
| BASE-02 | Fase 0 — Baseline Freeze y Bug Triage | Complete |
| BASE-03 | Fase 0 — Baseline Freeze y Bug Triage | Complete |
| BASE-04 | Fase 0 — Baseline Freeze y Bug Triage | Complete |
| BASE-05 | Fase 0 — Baseline Freeze y Bug Triage | Complete |
| BASE-06 | Fase 0 — Baseline Freeze y Bug Triage | Complete |
| STATS-01 | Fase 1 — Artefacto de Stats y Feature Calculator | Complete |
| STATS-02 | Fase 1 — Artefacto de Stats y Feature Calculator | Complete |
| FRAME-01 | Fase 1 — Artefacto de Stats y Feature Calculator | Complete |
| FRAME-02 | Fase 1 — Artefacto de Stats y Feature Calculator | Complete |
| FRAME-03 | Fase 1 — Artefacto de Stats y Feature Calculator | Complete |
| FRAME-04 | Fase 1 — Artefacto de Stats y Feature Calculator | Complete |
| CAL-01 | Fase 2 — Calibración Segmentada y Contrato API | Complete |
| CAL-02 | Fase 2 — Calibración Segmentada y Contrato API | Complete |
| API-01 | Fase 2 — Calibración Segmentada y Contrato API | Complete |
| API-02 | Fase 2 — Calibración Segmentada y Contrato API | Complete |
| PLAT-01 | Fase 3 — Wiring del Scorer e Integración Platform | Complete |
| PLAT-02 | Fase 3 — Wiring del Scorer e Integración Platform | Complete |
| PLAT-03 | Fase 3 — Wiring del Scorer e Integración Platform | Complete |
| SHAD-01 | Fase 4 — Shadow Dual-Run y Validación de Sesgo | Complete |
| SHAD-02 | Fase 4 — Shadow Dual-Run y Validación de Sesgo | Complete |
| SHAD-03 | Fase 4 — Shadow Dual-Run y Validación de Sesgo | Complete |
| HITL-01 | Fase 5 — Cola HITL y Captura de Etiquetas | Complete |
| HITL-02 | Fase 5 — Cola HITL y Captura de Etiquetas | Complete |
| HITL-03 | Fase 5 — Cola HITL y Captura de Etiquetas | Complete |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24/24
- Unmapped: 0

---
*Requirements defined: 2026-07-06*
*Last updated: 2026-07-06 — traceability confirmada por roadmapper*
