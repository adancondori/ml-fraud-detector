# Normalización de marcos para scoring de anomalías en pagos

## What This Is

Mejora operativa del sistema de scoring de anomalías de `ml-fraud-detector` (Isolation Forest sobre transacciones de pago de Playbypoint/TechSport). Hoy el modelo aísla artefactos de marco de referencia —montos absolutos grandes, facilities grandes, "madrugada" calculada en UTC— en lugar de comportamiento anómalo relativo al contexto de cada facility. Este proyecto normaliza esos marcos (magnitud relativa a la facility, hora local IANA, sin features con leakage) para que el ranking que revisa el equipo humano signifique desviación conductual real. Es trabajo de sistema operativo (producción shadow + human-in-the-loop), no de la tesis académica.

## Core Value

El ranking de anomalías que ven los revisores debe reflejar comportamiento anómalo relativo al contexto de la facility, no tamaño nominal ni artefactos horarios — medido por reducción material del sesgo de monto (top-5% de ~15,7× el promedio a <4×) y corrección del confound temporal (off-hours UTC ~30% → local ~4-5%), no por AUC contra proxies imperfectos.

## Requirements

### Validated

<!-- Inferido del código existente (ver .planning/codebase/). Ya funciona y se depende de ello. -->

- ✓ Pipeline de entrenamiento de 8 pasos orquestado por `run_pipeline.py` (extracción → features → preprocesamiento → modelado → evaluación → sensibilidad → reporte) — existing
- ✓ Feature engineering con estado y anti-leakage (`FeatureEngineer`, 31 features, warm-history carry-over, ventanas rolling que excluyen la fila actual) — existing
- ✓ Entrenamiento no supervisado de 3 modelos (IsolationForest, LOF, OC-SVM) con grid search en validación y multi-seed — existing
- ✓ Evaluación de hipótesis HE1–HE4 contra proxy (Mann-Whitney/rank-biserial, AUC-ROC/AP, enrichment factor, comparación de modelos) con bootstrap CI y estabilidad temporal — existing
- ✓ Scorer FastAPI con scoring single real-time (`POST /api/v1/score`) y batch (`POST /api/v1/score/batch`) reutilizando artefactos entrenados — existing
- ✓ Separación READ (producción) / WRITE (local) de ClickHouse con guardrail de fingerprint que aborta INSERT inseguro — existing
- ✓ Carga de artefactos versionada con validación de conteo de features (`artifact_loader.py`) — existing
- ✓ Integración con platform Rails: callback `after_commit` → `RealTimeScoringService` → scorer; `AlertManager` persiste alertas con `model_version`/`feature_version`/`threshold_version` — existing

### Active

<!-- Las 6 fases del plan plans/payment-anomaly-detection.md. Hipótesis hasta shipear+validar. -->

- [ ] Congelar baseline: métricas actuales, set dorado de 500-1.000 pagos, lista cerrada `FS-frame-operational-v1`, universo de stats y fórmula única de ventanas (Fase 0)
- [ ] Artefacto versionado de stats por facility (moneda primaria, currency_group, time_zone IANA, mediana/IQR local, fallback_level) computado sobre el universo exacto del scorer (Fase 1)
- [ ] `FS-frame-operational-v1`: magnitud relativa a facility (ratio + z-score robusto), hora local IANA, sin features de reversión ni `capture_delay_seconds`, con flags defensivos (Fase 1)
- [ ] Reentrenar modelo global con features de marco y aprobar gates de sesgo/estabilidad (Fase 1)
- [ ] Contrato API frame-v1: `amount_local`, `facility_time_zone_iana`, `fallback_level`, `frame_flags`; campos opcionales sin defaults silenciosos (Fase 2)
- [ ] Cálculo local-time y relative-amount idéntico en batch y single scorer; paridad batch↔real-time (Fase 2)
- [ ] Calibración de umbrales segmentada (`{segmento: bins}` con fallback facility→currency→group→global) con ventana rodante y versionado (Fase 2)
- [ ] Integración platform: payload real-time con zona horaria, metadata de alerta ampliada, `scorable?` alineado batch/real-time (Fase 3)
- [ ] Shadow mode con dual-run (modelo actual vs frame-v1): alert rate por segmento, Jaccard top-k, sesgo por monto comparado (Fase 4)
- [ ] Human-in-the-loop: top-k a revisión humana, captura de etiquetas, evaluar independencia del proxy de reembolso (Fase 5)

### Out of Scope

- Reabrir el confirmatorio académico HE1–HE4 o cambiar sus veredictos — este trabajo es operativo; la tesis mantiene su cierre honesto (rechazo bajo FS-clean-A-29 vs Tipo A)
- Reportar el AUC contra `pure_fraud` como validación de desempeño — es circularidad parcial declarada; solo diagnóstico
- Entrenar un modelo por moneda al inicio — muchas monedas sin volumen; primero validar calibración segmentada sobre modelo global
- Features de user_tokens — ya se probaron y degradaron el AUC (44% sin token → cluster artificial)
- USD absoluto como variable predictora central — reintroduce el sesgo de escala; USD solo para display/priorización

## Context

- **Origen:** hallazgo `docs/analisis-marcos-referencia.md` + plan `plans/payment-anomaly-detection.md` (revisado, validado y unificado en esta sesión). Evidencia experimental multi-seed en `output/revision/frames_improvement_results.json` y `scripts/exp_frames_improvement.py`.
- **Hallazgo clave validado:** normalizar marcos NO sube el AUC contra proxies (el techo lo pone el proxy) pero corrige la validez de constructo (qué detecta el modelo). Confirmado a escala global: sesgo monto 15,7×→3,3×, off-hours UTC 30%→local 4,4%.
- **Deuda técnica relevante descubierta en el mapa** (`.planning/codebase/CONCERNS.md`), afín a este objetivo:
  - Concern #8: el scorer real-time usa medias globales en vez de per-facility por un `getattr` con nombre de atributo equivocado (`_facility_avg` vs `_facility_avg_amount`; `_staff_stats` vs `_role_currency_stats`) → train/serve skew silencioso ya activo en producción.
  - Concern #3: `capture_delay_seconds` = 0 en real-time (captured_at nulo en after_commit) vs valor real en batch; bug de check `is pd.NaT`.
  - Concern #4: features temporales en UTC sin hora local (el bug central que este proyecto corrige).
  - Concern #6: threshold calibrado sobre test set.
  - Concern #1: arquitectura dual (run_pipeline legacy FS-baseline-31/unified vs confirmatorio disperso en scripts eval_*).
- **Datos:** 6,78M transacciones depuradas (gestión 2025) en ClickHouse; 21 monedas, ~1.876 facilities con zona horaria Rails; splits train Ene-Jun / val Jul-Ago / test Sep-Dic.

## Constraints

- **Metodología**: No supervisado — modelos entrenan sin etiquetas; proxy solo para evaluación. Nunca lenguaje causal.
- **Gobernanza**: El AUC contra `pure_fraud` es circular (proxy definido por features del modelo) — diagnóstico, nunca validación; no filtrar a dashboard ni a tesis.
- **Performance**: Presupuesto real-time del scorer ≈ 0,2s — descarta lookups a ClickHouse por request para stats y features pago-a-pago; artefacto de stats cargado en memoria.
- **Arquitectura**: Toda la lógica de features vive en el scorer; platform (Rails) solo envía payload factual y consume `risk_level`/`percentile`/`is_anomaly` ya calibrados. Paridad obligatoria batch↔real-time.
- **Dependencias**: Cross-repo `ml-fraud-detector` (Python 3.9+, scikit-learn, FastAPI, ClickHouse FINAL) + `platform` (Rails 6.1, pack `anomaly_detection`).
- **Datos**: `FINAL` obligatorio en queries ClickHouse; universo del artefacto de stats idéntico al del scorer (`_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free')`).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| v1 = las 6 fases completas (0–5) | El usuario quiere el sistema operativo maduro end-to-end, no solo el feature frame offline | — Pending |
| Proyecto SpecOps en `ml-fraud-detector/` | Coherente con la convención del workspace (openspec ya vive ahí para compartir git con el código); es el repo ancla del scorer | ✓ Good |
| Modelo global único + calibración segmentada, no un modelo por moneda | Muchas monedas sin volumen suficiente; menor complejidad operativa; validar segmentación primero | — Pending |
| Excluir `capture_delay_seconds` de `FS-frame-operational-v1` | Train/serve skew: ~0 en real-time (captured_at nulo) vs valor real en batch; discrimina poco (flag AUC 0,511) | — Pending |
| Gate de Fase 1 por reducción de sesgo, no por AUC | El techo de AUC lo pone el proxy; el valor real es la validez de constructo del ranking | — Pending |

---
*Last updated: 2026-07-06 after initialization*
