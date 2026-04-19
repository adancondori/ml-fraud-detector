# Plan Maestro Unificado — PLAN-FINAL

> Sintesis de Plan-CODEX (estrategia, gobernanza, edge cases) y Plan-CLAUDE (implementacion detallada, codigo, correcciones anti-leakage).

## Objetivo final

Completar la implementacion empirica de la tesis de deteccion de anomalias transaccionales en pagos digitales usando Isolation Forest, comparada contra LOF y One-Class SVM, con resultados reproducibles e integrables en `Tesis-Latex`.

## Resultado esperado

1. Snapshot de datos 2025 deduplicado y congelado con warm history.
2. Pipeline reproducible de extraccion a reporting (`run_pipeline.py`).
3. Tablas LaTeX y figuras PDF/PNG para Capitulo 2 y Capitulo 3.
4. Hipotesis HE1-HE4 contestadas contra proxy unificado (OR de 5 tipos) con evidencia y bootstrap CI 95%.
5. Analisis de sensibilidad (proxy unificado vs Tipo A individual, metricas desagregadas por tipo B-E, proxy amplio, Feature #18, estabilidad temporal, ablacion IF-31 vs IF-21, metricas por segmento, tipologia SHAP, perfil usuario).
6. Analisis post-hoc de anomalias por centro, actor operativo y moneda (concentracion de descuentos).
7. Repositorio limpio, probado y defendible.

## Indice de fases

| Fase | Archivo | Descripcion | Gate |
|------|---------|-------------|------|
| 0 | `01_CONTRATO_ALCANCE.md` | Contrato tesis-codigo, definiciones fijas | — |
| 1 | `02_DATOS_SNAPSHOT.md` | Extraccion ClickHouse + JOINs (is_staff, user_created_at), snapshot, warm history | Gate A |
| 2 | `03_EDA_CAPITULO2.md` | Diagnostico y EDA para OE2 | — |
| 3.5 | — | Normalizacion monetaria: `amount_usd` via `default.exchange_rates` (snapshot) | — |
| 4 | `04_FEATURE_ENGINEERING.md` | 31 features oficiales (8 grupos A-H) + variantes 30 y 21 | Gate B |
| 5 | `05_PREPROCESAMIENTO.md` | StandardScaler fit en train | — |
| 6 | `06_MODELADO_TUNING.md` | IF + LOF + OC-SVM con grid search | Gate C |
| 7 | `07_EVALUACION_HIPOTESIS.md` | HE1-HE4, bootstrap, temporal | Gate D |
| 8 | `08_SENSIBILIDAD.md` | Proxy, F#18, SHAP, post-hoc, ablacion 31/21, segmentos, tipologia, perfil usuario | — |
| 9 | `09_REPORTING.md` | Tablas LaTeX + figuras PDF/PNG | — |
| 10 | `10_ORQUESTADOR.md` | `run_pipeline.py` con CLI | — |
| 11 | `11_TESTS_CLEANUP_INTEGRACION.md` | Tests, limpieza, integracion tesis | Gate E |

**Fase 3.5 es NUEVA** — normaliza el 25.9% de transacciones non-USD usando `default.exchange_rates` (snapshot). Se acepta la tasa actual como aproximacion: la variacion vs. tasas historicas 2025 es < 10% para monedas volatiles (COP, PKR) y < 3% para estables (CAD, AUD, EUR), insuficiente para afectar la deteccion de anomalias. Se documenta como limitacion metodologica en la tesis.

## Documentos transversales

| Doc | Archivo | Proposito |
|-----|---------|-----------|
| T1 | `A1_ETL_LINEAGE_EDGE_CASES.md` | ETL idempotente, lineage, 20 edge cases |
| T2 | `A2_PROTOCOLO_RUNBOOK.md` | Protocolo de experimentos + runbook de ejecucion |
| T3 | `A3_RIESGOS_CHECKLIST.md` | Riesgos, decisiones cerradas, checklist por fase |
| T4 | `A4_GOBERNANZA_PRIVACIDAD_CONTRATOS.md` | Gobernanza de datos, privacidad, data contracts y contratos de artefactos |
| T5 | `A5_AUDITORIA_END_TO_END.md` | Auditoria final del plan y criterios de suficiencia end-to-end |
| T6 | `A6_VERIFICACION_CLICKHOUSE.md` | Ground truth: columnas, monedas, exchange rates, SQL con JOINs |

## Arquitectura objetivo del repositorio

> **Nota de coherencia:** esta seccion refleja la arquitectura objetivo alineada con los nombres reales del repo actual. Los modulos que aun no existen se marcan como **pendiente**; no se usan nombres obsoletos.

```
ml-fraud-detector/
├── config/
│   └── config.py                        # REESCRIBIR (Pydantic Settings, sin supervisado)
├── src/fraud_detector/
│   ├── data/
│   │   ├── clickhouse_connector.py      # MANTENER (con ajuste menor)
│   │   └── loader.py                    # REESCRIBIR → DataManager
│   ├── features/
│   │   ├── engineering.py               # REESCRIBIR → FeatureEngineer (31 features, 8 grupos)
│   │   └── preprocessor.py             # REESCRIBIR → UnsupervisedPreprocessor
│   ├── models/
│   │   └── trainer.py                   # REESCRIBIR/MANTENER → ModelTrainer
│   ├── evaluation/
│   │   ├── metrics.py                   # REESCRIBIR/MANTENER → evaluate_scores, bootstrap_ci, precision_at_k, enrichment_factor
│   │   ├── posthoc_analysis.py          # PENDIENTE → analisis centro/actor/moneda
│   │   └── segment_analysis.py          # PENDIENTE → metricas por rol/categoria
│   ├── reporting/                       # PENDIENTE
│   │   ├── __init__.py
│   │   ├── latex_tables.py             # ThesisTableGenerator
│   │   └── figures.py                  # ThesisFigureGenerator
│   └── utils/
│       ├── logger.py                    # MANTENER
│       └── currency.py                  # NUEVO → CurrencyNormalizer
├── notebooks/                           # OPCIONAL / pendiente
│   ├── 01_eda_capitulo2.ipynb          # Opcional
│   └── 02_resultados_capitulo3.ipynb   # Opcional
├── scripts/
│   ├── verify_counts.py                # NUEVO
│   └── inspect_clickhouse.py           # ACTUALIZAR
├── data/external/
│   └── exchange_rates.csv              # Tabla/csv de tasas usada por CurrencyNormalizer
├── tests/                               # ACTUALIZAR + NUEVOS
├── run_pipeline.py                      # NUEVO (orquestador principal)
├── data/processed/                      # Parquets (gitignored)
├── output/                              # Tablas, figuras, modelos, scores (gitignored)
├── requirements.txt                     # ACTUALIZAR (sin supervisado)
├── .env.example                         # ACTUALIZAR
├── CLAUDE.md                            # ACTUALIZAR
└── README.md                            # ACTUALIZAR
```

**ELIMINAR:** `balancing.py`, `run_simple_rf.py`, `check_git_status.py`, `verify_setup.py`, `QUICKSTART.md`, `EXPERT_AUDIT.md`, `01_exploratory_analysis.ipynb`

## Flujo de datos

```
ClickHouse (6.7M txns) + JOINs (facilities_users, users) + Warm History (Dic 2024)
  │
  ▼
[Fase 1] → data/processed/{warm,train,val,test}_raw.parquet
            (incl. is_staff, user_role, user_created_at)
            output/manifests/dataset_manifest.json
  │
  ▼
[Fase 3.5] → Normalizacion: amount_usd = reservation_paid_out / rate
              Fuente: default.exchange_rates (snapshot ClickHouse)
              src/fraud_detector/utils/currency.py (CurrencyNormalizer)
  │
  ▼
[Fase 4] → data/processed/{train,val,test}_features.parquet (31 features + metadata)
            output/models/feature_engineer.joblib
  │
  ▼
[Fase 4] → output/scores/X_{train,val,test}.npy
            output/models/scaler.joblib
  │
  ▼
[Fase 5] → output/models/{isolation_forest,lof,ocsvm}.joblib
            output/grid_search_{if,lof,ocsvm}.csv
            output/models/best_params_{if,lof,ocsvm}.json
  │
  ▼
[Fase 6] → output/scores/test_scores.parquet (id + created_at + 3 scores)
            output/results.json (HE1-HE4, bootstrap, temporal)
  │
  ▼
[Fase 7] → output/results_sensitivity.json
            output/results_posthoc.json
            output/figures/shap_summary.{pdf,png}
  │
  ▼
[Fase 8] → output/tables/table_3_*.tex (incluyendo 3.20-3.23 post-hoc)
            output/figures/*.{pdf,png} (incluyendo post-hoc centro/actor/moneda)
```

## Gates obligatorios

### Gate A — Universo de estudio

El snapshot debe reproducir (cifras de referencia de la tesis):

| Concepto | Tesis | Pipeline (real) | Diff |
|----------|-------|-----------------|------|
| N total depurado | 6,784,696 | 6,784,694 | -2 (OK) |
| Proxy estricto | 429,418 (6.33%) | 429,498 (6.33%) | +80 (OK) |
| Proxy amplio | 512,582 (7.55%) | 512,676 (7.56%) | +94 (OK) |
| Train (Ene-Jun) | 3,137,086 | 3,137,085 | -1 (OK) |
| Val (Jul-Ago) | 1,130,118 | 1,130,118 | 0 |
| Test (Sep-Dic) | 2,517,492 | 2,517,491 | -1 (OK) |

**Gate A: PASS** (todas las diferencias < 0.01%). No continuar si los conteos divergen mas de ±1%.

### Gate B — Validez metodologica (anti-leakage)

No se acepta ninguna feature que:
- use `status` de reembolso como predictor;
- use informacion futura (la fila actual o posteriores);
- dependa de la propia fila dentro de la ventana;
- convierta el proxy en senal circular.

Ademas, las features de `val` y `test` deben arrastrar historia previa correctamente (warm history).

### Gate C — Independencia del test

El test set NO se usa para seleccionar hiperparametros ni decisiones de ingenieria.

### Gate D — Robustez del resultado principal

Si el resultado colapsa al remover `user_reversal_ratio_30d` (delta AUC >= 0.02), el modelo de 31 features no puede presentarse como hallazgo principal. Se evaluan variantes IF-31, IF-30 (sin F18) e IF-21 (features originales) para cuantificar el aporte de los grupos F, G, H.

### Gate E — Reporting y tests

No cerrar Cap 2/3 con tablas manuales; todo artefacto debe salir del pipeline. Tests automaticos pasan.

## Principios de implementacion

1. `FINAL` no es opcional para reproducir la tesis.
2. El proxy se usa solo para evaluacion, nunca para entrenar.
3. Toda feature debe ser auditada contra leakage temporal y circularidad.
4. Pipeline offline y reproducible, no API productiva.
5. Todo resultado de Cap 2 y Cap 3 sale del codigo.
6. El test set temporal permanece intocable hasta evaluacion final.
7. La comparacion entre modelos usa misma base, features, proxy y split.
8. Cada fase deja artefactos serializados para no repetir extracciones.
9. Score alto = mas anomalo (convencion: `-decision_function(X)`).

## Correcciones criticas incorporadas (de ambos planes)

| # | Correccion | Origen |
|---|-----------|--------|
| 1 | Anti-leakage features 15, 16: `cumulative_nunique_shifted` O(n) en vez de `expanding().apply(nunique)` | CLAUDE |
| 2 | Feature 18 shift(1) dentro del grupo, no global | CLAUDE |
| 3 | Feature 18: `user_account_age_days` usa `first_txn` del training set | CLAUDE |
| 4 | Filtros post-extraccion: `user_id > 0` y `_peerdb_is_deleted = 0` | CODEX |
| 5 | Grid search IF incluye contamination: 4×4×5×3 = 240 combos | CLAUDE |
| 6 | Grid search para LOF y OC-SVM (comparacion justa en HE4) | CLAUDE |
| 7 | Rank-biserial r corregida: positiva cuando anomaly > normal | CLAUDE |
| 8 | Holm-Bonferroni aplicada a HE1-HE4 | CLAUDE |
| 9 | Estabilidad temporal: AUC mensual en test set | CLAUDE |
| 10 | Multiples k: top-K evaluado a 1%, 2%, 5%, 10% | CLAUDE |
| 11 | Scores como parquet con `id` y `created_at` | CLAUDE |
| 12 | Warm history obligatoria (Dic 2024) para features con ventana | CODEX |
| 13 | Epsilon 1e-8 en ratios (era 0.01) | CLAUDE |
| 14 | Checkpoint/resume en grid search | CLAUDE |
| 15 | Feature 16 consolidada como `user_distinct_facilities_30d` | CLAUDE |
| 16 | Multi-seed recomendado (42, 52, 62) con justificacion si solo una | CODEX |
| 17 | Sanity baselines (random, monto, z-score) para validacion interna | CODEX |
| 18 | Deduplicacion con `FINAL` obligatoria | CODEX |
| 19 | Dataset manifest JSON por split con trazabilidad | CODEX+CLAUDE |
| 20 | Prerequisite validation entre pasos del pipeline | CLAUDE |
| 21 | SQL canonico agrega `currency` y `paid_by_manager` para analisis post-hoc | CLAUDE |
| 22 | Analisis post-hoc: concentracion de anomalias por centro, manager, moneda y descuentos | CLAUDE |
| 23 | Tablas 3.20-3.23 y figuras post-hoc en Fase 8 Reporting | CLAUDE |
| 24 | Gate de identidad del actor agregado al post-hoc de managers/descuentos | CODEX |
| 25 | Gobernanza, privacidad y data contracts agregados al plan final | CODEX |
| 26 | 31 features en 8 grupos (A-H) alineados con tesis Cap. 3 (F06 y F21 eliminadas) | CLAUDE |
| 27 | Normalizacion multi-moneda a USD obligatoria antes de feature engineering | CLAUDE |
| 28 | Ablacion IF-31 vs IF-21 para cuantificar aporte de grupos F, G, H | CLAUDE |
| 29 | Metricas por rol de usuario y categoria de pago | CLAUDE |
| 30 | Tipologia de anomalias derivada de SHAP (9 tipos) | CLAUDE |
| 31 | Perfil de riesgo agregado por usuario | CLAUDE |
| 32 | Grid search IF incluye contamination (240 combos) | CLAUDE |
| 33 | Proxy unificado (OR 5 tipos A-E) como evaluacion principal; Tipo A individual como sensibilidad | REVISION |
| 34 | ProxyLabeler calcula Tipos B-E con reglas operacionales documentadas | REVISION |
| 35 | Metricas desagregadas por tipo de proxy (A, B, C, D, E) en sensibilidad | REVISION |
| 36 | HE2 criterio AP usa tasa base del proxy unificado (no fija 6.33%) | REVISION |
| 37 | Tabla 3.17 per-type y Tabla 4.1 resumen hipotesis agregadas al reporting | REVISION |

## Estimaciones

### Tiempo (end-to-end)

| Paso | Tiempo |
|------|--------|
| Extraccion (3 splits + warm) | 5-15 min |
| Feature engineering (por split) | 5-12 min |
| Preprocesamiento | < 1 min |
| Grid search IF (240 combos) | 30-45 min |
| Grid search LOF (3 combos) | 15-30 min |
| Grid search OC-SVM (6 combos) | 3 min |
| Scoring test | 2-5 min |
| Evaluacion + bootstrap (3 modelos) | 30-50 min |
| SHAP | 5-10 min |
| Sensibilidad completa | 20-30 min |
| Reportes | < 2 min |
| **Total** | **~2-3 horas** |

### Disco

| Componente | Tamano |
|-----------|--------|
| Parquets raw (3 splits + warm) | ~1.8 GB |
| Feature parquets | ~900 MB |
| Arrays numpy | ~1.5 GB |
| Modelos (IF ~200MB, LOF ~1GB, OC-SVM ~100MB) | ~1.5 GB |
| Otros (scores, tablas, figuras) | ~100 MB |
| **Total** | **~5.5 GB** |

### Memoria pico

~5-6 GB durante rolling groupby en feature engineering.

## Cronograma (alineado con tesis — 3 meses / 12 semanas)

El cronograma del perfil de tesis (UAGRM) define 9 actividades que se mapean a las fases del pipeline:

| Actividad (Tesis) | Fases Pipeline | Semana | Estado |
|---|---|---|---|
| 1. Elaboración del perfil | Fase 0 (Contrato) | S1-S2 | COMPLETADA |
| 2. Presentación y defensa del perfil | — (académico) | S3 | COMPLETADA |
| 3. Revisión documental y marco teórico | Cap 1 (.tex) | S2-S4 | COMPLETADA |
| 4. Diagnóstico del sistema actual | Fases 1-2 (Datos + EDA) | S3-S5 | COMPLETADA |
| 5. Recolección y preparación del dataset | Fases 1, 3, 4 (Snapshot + Features + Preproc) | S4-S6 | EN PROGRESO |
| 6. Diseño e implementación del modelo | Fase 5 (Modelado + Tuning) | S5-S8 | PENDIENTE |
| 7. Evaluación del modelo | Fases 6, 7 (Hipótesis + Sensibilidad) | S7-S9 | PENDIENTE |
| 8. Análisis de resultados y redacción | Fases 8, 9, 10 (Reporting + Integración) | S9-S11 | PENDIENTE |
| 9. Presentación y defensa de tesis final | — (académico) | S12 | PENDIENTE |

### Cronograma tecnico detallado (dentro de actividades 5-8)

| Semanas | Fases | Meta |
|---------|-------|------|
| 1 (actual) | 3 | Completar 31 features (Grupos F, G, H), ejecutar sobre parquets |
| 2 | 4 | Preprocesamiento (StandardScaler fit en train) |
| 3-4 | 5 | IF + LOF + OC-SVM con grid search |
| 5-6 | 6, 7 | HE1-HE4 cerradas, sensibilidad, SHAP |
| 7-8 | 8, 9, 10 | Reporting, tests, integración .tex, 61 [POR COMPLETAR] llenados |

## Secuencia minima viable

1. Cerrar contrato tesis-codigo.
2. Congelar snapshot con `FINAL` + warm history.
3. Completar EDA para Capitulo 2.
4. Implementar 31 features con anti-leakage.
5. Entrenar IF, LOF, OC-SVM.
6. Evaluar HE1-HE4 en test temporal.
7. Ejecutar sensibilidad.
8. Exportar tablas y figuras.
9. Pasar checklist de edge cases.
10. Integrar resultados en Tesis-LaTeX.

Si uno de estos diez hitos falla, la tesis aun no esta lista.
