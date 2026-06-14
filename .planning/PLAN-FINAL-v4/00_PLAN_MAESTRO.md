# Plan Maestro v4

## Objetivo

Construir una version defendible del sistema que alcance AUC >= 0.70 sobre reembolsos Tipo A en pagos transaccionales de la gestion 2025, sin leakage temporal y con utilidad operativa medida por precision top-k.

Gate A0 demuestra viabilidad, pero no reemplaza el pipeline final. La tesis solo puede declarar resultados finales despues de `run_pipeline_v4.py all --final` con USD normalizado, catalogo v4 completo y artefactos reproducibles.

## Resultado esperado

1. Dataset 2025 congelado con train, val, discovery y test final.
2. Extraccion enriquecida con columnas y tablas de alta senal.
3. Feature engineering point-in-time con catalogo v4.
4. Ranker supervisado HGB-CLEAN entrenado solo con train 2025.
5. Thresholds y metricas seleccionados solo con validacion.
6. Test final octubre-diciembre 2025 si septiembre se usa para discovery.
7. Baselines IF/LOF/OC-SVM conservados como comparacion academica.
8. Reporting reproducible para tesis y evaluacion operativa.

## Splits v4

| Split | Periodo | Uso |
|---|---|---|
| Train | 2025-01-01 a 2025-06-30 | Fit de features, encoders y modelo |
| Validation | 2025-07-01 a 2025-08-31 | Tuning, threshold, seleccion |
| Discovery | 2025-09-01 a 2025-09-30 | Busqueda de senales y stress temporal |
| Final test | 2025-10-01 a 2025-12-31 | Evaluacion principal no tocada |
| Legacy test | 2025-09-01 a 2025-12-31 | Solo comparabilidad con resultados previos |

No usar datos de 2024. Las ventanas rolling arrancan en enero 2025 y el cold-start de enero se documenta como limitacion.

## Fases

| Fase | Archivo | Gate |
|---|---|---|
| -1 | Preflight interno | Gaps materiales cerrados |
| 0 | `01_CONTRATO_ALCANCE.md` | Contrato v4 aprobado |
| 1 | `02_DATOS_SNAPSHOT.md` | Snapshot 2025 reproducible |
| 2 | `03_EDA_CAPITULO2.md` | Senales priorizadas y leakage audit |
| 3 | `04_FEATURE_ENGINEERING.md` | Features point-in-time |
| 4 | `05_PREPROCESAMIENTO.md` | Encoders train-only |
| 5 | `06_MODELADO_TUNING.md` | Modelo supera val gate |
| 6 | `07_EVALUACION_HIPOTESIS.md` | Test final supera gate |
| 7 | `08_SENSIBILIDAD.md` | Ablaciones completas |
| 8 | `09_REPORTING.md` | Artefactos generados |
| 9 | `10_ORQUESTADOR.md` | Pipeline end-to-end |
| 10 | `11_TESTS_CLEANUP_INTEGRACION.md` | Tests green |
| 11 | `12_SINGLE_TRANSACTION_SCORER.md` | Batch/single equivalentes |

## Gates principales

### Gate A: Universo 2025

- Solo transacciones con `created_at` en 2025.
- `payment_method != 'reversal'`.
- `payment_method != 'free'`.
- `user_id != 0`.
- `_peerdb_is_deleted = 0`.
- Tablas ReplacingMergeTree consultadas con `FINAL`.

### Gate A0: Pre-benchmark reproducible

No iniciar Fase 1 sin:

- `scripts/run_hgb_benchmark.py` versionado;
- resultados guardados en `output/v4/benchmarks/baseline_hgb_clean_strict.json`;
- feature list y seed guardados;
- metricas consistentes con el plan o gates actualizados.
- benchmark strict point-in-time para `reservations_users`, o exclusion explicita de esas features.
- baseline de regla manual simple reportado.

Estado Gate A0:

- V4-CLEAN strict as-of RU: AUC Oct-Dic 0.8339, AP/base 5.78, P@1% 73.4%.
- V4-CLEAN-NO-RU: AUC Oct-Dic 0.8285, AP/base 5.64, P@1% 71.3%.
- SIMPLE-RULE: AUC Oct-Dic 0.7052, AP/base 2.29, P@1% 35.8%.

Interpretacion: SIMPLE-RULE ya supera AUC 0.70, por lo que la contribucion de ML debe defenderse principalmente por utilidad top-k, AP/base y mejora operacional, no solo por cruzar el umbral AUC.

### Gate B: Anti-leakage

No se permite usar:

- `status` como feature directa.
- Estados posteriores de reserva: `incident_enum`, `reservation_status`, `payment_completed`.
- `membership_state=payment_refunded`.
- `comments_count` de pago como feature principal.
- Variables calculadas con filas futuras o la fila actual dentro de ventanas.
- Columnas mutables sin filtro temporal estricto: `last_change_at`, `updated_at`, `deleted_at`, `total_paid_out`, `most_recent_date`, `approval_status`, `status_enum`, `reservations_users.payment_id`, `user_tokens.updated_at`, `payment_discounts.updated_at`.
- Features de participantes (`participant_count`, `has_invited`, `has_free_pass`, `has_resident`, `teacher_rows`) sin condicion `reservations_users.created_at <= payment.created_at`.

### Gate C: Modelo principal limpio

El modelo v4 principal debe ser `V4-CLEAN`: sin historial directo de reembolso y sin target encoding. Debe cumplir en validation:

- AUC >= 0.78.
- AP/base >= 3.5.
- P@1% >= 35%.

No mirar test final si Gate C falla. Si validation cae por debajo de AUC 0.78 despues de USD y catalogo v4, revisar features y leakage antes de continuar.

### Gate D: Test final

El test final octubre-diciembre 2025 debe cumplir:

- AUC >= 0.70 obligatorio.
- AUC objetivo >= 0.80.
- AP/base >= 3.
- P@1% >= 40%.
- P@5% >= 25%.
- AUC mensual Oct/Nov/Dic >= 0.75.
- Rango mensual max(AUC)-min(AUC) <= 0.05, o drift explicado en sensibilidad.

### Gate E: Robustez

El modelo principal ya excluye historial directo de reembolso. Como control adicional, la variante boosted con target encoding debe reportarse separada y no puede sustituir al resultado limpio en la tesis.

La variante `V4-CLEAN` debe mantener:

- AUC >= 0.75.
- AP/base >= 2.5.

Si no se cumple, el resultado principal no puede presentarse como detector generalizable defendible.

## Kill Switches

Detener o replantear v4 si el pipeline final cumple cualquiera de estas condiciones:

- Test Oct-Dic AUC < 0.70 en V4-CLEAN.
- V4-CLEAN queda por debajo de SIMPLE-RULE en AUC y P@1% simultaneamente.
- Leakage detectado en features principales sin correccion.
- Validation AUC < 0.78 y no mejora tras revisar features, antes de mirar test.
- El pipeline final no reproduce desde cero los artefactos de datos, features, modelo y metricas.

## Arquitectura objetivo

```
src/fraud_detector/
├── data/
│   └── loader.py                    # SQL 2025 enriquecido
├── features/
│   ├── engineering.py               # features numericas point-in-time
│   ├── encoders.py                  # NUEVO: target/frequency encoders
│   └── supervised_pipeline.py       # NUEVO: matriz v4
├── models/
│   ├── trainer.py                   # baselines no supervisados
│   └── risk_ranker.py               # NUEVO: HGB supervisado
├── evaluation/
│   ├── metrics.py                   # AUC/AP/EF/P@k/bootstrap
│   └── leakage.py                   # NUEVO: leakage audit
├── reporting/
│   └── latex_tables.py
└── scoring/
    └── scorer.py                    # scorer batch/single v4
```

## Principios

1. Tipo A es proxy de reembolso, no fraude confirmado.
2. El ranker supervisado optimiza riesgo de reembolso, no anomalia universal.
3. IF/LOF/OC-SVM quedan como baselines, no como modelo operativo principal.
4. Las decisiones de tuning se hacen en validation, nunca en test final.
5. Septiembre puede usarse para descubrir senales; si se usa, test principal pasa a octubre-diciembre.
6. Toda feature debe tener definicion de tiempo de disponibilidad.
7. El scorer individual debe reproducir el pipeline batch.
8. El resultado principal no usa variables derivadas de reembolsos previos ni target encodings basados en la etiqueta.
