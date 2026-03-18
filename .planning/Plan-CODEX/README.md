# Plan-CODEX

Esta carpeta contiene la planificacion completa de implementacion de la tesis en formato ejecutable por fases.

## Estructura

- `PLAN.md`: plan maestro consolidado.
- `00_MASTER_ROADMAP.md`: roadmap resumido con dependencias, gates y cronograma.
- `01_FASE_0_CONTRATO_Y_ALCANCE.md`: cierre metodologico y contrato tesis-codigo.
- `02_FASE_1_DATOS_Y_CLICKHOUSE.md`: extraccion, deduplicacion, snapshot y decisiones ClickHouse.
- `03_FASE_2_DIAGNOSTICO_CAP2.md`: EDA y salidas para Capitulo 2.
- `04_FASE_3_FEATURE_ENGINEERING.md`: implementacion de features y control de leakage.
- `05_FASE_4_PREPROCESAMIENTO_Y_DATASETS.md`: datasets finales para modelado.
- `06_FASE_5_MODELADO_Y_TUNING.md`: `Isolation Forest`, `LOF`, `One-Class SVM`.
- `07_FASE_6_EVALUACION_Y_HIPOTESIS.md`: HE1-HE4, bootstrap y comparacion.
- `08_FASE_7_SENSIBILIDAD_REPORTING_Y_CIERRE.md`: robustez, reporting LaTeX y cierre tecnico.
- `09_ETL_Y_LINEAGE.md`: pipeline ETL completo, historia previa, idempotencia y manifests.
- `10_CASOS_BORDE_Y_VALIDACIONES.md`: edge cases, fallas previsibles y reglas de manejo.
- `11_PROTOCOLO_DE_EXPERIMENTOS.md`: protocolo de ejecucion, seeds, grids, aceptacion y reproducibilidad.
- `12_COMPARACION_DE_ALGORITMOS_Y_BASELINES.md`: marco de comparacion justo y baselines de sanidad.
- `13_RUNBOOK_DE_EJECUCION.md`: orden exacto de comandos, reruns y recuperacion ante fallos.
- `14_MULTIAGENT_REVIEW.md`: revision cruzada desde perspectivas de datos, ML, arquitectura y tesis.
- `90_RIESGOS_DECISIONES_Y_CHECKLIST.md`: riesgos, decisiones fijas y checklist de salida.

## Orden de ejecucion recomendado

1. `01_FASE_0_CONTRATO_Y_ALCANCE.md`
2. `02_FASE_1_DATOS_Y_CLICKHOUSE.md`
3. `03_FASE_2_DIAGNOSTICO_CAP2.md`
4. `04_FASE_3_FEATURE_ENGINEERING.md`
5. `05_FASE_4_PREPROCESAMIENTO_Y_DATASETS.md`
6. `06_FASE_5_MODELADO_Y_TUNING.md`
7. `07_FASE_6_EVALUACION_Y_HIPOTESIS.md`
8. `08_FASE_7_SENSIBILIDAD_REPORTING_Y_CIERRE.md`
9. `09_ETL_Y_LINEAGE.md`
10. `10_CASOS_BORDE_Y_VALIDACIONES.md`
11. `11_PROTOCOLO_DE_EXPERIMENTOS.md`
12. `12_COMPARACION_DE_ALGORITMOS_Y_BASELINES.md`
13. `13_RUNBOOK_DE_EJECUCION.md`
14. `14_MULTIAGENT_REVIEW.md`
15. `90_RIESGOS_DECISIONES_Y_CHECKLIST.md`

## Regla de uso

No avanzar de fase si el gate de salida de la fase anterior no esta cumplido. El plan esta pensado para terminar la tesis, no para solo producir codigo parcial.

## Reglas adicionales

- El snapshot canonico de tesis debe ser inmutable una vez aprobado.
- Ninguna feature temporal puede calcularse aislando train, val y test sin historia previa si usa ventanas moviles.
- Todo artefacto de salida debe quedar versionado con manifest, seed y timestamp.
