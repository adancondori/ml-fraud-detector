# 09 Reporting v4

## Objetivo

Generar tablas y figuras reproducibles que expliquen claramente el cambio de paradigma y la mejora empirica.

## Tablas principales

| Tabla | Contenido |
|---|---|
| T-v4-01 | Splits 2025 y tasas Tipo A |
| T-v4-02 | Hallazgos EDA por fuente/categoria |
| T-v4-03 | Catalogo de features v4 |
| T-v4-04 | Comparacion IF actual vs HGB v4 |
| T-v4-05 | Metricas top-k y carga operativa |
| T-v4-05b | Costo-beneficio operativo por capacidad de revision |
| T-v4-06 | Ablacion sin historial de reembolso |
| T-v4-07 | Estabilidad mensual Oct-Dic |
| T-v4-08 | Leakage audit |
| T-v4-09 | Comparacion HGB-CLEAN vs regla manual simple |
| T-v4-10 | Validacion humana de muestra Tipo A |
| T-v4-11 | Alineacion tesis: IF original vs HGB extension operativa |

## Figuras

- ROC: IF actual vs HGB v4.
- Precision-recall: IF actual vs HGB v4.
- Lift chart top-k.
- Tasa Tipo A mensual.
- Tasa Tipo A por `source_enum`.
- Tasa Tipo A por `payment_method`.
- Importancia de variables o permutation importance.

## Narrativa obligatoria

Incluir estas frases metodologicas:

- "El modelo v4 estima riesgo de reembolso Tipo A, no fraude confirmado."
- "El proxy se utiliza como etiqueta supervisada por necesidad operacional."
- "Los modelos no supervisados se conservan como baseline academico."
- "Las variables con leakage probable fueron excluidas del modelo principal."
- "La comparacion principal incluye una regla manual simple para evitar una mejora trivial por cambiar de no supervisado a supervisado."
- "Sin revision humana, el resultado no permite estimar que fraccion de Tipo A corresponde a fraude real."
- "Como SIMPLE-RULE ya supera AUC 0.70, la contribucion de V4-CLEAN se interpreta principalmente por precision top-k, AP/base y carga operativa."
- "Gate A0 prueba viabilidad; los resultados finales de tesis requieren pipeline completo con USD normalizado y catalogo v4."

## Costo-beneficio operativo

Reportar al menos:

- alertas mensuales esperadas por top 0.1%, 0.5%, 1%, 2% y 5%;
- horas operador = alertas * minutos_revision / 60;
- jornadas operador = horas / 8;
- reembolsos Tipo A capturados esperados;
- precision esperada por capacidad mensual;
- comparacion contra SIMPLE-RULE para la misma capacidad.

Supuesto base:

```text
minutos_revision_por_alerta = 4
jornada_horas = 8
```

Si se consigue dato de negocio:

```text
beneficio_estimado = reembolsos_evitable_estimados * costo_promedio_reembolso
costo_revision = horas_operador * costo_hora_operador
ROI_proxy = (beneficio_estimado - costo_revision) / costo_revision
```

Sin dato de costo real, reportar costo-beneficio como escenario y no como ahorro confirmado.

## Tesis-Latex

Actualizar en paralelo:

- problema;
- hipotesis;
- objetivo general;
- objetivos especificos;
- metodologia;
- conclusiones.

Usar dos lineas: IF no supervisado como resultado academico insuficiente, HGB V4-CLEAN como extension operativa para riesgo de reembolso.

## Artefactos

```text
output/v4/tables/*.tex
output/v4/figures/*.{png,pdf}
output/v4/reporting/report_manifest.json
```

## Gate

Ninguna tabla de resultados se llena manualmente.

No cerrar conclusiones operativas si:

- falta `run_pipeline_v4.py all --final`;
- falta USD normalizado en el pipeline final;
- falta A7 y se quiere usar lenguaje antifraude;
- no se reporta diciembre como el mes mas debil si se mantiene el patron de Gate A0.
