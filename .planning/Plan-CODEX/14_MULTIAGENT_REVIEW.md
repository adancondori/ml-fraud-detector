# Revision Multiagente

## Alcance

Esta revision sintetiza la evaluacion del plan desde cuatro perspectivas especializadas:

- Ingenieria de datos
- Machine Learning aplicado
- Arquitectura de software
- Integridad metodologica de tesis

## Agente 1. Ingenieria de datos

### Hallazgos

- El plan base era correcto, pero no cerraba la necesidad de `warm history`.
- Faltaba una estrategia de lineage y manifests.
- Faltaba politica de joins para dimensiones auxiliares.

### Cambios incorporados

- archivo `09_ETL_Y_LINEAGE.md`;
- reglas de idempotencia y reanudacion;
- warm history obligatoria;
- manifests por capa.

## Agente 2. Machine Learning

### Hallazgos

- El plan base no cerraba variabilidad por seed.
- Faltaban sanity baselines.
- Faltaba una regla de comparacion justa entre modelos.

### Cambios incorporados

- archivo `11_PROTOCOLO_DE_EXPERIMENTOS.md`;
- archivo `12_COMPARACION_DE_ALGORITMOS_Y_BASELINES.md`;
- requirement de estabilidad por seeds;
- baselines internos de sanidad.

## Agente 3. Arquitectura de software

### Hallazgos

- El plan base no definia reanudacion de pipeline;
- faltaban runbook y reglas de overwrite;
- faltaba aislamiento claro entre capas de datos.

### Cambios incorporados

- archivo `13_RUNBOOK_DE_EJECUCION.md`;
- definicion de capas source/raw/canonical/feature/model/results;
- criterio de reanudar desde la ultima capa valida.

## Agente 4. Tesis y QA metodologico

### Hallazgos

- El plan base no explicitaba suficientes edge cases que pueden invalidar HE1-HE4;
- faltaban reglas de bordes temporales entre splits;
- faltaba control sobre edicion manual de tablas.

### Cambios incorporados

- archivo `10_CASOS_BORDE_Y_VALIDACIONES.md`;
- gate de borde temporal;
- prohibicion de editar `.tex` manualmente.

## Conclusiones de la revision

### Estado anterior

Plan estructurado pero todavia vulnerable a:

- leakage por bordes de split;
- resultados no reproducibles;
- ETL no trazable;
- comparaciones injustas entre modelos.

### Estado actual

Plan apto para ejecucion disciplinada, con:

- fases;
- soporte transversal;
- runbook;
- edge cases;
- manifests;
- protocolo experimental;
- auditoria multi-perspectiva.

## Recomendacion final

Seguir el plan en este orden:

1. contrato;
2. ETL y snapshot;
3. edge cases;
4. feature engineering;
5. protocolo de experimentos;
6. modelado;
7. evaluacion;
8. sensibilidad;
9. integracion tesis.
