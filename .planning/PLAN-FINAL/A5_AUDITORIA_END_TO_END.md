# A5 — Auditoría End-to-End

## Objetivo

Verificar si `PLAN-FINAL` ya es suficientemente completo para ejecutar sin improvisación crítica.

## Estado por perspectiva

### Datos

- `FINAL` definido
- warm history definida
- manifests y lineage definidos
- edge cases principales cubiertos

### Feature engineering

- catálogo oficial de 20 features
- variante de 19 features
- tests TDD y anti-leakage
- bordes temporales explícitos

### Modelado

- IF, LOF y OC-SVM definidos
- comparación justa
- baselines internos
- bootstrap y estabilidad temporal

### Reporting post-hoc

- análisis por centro, moneda y descuentos
- gate explícito para identidad del actor

### Gobernanza

- política de privacidad
- contratos de entrada/salida
- regla de documento interno vs público

## Juicio final

El plan queda **apto para ejecución end-to-end**.

Lo único que permanece abierto por naturaleza y debe resolverse en discovery de datos:

- si la identidad del actor manager es validable;
- si `currency` requiere normalización adicional;
- si aparece una anomalía estructural no prevista que obligue a volver a Fase 0.

## Siguiente paso

La siguiente acción correcta ya no es seguir planificando, sino comenzar la implementación desde:

1. `01_CONTRATO_ALCANCE.md`
2. `02_DATOS_SNAPSHOT.md`
3. `10_ORQUESTADOR.md`
