# Roadmap Maestro

## Objetivo final

Completar la implementacion empirica de la tesis de deteccion de anomalias transaccionales en pagos digitales usando `Isolation Forest`, comparada contra `LOF` y `One-Class SVM`, con resultados reproducibles e integrables en `Tesis-Latex`.

## Resultado esperado

- snapshot de datos 2025 deduplicado y congelado;
- historia previa minima para ventanas y dimensiones auxiliares congelada;
- pipeline reproducible de extraccion a reporting;
- tablas y figuras listas para Capitulo 2 y Capitulo 3;
- hipotesis HE1-HE4 contestadas con evidencia;
- repositorio limpio, probado y defendible.

## Fases

| Fase | Nombre | Estado objetivo | Dependencia |
|---|---|---|---|
| 0 | Contrato y alcance | Tesis y codigo alineados | Ninguna |
| 1 | Datos y ClickHouse | Snapshot correcto congelado | Fase 0 |
| 2 | Diagnostico Cap. 2 | EDA reproducible | Fase 1 |
| 3 | Feature engineering | 20 features y variante de 19 | Fase 1 |
| 4 | Preprocesamiento | Matrices train/val/test cerradas | Fase 3 |
| 5 | Modelado y tuning | Scores reproducibles de 3 modelos | Fase 4 |
| 6 | Evaluacion e hipotesis | HE1-HE4 cerradas | Fase 5 |
| 7 | Sensibilidad, reporting y cierre | Tesis alimentada por artefactos | Fase 6 |

## Piezas transversales obligatorias

- ETL idempotente y reanudable
- manifests y lineage
- protocolo de experimentos
- manejo de edge cases
- runbook de ejecucion
- revision multi-perspectiva

## Gates obligatorios

### Gate A. Universo de estudio

El snapshot debe reproducir aproximadamente:

- `N = 6,784,695`
- `proxy_estricto = 429,442`
- `proxy_amplio = 512,609`
- `train = 3,137,086`
- `val = 1,130,118`
- `test = 2,517,491`

### Gate B. Validez metodologica

No se acepta ninguna feature que:

- use `status` de reembolso como predictor;
- use informacion futura;
- dependa de la propia fila dentro de la ventana;
- convierta el proxy en una senal circular.

### Gate B2. Bordes temporales

No se acepta un pipeline que calcule features con rolling windows sobre `val` o `test` sin arrastrar historia previa desde periodos anteriores.

### Gate C. Independencia del test

El conjunto test no se usa para seleccionar hiperparametros ni decisiones finales de ingenieria.

### Gate D. Robustez del resultado principal

Si el resultado colapsa al remover `user_reversal_ratio_30d`, el modelo de 20 features no puede presentarse como hallazgo principal.

## Cronograma sugerido

### Semanas 1-2

- Fase 0
- Fase 1

### Semanas 3-4

- Fase 2
- Fase 3

### Semanas 5-6

- Fase 4
- Fase 5

### Semanas 7-8

- Fase 6

### Semanas 9-10

- Fase 7

### Semanas 11-12

- integracion completa con `Tesis-Latex`
- revision final y preparacion de defensa

## Secuencia minima viable

1. Cerrar contrato tesis-codigo.
2. Congelar snapshot.
3. Cerrar EDA.
4. Implementar features.
5. Entrenar `Isolation Forest`.
6. Evaluar HE1-HE4.
7. Ejecutar sensibilidad.
8. Exportar tablas y figuras.
9. Ejecutar checklist de edge cases.
10. Integrar resultados a `Tesis-Latex`.

Si uno de esos diez hitos falla, la tesis aun no esta lista.
