# Riesgos, Decisiones y Checklist

## Decisiones ya cerradas

### Decision 1

El universo del estudio se extrae con `FINAL`.

### Decision 2

El proxy principal es estricto y el proxy amplio solo sirve para sensibilidad.

### Decision 3

El modelo principal es `Isolation Forest`.

### Decision 4

La comparacion minima obligatoria es contra `LOF` y `One-Class SVM`.

### Decision 5

El resultado principal debe contrastarse con una variante sin `user_reversal_ratio_30d`.

## Riesgos criticos

### Riesgo 1. Snapshot incorrecto

Impacto:
todo el estudio queda invalido.

Mitigacion:

- `scripts/verify_counts.py`
- `dataset_manifest.json`
- query canonica congelada

### Riesgo 2. Leakage

Impacto:
metricas artificiales e invalidez de la defensa.

Mitigacion:

- tests de leakage
- revision feature por feature
- ablacion obligatoria

### Riesgo 3. Sobrecarga computacional

Impacto:
retraso en entrenamiento y evaluacion.

Mitigacion:

- cache local
- extraccion por split
- subsampling para `OC-SVM`

### Riesgo 4. Sobreingenieria

Impacto:
perder tiempo en piezas que no aportan a la tesis.

Mitigacion:

- no construir API ni realtime scoring
- no introducir modelos extra sin cerrar primero la tesis base

## Checklist de salida por fase

### Fase 0

- [ ] Contrato tesis-codigo escrito
- [ ] Config limpia sin conceptos supervisados

### Fase 1

- [ ] Snapshot parquet creado
- [ ] Warm history creado
- [ ] Conteos validados
- [ ] Manifest generado

### Fase 2

- [ ] EDA reproducible
- [ ] Tablas de Capitulo 2 exportadas

### Fase 3

- [ ] 20 features implementadas
- [ ] Variante de 19 features implementada
- [ ] Tests de leakage pasando
- [ ] Bordes train/val/test validados

### Fase 4

- [ ] Matrices train/val/test congeladas
- [ ] Scaler y metadata guardados

### Fase 5

- [ ] IF tuneado en validation
- [ ] LOF entrenado
- [ ] OC-SVM entrenado con subsample
- [ ] Scores de test guardados
- [ ] Seeds o estabilidad del modelo principal documentadas

### Fase 6

- [ ] HE1 respondida
- [ ] HE2 respondida
- [ ] HE3 respondida
- [ ] HE4 respondida
- [ ] Bootstrap ejecutado

### Fase 7

- [ ] Sensibilidad completa
- [ ] Reporting LaTeX exportado
- [ ] README final escrito
- [ ] `Tesis-Latex` alimentada por resultados
- [ ] Runbook final probado
- [ ] Checklist de edge cases superado

## Checklist final de tesis

- [ ] OG respondido
- [ ] OE1 respondido
- [ ] OE2 respondido
- [ ] OE3 respondido
- [ ] OE4 respondido
- [ ] limitaciones redactadas
- [ ] trabajo futuro delimitado
- [ ] defensa apoyada por artefactos reproducibles
