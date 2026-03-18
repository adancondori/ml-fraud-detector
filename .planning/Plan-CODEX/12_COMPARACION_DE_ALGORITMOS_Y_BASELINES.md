# Comparacion de Algoritmos y Baselines

## Proposito

Definir una comparacion justa, completa y defendible entre algoritmos.

## Modelos principales del estudio

- `Isolation Forest`
- `LOF`
- `One-Class SVM`

## Baselines de sanidad interna

No son el centro de la tesis, pero sirven para detectar errores gruesos:

- ranking aleatorio;
- ranking por monto;
- z-score de monto;
- score heuristico simple si aparece un patron fuerte en EDA.

## Criterios de comparacion

### Criterios obligatorios

- AUC-ROC
- Average Precision
- Precision@5%
- Enrichment Factor

### Criterios recomendados

- Recall@5%
- runtime entrenamiento
- runtime scoring
- memoria pico

## Condiciones de equidad

Cada comparacion debe usar:

- el mismo snapshot;
- el mismo test set;
- el mismo proxy;
- el mismo set de features;
- la misma politica de nulos;
- la misma orientacion de scores.

## Presentacion recomendada

### Tabla principal

Una fila por modelo y columnas:

- AUC-ROC
- AP
- Precision@5%
- Recall@5%
- EF
- runtime
- memoria

### Tabla secundaria

Comparacion:

- 20 features
- 19 features
- proxy estricto
- proxy amplio

## Interpretacion

### Si IF gana

Se reporta como soporte de HE4.

### Si IF no gana

La tesis sigue siendo valida si:

- OE3 queda respondido;
- HE4 se rechaza con evidencia clara;
- se documenta cual metodo fue superior y por que.

## Regla para empates

Si dos modelos quedan muy cerca:

- priorizar el que mantenga mejor estabilidad;
- si aun asi hay empate, priorizar el mas simple o escalable;
- documentar la razon.
