# Fase 7. Sensibilidad, Reporting y Cierre

## Proposito

Blindar el resultado principal, producir artefactos finales para la tesis y cerrar el repositorio.

## Analisis de sensibilidad obligatorio

### Proxy

- resultado principal con proxy estricto
- contraste con proxy amplio

### Features

- modelo con 20 features
- modelo con 19 features sin `user_reversal_ratio_30d`

### Parametros

- verificar que el resultado principal no dependa de una unica configuracion extrema de `contamination`

### Subgrupos

Si el tiempo alcanza:

- por canal;
- por gateway principal;
- por rango de monto.

## Reporting

### Para la tesis

- tablas `.tex`
- figuras `.pdf`
- resumen de interpretacion por objetivo
- apendice con SQL y catalogo de features

### Para el repositorio

- `README` final
- guia de reproduccion
- inventario de artefactos generados

## Integracion con `Tesis-Latex`

### Capitulo 2

- insertar tablas y figuras del diagnostico

### Capitulo 3

- insertar pipeline, tuning, metricas, comparacion y sensibilidad

### Conclusiones

- responder OG y OE1-OE4 solo con base en resultados ya ejecutados

## Cierre tecnico

- limpiar codigo muerto;
- mantener solo el flujo valido para tesis;
- correr `pytest`;
- dejar un comando unico o secuencia minima reproducible.

## Entregables

- `output/tables/*.tex`
- `output/figures/*.pdf`
- `README.md`
- suite final de pruebas
- parrafos base para conclusiones y limitaciones

## Gate de salida

La tesis esta tecnicamente lista cuando:

- los resultados principales y de sensibilidad ya existen;
- Capitulo 2 y 3 pueden llenarse sin trabajo manual fuera del repo;
- el pipeline corre desde snapshot hasta reporting;
- la documentacion permite repetir el estudio.
