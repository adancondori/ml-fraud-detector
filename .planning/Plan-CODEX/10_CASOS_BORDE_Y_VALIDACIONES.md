# Casos Borde y Validaciones

## Proposito

Enumerar fallos previsibles del pipeline y dejar la respuesta esperada antes de codificar.

## Bordes de datos

### 1. Duplicados aun despues de `FINAL`

Accion:

- medir `countDistinct(id)` vs `count()`;
- si hay duplicados residuales, definir deduplicacion secundaria reproducible;
- documentar el criterio.

### 2. `status` vacio o desconocido

Accion:

- mapear a categoria explicita;
- nunca asumir que vacio es normal sin medir distribucion.

### 3. `reservation_paid_out = 0`

Accion:

- ya queda mayormente fuera por filtro de `free`, pero si aparece por otros medios, definir manejo en ratios y logs;
- proteger `log(0)` y divisiones.

### 4. Montos negativos o descuentos mayores al monto

Accion:

- registrar casos;
- decidir si se excluyen, winsorizan o conservan como anomalia operacional.

### 5. Multiples monedas

Accion:

- verificar si `reservation_paid_out` esta efectivamente en USD o no;
- si no lo esta, definir normalizacion por `currency` antes de comparar montos;
- no mezclar monedas ciegamente.

### 6. `captured_at` nulo o inconsistente

Accion:

- no usar `captured_at` en features criticas si no es confiable;
- si se usa, definir fallback a `created_at`.

### 7. Usuarios sin historial

Accion:

- usar defaults neutros;
- marcar cold-start y medir cuantas filas afecta.

### 8. Facilities nuevas o sin suficiente historial

Accion:

- baseline contextual con fallback global;
- registrar porcentaje afectado.

### 9. Gateways raros o categorias casi vacias

Accion:

- agrupar en `other` para reporting si es necesario;
- mantener raw original en snapshot.

### 10. `_peerdb_version` o backfills tardios

Accion:

- el snapshot queda congelado;
- no remezclar con una nueva extraccion sin versionar el estudio.

## Bordes temporales

### 11. Cambio train -> val

Accion:

- pruebas especificas para comprobar que `val` usa historia previa cuando corresponde.

### 12. Cambio val -> test

Accion:

- misma validacion que arriba.

### 13. Inicio del anio

Accion:

- usar `warm_history`;
- si falta historia para una feature, dejar el default y documentar.

## Bordes de modelado

### 14. Todos los scores iguales o casi iguales

Accion:

- registrar;
- revisar schema, leakage inverso o fallo en escalado;
- detener pipeline si sucede en el modelo principal.

### 15. `LOF` sin viabilidad en memoria

Accion:

- reducir sample o ajustar `n_neighbors`;
- documentar el cambio;
- no alterar unfairly la base de comparacion.

### 16. `OC-SVM` demasiado lento

Accion:

- mantener subsampling fijo y transparente;
- reportar costo computacional.

### 17. Alta variabilidad entre seeds

Accion:

- ejecutar varias seeds;
- reportar dispersión;
- si la variabilidad es alta, aumentar `n_estimators` o fijar protocolo final mas robusto.

### 18. Proxy muy desbalanceado en bootstrap

Accion:

- usar bootstrap estratificado si hace falta;
- evitar muestras sin clase positiva cuando rompan la metrica.

## Bordes de reporting

### 19. Overwrite accidental de artefactos

Accion:

- escribir por run;
- solo promover a `latest` al final.

### 20. Diferencia entre tablas del texto y tablas generadas

Accion:

- prohibido editar manualmente `.tex`;
- corregir el generador.

## Validaciones automáticas mínimas

- validacion de schema por capa;
- validacion de conteos por split;
- validacion de proxies;
- validacion de history windows;
- validacion de no leakage;
- validacion de ausencia de NaN e inf;
- validacion de reproducibilidad de seeds.

## Regla de parada

Si aparece un caso borde que cambie:

- el universo,
- el proxy,
- los splits,
- o el catalogo de features,

entonces se debe volver a Fase 0 y actualizar el contrato.
