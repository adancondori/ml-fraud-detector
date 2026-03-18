# Runbook de Ejecucion

## Proposito

Definir la secuencia exacta de ejecucion para pasar de cero a resultados finales.

## Preflight

1. Verificar entorno virtual.
2. Verificar credenciales ClickHouse.
3. Verificar espacio en disco.
4. Verificar que no haya un snapshot viejo siendo sobreescrito por error.
5. Verificar que el contrato de Fase 0 este vigente.

## Secuencia recomendada

### Paso 1. Verificacion del origen

- correr verificacion de conexion;
- correr `verify_counts`;
- guardar manifest inicial.

### Paso 2. Extraccion

- extraer `warm_history`;
- extraer `train_raw`;
- extraer `val_raw`;
- extraer `test_raw`;
- validar conteos.

### Paso 3. Canonicalizacion

- tipado;
- proxies;
- domains;
- manifests.

### Paso 4. EDA

- notebook o script de Capitulo 2;
- export de tablas y figuras.

### Paso 5. Features

- generar 20 features;
- generar 19 features;
- pasar tests de leakage.

### Paso 6. Preprocesamiento

- fit en train;
- transform en val y test;
- guardar scaler y schema.

### Paso 7. Tuning y training

- IF en validation;
- LOF;
- OCSVM;
- scores de test.

### Paso 8. Evaluacion

- HE1-HE4;
- bootstrap;
- tablas y figuras.

### Paso 9. Sensibilidad

- proxy amplio;
- 19 features;
- seeds si aplica;
- subgrupos si alcanza.

### Paso 10. Export final

- `.tex`
- `.pdf`
- manifests finales
- README y checklist

## Reruns

### Si falla extraccion

- rerun solo el split afectado.

### Si falla feature engineering

- rerun solo desde capa feature, no desde source.

### Si falla modelado

- rerun desde model input ya congelado.

### Si falla reporting

- rerun desde `final_results.json`.

## Criterio de recuperacion

Siempre reanudar desde la ultima capa validada, nunca desde cero sin motivo.
