# Plan Maestro de Implementacion de la Tesis

## Navegacion de esta carpeta

- `README.md`: indice y orden de uso.
- `00_MASTER_ROADMAP.md`: roadmap resumido.
- `01_FASE_0_CONTRATO_Y_ALCANCE.md` a `08_FASE_7_SENSIBILIDAD_REPORTING_Y_CIERRE.md`: detalle operativo por fase.
- `09_ETL_Y_LINEAGE.md`: ETL completo, historia previa y trazabilidad.
- `10_CASOS_BORDE_Y_VALIDACIONES.md`: manejo de edge cases y validaciones.
- `11_PROTOCOLO_DE_EXPERIMENTOS.md`: protocolo de ejecucion y reproducibilidad.
- `12_COMPARACION_DE_ALGORITMOS_Y_BASELINES.md`: comparacion justa entre algoritmos.
- `13_RUNBOOK_DE_EJECUCION.md`: orden exacto de ejecucion y reruns.
- `14_MULTIAGENT_REVIEW.md`: revision desde perspectivas especializadas.
- `90_RIESGOS_DECISIONES_Y_CHECKLIST.md`: riesgos, decisiones y checklist de control.

Fecha de elaboracion: 2026-03-14
Ubicacion objetivo del proyecto: `/Users/eidan/Documentation/Personal/Master/Perfil/ml-fraud-detector`
Tesis base: `Tesis-Latex`
Enfoque metodologico vigente: deteccion de anomalias no supervisada con `Isolation Forest`, validacion ex post mediante proxy de reembolso.

## 1. Conclusion ejecutiva

La tesis es ejecutable, pero el proyecto actual `ml-fraud-detector` no esta alineado con el alcance real del estudio. Hoy contiene una base util de configuracion, conexion a ClickHouse y estructura de paquete, pero el nucleo del codigo sigue respondiendo a un proyecto previo de fraude supervisado. La recomendacion no es "parchar" lo existente, sino rehacer el pipeline principal con enfoque greenfield y reutilizar solo lo que siga siendo correcto.

El objetivo operativo no es construir un sistema productivo en tiempo real. El objetivo real de la tesis es producir un pipeline reproducible que:

1. extraiga una poblacion depurada y deduplicada desde ClickHouse;
2. construya features sin leakage;
3. entrene `Isolation Forest` y compare contra `LOF` y `One-Class SVM`;
4. evalue HE1-HE4 en test temporal independiente;
5. genere tablas, figuras y resultados listos para insertarse en Capitulo 2 y Capitulo 3 de la tesis;
6. deje un repositorio defendible academica y tecnicamente.

## 2. Hallazgos validados antes de planificar

### 2.1 Tesis y perfil

El perfil y los capitulos vigentes ya estan razonablemente alineados a:

- Problema general: capacidad discriminativa del `anomaly_score` de `Isolation Forest` respecto de un proxy de anomalia.
- OE1-OE4: fundamentacion, diagnostico, evaluacion del modelo principal y comparacion metodologica.
- HE1-HE4: Mann-Whitney U, AUC/AP, top-5% y comparacion contra `LOF`/`OC-SVM`.

Eso significa que la tesis ya tiene direccion metodologica suficientemente clara; el deficit esta en la implementacion reproducible y en el cierre de evidencia empirica.

### 2.2 Estado real del codigo actual

El repositorio actual sirve como base estructural, pero no como base funcional final. Problemas criticos:

- `config/config.py` sigue modelando un proyecto supervisado: `model_type`, `SMOTE`, `class_weight`, `MLflow`, thresholds de fraude, API, costos de fraude.
- `src/fraud_detector/data/clickhouse_connector.py` crea un `is_fraud` sintetico basado en `reversed_id`, `refund status` y `debit_refund`. Eso contradice la tesis actual.
- `src/fraud_detector/data/loader.py` usa `train_test_split`, lo cual es incorrecto para la tesis.
- `src/fraud_detector/features/engineering.py` implementa loops por usuario/fila; a escala 6.7M filas no es viable.
- `src/fraud_detector/models/trainer.py` entrena clasificadores supervisados (`XGBoost`, `LightGBM`, `RF`, `Logistic`), no detectores de anomalias.
- `src/fraud_detector/evaluation/metrics.py` sigue orientado a etiquetas supervisadas y costos de clasificacion.
- `scripts/test_clickhouse_connection.py` conecta, pero una consulta de muestra usa una columna `amount` inexistente.

Conclusion: hay que reemplazar el pipeline principal casi por completo.

### 2.3 Validacion real contra ClickHouse

La conexion a ClickHouse funciona desde este entorno actual.

Hallazgos estructurales de `pbp_productionDB_optimized.payments`:

- Engine: `SharedReplacingMergeTree`
- `ORDER BY`: `(facility_id, created_at, id)`
- `PRIMARY KEY`: `(facility_id, created_at, id)`
- `PARTITION BY`: vacio
- Filas fisicas en la tabla: `85,399,167`
- Columnas: `63`

Esto importa mucho:

- Al ser `ReplacingMergeTree`, los conteos crudos sin deduplicacion quedan inflados.
- La tesis no puede basarse en `SELECT` simples sobre la tabla fisica; necesita `FINAL` o una estrategia equivalente de deduplicacion reproducible.

### 2.4 Universo correcto de la tesis

La poblacion del estudio se reproduce practicamente exacta con esta logica:

```sql
SELECT *
FROM pbp_productionDB_optimized.payments FINAL
WHERE created_at >= '2025-01-01'
  AND created_at < '2026-01-01'
  AND payment_method != 'reversal'
  AND payment_method != 'free'
  AND user_id != 0
  AND _peerdb_is_deleted = 0
```

Conteos validados con esa definicion:

- Poblacion depurada 2025: `6,784,695`
- Proxy estricto: `429,442` (`6.3296%`)
- Proxy amplio: `512,609` (`7.5554%`)
- Train (`2025-01-01` a `2025-06-30`): `3,137,086`
- Validation (`2025-07-01` a `2025-08-31`): `1,130,118`
- Test (`2025-09-01` a `2025-12-31`): `2,517,491`

Estos numeros coinciden practicamente con la tesis vigente. La diferencia de una fila en algunos totales no cambia el diseno, pero obliga a congelar el SQL exacto de extraccion.

### 2.5 Hallazgo ClickHouse que condiciona la estrategia

La tabla no tiene `PARTITION BY`, y el orden fisico esta liderado por `facility_id`, no por fecha. Segun el skill `clickhouse-best-practices`:

- Per `schema-pk-plan-before-creation`, cualquier tabla derivada nueva debe nacer a partir de patrones de consulta documentados; no hay que improvisar `ORDER BY`.
- Per `schema-pk-prioritize-filters`, si luego se crea una tabla analitica derivada, el `ORDER BY` debe responder a los filtros y ventanas reales del pipeline.
- Per `schema-pk-filter-on-orderby`, filtrar solo por `created_at` no es el escenario ideal si se omite el prefijo del key; aun asi, `EXPLAIN indexes = 1` mostro que ClickHouse logra cierto pruning de granulos para el rango anual, por lo que la extraccion es viable.
- Per `query-join-filter-before`, cualquier join auxiliar debe filtrar primero antes de unir.
- Per `query-mv-incremental`, una MV solo tiene sentido si se repiten agregaciones costosas y existen permisos de escritura; no es prerequisito para la tesis.

## 3. Decision estrategica

### 3.1 Que se conserva

Se puede conservar, con ajuste menor o reuso parcial:

- estructura de paquete Python;
- logger;
- conector base a ClickHouse;
- entorno virtual y dependencias base;
- carpeta `tests/`;
- carpeta `.planning/`.

### 3.2 Que se debe reemplazar

Se debe reescribir o eliminar del camino critico:

- configuracion supervisada;
- extractor que fabrique `is_fraud`;
- loader con split aleatorio;
- feature engineering actual;
- entrenamiento supervisado;
- metricas de clasificacion supervisada;
- `run_simple_rf.py`;
- notebooks viejos que asuman fraude etiquetado.

### 3.3 Posicion recomendada sobre "borrar y recrear"

No conviene borrar todo el repo al inicio. Conviene:

1. congelar el SQL y los conteos del universo valido;
2. crear el nuevo pipeline en paralelo;
3. retirar lo obsoleto cuando el pipeline nuevo pase smoke tests.

Razon: hoy el repositorio ya tiene conexion funcional a ClickHouse y staged changes que no deben tocarse sin necesidad.

## 4. Objetivo operativo del proyecto

El proyecto debe terminar produciendo cuatro salidas mayores:

1. `dataset snapshot` reproducible de la tesis.
2. `pipeline de modelado` reproducible de punta a punta.
3. `artefactos para la tesis` (tablas, figuras, JSON/CSV de resultados, apendices).
4. `repositorio defendible` (codigo limpio, pruebas, README y procedimiento de reproduccion).

## 5. Arquitectura objetivo del repositorio

```text
ml-fraud-detector/
├── .planning/
│   ├── Plan2
│   └── ...
├── config/
│   └── config.py
├── src/fraud_detector/
│   ├── data/
│   │   ├── clickhouse_connector.py
│   │   ├── extraction.py
│   │   ├── datasets.py
│   │   └── schema_contract.py
│   ├── features/
│   │   ├── engineering.py
│   │   ├── catalogs.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── trainers.py
│   │   ├── tuning.py
│   │   └── scoring.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── statistics.py
│   │   ├── sensitivity.py
│   │   └── comparison.py
│   ├── reporting/
│   │   ├── tables.py
│   │   ├── figures.py
│   │   └── exports.py
│   ├── pipeline/
│   │   ├── run_all.py
│   │   ├── run_eda.py
│   │   ├── run_training.py
│   │   └── run_reporting.py
│   └── utils/
│       └── logger.py
├── scripts/
│   ├── inspect_clickhouse.py
│   ├── extract_snapshot.py
│   └── verify_counts.py
├── notebooks/
│   ├── 01_cap2_eda.ipynb
│   └── 02_cap3_results.ipynb
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── output/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   ├── metrics/
│   └── manifests/
├── tests/
│   ├── test_extraction.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_reporting.py
└── README.md
```

## 6. Principios de implementacion

1. `FINAL` no es opcional para reproducir la tesis.
2. El proxy se usa solo para validacion, nunca para entrenar los modelos.
3. Toda feature debe ser auditada contra leakage temporal y circularidad metodologica.
4. La tesis necesita un pipeline offline y reproducible, no una API productiva.
5. Todo resultado de Capitulo 2 y Capitulo 3 debe salir del codigo, no de tablas manuales.
6. El test set temporal debe permanecer intocable hasta la fase final de evaluacion.
7. La comparacion entre modelos debe usar la misma base, mismas features, mismo proxy y mismo split.
8. Cada fase debe dejar artefactos serializados para no repetir extracciones costosas.

## 7. Plan maestro por fases

## Fase 0. Congelamiento metodologico y contrato tesis-codigo

Objetivo:
traducir la tesis a un contrato tecnico no ambiguo.

Tareas:

- fijar SQL canonico del universo 2025;
- fijar definicion del proxy estricto y amplio;
- fijar ventanas temporales train/val/test;
- fijar catalogo oficial de 20 features;
- fijar metrica de seleccion de hiperparametros;
- fijar reglas de analisis de sensibilidad;
- documentar una matriz `objetivo -> hipotesis -> modulo -> output`.

Entregables:

- `docs/thesis_contract.md` o equivalente;
- `config/config.py` con parametros del estudio;
- `scripts/verify_counts.py`.

Definition of Done:

- los conteos base reproducen la tesis;
- no quedan variables o metricas heredadas del proyecto supervisado;
- existe una unica fuente de verdad para filtros, proxies y splits.

Gate de salida:

- no empezar feature engineering hasta que el universo y los proxies esten cerrados.

## Fase 1. Reset estructural del proyecto

Objetivo:
dejar el repo limpio para la tesis actual.

Tareas:

- mover o eliminar codigo obsoleto del camino critico;
- reescribir `README` con alcance de tesis y no de fraude supervisado;
- reescribir configuracion;
- crear estructura `output/` y `data/interim/`;
- definir CLI o entrypoints del pipeline;
- dejar `Makefile` o comandos consistentes.

Entregables:

- esqueleto final del repositorio;
- configuracion saneada;
- comandos `extract`, `features`, `train`, `evaluate`, `report`.

Definition of Done:

- cualquier desarrollador entiende el flujo principal leyendo `README`;
- no hay rutas criticas apuntando a `is_fraud`, `SMOTE`, `RF`, `XGBoost` o `MLflow` como requisito central.

## Fase 2. Extraccion, deduplicacion y snapshot reproducible

Objetivo:
obtener el dataset definitivo y trazable del estudio.

Tareas:

- encapsular consulta `FINAL`;
- extraer por split temporal, no por anio completo a memoria;
- persistir `train_raw.parquet`, `val_raw.parquet`, `test_raw.parquet`;
- registrar `row_count`, checksum/log y fecha de extraccion;
- validar nulos, tipos y dominios base;
- fijar un `manifest.json` con conteos por split y por status.

SQL canonico minimo:

```sql
SELECT
    id,
    user_id,
    facility_id,
    facility_name,
    created_at,
    captured_at,
    payment_method,
    gateway,
    source_enum,
    status,
    reservation_paid_out,
    discount,
    tax,
    tip,
    card_brand,
    reversed_id,
    debit_refund,
    _peerdb_version
FROM pbp_productionDB_optimized.payments FINAL
WHERE created_at >= %(start)s
  AND created_at < %(end)s
  AND payment_method != 'reversal'
  AND payment_method != 'free'
  AND user_id != 0
  AND _peerdb_is_deleted = 0
ORDER BY created_at, id
```

Notas tecnicas:

- `FINAL` es obligatorio.
- Conviene extraer por split y cachear en parquet.
- Si la lectura completa resulta lenta, se puede extraer mes a mes y luego consolidar localmente.
- No usar joins pesados al inicio; primero snapshot base minimalista.

Definition of Done:

- los archivos parquet reproducen los conteos del universo;
- hay manifest con trazabilidad;
- el snapshot puede reutilizarse sin volver a consultar ClickHouse.

Gate de salida:

- no seguir a modelado si el snapshot no coincide con la tesis.

## Fase 3. EDA y diagnostico para Capitulo 2

Objetivo:
producir la evidencia empirica del OE2.

Tareas:

- distribucion por status, canal, gateway, metodo de pago, mes;
- estadisticos descriptivos de montos;
- comparacion proxy+ vs proxy-;
- outliers de monto;
- patrones de usuario, horario y velocidad;
- tablas y figuras exportables a LaTeX.

Entregables:

- notebook `01_cap2_eda.ipynb`;
- `output/tables/cap2_*.tex`;
- `output/figures/cap2_*.pdf`;
- `output/metrics/cap2_summary.json`.

Definition of Done:

- Capitulo 2 puede completarse sin hacer calculos manuales fuera del repo.

## Fase 4. Feature engineering oficial de la tesis

Objetivo:
implementar el catalogo final de features con garantia anti-leakage.

Catalogo esperado:

- Transaccionales: monto, log_amount, ratio global, discount_ratio, has_tip.
- Temporales: hour_sin, hour_cos, day_of_week, is_weekend, is_off_hours.
- Velocidad: user_txn_count_1h, user_txn_count_24h, time_since_last_txn, user_amount_24h.
- Comportamiento: user_distinct_facilities_30d, user_distinct_methods, user_reversal_ratio_30d, user_account_age_days.
- Contextuales: facility_avg_amount, amount_facility_ratio.

Tareas:

- disenar implementacion vectorizada;
- ordenar siempre por entidad y tiempo antes de rolling/expanding;
- usar `closed='left'`, `shift(1)` o estrategia equivalente;
- separar features obligatorias de features opcionales;
- marcar `user_reversal_ratio_30d` como feature de riesgo metodologico para ablacion obligatoria.

Decision critica:

- si una feature no puede justificarse sin leakage ni circularidad, se elimina aunque mejore metricas.

Entregables:

- `features/engineering.py`;
- `features/catalogs.py`;
- dataset `*_features.parquet`;
- pruebas unitarias de leakage.

Definition of Done:

- las 20 features se generan para train/val/test;
- existe prueba que confirme que una transaccion no usa informacion futura;
- existe version de 19 features sin `user_reversal_ratio_30d`.

## Fase 5. Preprocesamiento y datasets de modelado

Objetivo:
construir matrices de entrada consistentes para los tres algoritmos.

Tareas:

- imputacion de categoricas a `Unknown`;
- imputacion numerica con estadisticos de train;
- `StandardScaler` ajustado solo en train;
- definicion fija de columnas de entrada;
- serializacion de scaler y metadata de features.

Notas:

- `Isolation Forest` no necesita escalado estricto, pero `LOF` y `OC-SVM` si; por comparabilidad se mantiene una ruta comun.
- Evitar `one-hot` indiscriminado si no es necesario. El catalogo de tesis es mayormente numerico. Si entran categoricas, deben pasar por una estrategia acotada y documentada.

Entregables:

- `features/preprocessor.py`;
- `output/models/scaler.joblib`;
- matrices o parquets procesados por split.

Definition of Done:

- train/val/test comparten exactamente el mismo esquema final;
- no hay ajuste de scaler con datos de val/test.

## Fase 6. Modelado principal y comparadores

Objetivo:
entrenar `Isolation Forest`, `LOF` y `One-Class SVM`.

Tareas:

- implementar trainer no supervisado;
- grid search para `Isolation Forest` sobre validation;
- entrenamiento de `LOF` con `novelty=True`;
- entrenamiento de `OC-SVM` con subsampling controlado;
- guardar modelos y scores por split.

Parametros de arranque recomendados:

- `Isolation Forest`: `n_estimators=300`, `max_samples=1024`, `contamination` entre `0.02` y `0.08`, `max_features` entre `0.5` y `1.0`.
- `LOF`: `n_neighbors` entre `20` y `100`, `contamination` alineado al proxy.
- `OC-SVM`: `kernel='rbf'`, `nu` cercano a la tasa base, subsample `50k-100k`.

Decision metodologica:

- los hiperparametros se seleccionan con supervision indirecta ex post en validation. Eso debe documentarse tal como ya lo hace el perfil.

Entregables:

- `models/trainers.py`;
- `models/tuning.py`;
- `output/models/*.joblib`;
- `output/metrics/grid_search_if.csv`;
- `output/metrics/model_registry.json`.

Definition of Done:

- existen scores reproducibles para los tres modelos en el test set;
- el mejor modelo de `Isolation Forest` queda fijado antes de abrir sensibilidad amplia.

## Fase 7. Evaluacion estadistica y validacion de hipotesis

Objetivo:
cerrar OE3 y OE4 con evidencia cuantitativa defendible.

Tareas:

- HE1: Mann-Whitney U, p-value, rank-biserial;
- HE2: AUC-ROC y Average Precision vs tasa base;
- HE3: Precision@5%, Recall@5%, Enrichment Factor;
- HE4: comparacion multicriterio contra `LOF` y `OC-SVM`;
- bootstrap 1000 iteraciones para IC 95%;
- tablas resumen por modelo y proxy.

Adicionales obligatorios:

- curva ROC;
- curva PR;
- distribucion de scores por proxy;
- tabla de top-k;
- metricas del modelo con 20 features vs 19 features.

Entregables:

- `evaluation/metrics.py`;
- `evaluation/statistics.py`;
- `evaluation/comparison.py`;
- `output/metrics/final_results.json`;
- `output/tables/cap3_*.tex`;
- `output/figures/cap3_*.pdf`.

Definition of Done:

- cada HE queda marcada como `respaldada` o `rechazada` con evidencia;
- los resultados son reproducibles desde un solo comando.

## Fase 8. Analisis de sensibilidad y robustez

Objetivo:
blindar la tesis contra objeciones metodologicas.

Minimo obligatorio:

- proxy estricto vs proxy amplio;
- con y sin `user_reversal_ratio_30d`;
- estabilidad razonable de `contamination`;
- subgrupos de negocio relevantes: canal, gateway o rango de monto, si el tiempo alcanza.

Preguntas que esta fase debe contestar:

- el modelo conserva capacidad discriminativa si ampliamos proxy?
- las metricas colapsan al retirar la feature mas sospechosa?
- el desempeno esta concentrado solo en un segmento de negocio?

Entregables:

- `evaluation/sensitivity.py`;
- `output/metrics/sensitivity_results.json`;
- tablas comparativas para tesis.

Definition of Done:

- la tesis puede defender por que el resultado principal es robusto y donde deja de serlo.

## Fase 9. Reporting automatico para la tesis

Objetivo:
eliminar el trabajo manual de copiar resultados.

Tareas:

- exportar tablas en formato LaTeX;
- exportar figuras en PDF/PNG;
- generar archivos con nombres estables para `Tesis-Latex`;
- producir resumen ejecutivo tecnico de resultados;
- dejar apendice con SQL, configuracion y catalogo de features.

Entregables:

- `reporting/tables.py`;
- `reporting/figures.py`;
- `reporting/exports.py`;
- carpeta `output/tables/` y `output/figures/`.

Definition of Done:

- Capitulo 2 y Capitulo 3 se pueden actualizar importando archivos generados por el pipeline.

## Fase 10. Pruebas, reproducibilidad y limpieza final

Objetivo:
cerrar el proyecto con estandar defendible.

Tareas:

- pruebas unitarias de extraccion, proxies, features, preprocesamiento y metricas;
- smoke test de pipeline parcial;
- validacion de conteos y rangos;
- revisar logs y mensajes de error;
- documentar pasos de ejecucion end-to-end;
- limpiar codigo muerto.

Entregables:

- suite `pytest`;
- `README` final;
- lista de comandos reproducibles;
- changelog tecnico para apendice.

Definition of Done:

- cualquier ejecucion nueva reproduce el snapshot y resultados a partir de los mismos parquet/cache;
- no quedan modulos obsoletos en el flujo principal.

## Fase 11. Integracion con la tesis LaTeX

Objetivo:
cerrar la tesis completa, no solo el software.

Tareas por capitulo:

- Capitulo 1: completar literatura y sustento de IF/LOF/OC-SVM.
- Capitulo 2: insertar tablas EDA reales y diagnostico.
- Capitulo 3: insertar pipeline, hiperparametros, metricas, bootstrap y sensibilidad.
- Conclusiones: responder OG y OE1-OE4 con base en resultados reales.
- Apendices: SQL, catalogo de features, pseudocodigo, configuracion.

Definition of Done:

- `Tesis-Latex` compila con los resultados del proyecto;
- las conclusiones no contienen promesas, solo evidencia ya corrida.

## 8. Cronograma sugerido de 12 semanas

### Semanas 1-2

- Fase 0
- Fase 1
- Fase 2

Meta:
dataset y conteos cerrados.

### Semanas 3-4

- Fase 3
- inicio Fase 4

Meta:
Capitulo 2 casi completo y catalogo de features implementado.

### Semanas 5-6

- cierre Fase 4
- Fase 5
- inicio Fase 6

Meta:
matrices de modelado listas y primer `Isolation Forest` entrenado.

### Semanas 7-8

- cierre Fase 6
- Fase 7

Meta:
hipotesis evaluadas en test.

### Semanas 9-10

- Fase 8
- Fase 9

Meta:
sensibilidad y artefactos LaTeX cerrados.

### Semanas 11-12

- Fase 10
- Fase 11

Meta:
repo defendible + tesis compilada + material de defensa.

## 9. Gates criticos

### Gate A. Conteos

No continuar si la consulta base no reproduce:

- `N ~ 6,784,695`
- strict proxy `~ 429,442`
- wide proxy `~ 512,609`

### Gate B. Leakage

No continuar si una feature usa:

- status de reembolso;
- informacion futura;
- agregados que incluyan la propia fila actual;
- artefactos derivados del mismo evento posterior usado como proxy.

### Gate C. Test temporal

No tocar test para tuning fino.

### Gate D. Robustez

No cerrar tesis si el resultado principal depende por completo de `user_reversal_ratio_30d`.

### Gate E. Reporting

No cerrar Capitulo 2/3 con tablas manuales; deben salir del pipeline.

## 10. Riesgos y mitigaciones

### Riesgo 1. Duplicados / versiones en ClickHouse

Riesgo:
conteos inconsistentes si se omite `FINAL`.

Mitigacion:

- `FINAL` obligatorio en snapshot base;
- manifest de conteos;
- prueba automatica de consistencia.

### Riesgo 2. Performance de extraccion

Riesgo:
lecturas lentas desde tabla grande.

Mitigacion:

- extraer por split o mes;
- cachear parquet;
- no repetir queries completas innecesariamente.

### Riesgo 3. Leakage metodologico

Riesgo:
metricas artificialmente altas.

Mitigacion:

- auditoria feature por feature;
- ablacion de feature 17;
- validacion temporal estricta.

### Riesgo 4. Escalabilidad de `OC-SVM`

Riesgo:
coste cuadratico o cubico.

Mitigacion:

- subsampling controlado;
- tratarlo como baseline comparativo, no como modelo principal.

### Riesgo 5. Interpretabilidad de IF

Riesgo:
SHAP puede ser costoso o poco estable segun implementacion.

Mitigacion:

- prioridad a metricas y ranking;
- si SHAP complica, usar permutation importance o analisis descriptivo de top anomalies;
- no bloquear la tesis por explainability avanzada.

### Riesgo 6. Drift del origen de datos

Riesgo:
la tabla productiva cambia y los conteos se mueven.

Mitigacion:

- congelar snapshot parquet;
- registrar fecha de extraccion;
- desde ese punto trabajar sobre artefacto local.

### Riesgo 7. Sobrealcance

Riesgo:
meter API, realtime scoring, dashboards o despliegue productivo.

Mitigacion:

- fuera de alcance para esta tesis;
- solo se documenta como trabajo futuro.

## 11. Criterios de aceptacion finales

El trabajo se considera completo cuando se cumplan todos estos puntos:

- existe un snapshot deduplicado y reproducible del universo 2025;
- el pipeline corre de punta a punta desde snapshot hasta tablas/figuras;
- `Isolation Forest`, `LOF` y `OC-SVM` fueron evaluados sobre el mismo test temporal;
- las hipotesis HE1-HE4 quedaron contestadas con evidencia;
- existe analisis de sensibilidad;
- Capitulo 2 y Capitulo 3 se alimentan de artefactos generados por codigo;
- el repositorio tiene pruebas, README y comandos reproducibles;
- `Tesis-Latex` puede compilar con resultados reales;
- el alcance queda explicitamente limitado a evaluacion offline no experimental.

## 12. Orden exacto recomendado de ejecucion

1. Reescribir configuracion y contrato del estudio.
2. Corregir extractor para usar `FINAL` y universo canonico.
3. Congelar snapshot y manifest.
4. Producir EDA y tablas base de Capitulo 2.
5. Implementar features oficiales y pruebas de leakage.
6. Implementar preprocesamiento.
7. Entrenar `Isolation Forest` y hacer tuning en validation.
8. Entrenar `LOF`.
9. Entrenar `OC-SVM` con subsample.
10. Ejecutar evaluacion de HE1-HE4.
11. Ejecutar sensibilidad.
12. Exportar tablas/figuras.
13. Limpiar repo y escribir documentacion final.
14. Integrar resultados en `Tesis-Latex`.

## 13. Primera iteracion concreta recomendada

La primera iteracion no debe tocar modelado todavia. Debe cerrar solo esto:

1. `config/config.py` nuevo para tesis.
2. `data/extraction.py` con SQL canonico y `FINAL`.
3. `scripts/verify_counts.py` que imprima los conteos validados.
4. `data/processed/{train,val,test}_raw.parquet`.
5. `output/manifests/dataset_manifest.json`.

Si esa iteracion falla, el resto del proyecto no es confiable.

## 14. Recomendacion final

El proyecto debe tratarse como una implementacion academica reproducible basada en snapshot, no como producto SaaS de fraude. La mejor estrategia para llegar a tesis terminada es:

- greenfield en el pipeline central,
- reuse selectivo de infraestructura,
- disciplina estricta con `FINAL`, proxy y validacion temporal,
- generacion automatica de evidencia para LaTeX.

Ese enfoque minimiza riesgo metodologico, evita sobreingenieria y deja un resultado defendible frente a tribunal y tambien frente a una revision tecnica seria.
