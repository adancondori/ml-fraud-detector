# Anexo A2: Protocolo de Experimentación y Runbook

Documento transversal que define los modos de ejecución, el protocolo de versionado
de experimentos, la gestión de seeds, el protocolo de tuning, y el runbook operativo
para ejecutar el pipeline completo.

---

## Protocolo de Experimentación

### Modos de ejecución

#### Mode 1: Smoke

- Sample reducido o datos sintéticos
- Propósito: verificar integridad del pipeline end-to-end
- **NO usar para resultados de la tesis**
- Equivalente a `python run_pipeline.py --fast`

#### Mode 2: Full thesis

- Snapshot completo de ClickHouse
- Resultados oficiales para la tesis
- Exportar todos los artefactos finales
- Equivalente a `python run_pipeline.py` (sin flags de reducción)

---

### Versionado por ejecución (run)

Cada ejecución registra en `run_manifest_*.json`:

| Campo              | Ejemplo                                    |
|--------------------|--------------------------------------------|
| `timestamp`        | `2026-03-16T14:30:00Z`                     |
| `git_sha`          | `a1b2c3d`                                  |
| `snapshot_version` | `v2_20260315`                              |
| `seed`             | `42`                                       |
| `schema_hash`      | `sha256:abc123...`                         |
| `hyperparameters`  | `{n_estimators: 200, contamination: auto}` |
| `hostname`         | `macbook-eidan`                            |
| `duration_per_step`| `{1: "2:30", 2: "5:12", ...}`             |

---

### Seeds

| Contexto       | Seeds          | Notas                                         |
|----------------|----------------|-----------------------------------------------|
| Desarrollo     | 42             | Seed única para iteración rápida               |
| Ejecución final| 42, 52, 62     | O justificar uso de seed única en la tesis     |

Todas las seeds se establecen explícitamente en:
- `numpy.random.seed()`
- `random_state` de scikit-learn
- `PYTHONHASHSEED` como variable de entorno

---

### Protocolo de tuning

#### Isolation Forest (modelo principal)

- Tuning sobre conjunto de **validación** (nunca test)
- Métrica primaria: **AUC-ROC**
- Métrica secundaria: **Average Precision (AP)**
- Desempate: **runtime** (menor es mejor)
- Grid search sobre: `n_estimators`, `max_samples`, `max_features`, `contamination`

#### LOF (benchmark)

- Tuning ligero sobre misma base de validación
- Mismas métricas de evaluación
- Parámetros: `n_neighbors`, `contamination`

#### OC-SVM (benchmark)

- Tuning mínimo
- Subsample fijo por restricciones de memoria/tiempo
- Parámetros: `kernel`, `nu`, `gamma`

---

### Sanity baselines

Tres baselines que Isolation Forest **debe** superar para validar utilidad:

1. **Random ranking**: scores aleatorios uniformes
2. **Amount ranking**: ordenar por monto descendente (heurística naive)
3. **Z-score de amount**: anomalía por desviación estándar del monto

> Si IF no supera todas las baselines → **investigar** antes de reportar resultados.

---

### Prohibiciones del protocolo

| #  | Prohibición                                                                    |
|----|--------------------------------------------------------------------------------|
| 1  | Cambiar el proxy después de ver resultados en test sin actualizar el contrato  |
| 2  | Ajustar features basándose en el rendimiento en el test set                    |
| 3  | Comparar modelos entrenados/evaluados sobre datasets diferentes                |
| 4  | Regenerar tablas manualmente (siempre usar el generador)                       |

---

## Runbook

### Preflight (antes de ejecutar)

| #  | Verificación                                                    | Comando / método                     |
|----|-----------------------------------------------------------------|--------------------------------------|
| 1  | Entorno virtual activado                                        | `which python` apunta a `venv/`     |
| 2  | Credenciales de ClickHouse configuradas                         | `python -c "from config.config import settings; print(settings.clickhouse_host)"` |
| 3  | Espacio en disco suficiente (~6 GB necesarios)                  | `df -h .`                            |
| 4  | No se sobreescribe snapshot anterior sin intención              | Verificar `data/raw/` y manifests    |
| 5  | Contrato de Fase 0 vigente                                     | Revisar `00_CONTRATO_ALCANCE.md`     |

---

### Secuencia de ejecución (10 pasos)

| Paso | Acción                                    | Detalle                                                |
|------|-------------------------------------------|--------------------------------------------------------|
| 1    | `verify_counts.py`                        | Verificar conexión a ClickHouse + conteos esperados    |
| 2    | Extraer warm + train + val + test         | Parquets raw → validar por split                       |
| 3    | Canonicalizar                             | Tipos, proxies, manifests                              |
| 4    | EDA notebook                              | Generar artefactos para Capítulo 2 de la tesis         |
| 5    | Features                                  | 20 features + 19 variantes → ejecutar tests de leakage |
| 6    | Preprocesar                               | Fit en train, transform en todos los splits            |
| 7    | Tune + entrenar                           | IF, LOF, OC-SVM → generar test scores                 |
| 8    | Evaluar                                   | HE1-HE4, bootstrap CI, análisis temporal               |
| 9    | Sensibilidad                              | Proxy, feature 17, SHAP, análisis per-status           |
| 10   | Exportar                                  | `.tex`, `.pdf`, manifests, README de resultados        |

---

### Reruns (reanudación ante fallos)

| Fallo en...       | Acción                                                       |
|--------------------|--------------------------------------------------------------|
| Extracción         | Reejecutar solo el split afectado                            |
| Features           | Reejecutar desde la capa de features                         |
| Modelado           | Reejecutar desde model input (`.npy` ya existen)             |
| Reporting          | Reejecutar desde `results.json` (paso 8 del pipeline)       |

**Regla fundamental:** siempre reanudar desde la última capa validada, nunca desde
cero sin causa justificada. Usar `python run_pipeline.py --from-step N`.

---

### Aceptación de la ejecución final

| #  | Criterio                                                          | Estado |
|----|-------------------------------------------------------------------|--------|
| 1  | Snapshot correcto y documentado en manifest                       | [ ]    |
| 2  | Scores guardados como parquet con `id` y `created_at`             | [ ]    |
| 3  | HE1-HE4 ejecutados, resultados en `results.json`                 | [ ]    |
| 4  | Análisis de sensibilidad ejecutado (proxy, feature 17, SHAP)     | [ ]    |
| 5  | Tablas LaTeX exportadas y compilables                             | [ ]    |
| 6  | Figuras PDF/PNG generadas, tamaño > 0                            | [ ]    |
| 7  | Edge case checklist revisado (ver Anexo A1)                       | [ ]    |
| 8  | `run_manifest` completo con todos los campos                     | [ ]    |
| 9  | `requirements-lock.txt` generado                                  | [ ]    |
| 10 | Tesis-LaTeX compila sin errores con resultados reales             | [ ]    |
