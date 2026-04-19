# CLAUDE.md

Este archivo proporciona contexto a Claude Code sobre el proyecto ML Fraud Detector.

## Descripcion del Proyecto

Pipeline de deteccion de **anomalias transaccionales** (no supervisado) para la tesis de maestria UAGRM. Evalua Isolation Forest como modelo principal, comparado contra LOF y One-Class SVM, usando un proxy de reembolso para evaluacion (nunca para entrenamiento).

**Empresa:** TechSport Inc. (pseudonimo)
**Datos:** 6.784.696 transacciones depuradas (gestion 2025) desde ClickHouse
**Proxy unificado (evaluacion principal):** OR de 5 tipos (A+B+C+D+E) = 10.23%
**Tipo A (reembolso):** status IN ('totally_refunded', 'refunded_to_credit') = 6.33%

## Stack Tecnologico

- **Python 3.9+** con entorno virtual en `venv/`
- **ML**: scikit-learn (IsolationForest, LOF, OneClassSVM)
- **Data**: pandas, numpy, ClickHouse (clickhouse-connect)
- **Config**: Pydantic + python-dotenv
- **Logging**: Loguru
- **Explainability**: SHAP

## Estructura del Proyecto

```
src/fraud_detector/       # Codigo fuente principal
├── data/                 # Loaders y conectores (ClickHouse)
├── features/             # Feature engineering (33 features, 8 grupos A-H)
├── models/               # Entrenamiento no supervisado (IF, LOF, OC-SVM)
├── evaluation/           # Metricas, hipotesis (HE1-HE4), post-hoc, segmentos
└── utils/                # Logger, CurrencyNormalizer, utilidades

config/                   # Configuracion con Pydantic
scripts/                  # Scripts de utilidad (extraccion, verificacion)
notebooks/                # Jupyter notebooks
tests/                    # Tests unitarios
data/processed/           # Parquets (train, val, test, warm)
output/                   # Modelos, scores, tablas LaTeX, figuras
.planning/PLAN-FINAL/     # Plan detallado de implementacion (13 documentos)
```

## Comandos Frecuentes

```bash
source venv/bin/activate

# Extraccion de datos
python scripts/extract_full_dataset.py
python scripts/verify_counts.py

# Tests
make test                 # pytest --cov=src/fraud_detector
make format               # Black + isort
make lint                 # Flake8

# Notebooks
make notebook
```

## 31 Features (8 grupos)

F06 (is_free) y F21 (user_free_pct_30d) eliminadas porque payment_method='free' se excluye del universo.

| Grupo | Features | Nums |
|-------|----------|------|
| A) Transaccionales | reservation_paid_out, log_amount, amount_usd_ratio, discount_ratio, has_tip | #1-5 |
| B) Temporales | hour_sin, hour_cos, day_of_week, is_weekend, is_off_hours | #7-11 |
| C) Velocidad | user_txn_count_1h/24h, time_since_last_txn, user_amount_24h | #12-15 |
| D) Comportamiento | user_distinct_facilities_30d, user_distinct_methods, user_reversal_ratio_30d*, user_account_age_days, user_discount_ratio_30d | #16-20 |
| E) Contextuales | facility_avg_amount, amount_facility_ratio | #22-23 |
| F) Credito/Flujo | is_club_credit, user_debit_count_30d, user_debit_amount_30d, credit_flow_ratio | #24-27 |
| G) Rol/Staff | is_staff, paid_by_manager, staff_amount_zscore | #28-30 |
| H) Diversidad Operacional | category_entropy_30d, user_reversal_count_30d, user_merchandise_ratio_30d | #31-33 |

*Feature #18 (user_reversal_ratio_30d): correlacion mecanica con proxy. Analisis de sensibilidad obligatorio (delta AUC < 0.02).

Variantes: IF-31 (principal), IF-30 (sin F18), IF-21 (ablacion grupos F,G,H).

## Proxy Taxonomy (5 tipos)

| Tipo | Regla | Tasa |
|------|-------|------|
| A - Reembolso | status IN ('totally_refunded','refunded_to_credit') | 6.33% |
| B - Circuito credito | circuit_closure > 80% AND cash_loaded > $500 | 0% (datos no disponibles) |
| C - Descuento anomalo | user_discount_ratio_30d > 100% | 3.72% |
| D - Velocidad extrema | txn_count_1d > 100 | 0.34% |
| E - Gratuitas | free_pct_30d > 25% AND free_count > 10 | 0% (excluidas del universo) |
| **Unificado** | **OR(A,B,C,D,E)** | **10.23%** |

## Relacion con el Proyecto Tesis-Latex

El proyecto `Tesis-Latex/` (`../Tesis-Latex/`) es la **guia base academica** del trabajo de investigacion: define el marco teorico, la metodologia (Sampieri), las hipotesis (HE1-HE4) y la estructura formal requerida por la UAGRM para la presentacion del perfil y la defensa de tesis. Su funcion es exclusivamente documental y academica.

Este proyecto (`ml-fraud-detector/`) es la **implementacion tecnica completa** que va mas alla del alcance academico:

| Aspecto | Tesis-Latex (academico) | ml-fraud-detector (tecnico) |
|---------|------------------------|----------------------------|
| Proposito | Documento formal de investigacion para defensa de tesis | Pipeline operativo de deteccion de anomalias |
| Alcance | 3 modelos (IF, LOF, OC-SVM), metricas basicas | Mismos modelos + optimizacion avanzada, SHAP, perfiles de riesgo por usuario |
| Resultado | PDF/DOCX con tablas de resultados | Modelos entrenados, scores, alertas, artefactos MLflow |
| Features | Describe 33 features conceptualmente | Implementa extraccion, transformacion y validacion end-to-end |
| Datos | Cita cifras y distribuciones | Conecta a ClickHouse, extrae, depura y particiona 6.7M+ registros |
| Reproducibilidad | Describe el procedimiento | Pipeline reproducible con Makefile, configs, seeds fijos |

**En resumen:** la tesis guia *que* evaluar y *por que*; este proyecto implementa *como* hacerlo y extiende el alcance hacia un sistema de deteccion funcional que pueda orientar decisiones operativas reales en TechSport.

## Restricciones Criticas

- **NO supervisado**: Modelos entrenan SIN etiquetas; proxy SOLO para evaluacion
- **NO causal**: Usar "asociacion", "capacidad discriminativa", nunca "predice"
- **FINAL obligatorio** en queries ClickHouse (SharedReplacingMergeTree)
- **Anti-leakage**: Ventanas rolling excluyen fila actual; fit solo en train
- **Proxy != fraude**: El proxy de reembolso NO captura insider fraud
- **Test intocable**: No usar test set para seleccion de hiperparametros
- **Normalizacion USD**: Montos deben normalizarse a USD antes de feature engineering (21 monedas, 13 gateways)

## Datos Extraidos

| Split | Filas | Periodo |
|-------|-------|---------|
| Warm | 419,820 | Dic 2024 |
| Train | 3,137,086 | Ene-Jun 2025 |
| Val | 1,130,118 | Jul-Ago 2025 |
| Test | 2,517,492 | Sep-Dic 2025 |
