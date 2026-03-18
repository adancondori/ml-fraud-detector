# CLAUDE.md

Este archivo proporciona contexto a Claude Code sobre el proyecto ML Fraud Detector.

## Descripción del Proyecto

Sistema de detección de fraude utilizando Machine Learning con arquitectura modular. El proyecto está diseñado para un entorno académico (tesis/master) y utiliza ClickHouse como base de datos.

## Stack Tecnológico

- **Python 3.9+** con entorno virtual en `venv/`
- **ML**: scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Data**: pandas, numpy, ClickHouse
- **Tracking**: MLflow
- **Config**: Pydantic + python-dotenv
- **Logging**: Loguru
- **Explainability**: SHAP, LIME

## Estructura del Proyecto

```
src/fraud_detector/       # Código fuente principal
├── data/                 # Loaders y conectores (ClickHouse)
├── features/             # Preprocesamiento y feature engineering
├── models/               # Entrenamiento de modelos
├── evaluation/           # Métricas de evaluación
└── utils/                # Logger y utilidades

config/                   # Configuración con Pydantic
scripts/                  # Scripts de utilidad (conexión DB, exploración)
notebooks/                # Jupyter notebooks
tests/                    # Tests unitarios
```

## Comandos Frecuentes

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar prueba rápida de Random Forest
python run_simple_rf.py

# Probar conexión a ClickHouse
python scripts/test_clickhouse_connection.py

# Tests
make test                 # o: pytest --cov=src/fraud_detector

# Formateo y linting
make format               # Black + isort
make lint                 # Flake8
make type-check           # Mypy

# MLflow UI
make mlflow               # o: mlflow ui --port 5000

# Jupyter notebooks
make notebook
```

## Configuración

- Las variables de entorno están en `.env` (copiadas de `.env.example`)
- La configuración se valida con Pydantic en `config/config.py`
- Variables importantes: `ENVIRONMENT`, `MODEL_TYPE`, `RANDOM_SEED`, `LOG_LEVEL`
- Credenciales de ClickHouse: `CH_HOST`, `CH_PORT`, `CH_USER`, `CH_PASSWORD`, `CH_DATABASE`

## Convenciones de Código

- Usar `logger` de `fraud_detector.utils.logger` en lugar de `print()`
- Type hints requeridos
- Formateo con Black (line-length=100)
- Imports ordenados con isort (profile black)
- Pre-commit hooks configurados

## Datos

- Los datos crudos van en `data/raw/`
- Los datos procesados van en `data/processed/`
- Conexión a ClickHouse configurada en `src/fraud_detector/data/clickhouse_connector.py`

## Modelos

- Modelos guardados en `models/saved_models/`
- Experimentos en `models/experiments/`
- Tracking con MLflow (experimento: "fraud-detection")
