# ML Fraud Detector

Sistema robusto de detección de fraude utilizando Machine Learning con arquitectura modular, logging avanzado, y seguimiento de experimentos con MLflow.

## Características

- **Arquitectura Modular**: Código organizado en módulos reutilizables
- **Configuración Robusta**: Gestión de configuración con Pydantic y variables de entorno
- **Logging Avanzado**: Sistema de logs estructurado con Loguru (rotación, compresión, JSON)
- **Calidad de Código**: Pre-commit hooks, formateo automático, type checking
- **Seguimiento de Experimentos**: Integración con MLflow para tracking de modelos
- **Múltiples Algoritmos**: XGBoost, LightGBM, Random Forest, Logistic Regression
- **Notebooks Integrados**: Jupyter notebooks conectados al paquete Python

## Estructura del Proyecto

```
ml-fraud-detector/
├── config/                      # Configuración
│   ├── __init__.py
│   └── config.py               # Settings con Pydantic
├── src/fraud_detector/         # Código fuente principal
│   ├── __init__.py
│   ├── data/                   # Carga y manejo de datos
│   │   ├── __init__.py
│   │   └── loader.py          # DataLoader, split_data
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   └── preprocessor.py    # FeaturePreprocessor
│   ├── models/                 # Entrenamiento de modelos
│   │   ├── __init__.py
│   │   └── trainer.py         # ModelTrainer con MLflow
│   ├── evaluation/             # Evaluación de modelos
│   │   └── __init__.py
│   └── utils/                  # Utilidades
│       ├── __init__.py
│       └── logger.py          # Logging configurado
├── notebooks/                  # Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb
├── data/                       # Datos
│   ├── raw/                   # Datos crudos
│   ├── processed/             # Datos procesados
│   └── external/              # Datos externos
├── models/                     # Modelos guardados
│   ├── saved_models/          # Modelos productivos
│   └── experiments/           # Modelos experimentales
├── tests/                      # Tests unitarios
├── logs/                       # Archivos de log
├── .env.example               # Ejemplo de variables de entorno
├── .gitignore
├── .pre-commit-config.yaml    # Configuración pre-commit
├── pyproject.toml             # Configuración del proyecto
├── setup.py                   # Setup del paquete
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd ml-fraud-detector
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
# Dependencias básicas
pip install -r requirements.txt

# O con dependencias de desarrollo
pip install -r requirements-dev.txt

# Instalar el paquete en modo editable
pip install -e .
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env con tu configuración
```

### 5. (Opcional) Configurar pre-commit hooks

```bash
pre-commit install
```

## Configuración

La configuración se maneja a través de:

1. **Archivo `.env`**: Variables de entorno
2. **`config/config.py`**: Configuración con validación Pydantic

### Variables de Entorno Principales

```bash
ENVIRONMENT=development          # development, staging, production
MODEL_TYPE=xgboost              # xgboost, lightgbm, random_forest, logistic
RANDOM_SEED=42
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                 # text, json
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

## Uso Rápido

### 1. Exploratory Analysis (Notebook)

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 2. Uso Programático

```python
from fraud_detector.data.loader import DataLoader, split_data
from fraud_detector.features.preprocessor import FeaturePreprocessor
from fraud_detector.models.trainer import ModelTrainer
from fraud_detector.utils.logger import logger
from config.config import settings

# Cargar datos
loader = DataLoader()
df = loader.load_csv("data/raw/transactions.csv")

# Split data
train_df, val_df, test_df = split_data(df, target_col='is_fraud')

# Preprocesar
preprocessor = FeaturePreprocessor(
    numeric_features=['amount', 'distance_from_home'],
    categorical_features=['merchant_category']
)
X_train = preprocessor.fit_transform(train_df.drop('is_fraud', axis=1))
y_train = train_df['is_fraud']

# Entrenar modelo
trainer = ModelTrainer(
    model_type='xgboost',
    model_params={'max_depth': 6, 'n_estimators': 100}
)
trainer.train(X_train, y_train)

# Evaluar
metrics = trainer.evaluate(X_val, y_val)
logger.info(f"Validation metrics: {metrics}")

# Guardar modelo
trainer.save_model('models/saved_models/fraud_model.pkl')
```

## Logging

El sistema de logging está configurado con **Loguru** y proporciona:

- **Console output**: Con colores en desarrollo
- **File rotation**: Rotación cada 10 MB
- **Retention**: 30 días para logs generales, 60 para errores
- **Compression**: Compresión automática (zip)
- **Formato JSON**: Opcional para producción

Los logs se guardan en:
- `logs/fraud_detector.log`: Logs generales
- `logs/errors.log`: Solo errores

```python
from fraud_detector.utils.logger import logger

logger.info("Mensaje informativo")
logger.warning("Advertencia")
logger.error("Error")
logger.debug("Debug info")
```

## MLflow Tracking

Todos los experimentos se rastrean automáticamente con MLflow:

```bash
# Ver experimentos
mlflow ui
# Visita http://localhost:5000
```

## Calidad de Código

### Formateo automático

```bash
# Black (formateo de código)
black src/ tests/ notebooks/

# isort (ordenar imports)
isort src/ tests/

# Flake8 (linting)
flake8 src/ tests/
```

### Type Checking

```bash
mypy src/
```

### Tests

```bash
# Ejecutar todos los tests
pytest

# Con coverage
pytest --cov=src/fraud_detector --cov-report=html

# Ver reporte
open htmlcov/index.html
```

### Pre-commit Hooks

Los hooks se ejecutan automáticamente en cada commit:

```bash
# Ejecutar manualmente en todos los archivos
pre-commit run --all-files
```

## Desarrollo

### Agregar nuevas características

1. Crear módulo en `src/fraud_detector/`
2. Agregar tests en `tests/`
3. Actualizar documentación
4. Ejecutar tests y linters

### Agregar dependencias

```bash
# Agregar a requirements.txt
echo "nueva-libreria>=1.0.0" >> requirements.txt

# Reinstalar
pip install -r requirements.txt
```

## MLflow Experiments

### Visualizar experimentos

```bash
mlflow ui --port 5000
```

### Comparar modelos

1. Abre MLflow UI
2. Selecciona el experimento "fraud-detection"
3. Compara métricas entre runs
4. Descarga modelos

## Best Practices

1. **Siempre usa el logger** en lugar de `print()`
2. **Configura variables** en `.env` en lugar de hardcodear
3. **Usa type hints** para mejor documentación
4. **Escribe tests** para código crítico
5. **Commita con pre-commit** hooks activos
6. **Documenta funciones** con docstrings
7. **Versiona datos** y modelos importantes

## Troubleshooting

### Error: ModuleNotFoundError

```bash
# Asegúrate de instalar el paquete en modo editable
pip install -e .
```

### Error: MLflow database locked

```bash
# Detén el servidor MLflow
pkill -f "mlflow"
```

### Logs no aparecen

```bash
# Verifica el nivel de log en .env
LOG_LEVEL=INFO  # o DEBUG para más detalle
```

## Próximos Pasos

- [ ] Implementar feature engineering avanzado
- [ ] Agregar detección de anomalías
- [ ] Implementar balanceo de clases (SMOTE)
- [ ] Crear pipeline de deployment
- [ ] Agregar monitoreo en producción
- [ ] Implementar explicabilidad (SHAP)
- [ ] API REST con FastAPI

## Contribución

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Licencia

MIT License

## Contacto

Tu Nombre - your.email@example.com

---

**Nota**: Este proyecto está en desarrollo activo. Para tesis y proyectos académicos.
