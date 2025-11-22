# Saved Models Directory

Este directorio contiene **modelos entrenados listos para producción**.

## Propósito

- Almacenar modelos finales entrenados
- Versiones de modelos para deployment
- Modelos que han pasado validación y están aprobados

## Estructura recomendada

```
models/saved_models/
├── fraud_detector_v1.0.pkl
├── fraud_detector_v1.0_metadata.json
├── fraud_detector_v1.1.pkl
├── fraud_detector_v1.1_metadata.json
└── production/
    ├── current_model.pkl
    └── preprocessor.joblib
```

## Convención de nombres

Usa un esquema consistente:

```
{model_name}_{version}_{date}.{extension}
```

Ejemplos:
- `fraud_rf_v1.0_20240115.pkl`
- `fraud_xgboost_v2.3_20240220.joblib`
- `fraud_ensemble_v1.0_20240301.pkl`

## Metadata

Guarda metadata junto con cada modelo:

```json
{
  "model_name": "fraud_detector_v1.0",
  "model_type": "xgboost",
  "training_date": "2024-01-15",
  "framework": "scikit-learn",
  "metrics": {
    "test_accuracy": 0.95,
    "test_precision": 0.89,
    "test_recall": 0.87,
    "test_f1": 0.88,
    "test_roc_auc": 0.96
  },
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.1
  },
  "features": ["amount", "merchant_category", "..."],
  "preprocessing": "preprocessor_v1.0.joblib",
  "dataset_version": "2024-01-15",
  "mlflow_run_id": "abc123...",
  "notes": "Modelo inicial para producción"
}
```

## Guardar modelos

Usando el proyecto:

```python
from fraud_detector.models.trainer import ModelTrainer

trainer = ModelTrainer(model_type='xgboost')
trainer.train(X_train, y_train)

# Guardar
trainer.save_model('models/saved_models/fraud_xgboost_v1.0.pkl')
```

## Cargar modelos

```python
from fraud_detector.models.trainer import ModelTrainer

trainer = ModelTrainer()
trainer.load_model('models/saved_models/fraud_xgboost_v1.0.pkl')

# Predecir
predictions = trainer.predict(X_test)
```

## Versionado

### Estrategia de versionado

- **Major version** (1.0 → 2.0): Cambio de arquitectura/algoritmo
- **Minor version** (1.0 → 1.1): Mejoras, nuevos features
- **Patch version** (1.1.0 → 1.1.1): Bugfixes, reentrenamiento

### Git LFS (opcional)

Para versionar modelos grandes en Git:

```bash
git lfs install
git lfs track "models/saved_models/*.pkl"
git lfs track "models/saved_models/*.joblib"
```

Agrega al `.gitignore`:
```
# !models/saved_models/*_v*.pkl
```

## Gestión de almacenamiento

Los modelos **NO se versionan en Git** por defecto.

### Alternativas:
1. **MLflow**: Tracking automático de modelos
2. **Cloud Storage**: S3, GCS, Azure Blob
3. **Model Registry**: MLflow Model Registry, DVC
4. **Artifact stores**: Weights & Biases, Neptune.ai

## Deployment

Para producción:

```python
# Copiar modelo actual
import shutil
shutil.copy(
    'models/saved_models/fraud_xgboost_v1.0.pkl',
    'models/saved_models/production/current_model.pkl'
)
```

## CI/CD Integration

Considera automatizar:
- Validación de modelos antes de guardar
- Tests de regresión
- Benchmark contra modelo anterior
- Documentación automática de metadata

## Limpieza

Política de retención:
- Mantener últimas 3 versiones major
- Mantener todas las versiones de producción
- Archivar modelos obsoletos después de 6 meses
