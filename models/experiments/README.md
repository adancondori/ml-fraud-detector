# Experiments Directory

Este directorio contiene **modelos experimentales durante el desarrollo**.

## Propósito

- Almacenar modelos durante experimentación
- Pruebas de hiperparámetros
- Arquitecturas alternativas
- Modelos en desarrollo (no para producción)

## Diferencia con `saved_models/`

| Experiments | Saved Models |
|-------------|--------------|
| Modelos en desarrollo | Modelos validados |
| Múltiples variaciones | Versiones estables |
| Puede ser caótico | Organizado |
| Temporal | Permanente |

## Estructura sugerida

```
models/experiments/
├── 2024-01-15_xgboost_tuning/
│   ├── model_depth_3.pkl
│   ├── model_depth_5.pkl
│   ├── model_depth_10.pkl
│   └── results.json
├── 2024-01-20_random_forest/
│   ├── rf_100_trees.pkl
│   ├── rf_200_trees.pkl
│   └── comparison.csv
└── 2024-02-01_feature_selection/
    ├── model_top_10_features.pkl
    └── feature_importance.png
```

## Organización por fecha

Usa prefijo de fecha para experimentos:
```
YYYY-MM-DD_descripcion_experimento/
```

## Tracking con MLflow

**Recomendado**: Usa MLflow en lugar de guardar archivos manualmente.

```python
import mlflow
from fraud_detector.models.trainer import ModelTrainer

mlflow.set_experiment("hyperparameter-tuning")

with mlflow.start_run(run_name="xgboost_depth_10"):
    trainer = ModelTrainer(
        model_type='xgboost',
        model_params={'max_depth': 10}
    )
    trainer.train(X_train, y_train)

    # MLflow guarda automáticamente el modelo
    # No necesitas guardarlo manualmente aquí
```

Ver experimentos:
```bash
mlflow ui
```

## Cuando guardar manualmente

Guarda en este directorio solo si:
- Quieres compartir rápidamente con el equipo
- Necesitas backup temporal
- MLflow no está disponible

## Convención de nombres

```
{experiment_date}_{description}/{variant_name}.pkl
```

Ejemplo:
```
2024-01-15_lgbm_tuning/
├── lgbm_lr_0.01.pkl
├── lgbm_lr_0.05.pkl
└── lgbm_lr_0.1.pkl
```

## Metadata de experimentos

Guarda un archivo `README.md` o `notes.txt` en cada carpeta:

```
# Experiment: XGBoost Hyperparameter Tuning
Date: 2024-01-15
Objective: Find optimal max_depth

## Setup
- Dataset: transactions_v1.0
- Features: 15 numeric + 3 categorical
- Train size: 10,000 samples

## Variants tested
1. max_depth=3: F1=0.75
2. max_depth=5: F1=0.82
3. max_depth=10: F1=0.88 ← Best

## Conclusion
max_depth=10 gives best results without overfitting
Next: Test learning_rate variations
```

## Limpieza

### Política de retención

Limpia regularmente:
```bash
# Eliminar experimentos > 30 días (ejemplo)
find models/experiments/ -type d -mtime +30 -exec rm -rf {} +
```

### Antes de eliminar

1. Verifica que los resultados estén en MLflow
2. Documenta conclusiones en notebook o documento
3. Mueve modelos importantes a `saved_models/`

## Automatización

Script de limpieza (ejemplo):

```python
# cleanup_experiments.py
from pathlib import Path
from datetime import datetime, timedelta

experiments_dir = Path("models/experiments")
retention_days = 30

for exp_dir in experiments_dir.iterdir():
    if exp_dir.is_dir():
        # Parsear fecha del nombre
        date_str = exp_dir.name.split('_')[0]
        exp_date = datetime.strptime(date_str, '%Y-%m-%d')

        if datetime.now() - exp_date > timedelta(days=retention_days):
            print(f"Removing old experiment: {exp_dir}")
            # shutil.rmtree(exp_dir)  # Descomentar para eliminar
```

## Best Practices

1. **Usa MLflow** siempre que sea posible
2. **Documenta** cada experimento
3. **Nombra consistentemente** los archivos
4. **Limpia regularmente** experimentos antiguos
5. **Promueve** buenos modelos a `saved_models/`

## Git

Estos modelos **NO se versionan** en Git (muy pesados y temporales).
Usa MLflow o cloud storage para compartir.
