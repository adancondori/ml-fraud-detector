# Processed Data Directory

Este directorio contiene **datos procesados y transformados** listos para entrenamiento.

## Propósito

- Almacenar datos después de limpieza y preprocesamiento
- Resultados intermedios del pipeline de datos
- Datos particionados (train/val/test)
- Features engineered

## Tipos de archivos esperados

- `train.parquet` - Conjunto de entrenamiento
- `val.parquet` - Conjunto de validación
- `test.parquet` - Conjunto de prueba
- `features_*.parquet` - Features procesados
- `*.joblib` - Preprocessors guardados

## Ejemplo de estructura

```
data/processed/
├── train.parquet
├── val.parquet
├── test.parquet
├── preprocessor.joblib
└── feature_engineering_output.parquet
```

## Gestión

Estos archivos **NO se versionan en Git** porque:
- Son reproducibles desde `data/raw/` ejecutando el código
- Pueden ser grandes
- Se generan automáticamente

## Generación de datos procesados

Los datos procesados se generan con los scripts del proyecto:

```python
from fraud_detector.data.loader import DataLoader, split_data
from fraud_detector.features.preprocessor import FeaturePreprocessor

# Cargar raw data
loader = DataLoader()
df = loader.load_csv("data/raw/transactions.csv")

# Split
train_df, val_df, test_df = split_data(df, target_col='is_fraud')

# Guardar
loader.save_parquet(train_df, "data/processed/train.parquet")
loader.save_parquet(val_df, "data/processed/val.parquet")
loader.save_parquet(test_df, "data/processed/test.parquet")
```

## Configuración

El path se configura en `.env`:
```bash
PROCESSED_DATA_PATH=data/processed/processed_data.parquet
```

## Reproducibilidad

Para reproducir los datos procesados:

1. Asegúrate de tener los raw data
2. Ejecuta el notebook: `notebooks/01_exploratory_analysis.ipynb`
3. O ejecuta el script de preprocesamiento (cuando lo crees)

## Cache Strategy

Considera usar cache para acelerar el desarrollo:
- Primera ejecución: procesa desde raw
- Siguientes ejecuciones: carga desde processed
- Invalida cache cuando cambies el código de preprocessing
