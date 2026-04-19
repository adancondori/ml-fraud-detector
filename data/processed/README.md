# Processed Data Directory

Parquets generados por el pipeline de deteccion de anomalias. **No se versionan en Git.**

## Archivos esperados

| Archivo | Contenido | Generado por |
|---------|-----------|-------------|
| `warm_raw.parquet` | Dic 2024 — warm history | `DataManager.extract_from_clickhouse()` |
| `train_raw.parquet` | Ene-Jun 2025 | `DataManager.extract_from_clickhouse()` |
| `val_raw.parquet` | Jul-Ago 2025 | `DataManager.extract_from_clickhouse()` |
| `test_raw.parquet` | Sep-Dic 2025 | `DataManager.extract_from_clickhouse()` |
| `train_features.parquet` | 31 features + metadata | `FeatureEngineer.transform()` |
| `val_features.parquet` | 31 features + metadata | `FeatureEngineer.transform()` |
| `test_features.parquet` | 31 features + metadata | `FeatureEngineer.transform()` |

## Reproduccion

```python
from fraud_detector.data.loader import DataManager
from config.config import get_settings

dm = DataManager(get_settings())
dm.extract_from_clickhouse()  # genera los 4 _raw.parquet + manifests
```
