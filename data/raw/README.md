# Raw Data Directory

Este directorio contiene los **datos crudos originales** sin procesar.

## Propósito

- Almacenar datos tal como se obtienen de la fuente
- **NO MODIFICAR** estos archivos (solo lectura)
- Los datos aquí **NO se versionan en Git** (ver .gitignore)

## Tipos de archivos esperados

- `*.csv` - Archivos CSV de transacciones
- `*.json` - Datos en formato JSON
- `*.xlsx` - Archivos Excel (si aplica)
- `*.parquet` - Datos en formato Parquet

## Ejemplo de archivos

```
data/raw/
├── transactions_2024_01.csv
├── transactions_2024_02.csv
├── customer_data.json
└── merchant_info.csv
```

## Gestión de datos

### Datos pequeños (< 100 MB)
Si necesitas compartir datos de ejemplo pequeños, puedes:
1. Descomentarlas líneas en `.gitignore`:
   ```
   !data/raw/sample_*.csv
   !data/raw/demo_*.json
   ```
2. Nombrar tus archivos con prefijo `sample_` o `demo_`

### Datos grandes (> 100 MB)
Para datos grandes, usa:
- **Cloud Storage**: S3, Google Cloud Storage, Azure Blob
- **DVC (Data Version Control)**: Para versionar datasets
- **Shared Network Drive**: Para equipos locales

## Configuración

El path a estos datos se configura en `.env`:
```bash
RAW_DATA_PATH=data/raw/transactions.csv
```

## Carga de datos

Usa el `DataLoader` del proyecto:

```python
from fraud_detector.data.loader import DataLoader

loader = DataLoader()
df = loader.load_csv("data/raw/transactions.csv")
```

## Importante

⚠️ **Nunca commitees datos sensibles o grandes al repositorio Git**
- Usa `.gitignore` para excluir datos
- Documenta dónde obtener los datos en este README
- Considera usar datos sintéticos para demos
