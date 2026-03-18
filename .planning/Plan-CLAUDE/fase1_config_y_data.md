# Fase 1: Configuracion, Data Loading y Validacion

## Archivos a modificar

| Archivo | Accion |
|---------|--------|
| `config/config.py` | REESCRIBIR |
| `src/fraud_detector/data/loader.py` | REESCRIBIR |
| `requirements.txt` | ACTUALIZAR |
| `.env.example` | ACTUALIZAR |
| `config/__init__.py` | ACTUALIZAR |
| `src/fraud_detector/data/__init__.py` | ACTUALIZAR |

---

## 1.1 config/config.py

### Eliminar (params supervisados)

ALL of: `model_type`, `test_size`, `validation_size`, `mlflow_*`, `use_gpu`, `fraud_threshold`,
`high_risk_threshold`, `auto_decline_threshold`, `api_host`, `api_port`, `database_url`,
`fraud_cost_per_transaction`, `false_positive_cost`, `review_capacity_per_day`, `precision_target`,
`recall_target`, `use_smote`, `smote_sampling_strategy`, `use_class_weights`, `use_temporal_split`,
`temporal_split_date`, `embargo_days`, `aggregation_windows`, `velocity_windows`,
`min_transactions_for_aggregation`, `enable_drift_detection`, `drift_threshold`,
`performance_degradation_threshold`, `raw_data_path`, `processed_data_path`, `models_dir` (as direct field).

### New Settings class content

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path("data"))
    logs_dir: Path = Field(default=Path("logs"))
    output_dir: Path = Field(default=Path("output"))

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = Field(default=8443, ge=1, le=65535)
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "default"
    clickhouse_table: str = "payments"
    clickhouse_secure: bool = True

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["text", "json"] = "json"

    # Random seed
    random_seed: int = 42

    # Temporal Split Boundaries
    train_start: str = "2025-01-01"
    train_end: str = "2025-07-01"
    val_end: str = "2025-09-01"
    test_end: str = "2026-01-01"

    # Proxy Label Definitions
    strict_proxy_statuses: str = "totally_refunded,refunded_to_credit"
    wide_proxy_statuses: str = "totally_refunded,refunded_to_credit,partially_refunded"

    # Isolation Forest Defaults
    if_n_estimators: int = 300
    if_max_samples: int = 1024
    if_max_features: float = 1.0

    # Grid Search Ranges (CSV strings parsed via properties)
    # NOTE: contamination removed from grid search - it does NOT affect
    # rank-based metrics (AUC-ROC, AP). Set to "auto" during training.
    grid_n_estimators: str = "100,200,300,500"
    grid_max_samples: str = "256,512,1024,2048"
    grid_max_features: str = "0.5,0.75,1.0,auto"

    # LOF Parameters
    lof_n_neighbors: str = "20,50,100"  # Small grid for fair comparison
    lof_contamination: str = "auto"

    # OC-SVM Parameters
    ocsvm_kernel: str = "rbf"
    ocsvm_nu: str = "0.02,0.05,0.10"  # Small grid
    ocsvm_gamma: str = "scale,auto"    # Small grid
    ocsvm_subsample: int = 100_000

    # Evaluation
    bootstrap_n: int = 1000
    top_k_percents: str = "0.01,0.02,0.05,0.10"  # Multiple k values
    shap_sample_size: int = 5000

    # Feature Engineering
    n_jobs: int = -1
```

### Properties for parsing grids

```python
@property
def strict_proxy_list(self) -> List[str]:
    return [s.strip() for s in self.strict_proxy_statuses.split(",")]

@property
def wide_proxy_list(self) -> List[str]:
    return [s.strip() for s in self.wide_proxy_statuses.split(",")]

@property
def grid_n_estimators_list(self) -> List[int]:
    return [int(x.strip()) for x in self.grid_n_estimators.split(",")]

@property
def grid_max_samples_list(self) -> List:
    return ["auto" if x.strip() == "auto" else int(x.strip())
            for x in self.grid_max_samples.split(",")]

@property
def grid_max_features_list(self) -> List:
    return ["auto" if x.strip() == "auto" else float(x.strip())
            for x in self.grid_max_features.split(",")]

@property
def lof_n_neighbors_list(self) -> List[int]:
    return [int(x.strip()) for x in self.lof_n_neighbors.split(",")]

@property
def ocsvm_nu_list(self) -> List[float]:
    return [float(x.strip()) for x in self.ocsvm_nu.split(",")]

@property
def ocsvm_gamma_list(self) -> List[str]:
    return [x.strip() for x in self.ocsvm_gamma.split(",")]

@property
def top_k_percents_list(self) -> List[float]:
    return [float(x.strip()) for x in self.top_k_percents.split(",")]
```

### Directory properties

```python
@property
def processed_dir(self) -> Path:
    return self.get_absolute_path(self.data_dir / "processed")

@property
def figures_dir(self) -> Path:
    return self.get_absolute_path(self.output_dir / "figures")

@property
def tables_dir(self) -> Path:
    return self.get_absolute_path(self.output_dir / "tables")

@property
def models_dir(self) -> Path:
    return self.get_absolute_path(self.output_dir / "models")

@property
def scores_dir(self) -> Path:
    return self.get_absolute_path(self.output_dir / "scores")

@property
def manifests_dir(self) -> Path:
    return self.get_absolute_path(self.output_dir / "manifests")
```

### ensure_directories

Crea los siguientes directorios:

- `data/`
- `data/raw/`
- `data/processed/`
- `logs/`
- `output/figures/`
- `output/tables/`
- `output/models/`
- `output/scores/`
- `output/manifests/`

---

## 1.2 src/fraud_detector/data/loader.py — REESCRIBIR

### Eliminar

- Class `DataLoader` completa
- Function `split_data()`

### New class: DataManager

#### Data quality constants

```python
REQUIRED_NON_NULL = ["id", "user_id", "facility_id", "amount", "created_at", "status"]
SAFE_INT32_COLS = ["user_id", "facility_id"]
SAFE_FLOAT32_COLS = ["amount", "technology_fee", "tax", "tip", "discount"]
```

#### Key methods

1. **`__init__(self, settings)`** — Stores settings, lazy connector/extractor.

2. **`_get_extractor()`** — Lazy init con retry logic (tenacity: 3 attempts, exponential backoff).

3. **`extract_from_clickhouse()`** — Extrae 3 splits, cada uno independiente:
   - Para cada split: extract, validate, downcast, save parquet + metadata JSON sidecar.
   - Usa atomic writes (write to `.tmp.parquet` then rename).
   - Agrega filtro `AND user_id > 0` post-extraccion (o via custom query).
   - Agrega filtro `AND _peerdb_is_deleted = 0` (CRITICO para tablas replicadas por PeerDB).
   - Elimina la columna `is_fraud` (la genera BASE_QUERY pero usa una definicion distinta a nuestro proxy).
   - Guarda dataset manifest: `{start_date, end_date, row_count, extracted_at, columns, clickhouse_host, filters_used}`.
   - Guarda snapshot del SQL ejecutado.

4. **`_validate_extraction(df, name)`**:
   - Check duplicados en columna `id` (warn + deduplicate).
   - Check NULLs en columnas criticas: `id`, `user_id`, `facility_id`, `amount`, `created_at`, `status`.
   - Check `amount > 0` (deberia estar garantizado por query pero verificar).
   - Log distribucion de valores de `status`.
   - Log uso de memoria.

5. **`_downcast(df)`** — Downcast SELECTIVO:
   - `float64` -> `float32`: solo `amount`, `technology_fee`, `tax`, `tip`, `discount` (NO columnas de id).
   - `int64` -> `int32`: solo `user_id`, `facility_id` (despues de verificar que max < 2^31).
   - SKIP downcasting: `id`, `reversed_id` (pueden exceder int32).

6. **`load_splits()`** -> tuple de 3 DataFrames.

7. **`load_split(name)`** -> single DataFrame con existence check + clear error message.

8. **`assign_proxy_labels(df, proxy_type)`** -> Series (static method).

9. **`_save_manifest(name, start, end, df)`** — Guarda JSON sidecar con metadata de extraccion.

10. **`_check_stale_cache(name, start, end)`** — Advierte si el parquet existente fue extraido con boundaries de fecha distintas.

11. **`close()`** — Desconecta.

#### Post-extraction filters (aplicados en Python despues de la extraccion)

```python
# user_id > 0 (excluir transacciones de sistema/anonimas)
df = df[df["user_id"] > 0]
# Eliminar columna is_fraud (usa definicion distinta a nuestro proxy)
df = df.drop(columns=["is_fraud"], errors="ignore")
```

#### Nota sobre _peerdb_is_deleted

La tabla ClickHouse usa replicacion PeerDB (ReplacingMergeTree). Las filas soft-deleted tienen `_peerdb_is_deleted = 1`. Estas **DEBEN** ser excluidas. Opciones:

- Agregar `AND _peerdb_is_deleted = 0` a `BASE_QUERY` en `clickhouse_connector.py` (pero el plan dice MANTENER ese archivo).
- Agregar filtro post-extraccion: `df = df[df.get("_peerdb_is_deleted", 0) == 0]`.
- Usar un custom query en `DataManager` que envuelve `BASE_QUERY` con el filtro adicional.

**RECOMENDACION:** Usar un custom query en el metodo extract de `DataManager` que agrega el filtro, sin modificar `clickhouse_connector.py`.

---

## 1.3 requirements.txt

### Eliminar

`imbalanced-learn`, `xgboost`, `lightgbm`, `mlflow`, `optuna`, `evidently`, `pandera`,
`great-expectations`, `lime`, `plotly`, `category-encoders`, `clickhouse-driver`, `requests`,
`pandas-stubs`, `types-requests`.

### Mantener

```
numpy>=1.24.0
pandas>=2.0.0
clickhouse-connect>=0.7.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
python-dotenv>=1.0.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
joblib>=1.3.0
tqdm>=4.65.0
loguru>=0.7.0
shap>=0.42.0
```

### Agregar

```
scipy>=1.11.0       # Mann-Whitney U, bootstrap, KS test
pyarrow>=14.0.0     # parquet I/O
tenacity>=8.0.0     # retry logic for ClickHouse
```

### Dev (mantener)

```
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.5.0
ipykernel>=6.25.0
jupyter>=1.0.0
notebook>=7.0.0
```

---

## 1.4 .env.example

Contenido completo nuevo:

```env
# Environment
ENVIRONMENT=development

# Paths
DATA_DIR=data
LOGS_DIR=logs
OUTPUT_DIR=output

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Random Seed
RANDOM_SEED=42

# ClickHouse Configuration
CLICKHOUSE_HOST=your-host.clickhouse.cloud
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=your_user
CLICKHOUSE_PASSWORD=your_password
CLICKHOUSE_DATABASE=your_database
CLICKHOUSE_TABLE=payments
CLICKHOUSE_SECURE=true

# Temporal Split Boundaries
TRAIN_START=2025-01-01
TRAIN_END=2025-07-01
VAL_END=2025-09-01
TEST_END=2026-01-01

# Proxy Label Definitions
STRICT_PROXY_STATUSES=totally_refunded,refunded_to_credit
WIDE_PROXY_STATUSES=totally_refunded,refunded_to_credit,partially_refunded

# Isolation Forest Defaults
IF_N_ESTIMATORS=300
IF_MAX_SAMPLES=1024
IF_MAX_FEATURES=1.0

# Grid Search Ranges (contamination excluded - does not affect rank-based metrics)
GRID_N_ESTIMATORS=100,200,300,500
GRID_MAX_SAMPLES=256,512,1024,2048
GRID_MAX_FEATURES=0.5,0.75,1.0,auto

# LOF Grid
LOF_N_NEIGHBORS=20,50,100
LOF_CONTAMINATION=auto

# OC-SVM Grid
OCSVM_KERNEL=rbf
OCSVM_NU=0.02,0.05,0.10
OCSVM_GAMMA=scale,auto
OCSVM_SUBSAMPLE=100000

# Evaluation
BOOTSTRAP_N=1000
TOP_K_PERCENTS=0.01,0.02,0.05,0.10
SHAP_SAMPLE_SIZE=5000
```

---

## 1.5 `__init__.py` updates

### `config/__init__.py`

Sin cambios.

### `src/fraud_detector/data/__init__.py`

```python
"""Data loading and connector modules."""
from fraud_detector.data.clickhouse_connector import ClickHouseConnector, FraudDataExtractor
from fraud_detector.data.loader import DataManager

__all__ = ["ClickHouseConnector", "FraudDataExtractor", "DataManager"]
```

---

## 1.6 Verificacion (Gate A)

Antes de proceder a Fase 2, verificar:

1. `Settings` carga sin errores con `.env` actual.
2. Todos los directorios de output se crean.
3. ClickHouse connection test pasa (`SELECT 1`).
4. 3 splits extraidos con conteos razonables (+-10% de: train ~3.1M, val ~1.1M, test ~2.5M).
5. Proxy rate strict ~6.33%, wide ~7.55% (+-1%).
6. No NULLs en columnas criticas.
7. No duplicados en `id` (o menos de 0.01%).
8. Manifest JSON guardado para cada split.
9. Columna `is_fraud` eliminada de los parquets.
10. `user_id > 0` verificado (no hay `user_id=0` en los datos).
