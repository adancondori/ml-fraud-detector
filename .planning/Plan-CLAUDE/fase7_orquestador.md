# Fase 7: Orquestador Principal

## Archivo: `run_pipeline.py` (NUEVO, raiz del proyecto)

### CLI

```
python run_pipeline.py                    # All steps
python run_pipeline.py --step 3           # Single step
python run_pipeline.py --from-step 5      # Steps 5-8
python run_pipeline.py --skip-grid-search # IF with defaults
python run_pipeline.py --proxy wide       # Wide proxy
python run_pipeline.py --fast             # Reduced bootstrap (100), no grid search
python run_pipeline.py --dry-run          # Show what would run, validate inputs
```

---

## Correcciones criticas respecto a la version anterior

1. **Validacion de prerrequisitos** antes de cada paso (verificar que archivos de entrada existen).
2. **Logging de tiempo transcurrido** por paso.
3. **`gc.collect()`** despues de `del` de objetos grandes.
4. **`proxy_type` propagado** de step 6 a step 8 (antes estaba hardcodeado "strict" en step 8).
5. **Step 8 maneja gracefully** la ausencia de `grid_search_if.csv`.
6. **Sensibilidad de feature 17** ejecutada en step 6 (antes faltaba del orquestador).
7. **Scores guardados como parquet** con `id` y `created_at` (no `.npy` desnudo).
8. **Resumen del pipeline** impreso al final con metricas clave.

---

## Los 8 Pasos

### Step 1: Extraccion de datos

- `DataManager.extract_from_clickhouse()`
- Validar conteos (dentro de +-10% de lo esperado)
- Guardar dataset manifest (`dataset_manifest.json`)

**Output**: `data/processed/{train,val,test}_raw.parquet`, `output/manifests/dataset_manifest.json`

### Step 2: Feature engineering

- `FeatureEngineer.fit()` en train, `.transform()` en los 3 splits
- Guardar `FeatureEngineer` con `.save()`
- Loguear estadisticas de features para Tabla 3.7
- Guardar `feature_statistics.csv` para reporting

**Output**: `data/processed/{train,val,test}_features.parquet`, `output/models/feature_engineer.joblib`

### Step 3: Preprocesamiento

- `StandardScaler` fit en train
- Guardar arrays escalados como `.npy` (estos son anonimos, OK como npy)
- Guardar `scaler.joblib`

**Output**: `output/scores/X_{train,val,test}.npy`, `output/models/scaler.joblib`

### Step 4: Entrenamiento de modelos

- Grid search IF (64 combos con checkpoint/resume)
- Grid search LOF (3 combos)
- Grid search OC-SVM (6 combos)
- Guardar todos los modelos + best params JSON + grid CSVs
- Loguear tiempos de entrenamiento

**Output**: `output/models/{isolation_forest,lof,ocsvm}.joblib`, `output/grid_search_{if,lof,ocsvm}.csv`, `output/models/best_params_{if,lof,ocsvm}.json`

### Step 5: Scoring del test set

- Puntuar con los 3 modelos
- **FIX: Guardar scores como PARQUET** con `id` y `created_at` del `test_features.parquet`:

```python
test_df = pd.read_parquet(settings.processed_dir / "test_features.parquet")
scores_df = pd.DataFrame({
    "id": test_df["id"],
    "created_at": test_df["created_at"],
})
for name in ["isolation_forest", "lof", "ocsvm"]:
    scores = trainer.score(name, X_test)
    scores_df[f"{name}_score"] = scores
scores_df.to_parquet(settings.scores_dir / "test_scores.parquet", index=False)
```

**Output**: `output/scores/test_scores.parquet` (id + created_at + 3 columnas de scores)

### Step 6: Evaluacion

- `full_evaluation` para los 3 modelos (con fechas para estabilidad temporal)
- Comparacion HE4
- Sensibilidad: proxy (strict vs wide)
- Sensibilidad: feature 17 (AUC + Jaccard + Spearman) -- **antes faltaba, ahora agregada**
- Evaluacion por status
- Correccion de Holm-Bonferroni en p-values
- Guardar `results.json` con serializador numpy-safe
- **Almacenar `proxy_type` en results.json** para que step 8 lo use

**Output**: `output/results.json`

### Step 7: SHAP

- `TreeExplainer` sobre IF
- Subsample del test set (5000 filas)
- Guardar figura SHAP

**Output**: `output/figures/shap_summary.{pdf,png}`

### Step 8: Reportes

- **Leer `proxy_type` de `results.json`** (no hardcodeado)
- Generar todas las tablas LaTeX
- Generar todas las figuras (ROC, PR, distribuciones, curva de enriquecimiento, estabilidad temporal, heatmap de grid, correlacion)
- **Verificar existencia de `grid_search_if.csv`** antes del heatmap
- Computar `split_info` y `feature_statistics` desde parquet para tablas 3.5, 3.7

**Output**: `output/tables/table_3_*.tex`, `output/figures/*.{pdf,png}`

---

## Validacion de prerrequisitos

```python
STEP_INPUTS = {
    1: [],  # No prerequisites
    2: ["data/processed/train_raw.parquet", "data/processed/val_raw.parquet", "data/processed/test_raw.parquet"],
    3: ["data/processed/train_features.parquet"],
    4: ["output/scores/X_train.npy", "output/scores/X_val.npy"],
    5: ["output/models/isolation_forest.joblib", "output/scores/X_test.npy"],
    6: ["output/scores/test_scores.parquet", "data/processed/test_features.parquet"],
    7: ["output/models/isolation_forest.joblib", "output/scores/X_test.npy"],
    8: ["output/results.json"],
}

def validate_prerequisites(step, settings):
    for rel_path in STEP_INPUTS.get(step, []):
        abs_path = settings.get_absolute_path(Path(rel_path))
        if not abs_path.exists():
            raise FileNotFoundError(f"Step {step} requires {rel_path}. Run previous steps first.")
```

---

## Logging de tiempo por paso

```python
import time
import gc

def run_step(step_num, step_fn, *args, **kwargs):
    """Wrapper que loguea tiempo y limpia memoria."""
    logger.info("=" * 60)
    logger.info(f"STEP {step_num}: Starting")
    logger.info("=" * 60)

    validate_prerequisites(step_num, settings)
    t0 = time.time()

    result = step_fn(*args, **kwargs)

    elapsed = time.time() - t0
    minutes, seconds = divmod(int(elapsed), 60)
    logger.info(f"STEP {step_num}: Completed in {minutes}m {seconds}s")

    gc.collect()
    return result
```

---

## Resumen del pipeline (impreso al final)

```
============================================================
PIPELINE COMPLETE
============================================================
Model         | AUC-ROC | AP      | EF@5%  | HE1  | HE2  | HE3
--------------+---------+---------+--------+------+------+-----
Isol. Forest  | 0.7834  | 0.1521  | 2.41   | PASS | PASS | PASS
LOF           | 0.7456  | 0.1203  | 1.90   | PASS | PASS | FAIL
OC-SVM        | 0.7102  | 0.1045  | 1.65   | PASS | PASS | FAIL

HE4 (IF >= competitors in >=2/4 metrics): PASS

Total time: 2h 15m 32s
============================================================
```

---

## parse_args()

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
    parser.add_argument("--step", type=int, default=None,
                        help="Run only this step (1-8)")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from this step")
    parser.add_argument("--skip-grid-search", action="store_true",
                        help="Skip grid search, use default IF params")
    parser.add_argument("--proxy", type=str, default="strict",
                        choices=["strict", "wide"],
                        help="Proxy label type (default: strict)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: reduced bootstrap (100), no grid search")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run and validate inputs")
    return parser.parse_args()

def should_run(args, step_num: int) -> bool:
    if args.step is not None:
        return args.step == step_num
    return step_num >= args.from_step
```

---

## Imports del orquestador

```python
import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.config import settings
from fraud_detector.data.loader import DataManager
from fraud_detector.features.engineering import FeatureEngineer, FEATURE_NAMES
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
from fraud_detector.models.trainer import AnomalyModelTrainer
from fraud_detector.evaluation.metrics import HypothesisEvaluator
from fraud_detector.reporting.latex_tables import ThesisTableGenerator
from fraud_detector.reporting.figures import ThesisFigureGenerator
from fraud_detector.utils.logger import logger
```
