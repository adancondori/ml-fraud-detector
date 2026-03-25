# Fase 9: Orquestador del Pipeline

> **Estado:** esta fase sigue **pendiente de implementacion**. El documento define el contrato del orquestador futuro usando los nombres y modulos reales ya existentes en el repo.

## `run_pipeline.py` (pendiente, raíz del proyecto)

Punto de entrada único que ejecuta las 8 fases del pipeline de detección de anomalías
de forma secuencial, con soporte para ejecución parcial, validación de prerrequisitos
y resumen final de métricas.

---

## Notas de diseño (SOLID)

### SRP — Cada paso es una función standalone

Cada paso del pipeline debe implementarse como una **función independiente** (`step1_extract`, `step2_engineer`, etc.) que puede invocarse y testearse de forma aislada. El orquestador solo se encarga de: (1) parsear argumentos, (2) determinar qué pasos ejecutar, (3) invocar cada paso dentro del wrapper de timing/logging. La lógica de negocio vive en los módulos, no en el orquestador.

```python
def step1_extract(settings, proxy_type: str) -> None:
    """Extrae datos crudos. Testeable de forma independiente."""
    ...

def step2_engineer(settings) -> None:
    """Genera features. Testeable de forma independiente."""
    ...
```

### DIP — Inyección de dependencias, no imports globales

El orquestador actualmente importa clases concretas al inicio del archivo. Para cumplir con DIP:

- **Cada función de paso recibe sus dependencias como parámetros** en lugar de importarlas en el scope global. Esto permite inyectar mocks en tests.
- Como mínimo, las clases pesadas (`DataManager`, `FeatureEngineer`, `ModelTrainer`, etc.) deben importarse **dentro de cada función de paso** (lazy import), no al nivel del módulo.

```python
# MAL: import global de clase concreta
from fraud_detector.models.trainer import ModelTrainer

def step4_train(settings):
    trainer = ModelTrainer()  # acoplamiento fuerte
    ...

# BIEN: import local + parámetro inyectable
def step4_train(settings, trainer_cls=None):
    if trainer_cls is None:
        from fraud_detector.models.trainer import ModelTrainer
        trainer_cls = ModelTrainer
    trainer = trainer_cls()
    ...
```

Esto respeta DIP sin necesidad de un framework de inyección de dependencias completo.

---

## Contratos de test (`test_pipeline.py`, pendiente)

Antes de implementar, escribir estos tests (TDD red-green-refactor):

```python
def test_prerequisite_validation_fails_on_missing_file(tmp_path):
    """Si falta un artefacto prerrequisito, el paso debe fallar con mensaje claro."""
    # Configurar STEP_INPUTS para un paso que requiere un archivo inexistente
    missing = tmp_path / "nonexistent.parquet"
    with pytest.raises(FileNotFoundError, match="nonexistent.parquet"):
        validate_prerequisites(step=2, base_dir=tmp_path)

def test_step_wrapper_logs_timing(caplog):
    """El wrapper de paso debe registrar el tiempo de ejecución en formato MM:SS."""
    with caplog.at_level(logging.INFO):
        run_step(step_num=1, name="Test Step", fn=lambda: None)
    assert any(":" in r.message and "Step 1" in r.message for r in caplog.records)

def test_dry_run_does_not_execute_steps():
    """Con --dry-run, ningún paso debe ejecutarse realmente."""
    executed = []
    steps = {1: lambda: executed.append(1), 2: lambda: executed.append(2)}
    run_pipeline(steps, dry_run=True)
    assert executed == []

def test_from_step_skips_earlier_steps():
    """Con --from-step 3, los pasos 1 y 2 no se ejecutan."""
    executed = []
    steps = {1: lambda: executed.append(1), 2: lambda: executed.append(2), 3: lambda: executed.append(3)}
    run_pipeline(steps, from_step=3)
    assert executed == [3]

def test_single_step_runs_only_specified_step():
    """Con --step 3, solo se ejecuta el paso 3."""
    executed = []
    steps = {1: lambda: executed.append(1), 2: lambda: executed.append(2), 3: lambda: executed.append(3)}
    run_pipeline(steps, step=3)
    assert executed == [3]
```

---

## Interfaz CLI

```bash
python run_pipeline.py                    # Todos los pasos
python run_pipeline.py --step 3           # Un solo paso
python run_pipeline.py --from-step 5      # Desde el paso 5 en adelante
python run_pipeline.py --skip-grid-search # IF con hiperparámetros por defecto
python run_pipeline.py --proxy wide       # Proxy amplio (incluye parciales)
python run_pipeline.py --fast             # Bootstrap reducido (100), sin grid search
python run_pipeline.py --dry-run          # Solo validar inputs, no ejecutar
```

---

## Los 8 pasos del pipeline

| Paso | Nombre                        | Salida principal                                                  |
|------|-------------------------------|-------------------------------------------------------------------|
| 1    | Data Extraction               | raw parquets + manifest                                           |
| 2    | Feature Engineering           | feature parquets + `feature_engineer.joblib`                      |
| 3    | Preprocessing                 | `.npy` arrays + `scaler.joblib`                                   |
| 4    | Model Training                | model joblibs + grid search CSVs                                  |
| 5    | Test Set Scoring              | `test_scores.parquet`                                             |
| 6    | Evaluation                    | `results.json` (HE1-HE4, bootstrap, temporal)                     |
| 7    | Sensitivity + SHAP + Post-Hoc | `results_sensitivity.json`, `results_posthoc.json`, figuras SHAP  |
| 8    | Reports                       | tablas LaTeX + todas las figuras                                  |

---

## Validación de prerrequisitos por paso

Antes de ejecutar cada paso, el orquestador verifica que existan los artefactos
requeridos. Si falta alguno, el pipeline se detiene con mensaje claro indicando
qué paso previo debe ejecutarse.

```python
STEP_INPUTS = {
    1: [],
    2: ["data/processed/train_raw.parquet", ...],
    3: ["data/processed/train_features.parquet"],
    4: ["output/scores/X_train.npy", "output/scores/X_val.npy"],
    5: ["output/models/isolation_forest.joblib", "output/scores/X_test.npy"],
    6: ["output/scores/test_scores.parquet", "data/processed/test_features.parquet"],
    7: ["output/models/isolation_forest.joblib", "output/scores/X_test.npy"],
    8: ["output/results.json", "output/results_sensitivity.json", "output/results_posthoc.json"],
}
```

---

## Envoltura por paso (step wrapper)

Cada paso se ejecuta dentro de un wrapper que provee:

1. **Logger banner** con número y nombre del paso
2. **Validación de prerrequisitos** (verifica existencia de archivos en `STEP_INPUTS`)
3. **Timing** con formato `MM:SS` al finalizar
4. **`gc.collect()`** después de `del` de objetos grandes para liberar memoria

---

## Resumen final del pipeline

Al completar todos los pasos, se imprime un resumen que incluye:

- **Tabla de métricas del modelo**: AUC-ROC, AP, Enrichment Factor, HE1-HE4 con indicador pass/fail
- **Tiempo total de ejecución** del pipeline completo

---

## Imports

```python
import argparse, gc, json, time
from pathlib import Path

import joblib, numpy as np, pandas as pd

from config.config import settings
from fraud_detector.data.loader import DataManager
from fraud_detector.features.engineering import FeatureEngineer, FEATURE_NAMES
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
from fraud_detector.models.trainer import ModelTrainer
from fraud_detector.evaluation.metrics import evaluate_scores, bootstrap_ci
# Modulos de reporting pendientes
# from fraud_detector.reporting.latex_tables import ThesisTableGenerator
# from fraud_detector.reporting.figures import ThesisFigureGenerator
from fraud_detector.utils.logger import logger
```

> **Estado actual del repo:** `run_pipeline.py` y el paquete `reporting/` siguen pendientes. Este documento define el contrato del orquestador futuro sin contradecir la arquitectura ya implementada.

---

## Funciones auxiliares del orquestador

### `parse_args()` — Parser de argumentos CLI

```python
def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para el pipeline."""
    parser = argparse.ArgumentParser(description="Pipeline de detección de anomalías")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, choices=range(1, 9), help="Ejecutar solo este paso")
    group.add_argument("--from-step", type=int, choices=range(1, 9), help="Ejecutar desde este paso")
    parser.add_argument("--skip-grid-search", action="store_true", help="Omitir grid search de IF")
    parser.add_argument("--proxy", choices=["strict", "wide"], default="strict", help="Tipo de proxy")
    parser.add_argument("--fast", action="store_true", help="Bootstrap reducido, sin grid search")
    parser.add_argument("--dry-run", action="store_true", help="Solo validar, no ejecutar")
    return parser.parse_args()
```

### `should_run(step_num, args)` — Decide si un paso debe ejecutarse

```python
def should_run(step_num: int, args: argparse.Namespace) -> bool:
    """Retorna True si el paso debe ejecutarse dados los argumentos CLI.

    Lógica:
    - Si --step N: solo True cuando step_num == N
    - Si --from-step N: True cuando step_num >= N
    - Si ninguno: siempre True (ejecutar todos)
    """
    if args.step is not None:
        return step_num == args.step
    if args.from_step is not None:
        return step_num >= args.from_step
    return True
```

### `validate_prerequisites(step_num, base_dir)` — Verifica artefactos requeridos

```python
def validate_prerequisites(step_num: int, base_dir: Path) -> None:
    """Verifica que existan los archivos requeridos para el paso dado.

    Raises:
        FileNotFoundError: con mensaje indicando qué archivo falta y qué paso lo genera.
    """
    for path_str in STEP_INPUTS.get(step_num, []):
        full_path = base_dir / path_str
        if not full_path.exists():
            raise FileNotFoundError(
                f"Prerrequisito faltante para paso {step_num}: {path_str}. "
                f"Ejecute el paso anterior primero."
            )
```

### `run_step(step_num, name, fn)` — Wrapper de ejecución con timing

```python
def run_step(step_num: int, name: str, fn: Callable[[], None]) -> None:
    """Ejecuta un paso del pipeline con logging de banner y timing."""
    logger.info(f"{'='*60}")
    logger.info(f"Step {step_num}: {name}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    fn()
    elapsed = time.time() - t0
    minutes, seconds = divmod(int(elapsed), 60)
    logger.info(f"Step {step_num} completed in {minutes:02d}:{seconds:02d}")
    gc.collect()
```

---

## Correcciones críticas en el orquestador

| Aspecto                        | Detalle                                                              |
|--------------------------------|----------------------------------------------------------------------|
| Propagación de `proxy_type`    | Se pasa del paso 6 al paso 8 a través de `results.json`             |
| Sensibilidad Feature 17        | Se ejecuta dentro del paso 7 junto con proxy, baselines y per-status |
| Post-hoc centro/actor/moneda   | Se ejecuta dentro del paso 7 y serializa `results_posthoc.json`     |
| Scores guardados como parquet  | Incluyen columnas `id` y `created_at` para trazabilidad             |
| Grid search CSV                | Se verifica existencia antes de intentar generar heatmap en paso 8   |
| Gate de gobernanza             | Paso 8 revisa `actor_identity_validated` antes de exportar tablas o figuras post-hoc identificables |

---

## Notas de implementación

- `PYTHONHASHSEED=42` debe establecerse antes de la ejecución para reproducibilidad
- El flag `--dry-run` ejecuta toda la validación de prerrequisitos sin procesar datos
- El flag `--fast` es útil para desarrollo: reduce bootstrap a 100 iteraciones y omite grid search
- Cada paso es idempotente: si los artefactos de salida ya existen, se pueden regenerar sin efecto lateral
- El paso 7 debe decidir explicitamente `actor_identity_validated` y `actor_identifier_field` antes de escribir `results_posthoc.json`
- El paso 8 debe emitir version interna o publica de tablas y figuras segun la politica definida en `A4_GOBERNANZA_PRIVACIDAD_CONTRATOS.md`
