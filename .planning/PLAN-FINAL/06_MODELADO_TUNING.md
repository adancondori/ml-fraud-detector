# Fase 5: Modelado y Tuning

> **Gate C requerido antes de continuar a Fase 6.**
> El test set NO se toca en esta fase. La seleccion de hiperparametros se realiza exclusivamente sobre el validation set.

---

## Clase AnomalyModelTrainer

### Interfaz

```python
class AnomalyModelTrainer:
    """Entrena y evalua los tres modelos de deteccion de anomalias.

    Convencion de scores: higher = mas anomalo.
    Todos los modelos usan -decision_function(X).
    """

    def __init__(self, random_state: int = 42):
        self._random_state = random_state
        self._models: dict[str, object] = {}  # nombre -> modelo entrenado
        self._training_times: dict[str, float] = {}  # nombre -> segundos

    # --- Entrenamiento individual ---

    def train_isolation_forest(self, X_train: np.ndarray, params: dict) -> IsolationForest:
        """Entrena un Isolation Forest con los parametros dados."""
        ...

    def train_lof(self, X_train: np.ndarray, params: dict) -> LocalOutlierFactor:
        """Entrena LOF con novelty=True."""
        ...

    def train_ocsvm(self, X_train: np.ndarray, params: dict) -> OneClassSVM:
        """Entrena One-Class SVM con kernel RBF."""
        ...

    # --- Scoring ---

    def score(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Genera scores de anomalia. Higher = mas anomalo.

        Para TODOS los modelos: -decision_function(X)
        """
        model = self._models[model_name]
        return -model.decision_function(X)

    # --- Grid search ---

    def grid_search_if(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val_proxy: np.ndarray,
        param_grid: dict,
        checkpoint_path: str = None,
    ) -> dict:
        """Grid search para Isolation Forest. Optimiza AUC-ROC en val."""
        ...

    def grid_search_lof(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val_proxy: np.ndarray,
        param_grid: dict,
        checkpoint_path: str = None,
    ) -> dict:
        """Grid search para LOF."""
        ...

    def grid_search_ocsvm(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val_proxy: np.ndarray,
        param_grid: dict,
        checkpoint_path: str = None,
    ) -> dict:
        """Grid search para One-Class SVM."""
        ...

    # --- Persistencia ---

    def save_model(self, model_name: str, path: str) -> None:
        """Guardar modelo individual con joblib."""
        ...

    def load_model(self, model_name: str, path: str) -> None:
        """Cargar modelo individual."""
        ...
```

---

## Nota de diseno: Violacion SRP y OCP

### Problema SRP

`AnomalyModelTrainer` acumula cinco responsabilidades distintas: entrenamiento, scoring, grid search, multi-seed y persistencia. Esto dificulta el testing y hace que un cambio en grid search pueda romper el scoring.

**Descomposicion recomendada:**

| Clase | Responsabilidad |
|-------|----------------|
| `ModelFactory` | Crear instancias de modelos con parametros dados |
| `GridSearchRunner` | Ejecutar grid search con checkpoint/resume |
| `ModelScorer` | Convencion unificada de scoring (`-decision_function`) |
| `AnomalyModelTrainer` | **Facade** que orquesta las clases anteriores |

### Problema OCP

Agregar un nuevo modelo (e.g., `DBSCAN`, `AutoEncoder`) requiere modificar la clase para anadir un nuevo `train_xxx` y `grid_search_xxx`. Esto viola Open/Closed.

**Solucion: Model Registry con Strategy Pattern:**

```python
from abc import ABC, abstractmethod

class AnomalyModelStrategy(ABC):
    @abstractmethod
    def create(self, params: dict, random_state: int): ...
    @abstractmethod
    def default_grid(self) -> dict: ...

class IsolationForestStrategy(AnomalyModelStrategy):
    def create(self, params, random_state):
        return IsolationForest(contamination="auto", random_state=random_state, n_jobs=-1, **params)
    def default_grid(self):
        return {"n_estimators": [100, 200, 300, 500], "max_samples": [256, 512, 1024, 2048], ...}

MODEL_REGISTRY = {
    "isolation_forest": IsolationForestStrategy,
    "lof": LOFStrategy,
    "ocsvm": OCSVMStrategy,
}
```

Con esta estructura, agregar un modelo nuevo solo requiere crear una clase y registrarla, sin tocar `AnomalyModelTrainer`.

---

## Convencion de scores

Todos los modelos de scikit-learn usan `decision_function(X)` con la convencion de que valores **mas positivos** indican inliers. Para nuestra convencion (higher = mas anomalo), negamos:

| Modelo | `decision_function(X)` nativo | Score final |
|--------|-------------------------------|-------------|
| Isolation Forest | Positivo para inliers, negativo para outliers | `-decision_function(X)` |
| LOF (novelty=True) | Negativo para outliers, positivo para inliers | `-decision_function(X)` |
| One-Class SVM | Negativo para outliers, positivo para inliers | `-decision_function(X)` |

**Resultado:** En los tres casos, `score > 0` tiende a indicar anomalia, y scores mas altos = mas anomalo.

```python
def score(self, model_name: str, X: np.ndarray) -> np.ndarray:
    """Score de anomalia unificado. Higher = mas anomalo."""
    model = self._models[model_name]
    raw = model.decision_function(X)
    return -raw
```

---

## Isolation Forest

### Configuracion base

```python
IsolationForest(
    contamination="auto",  # SIEMPRE — no afecta ranking, solo el threshold
    random_state=self._random_state,
    n_jobs=-1,
)
```

### Grid de hiperparametros

| Parametro | Valores | Cantidad |
|-----------|---------|----------|
| `n_estimators` | [100, 200, 300, 500] | 4 |
| `max_samples` | [256, 512, 1024, 2048] | 4 |
| `max_features` | [0.5, 0.75, 1.0, "auto"] | 4 |

**Total: 4 x 4 x 4 = 64 combinaciones.**

> Nota: `max_features="auto"` en scikit-learn equivale a `1.0` para IsolationForest, pero se incluye explicitamente para documentacion. Si se confirma equivalencia exacta, reducir a 48 combos eliminando duplicados.

`contamination` se fija en `"auto"` y se **excluye del grid** porque:
- No afecta `decision_function(X)`, solo el threshold de `predict()`.
- No usamos `predict()`, solo scores continuos para AUC-ROC.
- Eliminar contamination del grid redujo las combinaciones de 256 a 64 (correccion critica).

### Metrica de optimizacion

**AUC-ROC** sobre el validation set usando el proxy estricto (`status IN ('totally_refunded', 'refunded_to_credit')`).

```python
from sklearn.metrics import roc_auc_score

scores_val = -model.decision_function(X_val)
auc = roc_auc_score(y_val_proxy, scores_val)
```

### Checkpoint y resume

```python
def grid_search_if(self, X_train, X_val, y_val_proxy, param_grid, checkpoint_path=None):
    """Grid search con checkpoint cada 10 combinaciones.

    Si checkpoint_path existe y contiene resultados previos, reanuda
    desde donde se quedo (util ante crashes o timeouts).
    """
    all_combos = list(ParameterGrid(param_grid))
    results = []

    # Cargar checkpoint si existe
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        results = existing.to_dict("records")
        completed = len(results)
        logger.info(f"Reanudando grid search desde combo {completed}/{len(all_combos)}")
    else:
        completed = 0

    for i, params in enumerate(all_combos[completed:], start=completed):
        t0 = time.time()
        model = IsolationForest(
            contamination="auto",
            random_state=self._random_state,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train)
        scores_val = -model.decision_function(X_val)
        auc = roc_auc_score(y_val_proxy, scores_val)
        elapsed = time.time() - t0

        results.append({**params, "auc_roc": auc, "time_seconds": elapsed})
        logger.info(f"IF combo {i+1}/{len(all_combos)}: AUC={auc:.4f} ({elapsed:.1f}s)")

        # Checkpoint cada 10 combos
        if checkpoint_path and (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            logger.info(f"Checkpoint guardado: {checkpoint_path}")

    # Guardar resultados finales
    results_df = pd.DataFrame(results)
    if checkpoint_path:
        results_df.to_csv(checkpoint_path, index=False)

    # Seleccionar mejor
    best_idx = results_df["auc_roc"].idxmax()
    best_params = results_df.iloc[best_idx].to_dict()

    return best_params
```

### Tiempo estimado

- ~10s por combinacion (3.1M filas).
- 64 combinaciones x 10s = ~10-15 minutos total.
- Con `n_jobs=-1` y paralelismo de arboles.

---

## LOF (Local Outlier Factor)

### Configuracion base

```python
LocalOutlierFactor(
    novelty=True,          # OBLIGATORIO para usar decision_function en datos nuevos
    contamination="auto",
    n_jobs=-1,
)
```

**Nota critica:** `novelty=True` es necesario porque:
- Sin `novelty=True`, LOF no expone `decision_function()` para datos nuevos (solo para training).
- Con `novelty=True`, se puede hacer `model.decision_function(X_val)` y `model.decision_function(X_test)`.

### Grid de hiperparametros

| Parametro | Valores | Cantidad |
|-----------|---------|----------|
| `n_neighbors` | [20, 50, 100] | 3 |

**Total: 3 combinaciones.**

LOF tiene un solo hiperparametro significativo. `metric` se deja en `"minkowski"` (default con `p=2`, i.e., euclidiana).

### Tiempo estimado

- LOF es O(n * k * log(n)) con ball tree.
- ~5-10 minutos por combinacion con 3.1M filas.
- 3 combinaciones = ~15-30 minutos total.

---

## One-Class SVM

### Configuracion base

```python
OneClassSVM(
    kernel="rbf",
)
```

### Grid de hiperparametros

| Parametro | Valores | Cantidad |
|-----------|---------|----------|
| `nu` | [0.02, 0.05, 0.10] | 3 |
| `gamma` | ["scale", "auto"] | 2 |

**Total: 3 x 2 = 6 combinaciones.**

### Subsampleo obligatorio

OC-SVM tiene complejidad O(n^2) a O(n^3) tanto en memoria como en tiempo. Con 3.1M filas es computacionalmente inviable. Se usa un subsample del training set:

```python
def _subsample_train(self, X_train, n_subsample=100_000):
    """Subsampleo temporal estratificado para OC-SVM.

    Se seleccionan 100K filas distribuidas uniformemente en el tiempo
    (no random puro) para preservar la distribucion temporal.
    """
    n = X_train.shape[0]
    if n <= n_subsample:
        return X_train
    # Indices equiespaciados (estratificacion temporal)
    indices = np.linspace(0, n - 1, n_subsample, dtype=int)
    return X_train[indices]
```

**Justificacion del tamano:**
- 100K filas: fit en ~20-30s por combinacion.
- 6 combinaciones = ~3 minutos total.
- Suficiente para capturar la estructura de los datos.
- El scoring (`decision_function`) se ejecuta sobre el dataset completo sin problema.

### Tiempo estimado

- Fit: ~20-30s por combinacion (100K subsample).
- Score val: ~2-3 minutos (1.1M filas, O(n * n_sv)).
- 6 combinaciones = ~3 minutos total (sin contar scoring).

---

## Multi-seed

### Estrategia

| Fase | Seeds | Proposito |
|------|-------|-----------|
| Desarrollo | 42 | Iteracion rapida, debugging |
| Ejecucion final | 42, 52, 62 | Verificar estabilidad del resultado |

### Implementacion

```python
def run_multi_seed(self, X_train, X_val, y_val_proxy, best_params, seeds=[42, 52, 62]):
    """Entrena el mejor IF con multiples seeds y reporta variabilidad."""
    results = []
    for seed in seeds:
        model = IsolationForest(
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
            **best_params,
        )
        model.fit(X_train)
        scores = -model.decision_function(X_val)
        auc = roc_auc_score(y_val_proxy, scores)
        results.append({"seed": seed, "auc_roc": auc})

    df_results = pd.DataFrame(results)
    mean_auc = df_results["auc_roc"].mean()
    range_auc = df_results["auc_roc"].max() - df_results["auc_roc"].min()

    logger.info(f"Multi-seed AUC: mean={mean_auc:.4f}, range={range_auc:.4f}")

    # Si la dispersion es trivial (< 0.005), justificar uso de single seed
    if range_auc < 0.005:
        logger.info("Dispersion trivial — se puede reportar con single seed (42)")
    else:
        logger.warning(
            f"Dispersion no trivial (range={range_auc:.4f}). "
            "Reportar mean y rango en la tesis."
        )

    return df_results
```

### Regla de decision

- Si `range(AUC) < 0.005`: reportar resultado con seed=42 y mencionar en tesis que las variaciones entre seeds son triviales.
- Si `range(AUC) >= 0.005`: reportar media y rango. Discutir en la seccion de limitaciones.

---

## Logging de tiempos de entrenamiento

Cada modelo registra su tiempo de entrenamiento:

```python
def train_isolation_forest(self, X_train, params):
    t0 = time.time()
    model = IsolationForest(
        contamination="auto",
        random_state=self._random_state,
        n_jobs=-1,
        **params,
    )
    model.fit(X_train)
    elapsed = time.time() - t0

    self._models["isolation_forest"] = model
    self._training_times["isolation_forest"] = elapsed
    logger.info(f"IF entrenado en {elapsed:.1f}s")

    return model
```

Los tiempos se exportan para la Tabla 3.X de la tesis:

```python
def get_training_times(self) -> dict:
    """Retorna dict {model_name: seconds} para reporting."""
    return self._training_times.copy()
```

---

## Flujo completo de la fase

```
X_train.npy + X_val.npy + y_val_proxy
    |
    v
[Grid search IF — 64 combos]
    |
    +--> grid_search_if.csv (64 filas con AUC y tiempos)
    +--> best_params_if.json
    |
    v
[Grid search LOF — 3 combos]
    |
    +--> grid_search_lof.csv
    +--> best_params_lof.json
    |
    v
[Grid search OC-SVM — 6 combos, subsample 100K]
    |
    +--> grid_search_ocsvm.csv
    +--> best_params_ocsvm.json
    |
    v
[Re-entrenar 3 modelos con best params en train completo]
    |
    +--> isolation_forest.joblib
    +--> lof.joblib
    +--> ocsvm.joblib
    |
    v
[Multi-seed (IF) — seeds 42, 52, 62]
    |
    +--> multi_seed_results.csv
    |
    v
[Gate C: verificar que test set no fue tocado]
```

---

## Gate C: Independencia del test set

Verificaciones obligatorias antes de continuar a Fase 6:

### 1. Mejor modelo seleccionado solo en validation

```python
def test_gate_c_no_test_leakage():
    """Verificar que la seleccion de hiperparametros se hizo en val, no en test."""
    # Los grid search CSV deben contener solo metricas sobre val
    gs_if = pd.read_csv("output/grid_search_if.csv")
    assert "auc_roc" in gs_if.columns  # Metrica calculada en val
    assert "test_auc" not in gs_if.columns  # NO debe haber metricas de test
```

### 2. Los tres modelos estan guardados

```python
def test_gate_c_models_saved():
    """Los tres modelos deben existir como archivos joblib."""
    assert os.path.exists("output/models/isolation_forest.joblib")
    assert os.path.exists("output/models/lof.joblib")
    assert os.path.exists("output/models/ocsvm.joblib")
```

### 3. El test set no fue cargado en esta fase

```python
def test_gate_c_test_untouched():
    """Verificar que X_test.npy no fue modificado (timestamp anterior a esta fase)."""
    # Implementacion: comparar hash MD5 de X_test.npy antes y despues de la fase
    assert hash_before == hash_after, "X_test.npy fue modificado durante la fase de modelado"
```

---

## Entregables

| Artefacto | Ruta | Descripcion |
|-----------|------|-------------|
| `trainer.py` | `src/fraud_detector/models/trainer.py` | Clase `AnomalyModelTrainer` |
| `isolation_forest.joblib` | `output/models/isolation_forest.joblib` | Mejor IF entrenado en train completo |
| `lof.joblib` | `output/models/lof.joblib` | Mejor LOF entrenado en train completo |
| `ocsvm.joblib` | `output/models/ocsvm.joblib` | Mejor OC-SVM entrenado en subsample |
| `grid_search_if.csv` | `output/grid_search_if.csv` | 64 combos con AUC-ROC y tiempos |
| `grid_search_lof.csv` | `output/grid_search_lof.csv` | 3 combos con AUC-ROC y tiempos |
| `grid_search_ocsvm.csv` | `output/grid_search_ocsvm.csv` | 6 combos con AUC-ROC y tiempos |
| `best_params_if.json` | `output/models/best_params_if.json` | Mejores hiperparametros IF |
| `best_params_lof.json` | `output/models/best_params_lof.json` | Mejores hiperparametros LOF |
| `best_params_ocsvm.json` | `output/models/best_params_ocsvm.json` | Mejores hiperparametros OC-SVM |
| `multi_seed_results.csv` | `output/multi_seed_results.csv` | AUC por seed (42, 52, 62) |

### Tamanos estimados de modelos

| Modelo | Tamano | Nota |
|--------|--------|------|
| Isolation Forest | ~200 MB | 500 arboles x 3.1M muestras |
| LOF | ~1 GB | Almacena distancias k-NN para 3.1M puntos |
| One-Class SVM | ~100 MB | Soporte vectorial sobre 100K subsample |
| **Total modelos** | **~1.3 GB** | |

---

## Bug: `np.random.RandomState` es legacy

En el bootstrap de `run_multi_seed` y en cualquier lugar donde se necesite generacion de numeros aleatorios fuera de scikit-learn, usar la API moderna:

```python
# Legacy (NO usar en codigo nuevo):
rng = np.random.RandomState(seed)

# Moderno (preferido):
rng = np.random.default_rng(seed)
```

**Nota:** Los modelos de scikit-learn aceptan `random_state=int` directamente, asi que el cambio aplica solo al codigo propio (e.g., subsampleo, bootstrap). `np.random.RandomState` sigue funcionando pero esta marcado como legacy desde NumPy 1.19.

---

## Contratos TDD

Tests a escribir **ANTES** de implementar `AnomalyModelTrainer`. Definen el comportamiento contractual:

| # | Test | Verifica |
|---|------|----------|
| 1 | `test_score_convention_higher_is_more_anomalous` | Que `-decision_function(X)` produce scores donde anomalias tienen valores mas altos |
| 2 | `test_if_trains_without_error` | Que IF se entrena sin errores con parametros default |
| 3 | `test_lof_requires_novelty_true` | Que LOF se crea con `novelty=True` (necesario para `decision_function` en datos nuevos) |
| 4 | `test_ocsvm_subsample_size_respected` | Que OC-SVM recibe un subsample de tamano <= `n_subsample` |
| 5 | `test_grid_search_checkpoint_resumes_correctly` | Que el grid search reanuda desde el checkpoint sin repetir combinaciones |
| 6 | `test_save_load_produces_same_scores` | Que un modelo guardado y cargado produce scores identicos |
| 7 | `test_multi_seed_reports_variability` | Que `run_multi_seed` retorna un DataFrame con columnas `seed` y `auc_roc` |

```python
# Ejemplo de test #1 (contrato critico de la convencion de scores)
def test_score_convention_higher_is_more_anomalous():
    """Anomalias sinteticas deben tener scores mas altos que inliers."""
    X_inliers = np.random.default_rng(42).normal(0, 1, (1000, 5))
    X_outliers = np.random.default_rng(42).normal(10, 1, (50, 5))
    X_train = X_inliers
    X_test = np.vstack([X_inliers[:100], X_outliers])

    trainer = AnomalyModelTrainer(random_state=42)
    trainer.train_isolation_forest(X_train, {"n_estimators": 100})
    scores = trainer.score("isolation_forest", X_test)

    mean_inlier = scores[:100].mean()
    mean_outlier = scores[100:].mean()
    assert mean_outlier > mean_inlier
```

---

## Resumen de tiempos estimados

| Paso | Tiempo |
|------|--------|
| Grid search IF (64 combos) | 10-15 min |
| Grid search LOF (3 combos) | 15-30 min |
| Grid search OC-SVM (6 combos) | ~3 min |
| Re-entrenamiento final (3 modelos) | ~5 min |
| Multi-seed IF (3 seeds) | ~1 min |
| **Total Fase 5** | **~35-55 min** |
