# Fase 5: Modelado y Tuning

> **Gate C requerido antes de continuar a Fase 6.**
> El test set NO se toca en esta fase. La seleccion de hiperparametros se realiza exclusivamente sobre el validation set.

---

## Clase `ModelTrainer`

### Interfaz

```python
class ModelTrainer:
    """Entrena y scorea modelos de deteccion de anomalias.

    Convencion de scores: higher = mas anomalo.
    En el repo actual se expone `score_samples(X)` y se normaliza
    la orientacion para que score alto = mas anomalo.
    """

    def __init__(self, model_type: str = "isolation_forest", model_params: dict | None = None):
        ...

    def fit(self, X_train: np.ndarray) -> "ModelTrainer":
        ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Genera scores de anomalia. Higher = mas anomalo."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save_model(self, output_path: str) -> None:
        ...

    def load_model(self, model_path: str) -> None:
        ...
```

---

## Nota de diseno: Violacion SRP y OCP

### Problema SRP

`ModelTrainer` no deberia acumular demasiadas responsabilidades. El repo actual ya resuelve entrenamiento, scoring y persistencia; si se agrega grid search/resume, conviene separarlo en utilidades o runners dedicados.

**Descomposicion recomendada:**

| Clase | Responsabilidad |
|-------|----------------|
| `ModelFactory` | Crear instancias de modelos con parametros dados |
| `GridSearchRunner` | Ejecutar grid search con checkpoint/resume |
| `ModelScorer` | Convencion unificada de scoring (`-decision_function`) |
| `ModelTrainer` | **Facade** liviana para entrenamiento, scoring y persistencia |

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

Con esta estructura, agregar un modelo nuevo solo requiere crear una clase y registrarla, sin tocar `ModelTrainer`.

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
    contamination=...,  # Se explora en el grid: [0.01, 0.03, 0.05, 0.06, 0.08]
    random_state=self._random_state,
    n_jobs=-1,
)
```

### Grid de hiperparametros

| Parametro | Valores | Cantidad |
|-----------|---------|----------|
| `n_estimators` | [100, 200, 300, 500] | 4 |
| `max_samples` | [256, 512, 1024, 2048] | 4 |
| `max_features` | [0.5, 0.75, 1.0] | 3 |
| `contamination` | [0.01, 0.03, 0.05, 0.06, 0.08] | 5 |

**Total: 4 x 4 x 5 x 3 = 240 combinaciones.**

> **Nota sobre contamination y ranking:**
> - `score_samples(X)` es INVARIANTE a `contamination` — produce los mismos scores raw
> - `decision_function(X)` = `score_samples(X) - offset_`, donde `offset_` depende de `contamination`
> - El RANKING se preserva (offset es constante), por lo que AUC-ROC es identico para cualquier `contamination`
> - En la practica: las 240 combos producen 48 AUC-ROC unicos (5 contamination × misma arquitectura = mismo ranking)
> - Se mantiene `contamination` en el grid para documentar su nulo impacto en el ranking, lo cual refuerza la tesis
> - **Scoring canonico:** usar `-model.score_samples(X)` (mayor = mas anomalo), NO `decision_function()`
>
> `max_features="auto"` equivale a `1.0` en IsolationForest, eliminado del grid para evitar duplicados.

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
    """Grid search con checkpoint cada 20 combinaciones.

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
            random_state=self._random_state,
            n_jobs=-1,
            **params,  # incluye contamination del grid
        )
        model.fit(X_train)
        scores_val = -model.decision_function(X_val)
        auc = roc_auc_score(y_val_proxy, scores_val)
        elapsed = time.time() - t0

        results.append({**params, "auc_roc": auc, "time_seconds": elapsed})
        logger.info(f"IF combo {i+1}/{len(all_combos)}: AUC={auc:.4f} ({elapsed:.1f}s)")

        # Checkpoint cada 20 combos
        if checkpoint_path and (i + 1) % 20 == 0:
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
- 240 combinaciones x 10s = ~30-45 minutos total.
- Con `n_jobs=-1` y paralelismo de arboles.

---

## Variantes de modelo (Isolation Forest)

Se entrenan tres variantes del Isolation Forest para analisis de sensibilidad y ablacion:

| Variante | Features | Descripcion |
|----------|----------|-------------|
| **IF-31** | 31 features (conjunto completo) | Modelo primario. Grid search completo (240 combos). Multi-seed con 42, 52, 62. |
| **IF-30** | 30 features (sin F18 `user_reversal_ratio_30d`) | Sensibilidad. Evalua si remover la feature con correlacion mecanica al proxy cambia el AUC significativamente (delta < 0.02). |
| **IF-21** | 21 features (ablacion: solo grupos A-E, sin F06/F21) | Ablacion. Evalua el aporte incremental de los grupos F, G y H. |

Las tres variantes se entrenan con los **mejores hiperparametros** obtenidos en el grid search de IF-31. Solo IF-31 pasa por grid search completo; IF-30 e IF-21 reutilizan esos hiperparametros.

### Flujo de variantes

```
[Grid search IF-31 — 240 combos] --> best_params_if.json
    |
    +--> [Entrenar IF-31 con best params] --> isolation_forest.joblib
    +--> [Entrenar IF-30 con best params, sin F18] --> isolation_forest_30.joblib
    +--> [Entrenar IF-21 con best params, 21 features base] --> isolation_forest_21.joblib
```

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
| `nu` | [0.01, 0.05, 0.10] | 3 |
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

Multi-seed se aplica exclusivamente al modelo **IF-31** (primario):

| Fase | Seeds | Proposito |
|------|-------|-----------|
| Desarrollo | 42 | Iteracion rapida, debugging |
| Ejecucion final (IF-31) | 42, 52, 62 | Verificar estabilidad del resultado del modelo primario |

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
[Grid search IF-31 — 240 combos]
    |
    +--> grid_search_if.csv (240 filas con AUC y tiempos)
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
[Re-entrenar modelos con best params en train completo]
    |
    +--> isolation_forest.joblib      (IF-31, modelo primario)
    +--> isolation_forest_30.joblib   (IF-30, sin F18)
    +--> isolation_forest_21.joblib   (IF-21, ablacion 21 features)
    +--> lof.joblib
    +--> ocsvm.joblib
    |
    v
[Multi-seed IF-31 — seeds 42, 52, 62]
    |
    +--> multi_seed_results.csv
    |
    v
[Gate C: verificar que test set no fue tocado y todas las variantes entrenadas]
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

### 2. Todos los modelos y variantes estan guardados

```python
def test_gate_c_models_saved():
    """Todos los modelos y variantes deben existir como archivos joblib."""
    # Modelos principales
    assert os.path.exists("output/models/isolation_forest.joblib")   # IF-31
    assert os.path.exists("output/models/lof.joblib")
    assert os.path.exists("output/models/ocsvm.joblib")
    # Variantes IF
    assert os.path.exists("output/models/isolation_forest_30.joblib")  # IF-30 (sin F18)
    assert os.path.exists("output/models/isolation_forest_21.joblib")  # IF-21 (ablacion)
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
| `trainer.py` | `src/fraud_detector/models/trainer.py` | Clase `ModelTrainer` |
| `isolation_forest.joblib` | `output/models/isolation_forest.joblib` | IF-31: modelo primario (31 features) |
| `isolation_forest_30.joblib` | `output/models/isolation_forest_30.joblib` | IF-30: sin F18 `user_reversal_ratio_30d` (sensibilidad) |
| `isolation_forest_21.joblib` | `output/models/isolation_forest_21.joblib` | IF-21: 21 features base (ablacion) |
| `lof.joblib` | `output/models/lof.joblib` | Mejor LOF entrenado en train completo |
| `ocsvm.joblib` | `output/models/ocsvm.joblib` | Mejor OC-SVM entrenado en subsample |
| `grid_search_if.csv` | `output/grid_search_if.csv` | 240 combos con AUC-ROC y tiempos |
| `grid_search_lof.csv` | `output/grid_search_lof.csv` | 3 combos con AUC-ROC y tiempos |
| `grid_search_ocsvm.csv` | `output/grid_search_ocsvm.csv` | 6 combos con AUC-ROC y tiempos |
| `best_params_if.json` | `output/models/best_params_if.json` | Mejores hiperparametros IF |
| `best_params_lof.json` | `output/models/best_params_lof.json` | Mejores hiperparametros LOF |
| `best_params_ocsvm.json` | `output/models/best_params_ocsvm.json` | Mejores hiperparametros OC-SVM |
| `multi_seed_results.csv` | `output/multi_seed_results.csv` | AUC por seed (42, 52, 62) |

### Tamanos estimados de modelos

| Modelo | Tamano | Nota |
|--------|--------|------|
| Isolation Forest (IF-31) | ~200 MB | 500 arboles x 3.1M muestras, 31 features |
| Isolation Forest (IF-30) | ~200 MB | 500 arboles x 3.1M muestras, 30 features |
| Isolation Forest (IF-21) | ~200 MB | 500 arboles x 3.1M muestras, 21 features |
| LOF | ~1 GB | Almacena distancias k-NN para 3.1M puntos |
| One-Class SVM | ~100 MB | Soporte vectorial sobre 100K subsample |
| **Total modelos** | **~1.7 GB** | |

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

Tests a escribir **ANTES** de ampliar `ModelTrainer` con grid search/resume. Definen el comportamiento contractual:

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

    trainer = ModelTrainer(model_type="isolation_forest", model_params={"n_estimators": 100})
    trainer.fit(X_train)
    scores = trainer.score_samples(X_test)

    mean_inlier = scores[:100].mean()
    mean_outlier = scores[100:].mean()
    assert mean_outlier > mean_inlier
```

---

## Resumen de tiempos estimados

| Paso | Tiempo |
|------|--------|
| Grid search IF-31 (240 combos) | 30-45 min |
| Grid search LOF (3 combos) | 15-30 min |
| Grid search OC-SVM (6 combos) | ~3 min |
| Re-entrenamiento final (5 modelos: IF-31, IF-30, IF-21, LOF, OC-SVM) | ~8 min |
| Multi-seed IF-31 (3 seeds) | ~1 min |
| **Total Fase 5** | **~60-90 min** |
