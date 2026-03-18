# Fase 4: Preprocesamiento

> Sin gate obligatorio. Paso simple entre Feature Engineering y Modelado.

---

## Objetivo

Escalar las 20 features numericas para que todos los modelos operen sobre la misma base estandarizada. Esto es **critico** para OC-SVM (kernel RBF es sensible a la escala) y buena practica para comparabilidad entre modelos.

---

## Justificacion de StandardScaler

| Modelo | Sensible a escala? | Justificacion |
|--------|-------------------|---------------|
| Isolation Forest | No | Basado en particiones aleatorias del espacio; invariante a escala |
| LOF | Si | Distancias k-NN se distorsionan con features de magnitudes dispares |
| One-Class SVM (RBF) | **Si** | El kernel RBF depende de `gamma * ||x - x'||^2`; features no escaladas dominan la norma |

Aunque IF es invariante, se aplica el mismo escalado a los tres modelos para garantizar que las comparaciones (HE4) sean sobre la misma representacion de los datos.

---

## No se necesita encoding ni imputacion

- **No one-hot encoding:** Todas las 20 features del catalogo son numericas (continuas o binarias int8). No hay categoricas textuales.
- **No imputacion:** La fase de Feature Engineering garantiza que no haya NaN (fillna en todas las features, test #5 de Gate B lo verifica). Si transform produce NaN, es un bug en engineering.py, no un problema de preprocesamiento.

---

## Clase UnsupervisedPreprocessor

```python
class UnsupervisedPreprocessor:
    """Estandarizacion fit-on-train para pipeline no supervisado.

    Aplica StandardScaler (media=0, std=1) ajustado exclusivamente en
    el training set. Produce arrays float32 para reducir memoria ~50%.
    """

    def __init__(self, scaler=None):
        """
        Args:
            scaler: Instancia de scaler (default: StandardScaler). Permite inyectar
                    alternativas como RobustScaler sin modificar la clase (OCP).
        """
        self._scaler = scaler or StandardScaler()
        self._feature_names: list[str] = None
        self._fitted: bool = False

    def fit(self, X_train: pd.DataFrame, feature_names: list[str]) -> "UnsupervisedPreprocessor":
        """Ajusta el StandardScaler sobre el training set.

        Args:
            X_train: DataFrame con al menos las columnas en feature_names.
            feature_names: Lista de 20 (o 19) nombres de features a escalar.
        """
        self._feature_names = feature_names
        self._scaler.fit(X_train[feature_names])
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforma el DataFrame a numpy array float32 escalado.

        Args:
            df: DataFrame con las columnas de feature_names.

        Returns:
            np.ndarray de shape (n_samples, n_features), dtype float32.

        Raises:
            ValueError: Si hay NaN en el input (fail fast).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        X = df[self._feature_names].values

        # Politica NaN: fail fast
        if np.isnan(X).any():
            nan_cols = [
                self._feature_names[i]
                for i in range(X.shape[1])
                if np.isnan(X[:, i]).any()
            ]
            raise ValueError(
                f"NaN detectados en columnas: {nan_cols}. "
                "Revisar FeatureEngineer.transform() — no deberian existir NaN."
            )

        X_scaled = self._scaler.transform(X).astype(np.float32)
        return X_scaled

    def fit_transform(self, X_train: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
        """Fit + transform en un solo paso (solo para train)."""
        return self.fit(X_train, feature_names).transform(X_train)

    def save(self, path: str) -> None:
        """Serializar con joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "UnsupervisedPreprocessor":
        """Cargar instancia previamente guardada."""
        return joblib.load(path)
```

---

## Flujo de ejecucion

```
train_features.parquet ──> fit_transform() ──> X_train.npy (float32)
                               |
                               +--> scaler.joblib
                               |
val_features.parquet   ──> transform()     ──> X_val.npy   (float32)
test_features.parquet  ──> transform()     ──> X_test.npy  (float32)
```

### Detalle del pipeline

```python
from fraud_detector.features.engineering import FEATURE_NAMES

preprocessor = UnsupervisedPreprocessor()

# Train
X_train = preprocessor.fit_transform(df_train_features, FEATURE_NAMES)
np.save("output/scores/X_train.npy", X_train)

# Val
X_val = preprocessor.transform(df_val_features)
np.save("output/scores/X_val.npy", X_val)

# Test
X_test = preprocessor.transform(df_test_features)
np.save("output/scores/X_test.npy", X_test)

# Guardar scaler
preprocessor.save("output/models/scaler.joblib")
```

---

## Reduccion de memoria con float32

| Dtype | Bytes/valor | Para 3.1M x 20 |
|-------|-------------|-----------------|
| float64 | 8 | ~496 MB |
| float32 | 4 | ~248 MB |
| **Ahorro** | | **~248 MB (~50%)** |

`float32` ofrece ~7 digitos de precision, mas que suficiente para features escaladas. scikit-learn acepta float32 sin problemas en IF, LOF, y OC-SVM.

---

## Politica de NaN: fail fast

A diferencia de un pipeline de produccion que podria imputar valores faltantes, este pipeline academico adopta una politica estricta:

- **Si hay NaN despues de Feature Engineering, es un bug.**
- `transform()` lanza `ValueError` indicando las columnas afectadas.
- Esto fuerza a corregir el problema en `engineering.py`, no a enmascararlo con imputacion.

La justificacion: todas las features del catalogo tienen reglas explicitas de `fillna(0)` o valores por defecto. Un NaN indica que una regla no se aplico correctamente.

---

## Nota de diseno: Open/Closed Principle (OCP)

El constructor acepta un `scaler` inyectable para que, si en el futuro se necesita un scaler diferente (e.g., `RobustScaler` para features con muchos outliers), no sea necesario modificar `UnsupervisedPreprocessor`. Basta con instanciarlo asi:

```python
from sklearn.preprocessing import RobustScaler

preprocessor = UnsupervisedPreprocessor(scaler=RobustScaler())
```

Cualquier objeto que implemente `fit(X)` y `transform(X)` es compatible.

---

## Contratos TDD

Tests a escribir **ANTES** de implementar `UnsupervisedPreprocessor`. Estos definen el comportamiento esperado y guian la implementacion:

| # | Test | Verifica |
|---|------|----------|
| 1 | `test_fit_transform_produces_correct_shape` | Que el output tiene shape `(n_samples, n_features)` |
| 2 | `test_output_is_float32` | Que el dtype del array resultante es `np.float32` |
| 3 | `test_nan_input_raises_valueerror` | Que `transform()` lanza `ValueError` si hay NaN en el input |
| 4 | `test_missing_columns_raises_valueerror` | Que `transform()` falla si el DataFrame no tiene las columnas esperadas |
| 5 | `test_save_load_produces_identical_output` | Que serializar y deserializar produce el mismo resultado numerico |
| 6 | `test_transform_without_fit_raises_runtime_error` | Que `transform()` sin `fit()` previo lanza `RuntimeError` |

```python
# Ejemplo de test #6 (el mas critico para evitar uso de assert en produccion)
def test_transform_without_fit_raises_runtime_error():
    prep = UnsupervisedPreprocessor()
    with pytest.raises(RuntimeError, match="Must call fit"):
        prep.transform(some_dataframe)
```

---

## Entregables

| Artefacto | Ruta | Descripcion |
|-----------|------|-------------|
| `preprocessor.py` | `src/fraud_detector/features/preprocessor.py` | Clase `UnsupervisedPreprocessor` |
| `X_train.npy` | `output/scores/X_train.npy` | Array float32, ~3.1M x 20 |
| `X_val.npy` | `output/scores/X_val.npy` | Array float32, ~1.1M x 20 |
| `X_test.npy` | `output/scores/X_test.npy` | Array float32, ~2.5M x 20 |
| `scaler.joblib` | `output/models/scaler.joblib` | StandardScaler ajustado en train |

### Tamanos estimados

- `X_train.npy`: ~248 MB
- `X_val.npy`: ~88 MB
- `X_test.npy`: ~200 MB
- `scaler.joblib`: < 1 KB
- **Total:** ~536 MB
