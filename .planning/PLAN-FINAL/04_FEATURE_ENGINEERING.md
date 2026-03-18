# Fase 3: Feature Engineering

> **Gate B requerido antes de continuar a Fase 4.**
> Esta fase es la mas critica del pipeline. Toda feature debe ser auditada contra leakage temporal y circularidad antes de aceptarse.

---

## Catalogo oficial de 20 features

### Grupo 1: Transaccionales (5 features)

| # | Nombre | Formula | Notas |
|---|--------|---------|-------|
| 1 | `amount` | `df["reservation_paid_out"]` (as-is) | Monto pagado en centavos o unidad original |
| 2 | `log_amount` | `np.log1p(df["amount"])` | Compresion logaritmica; `log1p` maneja amount=0 |
| 3 | `amount_usd_ratio` | `df["amount"] / self.global_avg_amount` | `global_avg_amount` se calcula en `fit()` sobre train |
| 4 | `discount_ratio` | `df["discount"] / (df["amount"] + 1e-8)` | Epsilon 1e-8 evita division por cero (correccion: era 0.01) |
| 5 | `has_tip` | `(df["tip"] > 0).astype(np.int8)` | Binaria: 1 si hay propina, 0 si no |

### Grupo 2: Temporales (5 features)

| # | Nombre | Formula | Notas |
|---|--------|---------|-------|
| 6 | `hour_sin` | `np.sin(2 * np.pi * hour / 24)` | Codificacion ciclica del componente horario |
| 7 | `hour_cos` | `np.cos(2 * np.pi * hour / 24)` | Complemento ciclico; juntos preservan distancia circular |
| 8 | `day_of_week` | `dt.dayofweek + 1` | 1=Lunes, 7=Domingo (convencion ISO) |
| 9 | `is_weekend` | `(day_of_week >= 6).astype(np.int8)` | Sabado (6) y Domingo (7) |
| 10 | `is_off_hours` | `hour.isin([23,0,1,2,3,4,5,6]).astype(np.int8)` | Horario nocturno/madrugada |

**Nota:** `hour` se extrae de `created_at` como `df["created_at"].dt.hour`.

### Grupo 3: Velocidad (4 features) — CRITICO ANTI-LEAKAGE

| # | Nombre | Formula | Notas |
|---|--------|---------|-------|
| 11 | `user_txn_count_1h` | Rolling count ventana 1h **MINUS 1** (excluir fila actual) | Ver implementacion abajo |
| 12 | `user_txn_count_24h` | Rolling count ventana 24h **MINUS 1** | Misma logica que #11 |
| 13 | `time_since_last_txn` | `groupby("user_id").diff()` en segundos | Primera transaccion del usuario = 0 |
| 14 | `user_amount_24h` | Rolling sum ventana 24h **MINUS monto propio** | Suma de montos ajenos en ventana |

**Implementacion rolling count (anti-leakage):**

```python
def _velocity_count(self, df, window, col_name):
    """Rolling count dentro de ventana temporal, excluyendo la fila actual.

    La fila actual NO debe contarse a si misma — eso seria
    usar informacion del presente como feature.
    """
    rolling_result = (
        df.groupby("user_id")
        .rolling(window, on="created_at")["id"]
        .count()
    )
    rolling_result = rolling_result.droplevel(0)
    return (rolling_result - 1).reindex(df.index).fillna(0)
```

**Implementacion rolling sum (anti-leakage):**

```python
def _velocity_sum(self, df, window, col_name):
    """Rolling sum de amount en ventana temporal, excluyendo el monto de la fila actual."""
    rolling_result = (
        df.groupby("user_id")
        .rolling(window, on="created_at")["amount"]
        .sum()
    )
    rolling_result = rolling_result.droplevel(0)
    return (rolling_result - df["amount"]).reindex(df.index).fillna(0)
```

**Implementacion time_since_last_txn:**

```python
def _time_since_last(self, df):
    """Segundos desde la ultima transaccion del mismo usuario.
    Primera transaccion = 0 (no NaN).
    """
    return (
        df.groupby("user_id")["created_at"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )
```

**Prerequisito:** DataFrame DEBE estar ordenado por `user_id, created_at` antes de cualquier operacion rolling.

### Grupo 4: Comportamentales (4 features) — CRITICO ANTI-LEAKAGE

| # | Nombre | Formula | Notas |
|---|--------|---------|-------|
| 15 | `user_distinct_facilities_cumul` | Conteo acumulado de facilities distintas **shifted** (excluir actual) | Renombrada de `_30d` a `_cumul`; O(n) |
| 16 | `user_distinct_methods` | Conteo acumulado de metodos de pago distintos **shifted** | Misma logica O(n) que #15 |
| 17 | `user_reversal_ratio_30d` | Rolling mean 30D de `is_reversal` **shifted por 1** dentro del grupo | Riesgo de circularidad con proxy; requiere analisis de sensibilidad |
| 18 | `user_account_age_days` | `(created_at - first_txn).dt.days` | `first_txn` se obtiene del **training set**; usuarios nuevos en val/test usan su primera aparicion dentro del split |

**Implementacion cumulative nunique shifted — O(n):**

```python
def _cumulative_nunique_shifted(series):
    """Conteo acumulado de valores unicos PREVIOS (excluye el valor actual).

    Complejidad O(n) — reemplaza expanding().apply(nunique) que era O(n^2)
    y tardaba 30-90 min. Ahora < 30s.

    Para la primera transaccion del usuario, retorna 0 (no ha visto nada).
    """
    seen = set()
    result = []
    for val in series:
        result.append(len(seen))  # cuantos distintos ANTES de esta fila
        seen.add(val)
    return pd.Series(result, index=series.index)
```

Uso:

```python
# Aplicar por grupo de usuario (ya ordenado por created_at)
df["user_distinct_facilities_cumul"] = (
    df.groupby("user_id")["facility_id"]
    .transform(_cumulative_nunique_shifted)
)

df["user_distinct_methods"] = (
    df.groupby("user_id")["payment_method"]
    .transform(_cumulative_nunique_shifted)
)
```

**Implementacion user_reversal_ratio_30d (con shift anti-leakage):**

```python
def _reversal_ratio_30d(self, df):
    """Proporcion de reversiones en los ultimos 30 dias, shifted por 1.

    ADVERTENCIA: Esta feature usa el proxy (status de reembolso) de forma
    indirecta. Se incluye como feature #17 pero se requiere analisis de
    sensibilidad (variante 19 features sin ella). Si delta AUC >= 0.02,
    Gate D no pasa.
    """
    df["is_reversal"] = df["status"].isin(
        ["totally_refunded", "refunded_to_credit"]
    ).astype(np.int8)

    # BUG FIX: Asignar el resultado del rolling al DataFrame con nombre explicito
    # antes de hacer shift. El .name de la Series tras droplevel/reindex puede
    # ser incorrecto o None, causando KeyError en el shift posterior.
    df["_reversal_ratio_30d_raw"] = (
        df.groupby("user_id")
        .rolling("30D", on="created_at")["is_reversal"]
        .mean()
        .droplevel(0)
        .reindex(df.index)
    )
    # Shift dentro del grupo para excluir la fila actual
    shifted = df.groupby("user_id")["_reversal_ratio_30d_raw"].shift(1).fillna(0)
    df.drop(columns=["_reversal_ratio_30d_raw"], inplace=True)
    return shifted
```

**Implementacion user_account_age_days:**

```python
def _account_age_days(self, df):
    """Dias desde la primera transaccion del usuario.

    - En fit(): se computa self._user_first_txn desde el training set.
    - En transform(): se hace lookup. Usuarios nuevos (no vistos en train)
      usan su primera aparicion dentro del split actual.
    """
    first_txn = df["user_id"].map(self._user_first_txn)
    # Usuarios nuevos: usar su primera aparicion en este split
    new_users = first_txn.isna()
    if new_users.any():
        split_first = df.loc[new_users].groupby("user_id")["created_at"].transform("min")
        first_txn.loc[new_users] = split_first
    return (df["created_at"] - first_txn).dt.days.clip(lower=0)
```

### Grupo 5: Contextuales (2 features)

| # | Nombre | Formula | Notas |
|---|--------|---------|-------|
| 19 | `facility_avg_amount` | Lookup en dict `{facility_id: avg_amount}` calculado en train | Facilities no vistas en train reciben `global_avg_amount` |
| 20 | `amount_facility_ratio` | `df["amount"] / (df["facility_avg_amount"] + 1e-8)` | Ratio del monto respecto a la media de su facility |

**Implementacion:**

```python
def _facility_features(self, df):
    """Features contextuales basadas en el comportamiento del facility."""
    df["facility_avg_amount"] = (
        df["facility_id"]
        .map(self._facility_avg_amount)
        .fillna(self._global_avg_amount)
    )
    df["amount_facility_ratio"] = (
        df["amount"] / (df["facility_avg_amount"] + 1e-8)
    )
    return df
```

---

## Clase FeatureEngineer

### Constante a nivel de modulo

```python
FEATURE_NAMES = [
    # Grupo 1: Transaccionales
    "amount",
    "log_amount",
    "amount_usd_ratio",
    "discount_ratio",
    "has_tip",
    # Grupo 2: Temporales
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    # Grupo 3: Velocidad
    "user_txn_count_1h",
    "user_txn_count_24h",
    "time_since_last_txn",
    "user_amount_24h",
    # Grupo 4: Comportamentales
    "user_distinct_facilities_cumul",
    "user_distinct_methods",
    "user_reversal_ratio_30d",
    "user_account_age_days",
    # Grupo 5: Contextuales
    "facility_avg_amount",
    "amount_facility_ratio",
]

# Validacion a nivel de modulo: usar excepcion en lugar de assert
if len(FEATURE_NAMES) != 20:
    raise ValueError(f"FEATURE_NAMES debe tener 20 elementos, tiene {len(FEATURE_NAMES)}")
```

### Interfaz de la clase

```python
class FeatureEngineer:
    """Genera las 20 features oficiales del catalogo.

    Patron fit/transform para evitar leakage:
    - fit() aprende estadisticas del training set.
    - transform() aplica las transformaciones sin reaprender.
    """

    def __init__(self):
        self._global_avg_amount: float = None
        self._facility_avg_amount: dict = None  # {facility_id: avg_amount}
        self._user_first_txn: dict = None        # {user_id: first_created_at}
        self._fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "FeatureEngineer":
        """Aprende estadisticas del training set.

        Calcula:
        - global_avg_amount: media global de amount en train
        - facility_avg_amount: media de amount por facility_id en train
        - user_first_txn: primera created_at por user_id en train
        """
        self._global_avg_amount = df_train["amount"].mean()
        self._facility_avg_amount = (
            df_train.groupby("facility_id")["amount"].mean().to_dict()
        )
        self._user_first_txn = (
            df_train.groupby("user_id")["created_at"].min().to_dict()
        )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera las 20 features. Requiere fit() previo."""
        # assert se desactiva con python -O; usar excepcion explicita
        if not self._fitted:
            raise RuntimeError("Llamar fit() antes de transform()")
        # 1. Ordenar por user_id, created_at (prerequisito rolling)
        df = df.sort_values(["user_id", "created_at"]).reset_index(drop=True)
        # 2. Transaccionales
        # 3. Temporales
        # 4. Velocidad (rolling)
        # 5. Comportamentales (cumulative + shift)
        # 6. Contextuales (lookup)
        # 7. Seleccionar solo FEATURE_NAMES + metadata
        return df[FEATURE_NAMES + ["id", "user_id", "created_at", "status"]]

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform en un solo paso (solo para train)."""
        return self.fit(df_train).transform(df_train)

    def save(self, path: str) -> None:
        """Serializar con joblib (incluye dicts y avg)."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        """Cargar instancia previamente guardada."""
        return joblib.load(path)

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna la lista oficial de 20 features."""
        return FEATURE_NAMES.copy()
```

### Descomposicion en FeatureGroup strategies (SRP)

> **Violacion SRP detectada:** `FeatureEngineer` calcula los 5 grupos de features en una sola clase monolitica. Esto viola el Principio de Responsabilidad Unica: cada grupo tiene logica distinta (vectorizada, rolling, cumulative, lookup) y deberia poder testearse y modificarse independientemente.

**Descomposicion recomendada (Strategy pattern):**

| Clase | Responsabilidad | Features |
|-------|----------------|----------|
| `TransactionalFeatures` | Transformaciones aritmeticas sobre montos | #1-5 |
| `TemporalFeatures` | Codificacion ciclica y flags temporales | #6-10 |
| `VelocityFeatures` | Rolling windows con anti-leakage | #11-14 |
| `BehavioralFeatures` | Cumulative nunique, reversal ratio, account age | #15-18 |
| `ContextualFeatures` | Lookup de estadisticas por facility | #19-20 |
| `FeatureEngineer` | **Compositor/fachada** que orquesta los FeatureGroups | Todas |

```python
class FeatureGroup(ABC):
    """Interfaz base para grupos de features."""
    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> "FeatureGroup": ...
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    @abstractmethod
    def feature_names(self) -> list[str]: ...

class FeatureEngineer:
    """Compositor que delega a FeatureGroups individuales."""
    def __init__(self, groups: list[FeatureGroup] = None):
        self._groups = groups or [
            TransactionalFeatures(),
            TemporalFeatures(),
            VelocityFeatures(),
            BehavioralFeatures(),
            ContextualFeatures(),
        ]
```

Cada grupo tiene sus propios tests unitarios, lo que facilita el ciclo TDD red-green-refactor por grupo.

---

## Manejo de warm history

### Proposito

Las features de ventana temporal (rolling 1h, 24h, 30D) necesitan historial previo al inicio de cada split. Sin warm history, las primeras transacciones de Enero tendrian features de velocidad artificialmente bajas.

### Estrategia

1. **Train:** Prepend `warm_history.parquet` (Diciembre 2024) al inicio del DataFrame antes de computar features. Despues de calcular features, eliminar las filas de warm history.

2. **Validation:** Prepend la cola del train (ultimas ~24h-30D segun la ventana maxima) antes de computar features de val. Eliminar filas prepended despues.

3. **Test:** Prepend la cola del val (misma logica) antes de computar features de test. Eliminar filas prepended despues.

### Implementacion

```python
def _compute_with_warm_history(self, df_split, df_warm, split_marker_col="_is_split"):
    """Computa features prepending warm history, luego elimina warm rows.

    Args:
        df_split: DataFrame del split actual.
        df_warm: DataFrame de warm history (periodo previo).
        split_marker_col: columna temporal para marcar filas del split.
    """
    # BUG FIX: Usar .assign() en lugar de mutar los DataFrames de entrada.
    # La asignacion directa df_split[col] = ... modifica el DataFrame del
    # llamador (side effect), lo que viola el principio de no-mutacion de inputs.
    df_split = df_split.assign(**{split_marker_col: True})
    df_warm = df_warm.assign(**{split_marker_col: False})

    combined = pd.concat([df_warm, df_split], ignore_index=True)
    combined = combined.sort_values(["user_id", "created_at"]).reset_index(drop=True)

    # Computar todas las features sobre el combined
    combined = self._compute_all_features(combined)

    # Filtrar solo las filas del split
    result = combined[combined[split_marker_col]].drop(columns=[split_marker_col])
    return result.reset_index(drop=True)
```

### Ventana maxima de warm history

- Rolling 30D (feature #17) requiere al menos 30 dias de historia previa.
- Para train (Ene 2025): warm_history = Diciembre 2024 completo.
- Para val (Jul 2025): cola de train = ultimos 30 dias de Junio 2025.
- Para test (Sep 2025): cola de val = ultimos 30 dias de Agosto 2025.

---

## Variante de 19 features

Para el analisis de sensibilidad (Gate D), se necesita una variante que excluya `user_reversal_ratio_30d` (feature #17):

```python
FEATURE_NAMES_19 = [f for f in FEATURE_NAMES if f != "user_reversal_ratio_30d"]
if len(FEATURE_NAMES_19) != 19:
    raise ValueError(f"FEATURE_NAMES_19 debe tener 19 elementos, tiene {len(FEATURE_NAMES_19)}")
```

Esta variante:
- Usa el mismo `FeatureEngineer` (ya calcula las 20 features).
- En el paso de preprocesamiento/modelado, simplemente selecciona las 19 columnas.
- Se evalua con los mismos hiperparametros del modelo de 20 features.
- Si `delta AUC >= 0.02`, el modelo de 20 features no puede ser el hallazgo principal (Gate D).

---

## Estimaciones de rendimiento (~3.1M filas, split mas grande)

| Grupo | Tiempo estimado | Nota |
|-------|----------------|------|
| Transaccionales + Temporales | < 2s | Operaciones vectorizadas puras |
| Rolling count/sum (Grupo 3) | 2-5 min | groupby + rolling es costoso en pandas |
| Cumulative nunique (Grupo 4, #15-16) | < 30s | O(n) con set(); era 30-90 min con expanding.apply |
| Reversal ratio + account age (#17-18) | < 30s | Rolling mean + lookup |
| Contextuales (Grupo 5) | < 1s | Map + aritmetica |
| **Total por split** | **5-12 min** | Dominado por rolling ventanas |

### Memoria pico

~4-5 GB durante las operaciones de rolling groupby. El DataFrame combinado (split + warm history) puede alcanzar ~3.5M filas para train.

---

## Reglas anti-leakage

Estas reglas son **obligatorias** y constituyen el Gate B:

1. **Orden temporal:** Siempre ordenar por `user_id, created_at` antes de cualquier operacion rolling o cumulative.

2. **Exclusion de la fila actual:** Rolling count/sum usa `-1` / `-self.amount`. Cumulative nunique usa `shifted` (append current AFTER counting). Reversal ratio usa `shift(1)` dentro del grupo.

3. **No usar `status` como feature:** El campo `status` solo se usa en feature #17 de forma derivada (`is_reversal`), y con shift. No se usa directamente como predictor. Esta es la razon del analisis de sensibilidad obligatorio.

4. **No informacion futura:** Ninguna ventana mira hacia adelante. `closed='left'` o equivalente aplica en todas las ventanas rolling.

5. **No reinicializar historias en limites de split:** Las features de val y test DEBEN arrastrar historia del split anterior via warm history. Cortar la historia en el limite del split crearia features artificialmente bajas.

6. **Estadisticas fit solo en train:** `global_avg_amount`, `facility_avg_amount`, y `user_first_txn` se calculan exclusivamente sobre el training set.

---

## Tests de validacion (Gate B)

> **Enfoque TDD (red-green-refactor):** Estos tests definen el CONTRATO que `FeatureEngineer` (y sus FeatureGroups) debe satisfacer. El orden de trabajo es:
>
> 1. **Red:** Escribir TODOS estos tests primero. Ejecutarlos y verificar que fallan (porque la implementacion aun no existe).
> 2. **Green:** Implementar la logica de feature engineering, grupo por grupo, hasta que cada test pase.
> 3. **Refactor:** Una vez que todos los tests pasan, refactorizar la implementacion (extraer metodos, mejorar performance) sin romper ningun test.
>
> Los tests NO son verificaciones post-hoc; son la especificacion ejecutable que guia la implementacion.

Estos 10 tests deben pasar antes de continuar a Fase 4:

### 1. Primera transaccion por usuario tiene count features == 0

```python
def test_first_txn_counts_zero():
    """La primera transaccion de cada usuario debe tener
    user_txn_count_1h == 0 y user_txn_count_24h == 0."""
    first_txns = df_features.groupby("user_id").first()
    assert (first_txns["user_txn_count_1h"] == 0).all()
    assert (first_txns["user_txn_count_24h"] == 0).all()
```

### 2. Cumulative nunique de la primera transaccion == 0

```python
def test_first_txn_distinct_facilities_zero():
    """Antes de su primera transaccion, un usuario no ha visto ninguna facility."""
    first_txns = df_features.groupby("user_id").first()
    assert (first_txns["user_distinct_facilities_cumul"] == 0).all()
```

### 3. facility_avg values coinciden con calculo en training

```python
def test_facility_avg_matches_train():
    """Los valores de facility_avg_amount deben coincidir con la media
    calculada sobre el training set."""
    train_avgs = df_train.groupby("facility_id")["amount"].mean()
    for fid, avg in train_avgs.items():
        assert np.isclose(fe._facility_avg_amount[fid], avg)
```

### 4. FEATURE_NAMES tiene exactamente 20 entradas

```python
def test_feature_names_count():
    assert len(FEATURE_NAMES) == 20
    assert len(set(FEATURE_NAMES)) == 20  # sin duplicados
```

### 5. No hay NaN en features despues de transform

```python
def test_no_nans():
    """Todas las features deben estar completas despues de transform."""
    for col in FEATURE_NAMES:
        assert df_features[col].isna().sum() == 0, f"NaN encontrados en {col}"
```

### 6. Matriz de correlacion (warn si |r| > 0.95)

```python
def test_correlation_matrix():
    """Advertencia si alguna pareja de features tiene |r| > 0.95.
    No es bloqueante pero debe documentarse."""
    corr = df_features[FEATURE_NAMES].corr()
    high_corr = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            if abs(corr.iloc[i, j]) > 0.95:
                high_corr.append((FEATURE_NAMES[i], FEATURE_NAMES[j], corr.iloc[i, j]))
    if high_corr:
        warnings.warn(f"Parejas con alta correlacion: {high_corr}")
```

### 7. Estadisticas de features para Tabla 3.7

```python
def test_feature_statistics():
    """Genera y valida estadisticas descriptivas de cada feature.
    Exporta a feature_statistics.csv."""
    stats = df_features[FEATURE_NAMES].describe().T
    stats.to_csv("output/feature_statistics.csv")
    # Verificar rangos esperados
    assert stats.loc["amount", "min"] >= 0
    assert stats.loc["has_tip", "max"] <= 1
    assert stats.loc["is_weekend", "max"] <= 1
```

### 8. Tests de limites de split

```python
def test_split_boundaries():
    """Verificar que las features en los limites de split (Jun 30 -> Jul 1,
    Ago 31 -> Sep 1) no muestran discontinuidades artificiales.

    Las features de velocidad del primer dia de val/test deben reflejar
    la historia arrastrada del split anterior."""
    jul1_first = df_val_features[
        df_val_features["created_at"].dt.date == pd.Timestamp("2025-07-01").date()
    ].head(100)
    # Al menos algunos usuarios deben tener count > 0 si tuvieron actividad reciente
    assert (jul1_first["user_txn_count_24h"] > 0).any()
```

### 9. Usuarios cold-start manejados correctamente

```python
def test_cold_start_users():
    """Usuarios que aparecen por primera vez en val/test deben:
    - Tener count features == 0 en su primera transaccion
    - Tener user_account_age_days basado en su primera aparicion en el split
    """
    new_val_users = set(df_val["user_id"]) - set(df_train["user_id"])
    if new_val_users:
        new_first = df_val_features[
            df_val_features["user_id"].isin(new_val_users)
        ].groupby("user_id").first()
        assert (new_first["user_txn_count_1h"] == 0).all()
        assert (new_first["user_account_age_days"] == 0).all()
```

### 10. Robustez para edge cases (amount=0, discount > amount)

```python
def test_edge_cases():
    """Verificar que features no explotan con edge cases."""
    # amount = 0
    zero_amount = df_features[df_features["amount"] == 0]
    if len(zero_amount) > 0:
        assert np.isfinite(zero_amount["log_amount"]).all()
        assert np.isfinite(zero_amount["discount_ratio"]).all()
        assert np.isfinite(zero_amount["amount_facility_ratio"]).all()

    # discount > amount (posible en datos reales)
    high_discount = df_features[df_features["discount_ratio"] > 1.0]
    if len(high_discount) > 0:
        assert np.isfinite(high_discount["discount_ratio"]).all()
```

---

## Entregables

| Artefacto | Ruta | Descripcion |
|-----------|------|-------------|
| `engineering.py` | `src/fraud_detector/features/engineering.py` | Clase `FeatureEngineer` + constante `FEATURE_NAMES` |
| `test_features.py` | `tests/test_features.py` | 10 tests de Gate B |
| `train_features.parquet` | `data/processed/train_features.parquet` | ~3.1M filas x 20 features + metadata |
| `val_features.parquet` | `data/processed/val_features.parquet` | ~1.1M filas x 20 features + metadata |
| `test_features.parquet` | `data/processed/test_features.parquet` | ~2.5M filas x 20 features + metadata |
| `feature_engineer.joblib` | `output/models/feature_engineer.joblib` | Instancia serializada con estadisticas de train |
| `feature_statistics.csv` | `output/feature_statistics.csv` | Estadisticas descriptivas para Tabla 3.7 |

---

## Diagrama de flujo

```
warm_history.parquet + train_raw.parquet
    |
    v
[FeatureEngineer.fit_transform(train_with_warm)]
    |
    +--> feature_engineer.joblib (estadisticas train)
    +--> train_features.parquet (solo filas de train, sin warm)
    |
    v
[FeatureEngineer.transform(val_with_train_tail)]
    |
    +--> val_features.parquet (solo filas de val)
    |
    v
[FeatureEngineer.transform(test_with_val_tail)]
    |
    +--> test_features.parquet (solo filas de test)
    |
    v
[Gate B: 10 tests de validacion]
    |
    +--> PASS --> Continuar a Fase 4
    +--> FAIL --> Revisar y corregir antes de continuar
```
