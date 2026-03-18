# Fase 2: Feature Engineering (archivo mas critico)

## Archivo: `src/fraud_detector/features/engineering.py`

### Que eliminar

La clase `FraudFeatureEngineer` completa. Tiene problemas fundamentales:
- Usa loops `for user_id ... for idx, row` (O(n^2), inutilizable en 3.1M filas)
- No tiene separacion fit/transform (leakage de test a train)
- Nombres de columnas incorrectos (timestamp hardcodeado en vez de created_at)
- No implementa las 20 features especificas de la tesis
- Feature set incorrecto

### Nueva clase: `FeatureEngineer`

```python
class FeatureEngineer:
    """
    Vectorized feature engineering for unsupervised anomaly detection.
    Implements 20 features aligned with thesis Chapter 2 methodology.
    fit() learns statistics from training data only.
    transform() applies features to any split without leakage.
    """

    def __init__(self):
        self.global_avg_amount: float = None
        self.facility_avg_amount: dict = None  # {facility_id: avg}
        self.user_first_txn: dict = None  # {user_id: Timestamp} - for account age
        self._fitted = False

    def fit(self, df_train: pd.DataFrame) -> "FeatureEngineer":
        """Learn statistics from training data only."""
        self.global_avg_amount = df_train["amount"].mean()
        self.facility_avg_amount = df_train.groupby("facility_id")["amount"].mean().to_dict()
        self.user_first_txn = df_train.groupby("user_id")["created_at"].min().to_dict()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "FeatureEngineer": ...
    @staticmethod
    def get_feature_names() -> list: return FEATURE_NAMES.copy()
```

---

## Las 20 Features

### Grupo 1: Transaccionales (5 features)

| # | Feature | Implementacion |
|---|---------|---------------|
| 1 | `amount` | Columna as-is (reservation_paid_out alias desde extraccion) |
| 2 | `log_amount` | `np.log1p(df["amount"])` |
| 3 | `amount_usd_ratio` | `df["amount"] / self.global_avg_amount` (train-fitted) |
| 4 | `discount_ratio` | `df["discount"] / (df["amount"] + 1e-8)` **FIX: usar 1e-8, NO 0.01** |
| 5 | `has_tip` | `(df["tip"] > 0).astype(np.int8)` |

**fit()**: calcula `self.global_avg_amount = df_train["amount"].mean()`

**FIX IMPORTANTE**: Epsilon cambiado de 0.01 a 1e-8. Con epsilon=0.01, una transaccion de $0.50 obtiene una distorsion del 2% en el ratio. Dado que amount > 0 esta garantizado por la clausula WHERE de la extraccion, 1e-8 es puramente una proteccion contra ceros de punto flotante.

### Grupo 2: Temporales (5 features)

| # | Feature | Implementacion |
|---|---------|---------------|
| 6 | `hour_sin` | `np.sin(2 * np.pi * hour / 24)` |
| 7 | `hour_cos` | `np.cos(2 * np.pi * hour / 24)` |
| 8 | `day_of_week` | `df["created_at"].dt.dayofweek + 1` (1=Mon, 7=Sun) |
| 9 | `is_weekend` | `(day_of_week >= 6).astype(np.int8)` |
| 10 | `is_off_hours` | `hour.isin([23, 0, 1, 2, 3, 4, 5, 6]).astype(np.int8)` |

**fit()**: No necesita nada (transformaciones deterministicas).

**Nota de timezone**: Verificar que ClickHouse almacena created_at en UTC o una zona horaria conocida. Si es timezone-aware, quitar timezone en transform():
```python
if df["created_at"].dt.tz is not None:
    df["created_at"] = df["created_at"].dt.tz_localize(None)
```

### Grupo 3: Velocidad (4 features)

| # | Feature | Implementacion |
|---|---------|---------------|
| 11 | `user_txn_count_1h` | rolling count - 1 (restar self) |
| 12 | `user_txn_count_24h` | rolling count - 1 (restar self) |
| 13 | `time_since_last_txn` | groupby diff en segundos |
| 14 | `user_amount_24h` | rolling sum - self amount (restar self) |

**FIX CRITICO**: El resultado del rolling tiene un MultiIndex (user_id, original_index) despues del groupby. Se debe aplanar antes de asignar:

```python
# Implementacion CORRECTA para features 11, 12:
def _velocity_count(self, df: pd.DataFrame, window: str, col_name: str) -> pd.Series:
    """Rolling transaction count within window, excluding current transaction."""
    rolling_result = (
        df.groupby("user_id")
        .rolling(window, on="created_at")["id"]
        .count()
    )
    # Drop groupby level from MultiIndex, keep original index
    rolling_result = rolling_result.droplevel(0)
    # Subtract 1 to exclude current transaction (rolling includes it)
    return (rolling_result - 1).reindex(df.index).fillna(0)

# Implementacion CORRECTA para feature 14:
def _velocity_sum(self, df: pd.DataFrame, window: str, amount_col: str) -> pd.Series:
    """Rolling amount sum within window, excluding current transaction's amount."""
    rolling_result = (
        df.groupby("user_id")
        .rolling(window, on="created_at")[amount_col]
        .sum()
    )
    rolling_result = rolling_result.droplevel(0)
    # Subtract self-amount to exclude current transaction
    return (rolling_result - df[amount_col]).reindex(df.index).fillna(0)
```

**Estrategia anti-leakage: "restar self"**
- La ventana rolling naturalmente incluye la fila actual
- Restamos 1 (para count) o df["amount"] (para sum) para excluir la transaccion actual
- Esto es mas robusto que shift(1) para features con ventana temporal
- **Caso borde**: si existen timestamps duplicados para el mismo usuario, rolling los incluye a todos. count()-1 solo restaria 1. Esto es aceptable (documentado).

**Feature 13 (time_since_last_txn)**:
```python
df["time_since_last_txn"] = (
    df.groupby("user_id")["created_at"]
    .diff()
    .dt.total_seconds()
)
# Primera transaccion por usuario = NaN -> rellenar con 0
df["time_since_last_txn"] = df["time_since_last_txn"].fillna(0)
```
Sin preocupacion de leakage: `.diff()` naturalmente mira solo la fila anterior.

### Grupo 4: Comportamiento (4 features)

| # | Feature | Implementacion |
|---|---------|---------------|
| 15 | `user_distinct_facilities_cumul` | Cumulative nunique (O(n) por grupo) |
| 16 | `user_distinct_methods` | Cumulative nunique (O(n) por grupo) |
| 17 | `user_reversal_ratio_30d` | Rolling mean con shift per-group |
| 18 | `user_account_age_days` | Usando fechas first_txn fitted en train |

**FIX CRITICO para Features 15, 16**:

El plan original usaba `expanding().apply(lambda x: x.nunique()).shift(1)`. Esto tiene DOS bugs:
1. `.shift(1)` despues de expanding opera sobre la serie APLANADA, causando leakage cross-user
2. `expanding().apply(nunique)` es O(n^2) por grupo -- tomaria 30-90 min para 3.1M filas

**Implementacion CORREGIDA usando enfoque cumulative O(n):**

```python
def _cumulative_nunique_shifted(series: pd.Series) -> pd.Series:
    """
    O(n) cumulative nunique, shifted by 1 to exclude current transaction.
    Returns the number of distinct values BEFORE the current row.
    """
    seen = set()
    result = []
    for val in series:
        result.append(len(seen))  # append count BEFORE adding current
        seen.add(val)
    return pd.Series(result, index=series.index)

# Aplicar por grupo:
df["user_distinct_facilities_cumul"] = (
    df.groupby("user_id")["facility_id"]
    .transform(_cumulative_nunique_shifted)
)

df["user_distinct_methods"] = (
    df.groupby("user_id")["payment_method"]
    .transform(_cumulative_nunique_shifted)
)
```

Esto es O(n) total, corre en <30 segundos para 3.1M filas, Y naturalmente maneja anti-leakage dentro de grupos (sin contaminacion cross-group).

**RENOMBRADO**: Feature 15 se renombra de `user_distinct_facilities_30d` a `user_distinct_facilities_cumul` porque usa expanding (all-time), no rolling de 30 dias. Documentar en la tesis.

**FIX CRITICO para Feature 17 (user_reversal_ratio_30d)**:

El `.shift(1)` original tenia leakage cross-group. Corregido:

```python
# Crear indicador auxiliar de reembolso
df["_is_refund"] = df["status"].isin(["totally_refunded", "refunded_to_credit"]).astype(np.int8)

# Rolling mean dentro de ventana de 30 dias por usuario
rolling_result = (
    df.groupby("user_id")
    .rolling("30D", on="created_at")["_is_refund"]
    .mean()
)
# El shift DEBE ser DENTRO de grupos, no global
rolling_result = rolling_result.groupby(level=0).shift(1)
df["user_reversal_ratio_30d"] = rolling_result.droplevel(0).reindex(df.index).values
df["user_reversal_ratio_30d"] = df["user_reversal_ratio_30d"].fillna(0)
df.drop(columns=["_is_refund"], inplace=True)
```

**Nota de circularidad**: Feature 17 usa el status de reembolso, que se solapa con la definicion del proxy label. El shift(1) asegura que el status de reembolso de la transaccion ACTUAL esta excluido. Esto es tecnicamente correcto (sin data leakage). La preocupacion filosofica (usuarios con historial de reembolsos tienden a reembolsar de nuevo) se mitiga con el analisis de sensibilidad en Fase 5 (comparar AUC con y sin feature 17). Documentar en la tesis.

**FIX CRITICO para Feature 18 (user_account_age_days)**:

El original usaba `groupby("user_id")["created_at"].transform("min")` por-split, lo que computa un baseline DIFERENTE para cada split. Un usuario activo desde ene-2025 tendria age=0 en el split de test (sep-2025).

**Corregido: usar fechas first_txn fitted en train:**

```python
# En fit():
self.user_first_txn = df_train.groupby("user_id")["created_at"].min().to_dict()

# En transform():
first_txn = df["user_id"].map(self.user_first_txn)
# Usuarios no vistos en training: usar su primera transaccion en este split
mask_new = first_txn.isna()
if mask_new.any():
    per_split_first = df.loc[mask_new].groupby("user_id")["created_at"].transform("min")
    first_txn.loc[mask_new] = per_split_first
df["user_account_age_days"] = (df["created_at"] - first_txn).dt.total_seconds() / 86400
```

### Grupo 5: Contextuales (2 features)

| # | Feature | Implementacion |
|---|---------|---------------|
| 19 | `facility_avg_amount` | Map desde dict fitted en train |
| 20 | `amount_facility_ratio` | `amount / (facility_avg + 1e-8)` **FIX: 1e-8, NO 0.01** |

```python
df["facility_avg_amount"] = df["facility_id"].map(self.facility_avg_amount)
df["facility_avg_amount"] = df["facility_avg_amount"].fillna(self.global_avg_amount)
df["amount_facility_ratio"] = df["amount"] / (df["facility_avg_amount"] + 1e-8)
```

---

## Constante FEATURE_NAMES (nivel de modulo)

```python
FEATURE_NAMES = [
    "amount",                           # 1  Transactional
    "log_amount",                       # 2  Transactional
    "amount_usd_ratio",                 # 3  Transactional
    "discount_ratio",                   # 4  Transactional
    "has_tip",                          # 5  Transactional
    "hour_sin",                         # 6  Temporal
    "hour_cos",                         # 7  Temporal
    "day_of_week",                      # 8  Temporal
    "is_weekend",                       # 9  Temporal
    "is_off_hours",                     # 10 Temporal
    "user_txn_count_1h",                # 11 Velocity
    "user_txn_count_24h",              # 12 Velocity
    "time_since_last_txn",             # 13 Velocity
    "user_amount_24h",                 # 14 Velocity
    "user_distinct_facilities_cumul",  # 15 Behavioral (expanding, not 30d)
    "user_distinct_methods",           # 16 Behavioral
    "user_reversal_ratio_30d",         # 17 Behavioral (SENSITIVITY)
    "user_account_age_days",           # 18 Behavioral
    "facility_avg_amount",             # 19 Contextual
    "amount_facility_ratio",           # 20 Contextual
]
```

---

## Metodos save() / load()

```python
def save(self, path: Path) -> None:
    joblib.dump({
        "global_avg_amount": self.global_avg_amount,
        "facility_avg_amount": self.facility_avg_amount,
        "user_first_txn": self.user_first_txn,
    }, path)

@classmethod
def load(cls, path: Path) -> "FeatureEngineer":
    data = joblib.load(path)
    obj = cls()
    obj.global_avg_amount = data["global_avg_amount"]
    obj.facility_avg_amount = data["facility_avg_amount"]
    obj.user_first_txn = data.get("user_first_txn", {})
    obj._fitted = True
    return obj
```

---

## Estructura del metodo transform()

```python
def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    if not self._fitted:
        raise RuntimeError("Must call fit() before transform()")

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    if df["created_at"].dt.tz is not None:
        df["created_at"] = df["created_at"].dt.tz_localize(None)
    df = df.sort_values(["user_id", "created_at"]).reset_index(drop=True)

    # 1. Transactional (vectorized, instant)
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_usd_ratio"] = df["amount"] / self.global_avg_amount
    df["discount_ratio"] = df["discount"] / (df["amount"] + 1e-8)
    df["has_tip"] = (df["tip"] > 0).astype(np.int8)

    # 2. Temporal (vectorized, instant)
    hour = df["created_at"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = df["created_at"].dt.dayofweek + 1
    df["is_weekend"] = (df["day_of_week"] >= 6).astype(np.int8)
    df["is_off_hours"] = hour.isin([23, 0, 1, 2, 3, 4, 5, 6]).astype(np.int8)

    # 3. Velocity (groupby + rolling, 2-5 min)
    df["user_txn_count_1h"] = self._velocity_count(df, "1h", "user_txn_count_1h")
    df["user_txn_count_24h"] = self._velocity_count(df, "24h", "user_txn_count_24h")
    df["time_since_last_txn"] = df.groupby("user_id")["created_at"].diff().dt.total_seconds().fillna(0)
    df["user_amount_24h"] = self._velocity_sum(df, "24h", "amount")

    # 4. Behavioral (cumulative + rolling, <30s + 2-5 min)
    df["user_distinct_facilities_cumul"] = df.groupby("user_id")["facility_id"].transform(_cumulative_nunique_shifted)
    df["user_distinct_methods"] = df.groupby("user_id")["payment_method"].transform(_cumulative_nunique_shifted)
    self._add_reversal_ratio(df)
    self._add_account_age(df)

    # 5. Contextual (map, instant)
    df["facility_avg_amount"] = df["facility_id"].map(self.facility_avg_amount).fillna(self.global_avg_amount)
    df["amount_facility_ratio"] = df["amount"] / (df["facility_avg_amount"] + 1e-8)

    # Fill any remaining NaN in feature columns
    df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(0)

    return df
```

---

## Estimaciones de Rendimiento (corregidas)

| Operacion | Complejidad | Tiempo estimado (3.1M filas) |
|-----------|-------------|------------------------------|
| Transactional (1-5) | O(n) vectorizado | < 1s |
| Temporal (6-10) | O(n) vectorizado | < 1s |
| Rolling count/sum (11,12,14) | O(n log n) por grupo | 2-5 min |
| diff (13) | O(n) por grupo | < 30s |
| Cumulative nunique (15,16) | O(n) por grupo | **< 30s** (era 30-90 min con expanding.apply) |
| Rolling mean (17) | O(n log n) por grupo | 2-5 min |
| Account age (18) | O(n) map | < 1s |
| Contextual (19,20) | O(n) map | < 1s |
| **Total estimado** | | **5-12 min** (era 10-25 min) |

Pico de memoria: ~4-5 GB (DataFrame + buffers de rolling)

---

## Verificacion (Gate B - Anti-Leakage)

Antes de proceder a la Fase 3:

1. Para la PRIMERA transaccion de cada usuario en el split, verificar que `user_txn_count_1h == 0` y `user_txn_count_24h == 0`
2. Verificar que `user_distinct_facilities_cumul[first_txn] == 0` para cada usuario
3. Verificar que `user_distinct_methods[first_txn] == 0` para cada usuario
4. Verificar que los valores de `facility_avg_amount` coincidan con los promedios calculados en training (no recomputados del split actual)
5. Verificar que `FEATURE_NAMES` tiene exactamente 20 entradas y todas las columnas existen en el DataFrame de salida
6. Verificar que no hay NaN en ninguna columna de features despues de transform
7. Loggear la matriz de correlacion de features (warning si algun par tiene |r| > 0.95)
8. Loggear estadisticas de features (min, max, mean, std, % zeros) para la Tabla 3.7 de la tesis
