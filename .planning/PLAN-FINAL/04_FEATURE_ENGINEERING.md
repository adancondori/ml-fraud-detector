# Fase 3: Feature Engineering

> **Gate B requerido antes de continuar a Fase 4.**
> Esta fase es la mas critica del pipeline. Toda feature debe ser auditada contra leakage temporal y circularidad antes de aceptarse.

---

## Catalogo oficial de 31 features (8 grupos, F06 y F21 eliminadas)

### Grupo A: Transaccionales (6 features)

| # | Codigo | Nombre | Formula | Notas |
|---|--------|--------|---------|-------|
| F01 | `amount` | Monto pagado | `df["reservation_paid_out"]` (normalizado a USD) | Monto ya en USD via `reservation_paid_out` |
| F02 | `log_amount` | Log-monto | `np.log1p(df["amount"])` | Compresion logaritmica; `log1p` maneja amount=0 |
| F03 | `amount_usd_ratio` | Ratio monto/media global | `df["amount"] / self.global_avg_amount` | `global_avg_amount` se calcula en `fit()` sobre train |
| F04 | `discount_ratio` | Ratio descuento/monto | `df["discount"] / (df["amount"] + 0.01)` | Epsilon 0.01 evita division por cero |
| F05 | `has_tip` | Tiene propina | `(df["tip"] > 0).astype(np.int8)` | Binaria: 1 si hay propina, 0 si no |
| ~~F06~~ | ~~`is_free`~~ | ~~Transaccion gratuita~~ | — | **ELIMINADA** — `payment_method='free'` excluido del universo; seria constante (=0) |

### Grupo B: Temporales (5 features)

| # | Codigo | Nombre | Formula | Notas |
|---|--------|--------|---------|-------|
| F07 | `hour_sin` | Hora (seno) | `np.sin(2 * np.pi * hour / 24)` | Codificacion ciclica del componente horario |
| F08 | `hour_cos` | Hora (coseno) | `np.cos(2 * np.pi * hour / 24)` | Complemento ciclico; juntos preservan distancia circular |
| F09 | `day_of_week` | Dia de la semana | `dt.dayofweek + 1` | 1=Lunes, 7=Domingo (convencion ISO) |
| F10 | `is_weekend` | Fin de semana | `(day_of_week >= 6).astype(np.int8)` | Sabado (6) y Domingo (7) |
| F11 | `is_off_hours` | Horario nocturno | `hour.isin([23,0,1,2,3,4,5,6]).astype(np.int8)` | Horario nocturno/madrugada |

**Nota:** `hour` se extrae de `created_at` como `df["created_at"].dt.hour`.

### Grupo C: Velocidad (4 features) -- CRITICO ANTI-LEAKAGE

| # | Codigo | Nombre | Formula | Notas |
|---|--------|--------|---------|-------|
| F12 | `user_txn_count_1h` | Conteo txn 1h | Rolling count ventana 1h **MINUS 1** (excluir fila actual) | Ver implementacion abajo |
| F13 | `user_txn_count_24h` | Conteo txn 24h | Rolling count ventana 24h **MINUS 1** | Misma logica que F12 |
| F14 | `time_since_last_txn` | Tiempo desde ultima txn | `groupby("user_id").diff()` en segundos | Primera transaccion del usuario = 0 |
| F15 | `user_amount_24h` | Monto acumulado 24h | Rolling sum ventana 24h **MINUS monto propio** | Suma de montos ajenos en ventana |

**Implementacion rolling count (anti-leakage):**

```python
def _velocity_count(self, df, window, col_name):
    """Rolling count dentro de ventana temporal, excluyendo la fila actual.

    La fila actual NO debe contarse a si misma -- eso seria
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

### Grupo D: Comportamentales (6 features) -- CRITICO ANTI-LEAKAGE

| # | Codigo | Nombre | Formula | Notas |
|---|--------|--------|---------|-------|
| F16 | `user_distinct_facilities_30d` | Facilities distintas 30d | Rolling nunique de `facility_id` en ventana 30d, **shifted por 1** | Ventana 30d (no cumulative); excluye fila actual |
| F17 | `user_distinct_methods` | Metodos de pago distintos | Conteo acumulado de metodos de pago distintos **shifted** | Logica O(n) con set() |
| F18 | `user_reversal_ratio_30d` | Ratio reversiones 30d | Rolling mean 30D de `is_reversal` **shifted por 1** dentro del grupo | Riesgo de circularidad con proxy; requiere analisis de sensibilidad |
| F19 | `user_account_age_days` | Antiguedad de cuenta (dias) | `(created_at - users.created_at).dt.days` | Usa `users.created_at` (fecha de creacion de cuenta) |
| F20 | `user_discount_ratio_30d` | Ratio descuento promedio 30d | Rolling mean 30D de discount_ratio shifted por 1 | Patron acumulado de descuentos por usuario |
| ~~F21~~ | ~~`user_free_pct_30d`~~ | ~~Porcentaje free 30d~~ | — | **ELIMINADA** — `payment_method='free'` excluido del universo; seria constante (=0) |

**Implementacion user_distinct_facilities_30d (ventana 30d con shift):**

```python
def _distinct_facilities_30d(self, df):
    """Conteo de facilities distintas en los ultimos 30 dias, shifted por 1.

    Ventana de 30 dias (no acumulativa) para capturar diversidad reciente.
    shift(1) excluye la fila actual para anti-leakage.
    """
    def _rolling_nunique_30d(group):
        """Nunique en ventana 30d para un grupo de usuario."""
        result = []
        created_ats = group["created_at"].values
        facilities = group["facility_id"].values
        for i in range(len(group)):
            cutoff = created_ats[i] - np.timedelta64(30, "D")
            window_facilities = facilities[
                (created_ats[:i] >= cutoff) & (np.arange(len(group)) < i)
            ]
            result.append(len(set(window_facilities)))
        return pd.Series(result, index=group.index)

    return df.groupby("user_id").apply(_rolling_nunique_30d).droplevel(0).reindex(df.index).fillna(0)
```

**Implementacion cumulative nunique shifted -- O(n) (para F17):**

```python
def _cumulative_nunique_shifted(series):
    """Conteo acumulado de valores unicos PREVIOS (excluye el valor actual).

    Complejidad O(n) -- reemplaza expanding().apply(nunique) que era O(n^2)
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
df["user_distinct_methods"] = (
    df.groupby("user_id")["payment_method"]
    .transform(_cumulative_nunique_shifted)
)
```

### Estado acumulado cross-split para F17

F17 es la unica feature acumulativa no acotada a ventana temporal. Para evitar subestimacion del historial en val/test:

1. Procesar splits secuencialmente: warm → train → val → test
2. Al finalizar cada split, persistir `user_cumulative_methods: Dict[int, Set[str]]` como estado
3. Al iniciar el siguiente split, inicializar con el estado persistido del split anterior
4. Serializar estado en `output/models/feature_state.joblib`

**Implementacion user_reversal_ratio_30d (con shift anti-leakage):**

```python
def _reversal_ratio_30d(self, df):
    """Proporcion de reversiones en los ultimos 30 dias, shifted por 1.

    ADVERTENCIA: Esta feature usa el proxy (status de reembolso) de forma
    indirecta. Se incluye como feature F18 pero se requiere analisis de
    sensibilidad (variante 30 features sin ella). Si delta AUC >= 0.02,
    Gate D no pasa.
    """
    df["is_reversal"] = df["status"].isin(
        ["totally_refunded", "refunded_to_credit"]
    ).astype(np.int8)

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

**Implementacion user_account_age_days (basada en users.created_at):**

```python
def _account_age_days(self, df):
    """Dias desde la creacion de la cuenta del usuario (users.created_at).

    - En fit(): se computa self._user_created_at desde la columna user_created_at
      (obtenida via JOIN con tabla users en la extraccion).
    - En transform(): se hace lookup. Usuarios nuevos (no vistos en train)
      usan su primera aparicion dentro del split actual como fallback.
    """
    created_at = df["user_id"].map(self._user_created_at)
    # Usuarios nuevos: fallback a primera aparicion en el split
    new_users = created_at.isna()
    if new_users.any():
        split_first = df.loc[new_users].groupby("user_id")["created_at"].transform("min")
        created_at.loc[new_users] = split_first
    return (df["created_at"] - created_at).dt.days.clip(lower=0)
```

### Grupo E: Contextuales (2 features)

| # | Codigo | Nombre | Formula | Notas |
|---|--------|--------|---------|-------|
| F22 | `facility_avg_amount` | Media de monto por facility | Lookup en dict `{facility_id: avg_amount}` calculado en train | Facilities no vistas en train reciben `global_avg_amount` |
| F23 | `amount_facility_ratio` | Ratio monto/media facility | `df["amount"] / (df["facility_avg_amount"] + 0.01)` | Ratio del monto respecto a la media de su facility |

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
        df["amount"] / (df["facility_avg_amount"] + 0.01)
    )
    return df
```

### Grupo F: Credito/Flujo (4 features) -- NUEVO

| # | Codigo | Nombre | Formula | Notas |
|---|--------|--------|---------|-------|
| F24 | `is_club_credit` | Credito club | `df["club_credit_flag"].astype(np.int8)` | Binaria: 1 si `club_credit_flag = true` |
| F25 | `user_debit_count_30d` | Conteo debitos 30d | Rolling count de txns con `category = 'debit'` en ventana 30d, **shifted por 1** | Requiere columna `category` |
| F26 | `user_debit_amount_30d` | Monto debitos 30d (USD) | Rolling sum de amount donde `category = 'debit'` en 30d, **shifted por 1** | Montos ya en USD |
| F27 | `credit_flow_ratio` | Ratio debito/prepago | `debit_amount_30d / (prepaid_spend_30d + 0.01)` | Desequilibrio entre debitos y prepago del usuario |

**Columnas requeridas en extraccion:** `club_credit_flag` (boolean), `category` (string: 'debit', 'prepaid', etc.)

**Implementacion F24:**

```python
def _is_club_credit(self, df):
    """Flag binaria de credito club."""
    return df["club_credit_flag"].fillna(False).astype(np.int8)
```

**Implementacion F25-F26 (rolling 30d sobre debitos, anti-leakage):**

```python
def _debit_features_30d(self, df):
    """Conteo y monto de debitos en ventana 30d, shifted por 1.

    Solo cuenta transacciones donde category = 'debit'.
    El shift(1) dentro del grupo excluye la fila actual.
    """
    df["_is_debit"] = (df["category"] == "debit").astype(np.int8)
    df["_debit_amount"] = df["amount"] * df["_is_debit"]

    # Rolling count de debitos en 30d
    raw_count = (
        df.groupby("user_id")
        .rolling("30D", on="created_at")["_is_debit"]
        .sum()
        .droplevel(0)
        .reindex(df.index)
    )
    debit_count = df.groupby("user_id")[raw_count.name if hasattr(raw_count, 'name') and raw_count.name else "_is_debit"].shift(1).fillna(0)
    # Alternativa mas segura: asignar resultado intermedio
    df["_debit_count_30d_raw"] = raw_count
    debit_count = df.groupby("user_id")["_debit_count_30d_raw"].shift(1).fillna(0)

    # Rolling sum de montos de debito en 30d
    raw_sum = (
        df.groupby("user_id")
        .rolling("30D", on="created_at")["_debit_amount"]
        .sum()
        .droplevel(0)
        .reindex(df.index)
    )
    df["_debit_amount_30d_raw"] = raw_sum
    debit_amount = df.groupby("user_id")["_debit_amount_30d_raw"].shift(1).fillna(0)

    # Limpiar columnas temporales
    df.drop(columns=["_is_debit", "_debit_amount", "_debit_count_30d_raw", "_debit_amount_30d_raw"], inplace=True)

    return debit_count, debit_amount
```

**Implementacion F27 (credit_flow_ratio):**

```python
def _credit_flow_ratio(self, df, debit_amount_30d):
    """Ratio entre monto de debitos y gasto prepago en 30d.

    credit_flow_ratio = debit_amount_30d / (prepaid_spend_30d + 0.01)
    Valores altos indican uso desproporcionado de debitos vs prepago.
    """
    df["_is_prepaid"] = (df["category"] == "prepaid").astype(np.int8)
    df["_prepaid_amount"] = df["amount"] * df["_is_prepaid"]

    raw_prepaid = (
        df.groupby("user_id")
        .rolling("30D", on="created_at")["_prepaid_amount"]
        .sum()
        .droplevel(0)
        .reindex(df.index)
    )
    df["_prepaid_30d_raw"] = raw_prepaid
    prepaid_spend_30d = df.groupby("user_id")["_prepaid_30d_raw"].shift(1).fillna(0)

    df.drop(columns=["_is_prepaid", "_prepaid_amount", "_prepaid_30d_raw"], inplace=True)

    return debit_amount_30d / (prepaid_spend_30d + 0.01)
```

### Grupo G: Rol/Staff (3 features) -- NUEVO

| # | Codigo | Nombre | Formula | Notas |
|---|--------|--------|---------|-------|
| F28 | `is_staff` | Es staff | `role.isin(["manager", "operator", "teacher"]).astype(np.int8)` | Binaria: 1 si el rol es manager, operator o teacher |
| F29 | `paid_by_manager` | Pagado por manager | Flag booleana desde SQL (campo derivado en extraccion) | Binaria: 1 si el pago fue realizado por un manager |
| F30 | `staff_amount_zscore` | Z-score monto por rol | `(amount - mu_role_currency) / sigma_role_currency` | Z-score per-currency Y per-role; estadisticas de `fit()` |

**Columnas requeridas en extraccion:** `role` (string), `paid_by_manager` (boolean), `currency` (string para F30)

**Implementacion F28:**

```python
def _is_staff(self, df):
    """Flag binaria para roles de staff (manager, operator, teacher)."""
    STAFF_ROLES = {"manager", "operator", "teacher"}
    return df["role"].isin(STAFF_ROLES).astype(np.int8)
```

**Implementacion F29:**

```python
def _paid_by_manager(self, df):
    """Flag binaria: pago realizado por un manager.

    Se extrae directamente del SQL de extraccion como campo derivado.
    """
    return df["paid_by_manager"].fillna(False).astype(np.int8)
```

**Implementacion F30 (z-score per-currency AND per-role):**

```python
def _staff_amount_zscore(self, df):
    """Z-score del monto normalizado por currency Y por role.

    Se computa como (amount - mu_role_currency) / sigma_role_currency.
    Las estadisticas (mu, sigma) se calculan en fit() sobre el training set,
    agrupadas por (role, currency) para evitar distorsion multi-moneda.

    Roles no vistos en train usan estadisticas globales por currency.
    Combinaciones (role, currency) con sigma = 0 reciben zscore = 0.
    """
    # Crear clave compuesta
    keys = list(zip(df["role"], df["currency"]))

    mu_vals = pd.Series(
        [self._role_stats.get(k, {}).get("mu", self._global_avg_amount) for k in keys],
        index=df.index,
    )
    sigma_vals = pd.Series(
        [self._role_stats.get(k, {}).get("sigma", 1.0) for k in keys],
        index=df.index,
    )

    # Evitar division por cero: sigma = 0 -> zscore = 0
    zscore = (df["amount"] - mu_vals) / sigma_vals.replace(0, np.nan)
    return zscore.fillna(0)
```

**Estadisticas de fit() para F30:**

```python
# En fit():
def _fit_role_stats(self, df_train):
    """Calcula mu y sigma de amount por (role, currency) en training set.

    Almacena en self._role_stats como:
    {
        ("manager", "USD"): {"mu": 45.2, "sigma": 12.3},
        ("operator", "BOB"): {"mu": 120.0, "sigma": 35.1},
        ...
    }
    """
    grouped = df_train.groupby(["role", "currency"])["amount"]
    self._role_stats = {}
    for (role, currency), group in grouped:
        self._role_stats[(role, currency)] = {
            "mu": group.mean(),
            "sigma": group.std(ddof=0),  # poblacional, no muestral
        }
```

### Grupo H: Diversidad Operacional (3 features) -- NUEVO

| # | Codigo | Nombre | Formula | Notas |
|---|--------|--------|---------|-------|
| F31 | `category_entropy_30d` | Entropia de categorias 30d | Shannon entropy: `-sum(p_i * log2(p_i))` sobre distribucion de `category` en 30d | Diversidad de tipos de transaccion; shifted por 1 |
| F32 | `user_reversal_count_30d` | Conteo reversiones 30d | Rolling count de `is_reversal` en 30d, **shifted por 1** | Conteo absoluto (complementa F18 que es ratio) |
| F33 | `user_merchandise_ratio_30d` | Porcentaje merchandise 30d | Rolling mean de `is_merchandise` en 30d, **shifted por 1** | Propension a transacciones de merchandise |

**Implementacion F31 (Shannon entropy, anti-leakage):**

```python
def _category_entropy_30d(self, df):
    """Shannon entropy de la distribucion de categorias de pago en ventana 30d.

    H = -sum(p_i * log2(p_i)) donde p_i es la proporcion de cada categoria.
    Se aplica shift(1) para anti-leakage (excluir fila actual).

    Valores:
    - 0: usuario usa una sola categoria (sin diversidad)
    - alto: usuario usa multiples categorias uniformemente (alta diversidad)

    Primera transaccion del usuario = 0 (sin historia previa).
    """
    def _entropy_for_group(group):
        """Calcula entropia rolling 30d para un grupo de usuario."""
        created_ats = group["created_at"].values
        categories = group["category"].values
        result = []

        for i in range(len(group)):
            if i == 0:
                result.append(0.0)
                continue
            cutoff = created_ats[i] - np.timedelta64(30, "D")
            # Ventana: transacciones ANTERIORES a la actual dentro de 30d
            mask = (created_ats[:i] >= cutoff)
            window_cats = categories[:i][mask]

            if len(window_cats) == 0:
                result.append(0.0)
                continue

            # Calcular distribucion y entropia
            _, counts = np.unique(window_cats, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            result.append(entropy)

        return pd.Series(result, index=group.index)

    return (
        df.groupby("user_id")
        .apply(_entropy_for_group)
        .droplevel(0)
        .reindex(df.index)
        .fillna(0)
    )
```

**Implementacion F32 (conteo reversiones 30d, anti-leakage):**

```python
def _reversal_count_30d(self, df):
    """Conteo absoluto de reversiones en ventana 30d, shifted por 1.

    Complementa F18 (ratio): F32 captura la magnitud absoluta de reversiones,
    mientras F18 captura la proporcion relativa.
    """
    df["is_reversal"] = df["status"].isin(
        ["totally_refunded", "refunded_to_credit"]
    ).astype(np.int8)

    df["_reversal_count_30d_raw"] = (
        df.groupby("user_id")
        .rolling("30D", on="created_at")["is_reversal"]
        .sum()
        .droplevel(0)
        .reindex(df.index)
    )
    shifted = df.groupby("user_id")["_reversal_count_30d_raw"].shift(1).fillna(0)
    df.drop(columns=["_reversal_count_30d_raw"], inplace=True)
    return shifted
```

**Implementacion F33 (merchandise ratio 30d, anti-leakage):**

```python
def _merchandise_ratio_30d(self, df):
    """Proporcion de transacciones de merchandise en ventana 30d, shifted por 1.

    Captura la propension de un usuario a realizar transacciones de tipo
    merchandise (compra de productos) vs otros tipos de transaccion.
    """
    df["_is_merchandise"] = (df["category"] == "merchandise").astype(np.int8)

    df["_merch_ratio_30d_raw"] = (
        df.groupby("user_id")
        .rolling("30D", on="created_at")["_is_merchandise"]
        .mean()
        .droplevel(0)
        .reindex(df.index)
    )
    shifted = df.groupby("user_id")["_merch_ratio_30d_raw"].shift(1).fillna(0)
    df.drop(columns=["_is_merchandise", "_merch_ratio_30d_raw"], inplace=True)
    return shifted
```

---

## Clase FeatureEngineer

### Constante a nivel de modulo

```python
FEATURE_NAMES = [
    # Grupo A: Transaccionales (F01-F05; F06 ELIMINADA)
    "amount",
    "log_amount",
    "amount_usd_ratio",
    "discount_ratio",
    "has_tip",
    # F06 (is_free) ELIMINADA: payment_method='free' excluido del universo
    # Grupo B: Temporales (F07-F11)
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    # Grupo C: Velocidad (F12-F15)
    "user_txn_count_1h",
    "user_txn_count_24h",
    "time_since_last_txn",
    "user_amount_24h",
    # Grupo D: Comportamentales (F16-F20; F21 ELIMINADA)
    "user_distinct_facilities_30d",
    "user_distinct_methods",
    "user_reversal_ratio_30d",
    "user_account_age_days",
    "user_discount_ratio_30d",
    # F21 (user_free_pct_30d) ELIMINADA: payment_method='free' excluido del universo
    # Grupo E: Contextuales (F22-F23)
    "facility_avg_amount",
    "amount_facility_ratio",
    # Grupo F: Credito/Flujo (F24-F27)
    "is_club_credit",
    "user_debit_count_30d",
    "user_debit_amount_30d",
    "credit_flow_ratio",
    # Grupo G: Rol/Staff (F28-F30)
    "is_staff",
    "paid_by_manager",
    "staff_amount_zscore",
    # Grupo H: Diversidad Operacional (F31-F33)
    "category_entropy_30d",
    "user_reversal_count_30d",
    "user_merchandise_ratio_30d",
]

# Validacion a nivel de modulo: usar excepcion en lugar de assert
# 31 features oficiales (F06 y F21 eliminadas por exclusion de free del universo)
if len(FEATURE_NAMES) != 31:
    raise ValueError(f"FEATURE_NAMES debe tener 31 elementos, tiene {len(FEATURE_NAMES)}")
```

### Variantes para analisis de sensibilidad

```python
# Variante sin F18 (user_reversal_ratio_30d) -- para analisis de sensibilidad Gate D
FEATURE_NAMES_30 = [f for f in FEATURE_NAMES if f != "user_reversal_ratio_30d"]
if len(FEATURE_NAMES_30) != 30:
    raise ValueError(f"FEATURE_NAMES_30 debe tener 30 elementos, tiene {len(FEATURE_NAMES_30)}")

# Variante legacy 21 features (F01-F23 sin F06/F21) -- para comparacion con diseno original
FEATURE_NAMES_21 = FEATURE_NAMES[:19]  # F01-F05, F07-F20 (sin F06, F21)
if len(FEATURE_NAMES_21) != 21:
    raise ValueError(f"FEATURE_NAMES_21 debe tener 21 elementos, tiene {len(FEATURE_NAMES_21)}")
```

**Uso de variantes:**

- `FEATURE_NAMES` (31): Modelo principal con catalogo completo (F06 y F21 eliminadas por exclusion de free).
- `FEATURE_NAMES_30` (30): Sensibilidad -- modelo sin F18 (circularidad con proxy). Si `delta AUC >= 0.02`, Gate D no pasa.
- `FEATURE_NAMES_21` (21): Comparacion retrocompatible con el diseno inicial (grupos A-E sin F06/F21).

### Interfaz de la clase

```python
class FeatureEngineer:
    """Genera las 31 features oficiales del catalogo (8 grupos, F06/F21 eliminadas).

    Patron fit/transform para evitar leakage:
    - fit() aprende estadisticas del training set.
    - transform() aplica las transformaciones sin reaprender.
    """

    def __init__(self):
        self._global_avg_amount: float = None
        self._facility_avg_amount: dict = None  # {facility_id: avg_amount}
        self._user_created_at: dict = None       # {user_id: users.created_at}
        self._role_stats: dict = None             # {(role, currency): {"mu": ..., "sigma": ...}}
        self._fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "FeatureEngineer":
        """Aprende estadisticas del training set.

        Calcula:
        - global_avg_amount: media global de amount en train
        - facility_avg_amount: media de amount por facility_id en train
        - user_created_at: users.created_at por user_id (desde JOIN en extraccion)
        - role_stats: mu y sigma de amount por (role, currency) en train
        """
        self._global_avg_amount = df_train["amount"].mean()
        self._facility_avg_amount = (
            df_train.groupby("facility_id")["amount"].mean().to_dict()
        )
        # user_created_at proviene de la columna user_created_at
        # (JOIN con users.created_at en la extraccion SQL)
        self._user_created_at = (
            df_train.groupby("user_id")["user_created_at"].first().to_dict()
        )
        # role_stats: mu y sigma por (role, currency) para F30
        self._fit_role_stats(df_train)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera las 31 features. Requiere fit() previo."""
        if not self._fitted:
            raise RuntimeError("Llamar fit() antes de transform()")
        # 1. Ordenar por user_id, created_at (prerequisito rolling)
        df = df.sort_values(["user_id", "created_at"]).reset_index(drop=True)
        # 2. Grupo A: Transaccionales
        # 3. Grupo B: Temporales
        # 4. Grupo C: Velocidad (rolling)
        # 5. Grupo D: Comportamentales (cumulative + shift)
        # 6. Grupo E: Contextuales (lookup)
        # 7. Grupo F: Credito/Flujo (rolling + flags)
        # 8. Grupo G: Rol/Staff (flags + z-score)
        # 9. Grupo H: Diversidad Operacional (entropy + rolling)
        # 10. Seleccionar solo FEATURE_NAMES + metadata
        return df[FEATURE_NAMES + ["id", "user_id", "created_at", "status"]]

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform en un solo paso (solo para train)."""
        return self.fit(df_train).transform(df_train)

    def save(self, path: str) -> None:
        """Serializar con joblib (incluye dicts, avg y role_stats)."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        """Cargar instancia previamente guardada."""
        return joblib.load(path)

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna la lista oficial de 31 features."""
        return FEATURE_NAMES.copy()
```

### Descomposicion en FeatureGroup strategies (SRP)

> **Violacion SRP detectada:** `FeatureEngineer` calcula los 8 grupos de features en una sola clase monolitica. Esto viola el Principio de Responsabilidad Unica: cada grupo tiene logica distinta (vectorizada, rolling, cumulative, lookup, z-score, entropy) y deberia poder testearse y modificarse independientemente.

**Descomposicion recomendada (Strategy pattern):**

| Clase | Responsabilidad | Features |
|-------|----------------|----------|
| `TransactionalFeatures` | Transformaciones aritmeticas sobre montos | F01-F06 |
| `TemporalFeatures` | Codificacion ciclica y flags temporales | F07-F11 |
| `VelocityFeatures` | Rolling windows con anti-leakage | F12-F15 |
| `BehavioralFeatures` | Rolling nunique, reversal ratio, account age | F16-F21 |
| `ContextualFeatures` | Lookup de estadisticas por facility | F22-F23 |
| `CreditFlowFeatures` | Flags de credito y rolling debitos/prepago | F24-F27 |
| `StaffRoleFeatures` | Flags de rol y z-score per-currency/per-role | F28-F30 |
| `OperationalDiversityFeatures` | Entropia, conteo reversiones, merchandise | F31-F33 |
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
            CreditFlowFeatures(),
            StaffRoleFeatures(),
            OperationalDiversityFeatures(),
        ]
```

Cada grupo tiene sus propios tests unitarios, lo que facilita el ciclo TDD red-green-refactor por grupo.

---

## Columnas requeridas en la extraccion SQL

Las nuevas features (F24-F33) requieren columnas adicionales en la extraccion desde ClickHouse:

| Columna | Origen | Features que la usan |
|---------|--------|---------------------|
| `club_credit_flag` | `reservations.club_credit_flag` | F24 |
| `category` | `reservations.category` | F25, F26, F27, F31, F33 |
| `role` | `users.role` (via JOIN) | F28, F30 |
| `paid_by_manager` | Derivado en SQL: logica de pago por manager | F29 |
| `currency` | `reservations.currency` | F30 |
| `user_created_at` | `users.created_at` (via JOIN) | F19 |

**Nota:** El script de extraccion (`scripts/extract_full_dataset.py`) debe actualizarse para incluir estas columnas en el SELECT y en el JOIN con la tabla `users`.

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

- Rolling 30D (features F16, F18, F20, F21, F25, F26, F27, F31, F32, F33) requiere al menos 30 dias de historia previa.
- Para train (Ene 2025): warm_history = Diciembre 2024 completo.
- Para val (Jul 2025): cola de train = ultimos 30 dias de Junio 2025.
- Para test (Sep 2025): cola de val = ultimos 30 dias de Agosto 2025.

---

## Estimaciones de rendimiento (~3.1M filas, split mas grande)

| Grupo | Tiempo estimado | Nota |
|-------|----------------|------|
| A) Transaccionales + B) Temporales | < 2s | Operaciones vectorizadas puras |
| C) Rolling count/sum (Velocidad) | 2-5 min | groupby + rolling es costoso en pandas |
| D) Comportamentales (nunique 30d, reversal, account age) | 1-3 min | Rolling nunique 30d es mas costoso que cumulative |
| E) Contextuales | < 1s | Map + aritmetica |
| F) Credito/Flujo (rolling debitos/prepago) | 1-2 min | Rolling sum/count adicionales sobre category |
| G) Rol/Staff | < 5s | Flags binarias + z-score vectorizado (lookup) |
| H) Diversidad Operacional (entropy + rolling) | 2-4 min | Entropy 30d requiere loop por fila; rolling count/mean |
| **Total por split** | **8-18 min** | Dominado por rolling ventanas y entropy |

### Memoria pico

~5-7 GB durante las operaciones de rolling groupby. El DataFrame combinado (split + warm history) puede alcanzar ~3.5M filas para train, con columnas adicionales para category, role, currency.

### Optimizaciones recomendadas

- **F31 (entropy):** El loop por fila es O(n*w). Para produccion, considerar implementacion en Cython o Numba.
- **F16 (distinct facilities 30d):** Tambien O(n*w). Alternativa: usar expanding con techo de 30d en ventanas discretas.
- **Paralelismo:** Los 8 grupos son independientes (excepto dependencias de columnas intermedias). Pueden ejecutarse en paralelo con `joblib.Parallel` o similar.

---

## Reglas anti-leakage

Estas reglas son **obligatorias** y constituyen el Gate B:

1. **Orden temporal:** Siempre ordenar por `user_id, created_at` antes de cualquier operacion rolling o cumulative.

2. **Exclusion de la fila actual:** Rolling count/sum usa `-1` / `-self.amount`. Cumulative nunique usa `shifted` (append current AFTER counting). Reversal ratio, debit count/amount, entropy, merchandise ratio usan `shift(1)` dentro del grupo.

3. **No usar `status` como feature directa:** El campo `status` solo se usa en features F18 y F32 de forma derivada (`is_reversal`), y con shift. No se usa directamente como predictor. Esta es la razon del analisis de sensibilidad obligatorio.

4. **No informacion futura:** Ninguna ventana mira hacia adelante. `closed='left'` o equivalente aplica en todas las ventanas rolling.

5. **No reinicializar historias en limites de split:** Las features de val y test DEBEN arrastrar historia del split anterior via warm history. Cortar la historia en el limite del split crearia features artificialmente bajas.

6. **Estadisticas fit solo en train:** `global_avg_amount`, `facility_avg_amount`, `user_created_at` y `role_stats` se calculan exclusivamente sobre el training set.

7. **Z-score per-currency AND per-role (F30):** Las estadisticas mu/sigma se segmentan por la combinacion `(role, currency)` para evitar que la mezcla de monedas distorsione el z-score. Combinaciones no vistas en train usan fallback a estadisticas globales.

8. **Columna `category` no es proxy:** La columna `category` (debit, prepaid, merchandise, etc.) describe el tipo de transaccion, NO el resultado (status). Es segura como feature base.

---

## Tests de validacion (Gate B)

> **Enfoque TDD (red-green-refactor):** Estos tests definen el CONTRATO que `FeatureEngineer` (y sus FeatureGroups) debe satisfacer. El orden de trabajo es:
>
> 1. **Red:** Escribir TODOS estos tests primero. Ejecutarlos y verificar que fallan (porque la implementacion aun no existe).
> 2. **Green:** Implementar la logica de feature engineering, grupo por grupo, hasta que cada test pase.
> 3. **Refactor:** Una vez que todos los tests pasan, refactorizar la implementacion (extraer metodos, mejorar performance) sin romper ningun test.
>
> Los tests NO son verificaciones post-hoc; son la especificacion ejecutable que guia la implementacion.

Estos 14 tests deben pasar antes de continuar a Fase 4:

### 1. Primera transaccion por usuario tiene count features == 0

```python
def test_first_txn_counts_zero():
    """La primera transaccion de cada usuario debe tener
    user_txn_count_1h == 0, user_txn_count_24h == 0,
    user_debit_count_30d == 0, y user_reversal_count_30d == 0."""
    first_txns = df_features.groupby("user_id").first()
    assert (first_txns["user_txn_count_1h"] == 0).all()
    assert (first_txns["user_txn_count_24h"] == 0).all()
    assert (first_txns["user_debit_count_30d"] == 0).all()
    assert (first_txns["user_reversal_count_30d"] == 0).all()
```

### 2. Primera transaccion tiene distinct facilities == 0

```python
def test_first_txn_distinct_facilities_zero():
    """Antes de su primera transaccion, un usuario no ha visto ninguna facility."""
    first_txns = df_features.groupby("user_id").first()
    assert (first_txns["user_distinct_facilities_30d"] == 0).all()
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

### 4. FEATURE_NAMES tiene exactamente 33 entradas

```python
def test_feature_names_count():
    assert len(FEATURE_NAMES) == 33
    assert len(set(FEATURE_NAMES)) == 33  # sin duplicados
```

### 5. Variantes tienen las dimensiones correctas

```python
def test_feature_name_variants():
    """Verificar que FEATURE_NAMES_32 y FEATURE_NAMES_23 tienen las dimensiones correctas."""
    assert len(FEATURE_NAMES_32) == 32
    assert "user_reversal_ratio_30d" not in FEATURE_NAMES_32
    assert len(FEATURE_NAMES_23) == 23
    assert all(f in FEATURE_NAMES for f in FEATURE_NAMES_23)
    assert all(f in FEATURE_NAMES for f in FEATURE_NAMES_32)
```

### 6. No hay NaN en features despues de transform

```python
def test_no_nans():
    """Todas las 31 features deben estar completas despues de transform."""
    for col in FEATURE_NAMES:
        assert df_features[col].isna().sum() == 0, f"NaN encontrados en {col}"
```

### 7. Matriz de correlacion (warn si |r| > 0.95)

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

### 8. Estadisticas de features para Tabla 3.7

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
    assert stats.loc["is_club_credit", "max"] <= 1
    assert stats.loc["is_staff", "max"] <= 1
    assert stats.loc["paid_by_manager", "max"] <= 1
    assert stats.loc["category_entropy_30d", "min"] >= 0
```

### 9. Tests de limites de split

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

### 10. Usuarios cold-start manejados correctamente

```python
def test_cold_start_users():
    """Usuarios que aparecen por primera vez en val/test deben:
    - Tener count features == 0 en su primera transaccion
    - Tener user_account_age_days >= 0
    """
    new_val_users = set(df_val["user_id"]) - set(df_train["user_id"])
    if new_val_users:
        new_first = df_val_features[
            df_val_features["user_id"].isin(new_val_users)
        ].groupby("user_id").first()
        assert (new_first["user_txn_count_1h"] == 0).all()
        assert (new_first["user_account_age_days"] >= 0).all()
```

### 11. Robustez para edge cases (amount=0, discount > amount)

```python
def test_edge_cases():
    """Verificar que features no explotan con edge cases."""
    # amount = 0
    zero_amount = df_features[df_features["amount"] == 0]
    if len(zero_amount) > 0:
        assert np.isfinite(zero_amount["log_amount"]).all()
        assert np.isfinite(zero_amount["discount_ratio"]).all()
        assert np.isfinite(zero_amount["amount_facility_ratio"]).all()
        assert np.isfinite(zero_amount["credit_flow_ratio"]).all()

    # discount > amount (posible en datos reales)
    high_discount = df_features[df_features["discount_ratio"] > 1.0]
    if len(high_discount) > 0:
        assert np.isfinite(high_discount["discount_ratio"]).all()
```

### 12. Features binarias son realmente binarias

```python
def test_binary_features():
    """Las features binarias deben contener solo 0 y 1."""
    binary_features = [
        "has_tip", "is_free", "is_weekend", "is_off_hours",
        "is_club_credit", "is_staff", "paid_by_manager",
    ]
    for col in binary_features:
        unique_vals = set(df_features[col].unique())
        assert unique_vals.issubset({0, 1}), f"{col} tiene valores no binarios: {unique_vals}"
```

### 13. Z-score de staff tiene media cercana a 0 por grupo (role, currency)

```python
def test_staff_zscore_distribution():
    """staff_amount_zscore debe tener media cercana a 0 para cada combinacion
    (role, currency) que existia en el training set."""
    for (role, currency), stats in fe._role_stats.items():
        mask = (df_train_features["role"] == role) & (df_train_features["currency"] == currency)
        subset = df_train_features.loc[mask, "staff_amount_zscore"]
        if len(subset) > 10:
            assert abs(subset.mean()) < 0.1, (
                f"Z-score medio para ({role}, {currency}) = {subset.mean():.4f}, "
                "deberia ser cercano a 0"
            )
```

### 14. Entropy es no-negativa y acotada

```python
def test_entropy_bounds():
    """category_entropy_30d debe ser >= 0 (entropia nunca negativa).
    Ademas, para n categorias distintas, max entropy = log2(n)."""
    assert (df_features["category_entropy_30d"] >= 0).all()
    # Con las categorias conocidas, la entropia maxima teorica es log2(n_categorias)
    n_cats = df_train["category"].nunique()
    max_theoretical = np.log2(n_cats) if n_cats > 0 else 0
    assert (df_features["category_entropy_30d"] <= max_theoretical + 0.01).all()
```

---

## Entregables

| Artefacto | Ruta | Descripcion |
|-----------|------|-------------|
| `engineering.py` | `src/fraud_detector/features/engineering.py` | Clase `FeatureEngineer` + constantes `FEATURE_NAMES`, `FEATURE_NAMES_32`, `FEATURE_NAMES_23` |
| `test_features.py` | `tests/test_features.py` | 14 tests de Gate B |
| `train_features.parquet` | `data/processed/train_features.parquet` | ~3.1M filas x 31 features + metadata |
| `val_features.parquet` | `data/processed/val_features.parquet` | ~1.1M filas x 31 features + metadata |
| `test_features.parquet` | `data/processed/test_features.parquet` | ~2.5M filas x 31 features + metadata |
| `feature_engineer.joblib` | `output/models/feature_engineer.joblib` | Instancia serializada con estadisticas de train (incluye `role_stats`) |
| `feature_statistics.csv` | `output/feature_statistics.csv` | Estadisticas descriptivas de 31 features para Tabla 3.7 |

---

## Diagrama de flujo

```
warm_history.parquet + train_raw.parquet
    |
    v
[FeatureEngineer.fit_transform(train_with_warm)]
    |
    +--> feature_engineer.joblib (estadisticas train: avg, facility_avg,
    |                              user_created_at, role_stats)
    +--> train_features.parquet (solo filas de train, sin warm; 31 features)
    |
    v
[FeatureEngineer.transform(val_with_train_tail)]
    |
    +--> val_features.parquet (solo filas de val; 31 features)
    |
    v
[FeatureEngineer.transform(test_with_val_tail)]
    |
    +--> test_features.parquet (solo filas de test; 31 features)
    |
    v
[Gate B: 14 tests de validacion]
    |
    +--> PASS --> Continuar a Fase 4
    +--> FAIL --> Revisar y corregir antes de continuar
```
