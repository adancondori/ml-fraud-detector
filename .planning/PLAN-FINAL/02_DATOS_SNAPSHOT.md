# Fase 1. Extraccion de Datos y Snapshot

> Sintesis de Plan-CODEX (estrategia de extraccion, warm history, edge cases, reglas ClickHouse) y Plan-CLAUDE (especificacion de DataManager, config, codigo detallado, filtros post-extraccion).

---

## 1. Tabla fuente

| Propiedad | Valor |
|-----------|-------|
| Base / Tabla | `pbp_productionDB_optimized.payments` |
| Engine | `SharedReplacingMergeTree` |
| ORDER BY | `(facility_id, created_at, id)` |
| PARTITION BY | Ninguno |
| Replicacion | PeerDB (columnas `_peerdb_version`, `_peerdb_is_deleted`) |

### Por que FINAL es obligatorio

La tabla es un `ReplacingMergeTree` replicado por PeerDB. Multiples versiones de la misma fila coexisten fisicamente hasta que ClickHouse ejecuta un merge en segundo plano. Sin `FINAL` en el `SELECT`, los conteos quedan inflados con filas duplicadas/obsoletas.

**Implicacion de rendimiento:** `FINAL` encarece la lectura (~2-5x mas lento). Se mitiga extrayendo por split (no el ano completo) y cacheando localmente en Parquet.

---

## 2. SQL canonico del snapshot

```sql
SELECT
    id,
    user_id,
    effective_user_id,
    facility_id,
    facility_name,
    created_at,
    captured_at,
    payment_method,
    gateway,
    source_enum,
    status,
    reservation_paid_out,
    discount,
    tax,
    tip,
    card_brand,
    currency,
    paid_by_manager,
    reversed_id,
    debit_refund,
    _peerdb_version
FROM pbp_productionDB_optimized.payments FINAL
WHERE created_at >= %(start)s
  AND created_at < %(end)s
  AND payment_method != 'reversal'
  AND payment_method != 'free'
  AND user_id != 0
  AND _peerdb_is_deleted = 0
ORDER BY created_at, id
```

Misma consulta para todos los splits y warm history; solo cambian `%(start)s` y `%(end)s`.

---

## 3. Estrategia de extraccion

### Extraccion por split (no full year)

Extraer cada segmento independientemente para:
- Reducir carga sobre ClickHouse (FINAL es costoso)
- Mantener pico de memoria manejable (~1-2 GB por split)
- Permitir reanudacion si una extraccion falla

| Segmento | start | end | N estimado |
|----------|-------|-----|------------|
| Warm history | 2024-12-01 | 2025-01-01 | ~550K (no entra a metricas) |
| Train | 2025-01-01 | 2025-07-01 | ~3,137,086 |
| Validation | 2025-07-01 | 2025-09-01 | ~1,130,118 |
| Test | 2025-09-01 | 2026-01-01 | ~2,517,491 |
| **Total universo** | | | **~6,784,695** |

### Warm history (2024-12-01 a 2024-12-31)

**Proposito:** Inicializar ventanas rolling de 30 dias para las features de velocidad y comportamiento en enero 2025.

**Reglas:**
- Las filas de warm history **nunca** entran al universo evaluado ni a metricas finales.
- Solo sirven para calcular rolling windows en el borde train (enero 2025) y bordes val/test.
- Se persiste como `warm_raw.parquet` con el mismo SQL canonico.

### Estrategia de account age (Feature #18)

`user_account_age_days` usa `first_txn` del **training set** como valor fitted:

```python
# En fit():
self.user_first_txn = df_train.groupby("user_id")["created_at"].min().to_dict()
```

- Usuarios no vistos en training: se usa su primera transaccion en el split actual como fallback.
- Se documenta la limitacion de censura izquierda (no se conoce historial anterior a 2025-01-01 salvo warm).
- No se requiere dimension `users` externa; `first_txn` se deriva del propio snapshot.

---

## 3.1. Test Contracts (TDD) — Escribir ANTES de implementar

> **Enfoque TDD:** Los siguientes tests definen el CONTRATO que la capa de datos debe satisfacer.
> Escribir estos tests PRIMERO (red), luego implementar el codigo para que pasen (green), luego refactorizar.

```python
# tests/test_data_manager.py

def test_extract_returns_correct_columns():
    """El DataFrame extraido contiene exactamente las columnas del SQL canonico."""
    expected_cols = {"id", "user_id", "effective_user_id", "facility_id", "facility_name", "created_at",
                     "captured_at", "payment_method", "gateway", "source_enum",
                     "status", "reservation_paid_out", "discount", "tax", "tip",
                     "card_brand", "currency", "paid_by_manager",
                     "reversed_id", "debit_refund", "_peerdb_version"}
    df = extractor.extract("2025-01-01", "2025-02-01")
    assert set(df.columns) == expected_cols

def test_extract_validates_row_counts():
    """Los conteos estan dentro de +-1% de los valores esperados."""
    df = extractor.extract("2025-01-01", "2025-07-01")
    expected = 3_137_086
    assert abs(len(df) - expected) / expected < 0.01

def test_proxy_labels_strict_matches_expected():
    """Proxy estricto marca exactamente totally_refunded y refunded_to_credit."""
    labels = ProxyLabeler.assign(df, "strict")
    positive = df.loc[labels]
    assert set(positive["status"].unique()) == {"totally_refunded", "refunded_to_credit"}

def test_proxy_labels_wide_includes_partial():
    """Proxy amplio incluye ademas partially_refunded."""
    labels = ProxyLabeler.assign(df, "wide")
    positive = df.loc[labels]
    assert "partially_refunded" in positive["status"].unique()

def test_downcast_preserves_large_ids():
    """id y reversed_id no se truncan a int32 (valores > 2^31 sobreviven)."""
    df_cast = validator.downcast(df_with_large_ids)
    assert df_cast["id"].max() > 2**31

def test_manifest_contains_required_fields():
    """El JSON sidecar contiene todos los campos requeridos."""
    required = {"name", "start_date", "end_date", "row_count",
                "extracted_at", "checksum_sha256", "columns",
                "status_distribution", "memory_mb"}
    manifest = json.loads(Path("output/manifests/train_manifest.json").read_text())
    assert required.issubset(manifest.keys())

def test_atomic_write_survives_interruption():
    """Si el proceso muere durante escritura, no queda Parquet corrupto en ruta final."""
    # Simular interrupcion: verificar que target_path no existe si tmp falla
    ...
```

---

## 4. Especificacion de DataManager

### Archivo: `src/fraud_detector/data/loader.py`

**Eliminar:** Clase `DataLoader` completa, funcion `split_data()`.

**Nueva clase:** `DataManager` (fachada que orquesta clases especializadas)

### Descomposicion en clases (Principio de Responsabilidad Unica — SRP)

> **Violacion SRP detectada:** El diseno original concentra extraccion, validacion, downcast, proxy labeling y manifests en una sola clase `DataManager`. Esto viola SRP y dificulta el testing unitario.

**Descomposicion recomendada:**

| Clase | Responsabilidad unica | Archivo |
|-------|----------------------|---------|
| `ClickHouseExtractor` | Conexion a ClickHouse, ejecucion de queries, retry logic | `data/extractors.py` |
| `DataValidator` | Validacion de duplicados, NULLs, rangos, dominios, downcast | `data/validators.py` |
| `ProxyLabeler` | Asignacion de proxy labels (metodo estatico/utilidad) | `data/proxy.py` |
| `ManifestWriter` | Generacion de JSON sidecars con metadata de extraccion | `data/manifest.py` |
| `DataManager` | **Fachada** que orquesta las clases anteriores en el flujo extract-validate-save | `data/loader.py` |

Cada clase tiene tests independientes, permitiendo verificar cada responsabilidad en aislamiento.

### Inversion de Dependencias (DIP)

> **Violacion DIP detectada:** `DataManager` crea internamente su conector ClickHouse, lo que impide inyectar un mock para testing.

**Correccion:** Inyectar el conector via constructor:

```python
class DataManager:
    def __init__(self, settings: Settings, extractor: ClickHouseExtractor = None):
        self._settings = settings
        # DIP: aceptar extractor inyectado; crear uno por defecto solo si no se provee
        self._extractor = extractor or ClickHouseExtractor(settings)
        self._validator = DataValidator()
        self._labeler = ProxyLabeler(settings)
        self._manifest_writer = ManifestWriter(settings)
```

Esto permite en tests:

```python
def test_extract_with_mock():
    mock_extractor = MockClickHouseExtractor(fake_df)
    dm = DataManager(settings, extractor=mock_extractor)
    result = dm.extract_from_clickhouse()
    # Verifica logica sin depender de ClickHouse real
```

### Constantes de calidad

```python
REQUIRED_NON_NULL = ["id", "user_id", "facility_id", "amount", "created_at", "status"]
SAFE_INT32_COLS = ["user_id", "facility_id"]
SAFE_FLOAT32_COLS = ["amount", "technology_fee", "tax", "tip", "discount"]
# Columnas de contexto para analisis post-hoc (no features del modelo)
CONTEXT_COLS = ["currency", "paid_by_manager", "effective_user_id"]
```

**Nota:** `id` y `reversed_id` NO se downcaestean a int32 porque pueden exceder 2^31.

### Metodos

| Metodo | Responsabilidad |
|--------|-----------------|
| `__init__(self, settings)` | Almacena settings; conector ClickHouse lazy |
| `extract_from_clickhouse()` | Extrae 4 segmentos (warm + 3 splits). Para cada uno: extract, validate, downcast, save parquet + manifest JSON sidecar |
| `_validate_extraction(df, name)` | Verifica duplicados en `id`, NULLs en columnas criticas, `amount > 0`, distribucion de `status`, uso de memoria. **IMPORTANTE:** Usar `raise ValueError(...)` en lugar de `assert` para validaciones — `assert` se desactiva con `python -O` y no es apropiado para logica de validacion de datos |
| `_downcast(df)` | Downcast selectivo: float64->float32 (solo SAFE_FLOAT32_COLS), int64->int32 (solo SAFE_INT32_COLS despues de verificar max < 2^31) |
| `load_splits()` | Retorna tupla de 3 DataFrames (train, val, test) |
| `load_split(name)` | Retorna un DataFrame con verificacion de existencia y error claro |
| `assign_proxy_labels(df, proxy_type)` | **Metodo estatico.** Retorna Series booleana segun proxy_type ("strict" o "wide") |
| `_save_manifest(name, start, end, df)` | Guarda JSON sidecar con metadata de extraccion |
| `close()` | Desconecta de ClickHouse |

### Filtros post-extraccion (en Python)

```python
# Excluir transacciones de sistema/anonimas
df = df[df["user_id"] > 0]

# Excluir filas soft-deleted por PeerDB
df = df[df.get("_peerdb_is_deleted", 0) == 0]

# Eliminar columna is_fraud (usa definicion distinta al proxy de la tesis)
df = df.drop(columns=["is_fraud"], errors="ignore")
```

**Nota critica:** La columna `is_fraud` no existe en la tabla ClickHouse nativa. Si algun codigo previo la genera durante la extraccion, se elimina. El proxy se asigna exclusivamente via `assign_proxy_labels()`.

> **assert vs. excepciones:** En toda la implementacion de DataManager y clases asociadas, las validaciones de datos deben usar `raise ValueError(msg)` o `raise RuntimeError(msg)` en lugar de `assert`. La sentencia `assert` se desactiva cuando Python se ejecuta con la flag `-O` (optimizado), lo que dejaria las validaciones inactivas silenciosamente. Ejemplo:
>
> ```python
> # MAL — assert se desactiva con python -O
> assert df["id"].duplicated().sum() == 0, "Duplicados encontrados"
>
> # BIEN — excepcion explicita, siempre activa
> if df["id"].duplicated().sum() > 0:
>     raise ValueError(f"Duplicados encontrados en {name}: {df['id'].duplicated().sum()}")
> ```

### Escrituras atomicas

```python
# Escribir a archivo temporal, luego renombrar (atomico en filesystem)
tmp_path = target_path.with_suffix(".tmp.parquet")
df.to_parquet(tmp_path, engine="pyarrow", compression="snappy")
tmp_path.rename(target_path)
```

### Manifest JSON sidecar

Un archivo JSON por cada segmento extraido:

```json
{
  "name": "train",
  "start_date": "2025-01-01",
  "end_date": "2025-07-01",
  "row_count": 3137086,
  "extracted_at": "2026-03-16T14:30:00Z",
  "columns": ["id", "user_id", "..."],
  "clickhouse_host": "host.clickhouse.cloud",
  "filters_used": "payment_method != reversal/free, user_id != 0, _peerdb_is_deleted = 0",
  "checksum_sha256": "abc123...",
  "status_distribution": {"captured": 2800000, "totally_refunded": 200000, "...": "..."},
  "memory_mb": 1250
}
```

### Retry logic con tenacity

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
def _execute_query(self, query, params):
    ...
```

3 intentos con backoff exponencial (4s, 8s, 16s... max 60s). Si la extraccion de un split falla, se reintenta ese split sin afectar los demas.

---

## 5. Cambios en config.py

### Agregar (temporal splits y proxy)

```python
# Temporal Split Boundaries
train_start: str = "2025-01-01"
train_end: str = "2025-07-01"      # exclusive
val_end: str = "2025-09-01"         # exclusive
test_end: str = "2026-01-01"        # exclusive
warm_start: str = "2024-12-01"      # warm history

# Proxy Label Definitions
strict_proxy_statuses: str = "totally_refunded,refunded_to_credit"
wide_proxy_statuses: str = "totally_refunded,refunded_to_credit,partially_refunded"
```

### Eliminar (parametros supervisados)

`model_type`, `test_size`, `validation_size`, `mlflow_*`, `use_gpu`, `fraud_threshold`, `high_risk_threshold`, `auto_decline_threshold`, `api_host`, `api_port`, `database_url`, `use_smote`, `smote_sampling_strategy`, `use_class_weights`, `use_temporal_split`, `temporal_split_date`, `embargo_days`, `enable_drift_detection`, `drift_threshold`, `performance_degradation_threshold`.

### Properties para parsing

```python
@property
def strict_proxy_list(self) -> List[str]:
    return [s.strip() for s in self.strict_proxy_statuses.split(",")]

@property
def wide_proxy_list(self) -> List[str]:
    return [s.strip() for s in self.wide_proxy_statuses.split(",")]
```

### Directorios de output

```python
@property
def processed_dir(self) -> Path:
    return self.project_root / self.data_dir / "processed"

@property
def manifests_dir(self) -> Path:
    return self.project_root / self.output_dir / "manifests"
```

---

## 6. Cambios en requirements.txt

### Eliminar

`imbalanced-learn`, `xgboost`, `lightgbm`, `mlflow`, `optuna`, `evidently`, `pandera`, `great-expectations`, `lime`, `plotly`, `category-encoders`, `clickhouse-driver` (se usa `clickhouse-connect`), `requests`, `pandas-stubs`, `types-requests`.

### Agregar

```
scipy>=1.11.0       # Mann-Whitney U, bootstrap, KS test
pyarrow>=14.0.0     # Parquet I/O eficiente
tenacity>=8.0.0     # Retry logic para ClickHouse
```

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

---

## 7. Conteos esperados y validacion

### Conteos objetivo

| Concepto | Valor esperado | Tolerancia |
|----------|---------------|------------|
| N total depurado | ~6,784,695 | +-1% |
| Proxy estricto | ~429,442 (6.33%) | +-1% |
| Proxy amplio | ~512,609 (7.55%) | +-1% |
| Train (Ene-Jun 2025) | ~3,137,086 | +-1% |
| Val (Jul-Ago 2025) | ~1,130,118 | +-1% |
| Test (Sep-Dic 2025) | ~2,517,491 | +-1% |

### Validaciones por split

1. **Duplicados:** `id` debe ser unico despues de `FINAL`. Si hay residuales, deduplicar y documentar.
2. **NULLs:** Cero NULLs en `REQUIRED_NON_NULL` = `[id, user_id, facility_id, amount, created_at, status]`.
3. **Amount > 0:** Verificar (deberia estar garantizado por el query, pero confirmar).
4. **Status distribution:** Loggear conteo por `status` y comparar con expectativas.
5. **Uso de memoria:** Loggear MB por split. Estimado: ~400-600 MB por split despues de downcast.
6. **Rango temporal:** `min(created_at)` y `max(created_at)` deben estar dentro de las boundaries del split.
7. **Dominios categoricos:** Verificar cardinalidad de `gateway`, `payment_method`, `source_enum`, `card_brand`.
8. **user_id > 0:** Confirmar que no hay `user_id=0` post-filtro.
9. **Strings vacios vs. NULLs:** Detectar y documentar si existen columnas con strings vacios.
10. **Monotonia de created_at:** Verificar que `ORDER BY created_at, id` se preserva.

---

## 8. Entregables

| Entregable | Ruta | Descripcion |
|------------|------|-------------|
| Train raw | `data/processed/train_raw.parquet` | Transacciones Ene-Jun 2025, snappy |
| Val raw | `data/processed/val_raw.parquet` | Transacciones Jul-Ago 2025, snappy |
| Test raw | `data/processed/test_raw.parquet` | Transacciones Sep-Dic 2025, snappy |
| Warm history | `data/processed/warm_raw.parquet` | Transacciones Dic 2024, snappy |
| Dataset manifest | `output/manifests/dataset_manifest.json` | Metadata consolidada: conteos, fechas, filtros |
| SQL snapshot | `output/manifests/query_snapshot.sql` | Copia del SQL ejecutado |
| verify_counts.py | `scripts/verify_counts.py` | Script que verifica conteos contra ClickHouse |

### Espacio estimado en disco

| Archivo | Tamano estimado |
|---------|----------------|
| train_raw.parquet | ~800 MB |
| val_raw.parquet | ~300 MB |
| test_raw.parquet | ~650 MB |
| warm_raw.parquet | ~100 MB |
| **Total** | **~1.8 GB** |

---

## 9. Riesgos y mitigaciones

### Performance de FINAL

`FINAL` encarece la lectura 2-5x.

**Mitigacion:** Extraccion por split, cache local en Parquet, no repetir lecturas completas. Si un split falla, extraer por mes y consolidar.

### Bordes temporales

Las features de val/test pueden quedar mal calculadas si se procesan sin contexto previo.

**Mitigacion:** Warm history para rolling windows. Pruebas especificas del cambio de split (borde 2025-06-30/07-01 y 2025-08-31/09-01).

### Drift del origen

La base productiva sigue recibiendo datos y actualizaciones.

**Mitigacion:** Congelar artefactos Parquet. Registrar fecha/hora de extraccion en manifest. No remezclar con nueva extraccion sin versionar el estudio.

### Duplicados residuales post-FINAL

En raras ocasiones, `FINAL` puede no eliminar todas las versiones si un merge esta en progreso.

**Mitigacion:** Medir `countDistinct(id)` vs `count()`. Si hay duplicados, aplicar deduplicacion secundaria (keep last by `_peerdb_version`) y documentar.

---

## 10. Gate A — Verificacion (10 criterios)

**No pasar a Fase 2 ni Fase 3 hasta que se cumplan TODOS:**

| # | Criterio | Verificacion |
|---|----------|-------------|
| 1 | `Settings` carga sin errores con `.env` actual | `python -c "from config.config import settings; print(settings)"` |
| 2 | Directorios de output creados | `settings.ensure_directories()` sin errores |
| 3 | Conexion ClickHouse funcional | `SELECT 1` exitoso |
| 4 | 4 Parquets extraidos (warm + 3 splits) | Archivos existen y no estan vacios |
| 5 | Conteos dentro de tolerancia +-1% | `verify_counts.py` pasa |
| 6 | Proxy rate estricto ~6.33%, amplio ~7.55% | Calculado post-extraccion |
| 7 | Cero NULLs en columnas criticas | `REQUIRED_NON_NULL` verificado |
| 8 | Cero duplicados en `id` (o < 0.01%) | `df["id"].duplicated().sum()` por split |
| 9 | Columna `is_fraud` eliminada de parquets | `"is_fraud" not in df.columns` |
| 10 | `user_id > 0` verificado | `df["user_id"].min() > 0` por split |
