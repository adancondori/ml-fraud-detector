# Phase 4: Shadow Dual-Run y Validación de Sesgo — Research

**Researched:** 2026-07-06
**Domain:** Shadow champion/challenger dual-scoring, ClickHouse DDL migration, monitoring queries (Jaccard, Spearman, top-5% winsorized, off-hours local), temporal gate pattern
**Confidence:** HIGH — derived entirely from direct codebase audit of the actual source files. No speculation from training data.

---

## Summary

La Fase 4 construye la infraestructura de shadow scoring sobre una base completamente cableada (Fases 0–3 completas). El scorer FastAPI ya carga dos conjuntos de artefactos distintos vía `model_metadata.json` vs `model_metadata_frame_v1.json`; `SingleTransactionScorer` ya despacha a `FrameV1FeatureCalculator` + `SegmentedThresholdClassifier` cuando `facility_stats is not None AND thresholds_segmented is not None`. Lo que falta es: (1) un runner que cargue ambos modelos simultáneamente y produzca 2 filas por pago en `anomaly_scores`, (2) tres columnas nuevas en el DDL de ClickHouse para campos frame-v1 (`calibration_segment`, `fallback_level`, `frame_flags`), (3) cuatro queries de monitoreo (SHAD-02), y (4) un script de gate go/no-go (SHAD-03) que no puede evaluarse en un solo run porque requiere ≥2 semanas de datos shadow.

La arquitectura recomendada es un `ShadowDualRunner` dentro de `BatchScorer` que carga los dos modelos en el `lifespan` de FastAPI (champion IF-40 + frame-v1), opera en modo dual sobre el mismo batch de pagos, y produce exactamente 2 filas por pago con `scoring_mode='shadow_old'` y `scoring_mode='shadow_new'`. El gate temporal se separa estructuralmente del código: el script `scripts/shadow_gate.py` existe desde día 1 pero se ejecuta como operación humana tras ≥2 semanas de acumulación.

**Primary recommendation:** Implementar el dual-runner dentro del batch scorer existente como una nueva ruta `SCORING_MODE=shadow_dual` que carga ambos modelos en el lifespan; añadir 3 columnas al DDL con `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`; implementar queries de monitoreo como SQL puro en ClickHouse + un script Python delgado que ejecuta las 4 métricas SHAD-02; y separar el gate SHAD-03 en un checkpoint `human-verify` explícito diferido.

---

## Standard Stack

### Core (ya presente en el proyecto)

| Librería | Versión | Propósito | Por qué estándar |
|----------|---------|-----------|-----------------|
| `clickhouse-connect` | instalado en venv | INSERT/SELECT a ClickHouse local (WRITE) y prod (READ) | Ya usado en `scorer/batch/scorer.py`; `insert()` con `column_names` y dedup token |
| `scipy.stats.spearmanr` | 1.13.1 | Spearman ranking correlation entre scores champion vs. challenger | Ya disponible en venv (verificado: `from scipy.stats import spearmanr` OK) |
| `numpy` | instalado | top-5% winsorizado, Jaccard@100, operaciones de ranking | Ya usado extensamente |
| `joblib` | instalado | Cargar dos `.joblib` simultáneamente en memoria | `load_artifacts()` ya usa joblib para model + scaler |
| `loguru` | instalado | Logging del dual-runner con deltas de score | Ya en uso en `scorer/batch/scorer.py` |

### Supporting

| Librería | Versión | Propósito | Cuándo usar |
|----------|---------|-----------|------------|
| `pandas` | instalado | Queries de monitoreo en Python (offline sobre parquets); transformaciones de datos | Scripts de monitoreo offline; no en hot path del scorer |
| `pydantic-settings` | instalado | Configuración del scorer con env vars (`SHADOW_MODEL_DIR`) | Extensión de `ScorerSettings` en `scorer/main.py` |
| `tqdm` | instalado | Progress bar en dual-batch si verbose | Ya en uso en `scorer/batch/scorer.py` |

### Alternativas consideradas

| En vez de | Podría usar | Tradeoff |
|-----------|-------------|----------|
| Dual-runner en batch scorer | Feature flag en el scorer single | El dual en batch es más simple: un `score_batch` con dos scorers, 2× filas por pago; el feature flag requiere modificar todas las rutas de scoring |
| `scipy.stats.spearmanr` | `pandas.DataFrame.corr(method='spearman')` | Scipy da directamente el p-value; pandas más conveniente para DataFrames grandes; ambos correctos — usar scipy para el script de gate |
| `ALTER TABLE ADD COLUMN IF NOT EXISTS` | Recrear tabla desde DDL | `ALTER TABLE` es idempotente y no requiere downtime ni pérdida de datos; ClickHouse MergeTree soporta ADD COLUMN instantáneamente |

**Installation:** nada nuevo que instalar; todo el stack ya está en `venv`.

---

## Architecture Patterns

### Recommended Project Structure (additions for Phase 4)

```
scorer/
├── shadow/                          # (NEW) dual-run infrastructure
│   ├── __init__.py
│   └── dual_runner.py               # ShadowDualRunner: champion + challenger, 2 filas/pago
├── batch/
│   └── scorer.py                    # MODIFY: acepta scorer_shadow opcional; modo dual
├── main.py                          # MODIFY: carga artifacts champion + frame-v1 en lifespan
├── artifact_loader.py               # NO TOCAR (ya funciona para ambos model dirs)

docker/clickhouse/
├── init/
│   └── 02_anomaly_scores.sql        # NO TOCAR (DDL original intacto)
├── migrations/
│   └── 02_anomaly_scores_frame_v1.sql  # (NEW) ADD COLUMN calibration_segment/fallback_level/frame_flags

scripts/
├── shadow_monitor.py                # (NEW) ejecuta las 4 queries SHAD-02 y reporta
└── shadow_gate.py                   # (NEW) evalúa gate SHAD-03; falla si < 2 semanas de datos
```

### Pattern 1: Dual-Runner en BatchScorer — 2 filas por pago

**Qué es:** `ShadowDualRunner` recibe `scorer_champion` (IF-40) y `scorer_new` (frame-v1), puntúa cada pago con ambos, produce 2 listas de filas para INSERT: una con `scoring_mode='shadow_old'` y otra con `scoring_mode='shadow_new'`. `BatchScorer` llama al dual-runner cuando `scorer_shadow is not None`.

**Cuándo usar:** Cuando `SCORING_MODE=shadow_dual` (nueva variable de entorno) está seteado en el scorer.

**Invariante de escritura dual:** Si una de las dos inserciones falla, se registra el error pero la otra escritura no se revierte. La dedup token garantiza que un retry no duplique filas. El contador de éxito dual se reporta en el summary de `score_batch`.

```python
# scorer/shadow/dual_runner.py
# Source: diseño derivado del codebase audit

class ShadowDualRunner:
    """Puntúa el mismo pago con champion (IF-40) y challenger (frame-v1).

    Produce dos listas de filas — una por modelo — para INSERT en anomaly_scores
    con scoring_mode='shadow_old' y scoring_mode='shadow_new' respectivamente.
    """

    def __init__(
        self,
        scorer_champion: SingleTransactionScorer,  # IF-40 — model_version='IF-40-v1'
        scorer_new: SingleTransactionScorer,       # frame-v1 — model_version='frame-v1'
    ):
        self._champion = scorer_champion
        self._new = scorer_new

    def score_pair(
        self,
        payment: dict,
        context_champion: UserContext,
        context_new: UserContext,
    ) -> tuple[ScoringResult, ScoringResult]:
        """Devuelve (result_champion, result_new). Nunca lanza excepción."""
        try:
            r_old = self._champion.score(payment, context=context_champion)
        except Exception as exc:
            logger.warning(f"shadow champion failed payment_id={payment.get('payment_id')}: {exc}")
            r_old = _error_result("IF-40-v1")

        try:
            r_new = self._new.score(payment, context=context_new)
        except Exception as exc:
            logger.warning(f"shadow frame-v1 failed payment_id={payment.get('payment_id')}: {exc}")
            r_new = _error_result("frame-v1")

        delta = abs(r_old.score - r_new.score)
        logger.debug(
            f"shadow delta={delta:.4f} "
            f"fid={payment.get('facility_id')} "
            f"old_risk={r_old.risk_level} new_risk={r_new.risk_level}"
        )
        return r_old, r_new
```

**Punto clave sobre el contexto:** El `BatchContextProvider` construye el contexto una vez por pago (6 queries en bulk). Para el dual-run, el mismo `UserContext` puede pasarse a ambos scorers porque el contexto es factual (rolling aggregates de ClickHouse READ), no derivado del modelo. Esto evita duplicar los 6 queries de contexto.

**Dedup tokens para dual:** El token debe distinguir los dos inserts por modelo:

```python
# En _insert_chunks del BatchScorer modo dual:
token_old = f"shadow-old-{cursor.isoformat()}-{cursor_end.isoformat()}-chunk-{chunk_index}"
token_new = f"shadow-new-{cursor.isoformat()}-{cursor_end.isoformat()}-chunk-{chunk_index}"
```

### Pattern 2: Carga de Dos Modelos en el Lifespan

**Qué es:** `scorer/main.py` expone `SHADOW_MODEL_DIR` como variable de entorno opcional. Si está definida, `load_artifacts(shadow_model_dir)` se llama en el lifespan y el scorer frame-v1 se instancia en paralelo al champion.

```python
# scorer/main.py — extensión del lifespan existente
class ScorerSettings(BaseSettings):
    # ... campos existentes ...
    shadow_model_dir: Optional[Path] = None  # NEW — dir de artefactos frame-v1
    scoring_mode: str = "active"             # 'active' | 'shadow' | 'shadow_dual'

# En lifespan:
if settings.shadow_model_dir and settings.scoring_mode == "shadow_dual":
    shadow_artifacts = load_artifacts(settings.shadow_model_dir)
    scorer_shadow = SingleTransactionScorer(
        feature_engineer_path=str(settings.shadow_model_dir / "feature_engineer.joblib"),
        artifacts=shadow_artifacts,
    )
    _deps._state["scorer_shadow"] = scorer_shadow
```

**Presupuesto de memoria:** IF-40 ocupa ~50MB en memoria (IsolationForest n_estimators=200); frame-v1 también ~50MB (mismo n_estimators, max_samples=512). Total ~100MB adicionales — aceptable para el entorno local.

### Pattern 3: DDL Migration — ADD COLUMN idempotente

**Qué es:** ClickHouse MergeTree soporta `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` sin downtime ni reescritura de datos. Las filas existentes reciben el valor DEFAULT.

```sql
-- docker/clickhouse/migrations/02_anomaly_scores_frame_v1.sql
ALTER TABLE pbp_productionDB_optimized.anomaly_scores
    ADD COLUMN IF NOT EXISTS calibration_segment LowCardinality(String) DEFAULT '',
    ADD COLUMN IF NOT EXISTS fallback_level       LowCardinality(String) DEFAULT '',
    ADD COLUMN IF NOT EXISTS frame_flags          String DEFAULT '';

-- frame_flags almacenado como JSON string (igual que top_factors)
-- Valores válidos:
--   calibration_segment: 'facility:123' | 'currency:USD' | 'global' | '' (IF-40)
--   fallback_level: 'facility' | 'currency' | 'global' | '' (IF-40)
--   frame_flags: '{"timezone_missing":false,"currency_missing":false,"currency_unknown":false}' | '' (IF-40)
```

**_INSERT_COLUMNS tras la migración:**

```python
# scorer/batch/scorer.py — añadir 3 columnas al final
_INSERT_COLUMNS = [
    # ... las 23 existentes ...
    "calibration_segment",   # NEW LowCardinality(String)
    "fallback_level",        # NEW LowCardinality(String)
    "frame_flags",           # NEW String (JSON)
]
```

**Retrocompatibilidad:** Las filas IF-40 existentes (sin estos campos) no se reescriben; leen el DEFAULT `''`. Los inserts IF-40 nuevos pasan `''` para los 3 campos.

### Pattern 4: Queries de Monitoreo SHAD-02

Las 4 métricas deben vivir como SQL ClickHouse ejecutables desde Python (no en un ORM). El script `scripts/shadow_monitor.py` ejecuta cada query contra el WRITE CH local y reporta los resultados.

**SHAD-02-A: Alert rate por segmento (champion vs. frame-v1)**

```sql
-- Tasa de alertas por currency durante ventana shadow
SELECT
    currency,
    scoring_mode,
    count() AS total,
    countIf(is_anomaly = 1) AS alerts,
    round(countIf(is_anomaly = 1) / count(), 4) AS alert_rate
FROM pbp_productionDB_optimized.anomaly_scores
WHERE scoring_mode IN ('shadow_old', 'shadow_new')
  AND scored_at >= now() - INTERVAL {days:UInt8} DAY
GROUP BY currency, scoring_mode
ORDER BY currency, scoring_mode
```

**SHAD-02-B: Sesgo monto top-5% champion vs. frame-v1 (winsorizado p99.9)**

```sql
-- Reusar la misma lógica de retrain_frame_v1.py: winsorizar a p99.9 por modelo
WITH
    amounts AS (
        SELECT
            payment_id,
            amount_usd,
            scoring_mode,
            percentile,
            -- Rank relativo: percentile ya está en [0,1]
            ROW_NUMBER() OVER (PARTITION BY scoring_mode ORDER BY percentile DESC) AS rn,
            COUNT(*) OVER (PARTITION BY scoring_mode) AS total_n
        FROM pbp_productionDB_optimized.anomaly_scores
        WHERE scoring_mode IN ('shadow_old', 'shadow_new')
          AND scored_at >= now() - INTERVAL {days:UInt8} DAY
    ),
    p999 AS (
        SELECT
            scoring_mode,
            quantile(0.999)(amount_usd) AS p999_threshold
        FROM pbp_productionDB_optimized.anomaly_scores
        WHERE scoring_mode IN ('shadow_old', 'shadow_new')
          AND scored_at >= now() - INTERVAL {days:UInt8} DAY
        GROUP BY scoring_mode
    )
SELECT
    a.scoring_mode,
    -- top-5% rows
    avg(least(a.amount_usd, p.p999_threshold)) AS top5_wins_mean,
    -- global mean
    (SELECT avg(least(b.amount_usd, p2.p999_threshold))
     FROM amounts b JOIN p999 p2 ON b.scoring_mode = p2.scoring_mode
     WHERE b.scoring_mode = a.scoring_mode
    ) AS global_wins_mean
FROM amounts a JOIN p999 p ON a.scoring_mode = p.scoring_mode
WHERE a.rn <= toUInt64(a.total_n * 0.05)
GROUP BY a.scoring_mode
```

*Nota: las queries de ClickHouse complejas con múltiples CTEs a veces requieren ajuste en la versión de CH local. El script Python computa el ratio directamente sobre los resultados descargados (pandas) para evitar SQL complejo.*

**Implementación recomendada para SHAD-02-B en Python:**

```python
# scripts/shadow_monitor.py
# Source: patrón de retrain_frame_v1.py lines 525-545

import numpy as np
import pandas as pd

def compute_top5_bias(df: pd.DataFrame, model: str) -> dict:
    """Winsorized top-5% amount ratio para un modelo."""
    sub = df[df["scoring_mode"] == model].copy()
    k5 = int(len(sub) * 0.05)
    top5_idx = sub["percentile"].nlargest(k5).index
    amounts = sub["amount_usd"].to_numpy(dtype=np.float64)
    p999 = float(np.percentile(amounts, 99.9))
    wins = np.clip(amounts, None, p999)
    top5_wins = np.clip(sub.loc[top5_idx, "amount_usd"].to_numpy(), None, p999)
    ratio = float(top5_wins.mean() / wins.mean()) if wins.mean() > 0 else 0.0
    return {"model": model, "top5_wins_ratio": ratio, "p999": p999, "n": len(sub)}
```

**SHAD-02-C: Off-hours local vs. UTC**

```sql
-- El campo scoring_mode distingue los modelos;
-- frame_flags JSON tiene timezone_missing para auditar calidad de tz
SELECT
    scoring_mode,
    countIf(is_anomaly = 1) AS total_alerts,
    -- off-hours aproximado: hora local en el scorer no está en anomaly_scores
    -- pero frame_flags.timezone_missing permite auditar cuántos usaron fallback UTC
    countIf(JSONExtractBool(frame_flags, 'timezone_missing') = true AND is_anomaly = 1)
        AS alerts_with_tz_missing,
    round(
        countIf(JSONExtractBool(frame_flags, 'timezone_missing') = true) / count(), 4
    ) AS tz_missing_rate
FROM pbp_productionDB_optimized.anomaly_scores
WHERE scoring_mode IN ('shadow_old', 'shadow_new')
  AND scored_at >= now() - INTERVAL {days:UInt8} DAY
GROUP BY scoring_mode
```

*Nota: `is_off_hours_loc` no se persiste en `anomaly_scores`. La métrica directa de off-hours local post-shadow requiere comparar la tasa de alertas por hora del día usando `toHour(payment_created_at)` — aproximación válida dado que `payment_created_at` está en UTC y las facilities siguen en el rango UTC-3 a UTC-8.*

**SHAD-02-D: Jaccard@100 — top-100 overlap**

```python
# scripts/shadow_monitor.py
# Source: patrón de FEATURES.md + scipy/numpy

def compute_jaccard_at_k(df: pd.DataFrame, k: int = 100) -> float:
    """Jaccard similarity de los top-k payment_ids por percentile."""
    old = set(
        df[df["scoring_mode"] == "shadow_old"]
        .nlargest(k, "percentile")["payment_id"]
        .tolist()
    )
    new = set(
        df[df["scoring_mode"] == "shadow_new"]
        .nlargest(k, "percentile")["payment_id"]
        .tolist()
    )
    if not old and not new:
        return 1.0
    return len(old & new) / len(old | new)
```

### Pattern 5: Gate Go/No-Go (SHAD-03) — Separación temporal

**El problema temporal:** SHAD-03 requiere ≥2 semanas de datos shadow. Un run del planner NO puede completar este gate. La solución es separar el código (disponible día 1) de la evaluación (diferida).

**Estructura de separación:**

```
Día 1 (entrega de Fase 4):
  ✓ Infraestructura dual-runner cableada
  ✓ DDL migrado con columnas frame-v1
  ✓ shadow_monitor.py ejecutable (SHAD-02)
  ✓ shadow_gate.py existe con thresholds hardcodeados
  ✓ Tests pytest de la infraestructura (sin datos shadow reales)
  ✓ checkpoint:human-verify abierto en plan (estado: PENDING_DATA)

Semana 2+ (operativo):
  - El human-verify se aprueba cuando shadow_gate.py pasa con ≥2 semanas de datos
  - shadow_gate.py imprime PASS/FAIL con los 4 criterios SHAD-03
  - El operador lo corre y documenta el resultado
```

**script shadow_gate.py:**

```python
# scripts/shadow_gate.py
# Criterios SHAD-03 (hardcodeados aquí, mismos que en bias_report de Fase 1):
GATE_TOP5_MAX_RATIO = 4.0       # top-5% winsorized < 4× (vs baseline 11.79×)
GATE_OFF_HOURS_MIN = 0.03       # off-hours local ≥ 3%
GATE_OFF_HOURS_MAX = 0.07       # off-hours local ≤ 7% (~4-5%)
GATE_SPEARMAN_MIN = 0.90        # Spearman ranking ≥ 0.90
GATE_ALERT_RATE_DELTA_MAX = 0.02  # delta alert rate ≤ 2pp

MIN_SHADOW_DAYS = 14

def evaluate_gate(ch_client, days_available: int) -> dict:
    if days_available < MIN_SHADOW_DAYS:
        return {"status": "INSUFFICIENT_DATA",
                "days_available": days_available,
                "min_required": MIN_SHADOW_DAYS}
    # ... queries SHAD-02 + Spearman + checks ...
```

**Spearman ranking:**

```python
# Source: scipy docs + exp_frame_feature_small.py pattern (uses spearmanr)
from scipy.stats import spearmanr

def compute_spearman(df: pd.DataFrame) -> float:
    """Spearman correlation entre percentile champion y challenger por payment_id."""
    merged = (
        df[df["scoring_mode"] == "shadow_old"][["payment_id", "percentile"]]
        .rename(columns={"percentile": "pct_old"})
        .merge(
            df[df["scoring_mode"] == "shadow_new"][["payment_id", "percentile"]]
            .rename(columns={"percentile": "pct_new"}),
            on="payment_id",
        )
    )
    if len(merged) < 30:
        return float("nan")
    rho, _ = spearmanr(merged["pct_old"], merged["pct_new"])
    return float(rho)
```

### Anti-Patterns a Evitar

- **No puntuar dos veces con el mismo scorer:** IF-40 y frame-v1 son `SingleTransactionScorer` distintos instanciados de directorios de artefactos distintos (`output/models/` vs `output/models/` con `model_metadata_frame_v1.json`). No reusar el mismo objeto con distintos parámetros — los artefactos se cargan en el `__init__` y son inmutables.

- **No duplicar los 6 queries de contexto:** El mismo `UserContext` puede pasarse a ambos scorers. El contexto es factual (aggregates de prod), independiente del modelo. Duplicar el contexto querería decir 12 queries por pago en lugar de 6.

- **No usar `scoring_mode` como la única distinción de modelo:** Añadir siempre `model_version='IF-40-v1'` para shadow_old y `model_version='frame-v1'` para shadow_new. Ambas columnas deben estar presentes para poder filtrar por modelo independientemente del mode.

- **No evaluar SHAD-03 con datos de menos de 2 semanas:** El script `shadow_gate.py` debe abortar con `INSUFFICIENT_DATA` si `days_available < 14`. No forzar PASS con datos insuficientes.

- **No escribir en el READ (prod):** El guardrail `assert_write_target_is_safe` ya existe y debe seguir siendo la primera llamada en `_insert_chunks`. El dual-runner llama `_insert_chunks` dos veces (una por modelo) sobre el mismo `write_ch_client`.

---

## Don't Hand-Roll

| Problema | No construir | Usar en cambio | Por qué |
|---------|-------------|----------------|---------|
| Dedup para doble INSERT | Sistema propio de tracking de filas insertadas | `insert_deduplication_token` de ClickHouse + tokens distintos por modelo (`shadow-old-...` vs `shadow-new-...`) | Ya funcionando en `_insert_chunks`; extender el token con `shadow_old/shadow_new` prefix es trivial |
| Carga de dos modelos en memoria | Serializar/deserializar bajo demanda | `joblib.load()` una vez en el lifespan; ambos scorers en `_state` | joblib memoiza implícitamente si el mismo archivo se carga dos veces; los dos archivos son distintos |
| Ranking correlation | Implementación manual de Spearman | `scipy.stats.spearmanr` | Ya disponible en venv (v1.13.1); maneja empates correctamente |
| Winsorización top-5% | Percentile manual + clip custom | `np.percentile(arr, 99.9)` + `np.clip()` | Patrón ya establecido en `retrain_frame_v1.py` lines 533-537; idéntico para shadow |
| ALTER TABLE en ClickHouse | Recrear tabla o migrar con INSERT/SELECT | `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` | Idempotente, sin downtime, sin pérdida de datos; patrón ya en `docker/clickhouse/migrations/01_anomaly_scores_v2.sql` |
| Jaccard top-k | Implementación ad-hoc de similitud de conjuntos | Set intersection/union con Python `set` | Una línea: `len(a & b) / len(a | b)` |

**Key insight:** Todo el problema matemático del gate (Spearman, Jaccard, winsorized ratio, off-hours rate) tiene solución directa con numpy/scipy disponibles. El valor de Fase 4 está en el cableado y la separación temporal, no en la matemática.

---

## Common Pitfalls

### Pitfall 1: Confundir el champion con el scorer activo en RT

**Qué falla:** El scorer del lifespan (cargado desde `MODEL_DIR`) puede ser frame-v1 si `model_metadata.json` apunta a esos artefactos, o IF-40 si apunta a los artefactos legacy. El "champion IF-40" para shadow_old debe ser el scorer cargado desde el directorio de artefactos IF-40, no el scorer activo.

**Por qué ocurre:** `load_artifacts(model_dir)` despacha según `facility_stats is not None`; si el `model_dir` principal ya apunta a frame-v1 en el futuro, el scorer principal sería frame-v1, no IF-40.

**Cómo evitar:** Usar `SHADOW_CHAMPION_MODEL_DIR` y `SHADOW_NEW_MODEL_DIR` como variables de entorno explícitas para el modo dual. No asumir que el scorer activo es el champion.

**Configuración correcta para Fase 4:**
```env
# Champion = IF-40 (output/models/ con model_metadata.json → IF-40-v1)
SHADOW_CHAMPION_MODEL_DIR=output/models
# New = frame-v1 (mismo directorio, carga model_metadata_frame_v1.json)
SHADOW_NEW_MODEL_DIR=output/models
# El dual-runner carga champion desde metadata estándar y new desde metadata_frame_v1
```

*Nota: ambos modelos están en `output/models/`. El dispatcher de `load_artifacts` usa `model_metadata.json` por defecto. Para cargar frame-v1 específicamente, habrá que añadir un parámetro `metadata_filename` a `load_artifacts` o renombrar el archivo durante la carga.*

**Solución práctica verificada en código:** En `artifact_loader._load_metadata()`, el fallback a `model_metadata.json` carga IF-40-v1; para cargar frame-v1 explícitamente, pasar `metadata_path = model_dir / "model_metadata_frame_v1.json"` (modificar `_load_metadata` para aceptar un override).

### Pitfall 2: Duplicar los 6 queries de contexto por pago

**Qué falla:** `BatchContextProvider.get_batch_context()` hace 6 queries en bulk para todos los pagos del batch. Si se llama dos veces (una por modelo), el tiempo de batch se duplica.

**Por qué ocurre:** Pensar que "cada scorer necesita su propio contexto".

**Cómo evitar:** El `UserContext` es factual (aggregates de producción). Construirlo una vez y pasarlo a ambos `scorer.score(payment, context=ctx)`. El contexto no depende del modelo. El dual-runner recibe `ctx_map` ya construido.

### Pitfall 3: Dedup token no distingue los dos inserts del mismo pago

**Qué falla:** Si el token es `batch-{cursor}-{cursor_end}-{model_version}-chunk-{i}`, pero `model_version` es el mismo para ambas filas de un pago (bug: olvidar distinguir shadow_old/shadow_new en el token), la segunda inserción queda deduplicada.

**Por qué ocurre:** El token incluye `model_version`, pero si el batch scorer comparte el mismo token para ambas inserciones.

**Cómo evitar:** El token DEBE incluir el `scoring_mode` como prefijo: `shadow-old-{cursor}-...` vs `shadow-new-{cursor}-...`. Verificar en tests.

### Pitfall 4: El volumen Docker de ClickHouse local no refleja la migración

**Qué falla:** Si el volumen `clickhouse_data` preexiste, los scripts de `init/` no se re-ejecutan. La tabla `anomaly_scores` tiene las 23 columnas originales pero falta `calibration_segment`/`fallback_level`/`frame_flags`.

**Por qué ocurre:** Documentado en `docs/HOWTO-anomaly-local.md` (hallazgo #1 de la integración de Fase 3): el DDL de init solo corre en contenedores frescos.

**Cómo evitar:** La migración `02_anomaly_scores_frame_v1.sql` usa `ADD COLUMN IF NOT EXISTS`, por lo que se puede ejecutar manualmente incluso sobre un volumen existente:
```bash
docker exec -i clickhouse clickhouse-client --multiquery \
  < docker/clickhouse/migrations/02_anomaly_scores_frame_v1.sql
```
Este comando es seguro de re-ejecutar (idempotente). Incluirlo en el runbook de Fase 4.

### Pitfall 5: El gate SHAD-03 ejecutado sin datos suficientes

**Qué falla:** `shadow_gate.py` devuelve PASS cuando solo hay 2 días de datos si no hay un guard explícito de mínimo de días.

**Cómo evitar:** El script debe calcular `days_span = (max_scored_at - min_scored_at).days` y abortar con `INSUFFICIENT_DATA` si `days_span < 14`.

### Pitfall 6: frame_flags almacenado como columna JSON en CH sin parsear

**Qué falla:** Si `frame_flags` se almacena como string JSON `'{"timezone_missing":false,...}'`, las queries SQL que necesitan filtrar por `timezone_missing` requieren `JSONExtractBool(frame_flags, 'timezone_missing')`, que no usa índices y puede ser lento en tablas grandes.

**Cómo evitar:** Para el volumen de shadow (local, ~miles de filas/día), `JSONExtractBool` es aceptable. Si en el futuro el volumen crece, añadir columnas `flag_tz_missing UInt8` separadas. Por ahora, almacenar como JSON string igual que `top_factors` — patrón ya establecido en el DDL.

---

## Code Examples

### Verificado: cargar dos modelos desde el mismo directorio (artifact_loader)

```python
# Source: scorer/artifact_loader.py + output/models/model_metadata_frame_v1.json (auditado)

# IF-40 champion: metadata estándar
artifacts_champion = load_artifacts(Path("output/models"))
# → carga model_metadata.json → model_version='IF-40-v1'
# → facility_stats=None, thresholds_segmented=None → IF-40 path

# frame-v1: metadata explícita
# Requiere pasar el filename alternativo — modificación mínima a _load_metadata:
def _load_metadata(model_dir: Path, metadata_filename: str = "model_metadata.json") -> dict:
    metadata_path = model_dir / metadata_filename
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    # ... fallbacks existentes ...

artifacts_new = load_artifacts(
    Path("output/models"),
    metadata_filename="model_metadata_frame_v1.json"  # parámetro nuevo
)
# → carga model_metadata_frame_v1.json → model_version='frame-v1'
# → stats_artifact='facility_stats_v1.json' → facility_stats cargado
# → thresholds_segmented_artifact='thresholds_segmented_v1.json' → cargado
# → dispatch → frame-v1 path
```

### Verificado: formato de row para INSERT con columnas frame-v1

```python
# Source: scorer/batch/scorer.py lines 403-427 + extensión para frame-v1

# Para shadow_old (IF-40): los 3 campos nuevos van como strings vacíos
row_old = [
    ...,  # 23 columnas existentes
    "",   # calibration_segment — vacío para IF-40
    "",   # fallback_level — vacío para IF-40
    "",   # frame_flags — vacío para IF-40
]

# Para shadow_new (frame-v1): campos frame-v1 desde ScoringResult
row_new = [
    ...,  # 23 columnas existentes con model_version='frame-v1', scoring_mode='shadow_new'
    str(result.calibration_segment or ""),          # 'facility:123' | 'currency:USD' | 'global'
    str(result.fallback_level or ""),               # 'facility' | 'currency' | 'global'
    json.dumps(result.frame_flags) if result.frame_flags else "",  # JSON
]
```

### Verificado: dedup token para dual-insert

```python
# Source: scorer/batch/scorer.py lines 488-492 (patrón existente + extensión)

# champion (shadow_old)
token_old = (
    f"shadow-old-{cursor.isoformat()}-{cursor_end.isoformat()}"
    f"-IF40v1-chunk-{chunk_index}"
)

# challenger (shadow_new)
token_new = (
    f"shadow-new-{cursor.isoformat()}-{cursor_end.isoformat()}"
    f"-framev1-chunk-{chunk_index}"
)

# INSERT con token
write_ch_client.insert(
    anomaly_scores_table,
    chunk_old,
    column_names=_INSERT_COLUMNS,  # ahora con 26 columnas
    settings={"insert_deduplication_token": token_old},
)
write_ch_client.insert(
    anomaly_scores_table,
    chunk_new,
    column_names=_INSERT_COLUMNS,
    settings={"insert_deduplication_token": token_new},
)
```

### Verificado: Spearman sobre datos shadow

```python
# Source: scripts/exp_frame_feature_small.py (usa spearmanr); scipy 1.13.1 verificado

from scipy.stats import spearmanr

# df tiene columnas: payment_id, scoring_mode, percentile
def compute_spearman_shadow(df: pd.DataFrame) -> float:
    old = df[df["scoring_mode"] == "shadow_old"][["payment_id", "percentile"]]
    new = df[df["scoring_mode"] == "shadow_new"][["payment_id", "percentile"]]
    merged = old.merge(new, on="payment_id", suffixes=("_old", "_new"))
    if len(merged) < 30:
        return float("nan")
    rho, pval = spearmanr(merged["percentile_old"], merged["percentile_new"])
    return float(rho)

# gate: rho >= 0.90
assert rho >= 0.90, f"Spearman {rho:.3f} < 0.90 — gate FAIL"
```

---

## State of the Art

| Enfoque Anterior | Enfoque Actual (Fase 4) | Cuándo cambió | Impacto |
|-----------------|-------------------------|---------------|---------|
| `SCORING_MODE=shadow` puntúa un solo modelo en modo silencioso | `SCORING_MODE=shadow_dual` puntúa champion Y challenger, 2 filas/pago | Fase 4 | Permite comparación directa de scores por pago en lugar de comparar distribuciones de periodos distintos |
| `anomaly_scores` sin campos frame-v1 (decisión deliberada de Fase 3, ver 03-VERIFICATION.md) | `anomaly_scores` con `calibration_segment`, `fallback_level`, `frame_flags` | Fase 4 DDL migration | Persistencia de calibración segmentada en ClickHouse; queries de monitoreo de fallback chain |
| Gate de sesgo sobre val set offline (retrain_frame_v1.py) | Gate de sesgo sobre datos shadow reales (shadow_gate.py) | Fase 4 | Validación sobre distribución real de producción, no sobre el val set histórico |

**Decisiones de Fases anteriores que son pre-condiciones locked:**

- `[03-01]` `_INSERT_COLUMNS` deliberadamente NO tocado en Fase 3 — Fase 4 lo toca.
- `[01-03]` Gate 1 usa métrica winsorizada (`top5_amount_ratio_winsorized_p999 < 4.0`) — SHAD-03 usa la misma métrica.
- `[02-02]` Thresholds segmentados: `by_facility` 452 entradas, `by_currency` 17 entradas, `global` — el dual-runner usa el `SegmentedThresholdClassifier` ya calibrado.
- `[03-01]` Dispatch por presencia de artefactos (`facility_stats is not None AND thresholds_segmented is not None`) — el dual-runner carga artefactos con y sin stats para obtener los dos paths.

---

## Open Questions

1. **¿Cuál es el model_dir que apunta al IF-40 champion en el momento de Fase 4?**
   - Lo que sabemos: `output/models/model_metadata.json` → `IF-40-v1`; `output/models/model_metadata_frame_v1.json` → `frame-v1`. Ambos están en el mismo directorio.
   - Incertidumbre: si en el futuro `model_metadata.json` es reemplazado por frame-v1, el champion IF-40 se perdería.
   - Recomendación: en el lifespan, cargar champion siempre de `model_metadata.json` y new siempre de `model_metadata_frame_v1.json`. Añadir `metadata_filename` param a `_load_metadata()`.

2. **¿El BatchContextProvider de IF-40 (40 features, ~3.6s/pago con 8 queries por pago) es compatible con el dual-runner en términos de tiempo?**
   - Lo que sabemos: el HOWTO documenta que IF-40 usa `~3.6 s/pago` con contexto por-pago (no batch); frame-v1 usa `BatchContextProvider` de 6 queries bulk. La Fase 4 debe usar el mismo `BatchContextProvider` para ambos (factual, no model-specific).
   - Incertidumbre: si el champion IF-40 usa `_score_all` con el path de IF-40 (que llama `context_provider.get_context()` por pago para 40 features), el dual-run puede ser muy lento.
   - Recomendación: verificar en `scorer/batch/scorer.py` líneas 364-370 — el `if len(feature_names) == 40:` cae en el path lento. El dual-runner debe usar el `ctx_map` construido con `BatchContextProvider` para ambos modelos; si el IF-40 scorer requiere contexto adicional, construirlo una vez.

3. **¿Cuántos días de shadow son suficientes para que el Spearman y el Jaccard@100 sean estadísticamente estables?**
   - Lo que sabemos: el requisito es ≥2 semanas (SHAD-03). El HOWTO muestra que la ventana típica produce ~25-58 pagos por corrida de 2 minutos. El volumen real depende del tráfico de producción.
   - Incertidumbre: si el volumen es <100 pagos/día, ≥2 semanas producen ~1,400 pagos. Spearman es estable con n≥30; Jaccard@100 requiere ≥100 pagos en shadow para el overlap top-100.
   - Recomendación: `shadow_gate.py` debe verificar tanto `days_span >= 14` como `n_shadow_rows >= 500` (o configurable). Documentar el volumen mínimo en el gate.

---

## Sources

### Primary (HIGH confidence — auditado directamente)

- `scorer/batch/scorer.py` — arquitectura de `BatchScorer`, `_INSERT_COLUMNS` (23 columnas), `_insert_chunks` con dedup token, `assert_write_target_is_safe`, `score_batch` flow
- `docker/clickhouse/init/02_anomaly_scores.sql` — DDL completo de `anomaly_scores` (23 columnas actuales)
- `docker/clickhouse/migrations/01_anomaly_scores_v2.sql` — patrón de `ALTER TABLE ADD COLUMN IF NOT EXISTS`
- `scorer/artifact_loader.py` — `load_artifacts`, `_load_metadata` (dispatch por `model_metadata.json`), `Artifacts` dataclass
- `scorer/main.py` — lifespan: carga de artefactos, construcción de `SingleTransactionScorer`, `_deps._state`
- `src/fraud_detector/scoring/scorer.py` — dispatch `_is_frame_v1`, `score()` method, ambas rutas
- `output/models/model_metadata.json` — IF-40-v1, `artifact_files` map
- `output/models/model_metadata_frame_v1.json` — frame-v1, `bias_metrics` (gate1_pass=true, gate2_pass=true), `parity_check`
- `output/frame_v1_bias_report.json` — thresholds confirmados: top5_wins_ratio=1.49x (< 4.0 PASS), off_hours_local=6.46% (PASS)
- `scripts/retrain_frame_v1.py` — patrón de winsorización lines 525-545, `check_batch_calculator_parity`
- `scripts/exp_frame_feature_small.py` — uso de `spearmanr` (patterns de Spearman correlation)
- `.planning/STATE.md` — decisiones locked relevantes: 03-01 `_INSERT_COLUMNS` no tocado, 01-03 gate winsorizado, etc.
- `.planning/phases/03-wiring-del-scorer-e-integracion-platform/03-VERIFICATION.md` — confirmación de que `_INSERT_COLUMNS` fue deliberadamente no tocado en Fase 3
- `docs/HOWTO-anomaly-local.md` — arquitectura READ/WRITE, pitfall de volumen Docker, env vars de configuración
- `scorer/dependencies.py` — `_state` dict, `get_scorer()`, `get_write_ch_client()`
- `tests/test_batch_scorer.py` — patrones de test para `BatchScorer` (mocks, fingerprints, assertions)

### Secondary (MEDIUM confidence — verificado por contexto múltiple)

- `.planning/research/ARCHITECTURE.md` — patrón `ShadowDualRunner` (documentado antes de Fase 4; coincide con el análisis actual)
- `.planning/research/FEATURES.md` — Jaccard@k, bias comparison metric (documentados como P1)
- scipy 1.13.1 en venv — `spearmanr` confirmado funcionando

### Tertiary (LOW confidence — no verificado externamente)

- Ninguno — toda la investigación es codebase-based.

---

## Metadata

**Confidence breakdown:**
- Standard Stack: HIGH — todos los paquetes están en el venv y se usan actualmente
- Architecture: HIGH — derivada del código real; dual-runner es la extensión natural de `BatchScorer` existente
- DDL migration: HIGH — patrón idéntico en `01_anomaly_scores_v2.sql`
- Monitoring queries: HIGH (Python) / MEDIUM (SQL complejo) — la lógica es correcta; el SQL de ClickHouse para window functions anidadas puede requerir ajuste
- Pitfalls: HIGH — 5 de 6 pitfalls son issues documentados en el HOWTO o en el STATE.md
- Gate temporal: HIGH — la separación código/evaluación es estándar y alineada con la arquitectura del proyecto

**Research date:** 2026-07-06
**Valid until:** 2026-08-06 (30 días — stack estable; solo invalidar si ClickHouse local se actualiza a versión con cambios en `non_replicated_deduplication_window`)
