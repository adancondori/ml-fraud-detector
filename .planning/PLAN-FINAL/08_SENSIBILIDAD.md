# Fase 7: Analisis de Sensibilidad y Robustez

> Blindar el resultado principal verificando que no depende de una unica decision metodologica (proxy, features, contamination). Ademas, proveer interpretabilidad via SHAP.

---

## Sensibilidad del Proxy

### Proposito

Demostrar que los resultados son consistentes independientemente de la definicion operacional del proxy de anomalia.

### Proxies

| Proxy | Definicion | Tasa base |
|-------|-----------|-----------|
| Estricto | `status IN ('totally_refunded', 'refunded_to_credit')` | ~6.33% |
| Amplio | `status IN ('totally_refunded', 'refunded_to_credit', 'partially_refunded')` | ~7.55% |

### Procedimiento

1. Tomar los scores ya generados en Fase 6 (no reentrenar).
2. Evaluar `full_evaluation()` con proxy estricto (resultado principal, ya disponible).
3. Evaluar `full_evaluation()` con proxy amplio.
4. Calcular deltas:

```
delta_auc = |AUC_estricto - AUC_amplio|
delta_ap  = |AP_estricto  - AP_amplio|
```

### Criterio

```
delta_auc < 0.05
```

Si se cumple, el resultado principal es robusto frente a la definicion del proxy.

### Reporte

Ambos resultados se reportan en la tesis para transparencia metodologica, independientemente de si el criterio se cumple o no.

---

## Sensibilidad de Feature #17 (`user_reversal_ratio_30d`)

### Proposito

Verificar que el modelo no depende excesivamente de una feature que podria ser circular (tasa de reversiones del usuario en los ultimos 30 dias). Si esta feature domina, el modelo estaria capturando un patron trivial.

### Procedimiento

1. Entrenar IF con **20 features** (resultado principal, ya disponible).
2. Entrenar IF con **19 features** (sin `user_reversal_ratio_30d`).
3. Evaluar ambos en el test set con proxy estricto.
4. Comparar:

```
delta_auc = |AUC_20 - AUC_19|
```

### Criterio

```
delta_auc < 0.02 → modelo de 20 features aceptable
delta_auc >= 0.02 → reportar ambos; usar 19 features como primario
```

### Metricas adicionales de comparacion

#### Jaccard similarity en top-5%

```python
k = int(len(scores) * 0.05)
top5_20 = set(np.argsort(scores_20)[-k:])
top5_19 = set(np.argsort(scores_19)[-k:])
jaccard = len(top5_20 & top5_19) / len(top5_20 | top5_19)
```

Mide cuanto coinciden las transacciones flaggeadas entre ambas variantes. `jaccard > 0.80` indica alta coincidencia.

#### Spearman rank correlation

```python
from scipy.stats import spearmanr
rho, p = spearmanr(scores_20, scores_19)
```

Mide si el ranking de anomalia es similar entre ambas variantes, independientemente de la escala de scores.

### Retorno esperado

```python
{
    "auc_20_features": float,
    "auc_19_features": float,
    "delta_auc": float,
    "low_sensitivity": bool,     # delta < 0.02
    "jaccard_top5pct": float,
    "spearman_r": float,
    "spearman_p": float
}
```

---

## Evaluacion Per-Status

> **Nota de diseno:** Este analisis requiere el metodo `per_status_evaluation` que se ha agregado a la interfaz de `HypothesisEvaluator` en la Fase 6. Alternativamente, si se aplica la descomposicion SRP recomendada, este metodo pertenece a `DiscriminationEvaluator`. Si se prefiere mantener la separacion de fases, se puede crear una clase `SensitivityAnalyzer` dedicada que contenga tanto esta logica como la sensibilidad de proxy y Feature #17.

### Proposito

Determinar que tipo de reembolso captura mejor el modelo. Esto informa sobre la naturaleza de las anomalias detectadas.

### Procedimiento

Para cada status de reembolso, generar un proxy binario individual y calcular AUC:

```python
statuses = ["totally_refunded", "refunded_to_credit", "partially_refunded"]
for status in statuses:
    proxy_status = (df_test["status"] == status).astype(int)
    if proxy_status.sum() >= 10:
        auc = roc_auc_score(proxy_status, scores)
```

### Retorno esperado

```python
{
    "totally_refunded": {"auc_roc": float, "count": int},
    "refunded_to_credit": {"auc_roc": float, "count": int},
    "partially_refunded": {"auc_roc": float, "count": int}
}
```

### Interpretacion

Diferencias grandes entre tipos de reembolso pueden indicar que el modelo discrimina un patron operacional especifico (e.g., reembolsos totales vs parciales) y no anomalias en general. Esto se discute pero no invalida el estudio.

---

## Interpretabilidad SHAP

### Proposito

Identificar que features contribuyen mas a los scores de anomalia del Isolation Forest, proporcionando interpretabilidad al modelo.

### TreeExplainer con fallback

```python
import shap

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
except Exception:
    background = shap.kmeans(X_sample, 50)
    explainer = shap.KernelExplainer(model.decision_function, background)
    shap_values = explainer.shap_values(X_sample[:500])
```

- `TreeExplainer` es compatible con `IsolationForest` desde SHAP v0.39+.
- Si falla por cualquier razon, `KernelExplainer` como fallback (mas lento, por eso se limita a 500 filas).

> **Nota DIP (Dependency Inversion):** El patron fallback `TreeExplainer -> KernelExplainer` es pragmatico pero el codigo depende directamente de `shap.TreeExplainer` y `shap.KernelExplainer`. Para formalizar, se podria definir una estrategia:
>
> ```python
> class ExplainerStrategy(ABC):
>     @abstractmethod
>     def explain(self, model, X_sample) -> np.ndarray: ...
>
> class TreeExplainerStrategy(ExplainerStrategy): ...
> class KernelExplainerStrategy(ExplainerStrategy): ...
> class FallbackExplainerStrategy(ExplainerStrategy):
>     """Intenta TreeExplainer; si falla, usa KernelExplainer."""
> ```
>
> Esto permitiria inyectar explainers alternativos (e.g., `PermutationExplainer`) sin modificar el codigo de sensibilidad.

### Subsample

- Tomar 5,000 filas del test set como muestra representativa.
- Seed fijo (`random_seed=42`) para reproducibilidad.

### Salida

| Artefacto | Formato | Contenido |
|-----------|---------|-----------|
| `shap_summary.pdf` | PDF | Summary plot top 10 features |
| `shap_summary.png` | PNG | Mismo, para notebooks |
| Tabla top 10 | dict/DataFrame | Feature, mean |SHAP|, ranking |

### Integracion con tesis

El summary plot se incluye en Capitulo 3 como evidencia de interpretabilidad. La tabla de importancia se usa en la discusion para vincular features con patrones operacionales.

---

## Estabilidad de Contamination (sanity check)

### Proposito

Verificar que el ranking de modelos no cambia al variar el parametro `contamination` de IF. No es un requisito de la tesis, pero aporta confianza.

### Procedimiento

1. Tomar el mejor IF ya entrenado.
2. Verificar que los scores (y por tanto el ranking) no dependen de `contamination` — en scikit-learn, `contamination` solo afecta el offset de `decision_function`, no los scores relativos.
3. Si los resultados de HE1-HE4 fueran sensibles, investigar antes de reportar.

### Nota

Dado que usamos `-decision_function(X)` como score y nunca el threshold binario, `contamination` no deberia afectar los resultados de evaluacion. Se documenta esta decision.

---

## Sanity Baselines

### Proposito

Confirmar que IF (y LOF, OC-SVM) superan baselines triviales. Si un baseline trivial iguala o supera al modelo, hay un problema fundamental.

### Baselines

| # | Baseline | Score | AUC esperado |
|---|----------|-------|-------------|
| 1 | Random ranking | `np.random.rand(n)` | ~0.50 |
| 2 | Amount-based ranking | `reservation_paid_out` (normalizado) | Variable |
| 3 | Z-score de monto | `(amount - mean) / std` | Variable |

### Criterio

```
IF debe superar TODOS los baselines en AUC-ROC y AP
```

Si IF no supera un baseline trivial:
1. Investigar posible error en el pipeline.
2. Revisar orientacion de scores.
3. Si el resultado es genuino, reportar honestamente y discutir en limitaciones.

### Implementacion

Los baselines se evaluan con `full_evaluation()` como cualquier modelo, asegurando la misma metodologia.

---

## Analisis Post-Hoc: Anomalias por Centro, Actor Operativo y Moneda

### Proposito

Una vez obtenidos los scores de anomalia del Isolation Forest, agrupar las transacciones anomalas por **centro (facility)**, **actor operativo asociado** y **moneda** para identificar donde se concentran los patrones anomalos. Este analisis responde a la pregunta operacional: *¿que centros y actores presentan mayor concentracion de anomalias, particularmente en el uso de descuentos?*

**Importante:** Este es un analisis descriptivo post-hoc sobre los scores del modelo ya entrenado. No implica reentrenamiento ni modifica las hipotesis HE1-HE4. Se presenta como hallazgo complementario en la discusion del Capitulo 3.

### Gate previo: identidad del actor

Antes de exportar cualquier tabla o figura atribuida a un manager/actor individual, validar si existe un identificador operacional confiable. La prioridad de campos es:

1. `effective_user_id` validado contra logs o semantica operacional conocida.
2. `user_id`, solo si se demuestra que representa al actor que ejecuto la accion.
3. Si ninguno es confiable, degradar el analisis a nivel agregado: **"pagos con intervencion de manager"** sin atribucion individual.

Esto implica:

- `actor_identity_validated = true`: se permite ranking de actores/usuarios asociados.
- `actor_identity_validated = false`: se prohibe exportar top 10 de managers identificables; solo se reportan agregados anonimizados por `paid_by_manager`, centro y moneda.

### Datos requeridos

Las columnas `currency`, `paid_by_manager` y `effective_user_id` se agregan al SQL canonico de extraccion (Fase 1). Estas columnas **no** se usan como features del modelo; son exclusivamente para el analisis post-hoc.

| Columna | Origen | Uso |
|---------|--------|-----|
| `facility_id` | SQL canonico (ya existente) | Agrupar anomalias por centro |
| `facility_name` | SQL canonico (ya existente) | Nombre legible del centro |
| `user_id` | SQL canonico (ya existente) | Fallback solo si se valida que representa al actor operativo |
| `effective_user_id` | SQL canonico (NUEVO) | Identificador preferido para actor/usuario asociado |
| `paid_by_manager` | SQL canonico (NUEVO) | Filtrar transacciones donde el manager intervino en el pago |
| `currency` | SQL canonico (NUEVO) | Identificar moneda afectada |
| `discount` | SQL canonico (ya existente) | Analizar patrones de descuentos |

### Procedimiento

#### Paso 1: Seleccionar transacciones anomalas (top-K)

```python
# Usar scores del mejor modelo IF (ya disponible de Fase 6)
k_pct = 0.05  # top 5% mas anomalo
k = int(len(scores_if) * k_pct)
threshold = np.sort(scores_if)[-k]
df_test["is_top_anomaly"] = scores_if >= threshold
```

#### Paso 2: Concentracion de anomalias por centro

```python
# Tasa de anomalias por facility
facility_stats = df_test.groupby(["facility_id", "facility_name"]).agg(
    n_transactions=("id", "count"),
    n_anomalies=("is_top_anomaly", "sum"),
    anomaly_rate=("is_top_anomaly", "mean"),
    mean_discount=("discount", "mean"),
    total_discount=("discount", "sum"),
    mean_score=("score_if", "mean"),
).sort_values("anomaly_rate", ascending=False)

# Centros con tasa de anomalias significativamente superior a la base (5%)
facility_stats["anomaly_enrichment"] = facility_stats["anomaly_rate"] / k_pct
top_facilities = facility_stats[facility_stats["anomaly_enrichment"] > 2.0]
```

#### Paso 3: Concentracion de anomalias por actor operativo

```python
actor_col = "effective_user_id" if actor_identity_validated else None

# Filtrar transacciones pagadas por manager
df_manager = df_test[df_test["paid_by_manager"] == True]

if actor_col is not None:
    actor_stats = df_manager.groupby(actor_col).agg(
        n_transactions=("id", "count"),
        n_anomalies=("is_top_anomaly", "sum"),
        anomaly_rate=("is_top_anomaly", "mean"),
        mean_discount=("discount", "mean"),
        total_discount=("discount", "sum"),
        n_facilities=("facility_id", "nunique"),
        mean_score=("score_if", "mean"),
    ).sort_values("anomaly_rate", ascending=False)

    actor_stats = actor_stats[actor_stats["n_transactions"] >= 10]
    actor_stats["anomaly_enrichment"] = actor_stats["anomaly_rate"] / k_pct
    top_actors = actor_stats[actor_stats["anomaly_enrichment"] > 2.0]
else:
    manager_aggregate = {
        "n_transactions_with_manager_intervention": int(df_manager["id"].count()),
        "n_anomalies_with_manager_intervention": int(df_manager["is_top_anomaly"].sum()),
        "anomaly_rate_with_manager_intervention": float(df_manager["is_top_anomaly"].mean()),
        "mean_discount_with_manager_intervention": float(df_manager["discount"].mean()),
        "total_discount_with_manager_intervention": float(df_manager["discount"].sum()),
    }
```

#### Paso 4: Analisis por moneda

```python
currency_stats = df_test.groupby("currency").agg(
    n_transactions=("id", "count"),
    n_anomalies=("is_top_anomaly", "sum"),
    anomaly_rate=("is_top_anomaly", "mean"),
    mean_discount=("discount", "mean"),
    total_discount=("discount", "sum"),
    mean_score=("score_if", "mean"),
).sort_values("anomaly_rate", ascending=False)

currency_stats["anomaly_enrichment"] = currency_stats["anomaly_rate"] / k_pct
```

#### Paso 5: Analisis cruzado centro x actor x descuento

```python
# Transacciones anomalas con descuento > 0 pagadas por manager
df_discount_abuse = df_test[
    (df_test["is_top_anomaly"]) &
    (df_test["discount"] > 0) &
    (df_test["paid_by_manager"] == True)
]

if actor_identity_validated:
    cross_stats = df_discount_abuse.groupby(
        ["facility_id", "facility_name", actor_col]
    ).agg(
        n_discounted_anomalies=("id", "count"),
        total_discount=("discount", "sum"),
        mean_discount=("discount", "mean"),
        currencies=("currency", lambda x: list(x.unique())),
    ).sort_values("total_discount", ascending=False)
else:
    cross_stats = df_discount_abuse.groupby(
        ["facility_id", "facility_name"]
    ).agg(
        n_discounted_anomalies=("id", "count"),
        total_discount=("discount", "sum"),
        mean_discount=("discount", "mean"),
        currencies=("currency", lambda x: list(x.unique())),
    ).sort_values("total_discount", ascending=False)
```

### Metricas de salida

```python
{
    "posthoc_analysis": {
        "top_k_pct": 0.05,
        "n_anomalies": int,
        "actor_identity_validated": bool,
        "actor_identifier_field": "effective_user_id|null",
        "facility_concentration": {
            "n_facilities_with_enrichment_gt_2": int,
            "top_10_facilities": [
                {
                    "facility_id": int,
                    "facility_name": str,
                    "n_transactions": int,
                    "anomaly_rate": float,
                    "anomaly_enrichment": float,
                    "mean_discount": float,
                    "total_discount": float
                }
            ]
        },
        "manager_concentration": {
            "mode": "identified_actor|aggregated_manager_intervention",
            "n_managers_with_enrichment_gt_2": int,
            "top_10_managers": [
                {
                    "actor_id": int,
                    "n_transactions": int,
                    "anomaly_rate": float,
                    "anomaly_enrichment": float,
                    "mean_discount": float,
                    "total_discount": float,
                    "n_facilities": int
                }
            ],
            "aggregate_only": {
                "n_transactions_with_manager_intervention": int,
                "n_anomalies_with_manager_intervention": int,
                "anomaly_rate_with_manager_intervention": float,
                "mean_discount_with_manager_intervention": float,
                "total_discount_with_manager_intervention": float
            }
        },
        "currency_concentration": {
            "currencies_affected": [
                {
                    "currency": str,
                    "n_transactions": int,
                    "anomaly_rate": float,
                    "anomaly_enrichment": float,
                    "total_discount": float
                }
            ]
        },
        "discount_abuse_pattern": {
            "n_anomalies_with_discount_by_manager": int,
            "total_discount_amount": float,
            "top_10_facility_manager_pairs": list
        }
    }
}
```

### Lenguaje correlacional

En la tesis, este analisis se presenta con lenguaje no causal:

- "Los centros X, Y, Z presentan una concentracion de anomalias N veces superior a la tasa base"
- "Los actores/usuarios asociados a mayor volumen de descuentos en transacciones anomalas son..."
- "La moneda M presenta la mayor tasa de transacciones anomalas con descuento"
- Si `actor_identity_validated = false`: "Los pagos con intervencion de manager muestran una tasa agregada de anomalias superior a la base"

**Nunca:** "El manager X esta cometiendo fraude" o "El centro Y tiene fraude".

### Archivo de implementacion

`src/fraud_detector/evaluation/posthoc_analysis.py`

Clase: `PostHocAnalyzer`

```python
class PostHocAnalyzer:
    def __init__(self, top_k_pct: float = 0.05, min_transactions: int = 10):
        self.top_k_pct = top_k_pct
        self.min_transactions = min_transactions

    def analyze_facility_concentration(self, df: pd.DataFrame, scores: np.ndarray) -> dict: ...
    def analyze_manager_concentration(
        self,
        df: pd.DataFrame,
        scores: np.ndarray,
        actor_col: str | None = None,
        actor_identity_validated: bool = False,
    ) -> dict: ...
    def analyze_currency_concentration(self, df: pd.DataFrame, scores: np.ndarray) -> dict: ...
    def analyze_discount_patterns(
        self,
        df: pd.DataFrame,
        scores: np.ndarray,
        actor_col: str | None = None,
        actor_identity_validated: bool = False,
    ) -> dict: ...
    def full_posthoc_analysis(
        self,
        df: pd.DataFrame,
        scores: np.ndarray,
        actor_col: str | None = None,
        actor_identity_validated: bool = False,
    ) -> dict: ...
```

### Relacion con la tesis

Este analisis se ubica en la **seccion de discusion del Capitulo 3**, despues de las hipotesis formales (HE1-HE4) y la sensibilidad. Demuestra la utilidad practica del modelo: no solo detecta anomalias, sino que permite identificar **donde** se concentran. Si la identidad del actor no es confiable, el valor operacional sigue existiendo a nivel agregado por centro, intervencion de manager y moneda.

---

## Contratos TDD

Tests a escribir **ANTES** de implementar los analisis de sensibilidad:

| # | Test | Verifica |
|---|------|----------|
| 1 | `test_proxy_sensitivity_both_proxies_evaluated` | Que el resultado contiene metricas para proxy estricto y proxy amplio |
| 2 | `test_feature17_sensitivity_delta_computed` | Que `delta_auc` se computa como `abs(AUC_20 - AUC_19)` |
| 3 | `test_jaccard_similarity_range_0_to_1` | Que Jaccard esta en `[0.0, 1.0]` para cualquier input |
| 4 | `test_shap_produces_feature_importance_ranking` | Que SHAP retorna un ranking de features ordenado por `mean(abs(shap_values))` |
| 5 | `test_baselines_random_auc_near_half` | Que el baseline aleatorio produce `AUC ~ 0.5` (tolerancia 0.05) |
| 6 | `test_posthoc_degrades_to_aggregate_when_actor_not_validated` | Que el post-hoc no exporta ranking identificable si `actor_identity_validated=False` |

```python
# Test #7 — PostHocAnalyzer
def test_posthoc_facility_enrichment_computed():
    """Que el enrichment por facility se calcula correctamente."""
    analyzer = PostHocAnalyzer(top_k_pct=0.10)
    df = pd.DataFrame({
        "id": range(1000),
        "facility_id": [1]*100 + [2]*900,
        "facility_name": ["Fac1"]*100 + ["Fac2"]*900,
        "effective_user_id": range(1000),
        "discount": [50.0]*100 + [0.0]*900,
        "paid_by_manager": [True]*100 + [False]*900,
        "currency": ["USD"]*1000,
    })
    scores = np.concatenate([np.ones(100)*10, np.zeros(900)])
    result = analyzer.analyze_facility_concentration(df, scores)
    assert result["top_10_facilities"][0]["anomaly_enrichment"] > 1.0

def test_posthoc_currency_returns_all_currencies():
    """Que todas las monedas presentes aparecen en el resultado."""
    analyzer = PostHocAnalyzer(top_k_pct=0.10)
    df = pd.DataFrame({
        "id": range(100),
        "currency": ["USD"]*50 + ["MXN"]*50,
        "discount": [10.0]*100,
    })
    scores = np.random.default_rng(42).random(100)
    result = analyzer.analyze_currency_concentration(df, scores)
    currencies = {c["currency"] for c in result["currencies_affected"]}
    assert currencies == {"USD", "MXN"}

def test_posthoc_degrades_to_aggregate_when_actor_not_validated():
    analyzer = PostHocAnalyzer(top_k_pct=0.10)
    df = pd.DataFrame({
        "id": range(20),
        "effective_user_id": [101] * 10 + [202] * 10,
        "facility_id": [1] * 20,
        "facility_name": ["Fac1"] * 20,
        "discount": [10.0] * 20,
        "paid_by_manager": [True] * 20,
        "currency": ["USD"] * 20,
    })
    scores = np.linspace(0, 1, 20)
    result = analyzer.analyze_manager_concentration(
        df,
        scores,
        actor_col=None,
        actor_identity_validated=False,
    )
    assert result["mode"] == "aggregated_manager_intervention"
    assert result["top_10_managers"] == []

# Ejemplo de test #3 (Jaccard siempre en rango)
def test_jaccard_similarity_range_0_to_1():
    scores_a = np.random.default_rng(42).random(1000)
    scores_b = np.random.default_rng(99).random(1000)
    k = int(len(scores_a) * 0.05)
    top_a = set(np.argsort(scores_a)[-k:])
    top_b = set(np.argsort(scores_b)[-k:])
    jaccard = len(top_a & top_b) / len(top_a | top_b)
    assert 0.0 <= jaccard <= 1.0

# Ejemplo de test #5 (baseline aleatorio)
def test_baselines_random_auc_near_half():
    rng = np.random.default_rng(42)
    proxy = rng.integers(0, 2, size=10_000)
    scores = rng.random(10_000)
    auc = roc_auc_score(proxy, scores)
    assert abs(auc - 0.5) < 0.05
```

---

## Entregables

| Artefacto | Ruta | Contenido |
|-----------|------|-----------|
| Resultados sensibilidad | `output/results_sensitivity.json` | Proxy, Feature #17, per-status, baselines |
| Resultados post-hoc | `output/results_posthoc.json` | Concentracion por centro, actor, moneda y descuentos |
| SHAP summary | `output/figures/shap_summary.pdf` | Summary plot top 10 |
| SHAP summary (PNG) | `output/figures/shap_summary.png` | Mismo, para notebooks |
| Modulo post-hoc | `src/fraud_detector/evaluation/posthoc_analysis.py` | PostHocAnalyzer |
| Tablas para tesis | (generadas en Fase 8) | Sensibilidad proxy, Feature #17, post-hoc |

### Estructura de `results_sensitivity.json`

```json
{
  "proxy_sensitivity": {
    "strict": {"auc_roc": ..., "ap": ...},
    "wide": {"auc_roc": ..., "ap": ...},
    "delta_auc": ...,
    "delta_ap": ...,
    "robust": true
  },
  "feature17_sensitivity": {
    "auc_20_features": ...,
    "auc_19_features": ...,
    "delta_auc": ...,
    "low_sensitivity": true,
    "jaccard_top5pct": ...,
    "spearman_r": ...
  },
  "per_status": {
    "totally_refunded": {"auc_roc": ..., "count": ...},
    "refunded_to_credit": {"auc_roc": ..., "count": ...},
    "partially_refunded": {"auc_roc": ..., "count": ...}
  },
  "baselines": {
    "random": {"auc_roc": ..., "ap": ...},
    "amount_ranking": {"auc_roc": ..., "ap": ...},
    "zscore_amount": {"auc_roc": ..., "ap": ...}
  }
}
```

### Estructura de `results_posthoc.json`

```json
{
  "posthoc_analysis": {
    "top_k_pct": 0.05,
    "n_total_anomalies": "...",
    "actor_identity_validated": true,
    "actor_identifier_field": "effective_user_id",
    "facility_concentration": {
      "n_facilities_with_enrichment_gt_2": "...",
      "top_10_facilities": ["..."]
    },
    "manager_concentration": {
      "mode": "identified_actor",
      "n_managers_with_enrichment_gt_2": "...",
      "top_10_managers": ["..."],
      "aggregate_only": null
    },
    "currency_concentration": {
      "currencies_affected": ["..."]
    },
    "discount_abuse_pattern": {
      "n_anomalies_with_discount_by_manager": "...",
      "total_discount_amount": "...",
      "top_10_facility_manager_pairs": ["..."]
    }
  }
}
```
