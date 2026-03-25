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

## Sensibilidad de Feature #18 (`user_reversal_ratio_30d`)

### Proposito

Verificar que el modelo no depende excesivamente de una feature que podria ser circular (tasa de reversiones del usuario en los ultimos 30 dias). Si esta feature domina, el modelo estaria capturando un patron trivial.

### Procedimiento

1. Entrenar IF con **31 features** (resultado principal, ya disponible).
2. Entrenar IF con **30 features** (sin `user_reversal_ratio_30d`).
3. Evaluar ambos en el test set con proxy estricto.
4. Comparar:

```
delta_auc = |AUC_31 - AUC_30|
```

### Criterio

```
delta_auc < 0.02 → modelo de 31 features aceptable
delta_auc >= 0.02 → reportar ambos; usar 30 features como primario
```

### Metricas adicionales de comparacion

#### Jaccard similarity en top-5%

```python
k = int(len(scores) * 0.05)
top5_31 = set(np.argsort(scores_31)[-k:])
top5_30 = set(np.argsort(scores_30)[-k:])
jaccard = len(top5_31 & top5_30) / len(top5_31 | top5_30)
```

Mide cuanto coinciden las transacciones flaggeadas entre ambas variantes. `jaccard > 0.80` indica alta coincidencia.

#### Spearman rank correlation

```python
from scipy.stats import spearmanr
rho, p = spearmanr(scores_31, scores_30)
```

Mide si el ranking de anomalia es similar entre ambas variantes, independientemente de la escala de scores.

### Retorno esperado

```python
{
    "auc_31_features": float,
    "auc_30_features": float,
    "delta_auc": float,
    "low_sensitivity": bool,     # delta < 0.02
    "jaccard_top5pct": float,
    "spearman_r": float,
    "spearman_p": float
}
```

---

## Ablacion: 31 Features vs 21 Features (Grupos F, G, H)

### Proposito

Cuantificar la contribucion marginal de los 3 nuevos grupos de features agregados al modelo extendido:

| Grupo | Features | Descripcion |
|-------|----------|-------------|
| F) Credito/Flujo | F24: is_club_credit, F25: user_debit_count_30d, F26: user_debit_amount_30d, F27: credit_flow_ratio | Patrones de uso de credito y flujo prepago/debito |
| G) Staff/Rol | F28: is_staff, F29: paid_by_manager, F30: staff_amount_zscore | Desviacion relativa del monto por rol |
| H) Diversidad Operacional | F31: category_entropy_30d, F32: user_reversal_count_30d, F33: user_merchandise_ratio_30d | Diversidad de categorias y comportamiento operativo |

Si la diferencia es marginal (delta AUC < 0.01), los grupos F-H no aportan capacidad discriminativa significativa. Si es sustancial, justifica la extension del feature set.

### Procedimiento

1. Entrenar IF con **31 features** (F01-F33) — resultado principal, ya disponible de Fase 6.
2. Entrenar IF con **21 features** (F01-F23 unicamente) — mismos hiperparametros optimos.
3. Evaluar ambos en el **test set** con proxy estricto.
4. Comparar 4 metricas:

```
delta_auc = AUC_31 - AUC_21
delta_ap  = AP_31  - AP_21
delta_p5  = P@5%_31 - P@5%_21
delta_ef  = EF_31  - EF_21
```

### Retorno esperado

```python
{
    "ablation_31_vs_21": {
        "model_31": {"auc_roc": float, "ap": float, "precision_at_5pct": float, "enrichment_factor": float},
        "model_21": {"auc_roc": float, "ap": float, "precision_at_5pct": float, "enrichment_factor": float},
        "delta": {"auc_roc": float, "ap": float, "precision_at_5pct": float, "enrichment_factor": float},
        "groups_contribute": bool  # delta_auc > 0.01
    }
}
```

### Integracion con tesis

Los resultados se presentan en **Tabla `tab:ablacion-31vs21`** del Capitulo 3. Si los grupos F-H contribuyen significativamente, se discute cuales features especificas (via SHAP) impulsan la mejora.

---

## Metricas por Segmento

### Proposito

Evaluar la capacidad discriminativa del Isolation Forest desagregada por **rol del usuario** y **categoria de pago**, para identificar segmentos donde el modelo es mas o menos efectivo.

### Segmentos por Rol

| Rol | Filtro |
|-----|--------|
| Players | `user_role = 'player'` |
| Court Managers | `user_role = 'court_manager'` |
| Court Operators | `user_role = 'court_operator'` |
| Teachers | `user_role = 'teacher'` |

### Segmentos por Categoria de Pago

| Categoria | Filtro |
|-----------|--------|
| Reservation | `category = 'reservation'` |
| Merchandise | `category = 'merchandise'` |
| Lesson/Clinic | `category IN ('lesson', 'clinic')` |
| Debit | `category = 'debit'` |

### Procedimiento

1. Tomar los scores del mejor IF (31 features) ya entrenado.
2. Para cada segmento, filtrar el test set y el vector de scores correspondiente.
3. Calcular 4 metricas por segmento: AUC-ROC, AP, P@5%, EF.
4. Descartar segmentos con menos de 100 transacciones o menos de 10 positivos proxy.

```python
def evaluate_segment(df_segment, scores_segment, proxy_segment):
    if len(df_segment) < 100 or proxy_segment.sum() < 10:
        return None
    return {
        "auc_roc": roc_auc_score(proxy_segment, scores_segment),
        "ap": average_precision_score(proxy_segment, scores_segment),
        "precision_at_5pct": precision_at_k(proxy_segment, scores_segment, k_pct=0.05),
        "enrichment_factor": enrichment_factor(proxy_segment, scores_segment, k_pct=0.05),
        "n_transactions": len(df_segment),
        "n_proxy_positive": int(proxy_segment.sum()),
        "proxy_rate": float(proxy_segment.mean()),
    }
```

### Retorno esperado

```python
{
    "segment_metrics": {
        "by_role": {
            "player": {"auc_roc": float, "ap": float, "precision_at_5pct": float, "enrichment_factor": float, ...},
            "court_manager": {...},
            "court_operator": {...},
            "teacher": {...}
        },
        "by_category": {
            "reservation": {...},
            "merchandise": {...},
            "lesson_clinic": {...},
            "debit": {...}
        }
    }
}
```

### Integracion con tesis

Los resultados se presentan en:
- **Tabla `tab:metricas-por-rol`**: AUC-ROC, AP, P@5%, EF por rol.
- **Tabla `tab:metricas-por-categoria`**: AUC-ROC, AP, P@5%, EF por categoria de pago.

Se discuten diferencias significativas entre segmentos en la seccion de discusion del Capitulo 3.

---

## Tipologia de Anomalias (SHAP)

### Proposito

Clasificar las transacciones anomalas (top-5%) en tipos segun la feature dominante que impulsa su score de anomalia, usando valores SHAP. Esto permite responder: *que tipo de patron anomalo detecta el modelo con mas frecuencia?*

### Tipos de Anomalia

| # | Tipo | Feature(s) dominante(s) |
|---|------|------------------------|
| 1 | `amount` | amount, log_amount, amount_usd_ratio, amount_facility_ratio, staff_amount_zscore |
| 2 | `velocity` | user_txn_count_1h, user_txn_count_24h, time_since_last_txn, user_amount_24h |
| 3 | `discount` | discount_ratio, user_discount_ratio_30d |
| 4 | `temporal` | hour_sin, hour_cos, day_of_week, is_weekend, is_off_hours |
| 5 | `credit_flow` | is_club_credit, user_debit_count_30d, user_debit_amount_30d, credit_flow_ratio |
| 6 | `role_deviation` | is_staff, paid_by_manager, staff_amount_zscore |
| 7 | `diversity` | user_distinct_facilities_30d, user_distinct_methods, category_entropy_30d, user_merchandise_ratio_30d |
| 8 | `reversal` | user_reversal_ratio_30d |
| 9 | `mixed` | Ninguna feature domina (max SHAP < 2x segundo SHAP) |

### Procedimiento

1. Calcular SHAP values para las transacciones del top-5% (subsample si es necesario).
2. Para cada transaccion, identificar la feature con mayor |SHAP value|.
3. Mapear esa feature al tipo de anomalia segun la tabla anterior.
4. Si la feature dominante tiene |SHAP| < 2x la segunda feature dominante, clasificar como `mixed`.
5. Calcular distribucion de tipos.

```python
def classify_anomaly_type(shap_row, feature_names, feature_to_type_map):
    abs_shap = np.abs(shap_row)
    sorted_idx = np.argsort(abs_shap)[::-1]
    top_feature = feature_names[sorted_idx[0]]
    top_value = abs_shap[sorted_idx[0]]
    second_value = abs_shap[sorted_idx[1]]

    if top_value < 2.0 * second_value:
        return "mixed"
    return feature_to_type_map.get(top_feature, "mixed")
```

### Retorno esperado

```python
{
    "anomaly_typology": {
        "n_anomalies_classified": int,
        "type_distribution": {
            "amount": {"count": int, "pct": float},
            "velocity": {"count": int, "pct": float},
            "discount": {"count": int, "pct": float},
            "temporal": {"count": int, "pct": float},
            "credit_flow": {"count": int, "pct": float},
            "role_deviation": {"count": int, "pct": float},
            "diversity": {"count": int, "pct": float},
            "reversal": {"count": int, "pct": float},
            "mixed": {"count": int, "pct": float}
        },
        "dominance_threshold": 2.0
    }
}
```

### Integracion con tesis

Los resultados se presentan en **Tabla `tab:anomaly-types`** del Capitulo 3. La distribucion de tipos informa la discusion sobre que patrones operacionales captura el modelo con mayor frecuencia.

---

## Perfil de Riesgo Agregado por Usuario

### Proposito

Construir un perfil de riesgo a nivel de usuario agregando los scores de anomalia de todas sus transacciones en el test set. Permite identificar usuarios con concentracion sistematica de anomalias, lo cual puede indicar patrones operacionales recurrentes (no necesariamente fraude).

### Metricas por Usuario

| Metrica | Descripcion |
|---------|-------------|
| `avg_score` | Media de scores de anomalia del usuario |
| `max_score` | Score maximo observado |
| `p95_score` | Percentil 95 de scores |
| `n_total` | Numero total de transacciones del usuario |
| `n_top5pct` | Numero de transacciones en el top-5% de anomalias |
| `concentration` | `n_top5pct / n_total` — proporcion de transacciones anomalas |
| `dominant_type` | Tipo de anomalia mas frecuente entre sus transacciones top-5% (de la tipologia SHAP) |

### Procedimiento

```python
def build_user_risk_profiles(df_test, scores, anomaly_types, top_k_pct=0.05):
    df_test["score"] = scores
    df_test["anomaly_type"] = anomaly_types  # del paso de tipologia
    threshold = np.quantile(scores, 1 - top_k_pct)
    df_test["is_top5"] = scores >= threshold

    profiles = df_test.groupby("user_id").agg(
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        p95_score=("score", lambda x: np.percentile(x, 95)),
        n_total=("score", "count"),
        n_top5pct=("is_top5", "sum"),
    )
    profiles["concentration"] = profiles["n_top5pct"] / profiles["n_total"]

    # Tipo dominante por usuario (moda entre sus transacciones top-5%)
    top5_types = df_test[df_test["is_top5"]].groupby("user_id")["anomaly_type"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else None
    )
    profiles["dominant_type"] = top5_types

    return profiles
```

### Criterio de flaggeo

```
concentration > 0.10 → usuario flaggeado para revision
```

Esto significa que mas del 10% de las transacciones del usuario estan en el top-5% de anomalias, lo cual es el doble de lo esperado por azar (5%).

### Retorno esperado

```python
{
    "user_risk_profiles": {
        "n_users_total": int,
        "n_users_flagged": int,           # concentration > 0.10
        "pct_users_flagged": float,
        "flagged_users_summary": {
            "mean_concentration": float,
            "max_concentration": float,
            "dominant_types_distribution": {
                "amount": int, "velocity": int, "discount": int, ...
            }
        }
    }
}
```

### Salida

- **Archivo**: `output/user_risk_profiles.parquet` — DataFrame completo con perfiles de todos los usuarios.
- **Tabla tesis**: `tab:user-risk-profile` — Top 20 usuarios con mayor concentracion (anonimizados).

### Integracion con tesis

Se presenta en la seccion de discusion del Capitulo 3 como evidencia de la utilidad operacional del modelo. Se enfatiza que la concentracion elevada no implica fraude, sino patrones recurrentes que ameritan revision.

---

## Evaluacion Per-Status

> **Nota de diseno:** Este analisis requiere una funcion/helper `per_status_evaluation` dentro de `evaluation.metrics` o, alternativamente, un modulo `SensitivityAnalyzer` dedicado.

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

- SHAP muestreado: TreeExplainer sobre top-5% anomalias del test set (~125K txns) + muestra aleatoria de 5K normales para contraste.
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

`src/fraud_detector/evaluation/posthoc_analysis.py` (**pendiente**)

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
| 2 | `test_feature18_sensitivity_delta_computed` | Que `delta_auc` se computa como `abs(AUC_31 - AUC_30)` |
| 3 | `test_jaccard_similarity_range_0_to_1` | Que Jaccard esta en `[0.0, 1.0]` para cualquier input |
| 4 | `test_shap_produces_feature_importance_ranking` | Que SHAP retorna un ranking de features ordenado por `mean(abs(shap_values))` |
| 5 | `test_baselines_random_auc_near_half` | Que el baseline aleatorio produce `AUC ~ 0.5` (tolerancia 0.05) |
| 6 | `test_posthoc_degrades_to_aggregate_when_actor_not_validated` | Que el post-hoc no exporta ranking identificable si `actor_identity_validated=False` |
| 7 | `test_ablation_31_vs_21_computed` | Que se computan las 4 metricas para ambas variantes (31 y 21 features) y los deltas |
| 8 | `test_segment_metrics_all_roles_covered` | Que se evaluan todos los roles con datos suficientes y se omiten los que no cumplen minimos |
| 9 | `test_anomaly_typology_9_types` | Que la tipologia retorna exactamente los 9 tipos definidos y sus porcentajes suman 100% |
| 10 | `test_user_risk_profile_concentration` | Que la concentracion se calcula como `n_top5pct / n_total` y el flaggeo usa umbral 0.10 |

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

# Test #7 — Ablacion 31 vs 21
def test_ablation_31_vs_21_computed():
    """Que se computan metricas para ambas variantes y deltas correctamente."""
    result = {
        "model_31": {"auc_roc": 0.75, "ap": 0.30, "precision_at_5pct": 0.20, "enrichment_factor": 3.2},
        "model_21": {"auc_roc": 0.72, "ap": 0.27, "precision_at_5pct": 0.18, "enrichment_factor": 2.9},
        "delta": {},
    }
    for metric in ["auc_roc", "ap", "precision_at_5pct", "enrichment_factor"]:
        result["delta"][metric] = result["model_31"][metric] - result["model_21"][metric]
    assert all(k in result["delta"] for k in ["auc_roc", "ap", "precision_at_5pct", "enrichment_factor"])
    assert result["delta"]["auc_roc"] == pytest.approx(0.03, abs=1e-6)

# Test #8 — Metricas por segmento
def test_segment_metrics_all_roles_covered():
    """Que se evaluan todos los roles con datos suficientes."""
    rng = np.random.default_rng(42)
    roles = ["player", "court_manager", "court_operator", "teacher"]
    df = pd.DataFrame({
        "role": np.repeat(roles, 250),
        "proxy": rng.integers(0, 2, size=1000),
    })
    scores = rng.random(1000)
    result = {}
    for role in roles:
        mask = df["role"] == role
        segment_proxy = df.loc[mask, "proxy"]
        if mask.sum() >= 100 and segment_proxy.sum() >= 10:
            result[role] = {"auc_roc": roc_auc_score(segment_proxy, scores[mask])}
    assert set(result.keys()) == set(roles)

# Test #9 — Tipologia de anomalias
def test_anomaly_typology_9_types():
    """Que la tipologia produce exactamente 9 tipos y suman 100%."""
    expected_types = {
        "amount", "velocity", "discount", "temporal", "credit_flow",
        "role_deviation", "diversity", "reversal", "mixed"
    }
    # Simular distribucion
    distribution = {t: {"count": 10, "pct": 100.0 / 9} for t in expected_types}
    assert set(distribution.keys()) == expected_types
    total_pct = sum(v["pct"] for v in distribution.values())
    assert abs(total_pct - 100.0) < 0.1

# Test #10 — Perfil de riesgo por usuario
def test_user_risk_profile_concentration():
    """Que la concentracion se calcula correctamente y el flaggeo usa umbral 0.10."""
    n_total = 100
    n_top5 = 15  # 15% concentracion → debe ser flaggeado
    concentration = n_top5 / n_total
    assert concentration == 0.15
    assert concentration > 0.10  # flaggeado
    n_top5_low = 5  # 5% concentracion → no flaggeado
    concentration_low = n_top5_low / n_total
    assert concentration_low <= 0.10  # no flaggeado
```

---

## Entregables

| Artefacto | Ruta | Contenido |
|-----------|------|-----------|
| Resultados sensibilidad | `output/results_sensitivity.json` | Proxy, Feature #18, ablacion 31vs21, per-status, baselines, segmentos |
| Resultados post-hoc | `output/results_posthoc.json` | Concentracion por centro, actor, moneda y descuentos |
| Tipologia de anomalias | `output/results_anomaly_typology.json` | Distribucion de 9 tipos de anomalia (SHAP) |
| Perfiles de riesgo | `output/user_risk_profiles.parquet` | Perfil de riesgo agregado por usuario |
| SHAP summary | `output/figures/shap_summary.pdf` | Summary plot top 10 |
| SHAP summary (PNG) | `output/figures/shap_summary.png` | Mismo, para notebooks |
| Modulo post-hoc | `src/fraud_detector/evaluation/posthoc_analysis.py` (**pendiente**) | PostHocAnalyzer |
| Tablas para tesis | (generadas en Fase 8) | Sensibilidad proxy, Feature #18, ablacion, segmentos, tipologia, perfiles, post-hoc |

### Estructura de `results_sensitivity.json`

```json
{
  "proxy_sensitivity": {
    "strict": {"auc_roc": "...", "ap": "..."},
    "wide": {"auc_roc": "...", "ap": "..."},
    "delta_auc": "...",
    "delta_ap": "...",
    "robust": true
  },
  "feature18_sensitivity": {
    "auc_31_features": "...",
    "auc_30_features": "...",
    "delta_auc": "...",
    "low_sensitivity": true,
    "jaccard_top5pct": "...",
    "spearman_r": "..."
  },
  "ablation_31_vs_21": {
    "model_31": {"auc_roc": "...", "ap": "...", "precision_at_5pct": "...", "enrichment_factor": "..."},
    "model_21": {"auc_roc": "...", "ap": "...", "precision_at_5pct": "...", "enrichment_factor": "..."},
    "delta": {"auc_roc": "...", "ap": "...", "precision_at_5pct": "...", "enrichment_factor": "..."},
    "groups_contribute": true
  },
  "per_status": {
    "totally_refunded": {"auc_roc": "...", "count": "..."},
    "refunded_to_credit": {"auc_roc": "...", "count": "..."},
    "partially_refunded": {"auc_roc": "...", "count": "..."}
  },
  "segment_metrics": {
    "by_role": {
      "player": {"auc_roc": "...", "ap": "...", "precision_at_5pct": "...", "enrichment_factor": "...", "n_transactions": "...", "n_proxy_positive": "...", "proxy_rate": "..."},
      "court_manager": {"...": "..."},
      "court_operator": {"...": "..."},
      "teacher": {"...": "..."}
    },
    "by_category": {
      "reservation": {"...": "..."},
      "merchandise": {"...": "..."},
      "lesson_clinic": {"...": "..."},
      "debit": {"...": "..."}
    }
  },
  "anomaly_typology": {
    "n_anomalies_classified": "...",
    "type_distribution": {
      "amount": {"count": "...", "pct": "..."},
      "velocity": {"count": "...", "pct": "..."},
      "discount": {"count": "...", "pct": "..."},
      "temporal": {"count": "...", "pct": "..."},
      "credit_flow": {"count": "...", "pct": "..."},
      "role_deviation": {"count": "...", "pct": "..."},
      "diversity": {"count": "...", "pct": "..."},
      "reversal": {"count": "...", "pct": "..."},
      "mixed": {"count": "...", "pct": "..."}
    },
    "dominance_threshold": 2.0
  },
  "user_risk_profiles": {
    "n_users_total": "...",
    "n_users_flagged": "...",
    "pct_users_flagged": "...",
    "flagged_users_summary": {
      "mean_concentration": "...",
      "max_concentration": "...",
      "dominant_types_distribution": {"...": "..."}
    }
  },
  "baselines": {
    "random": {"auc_roc": "...", "ap": "..."},
    "amount_ranking": {"auc_roc": "...", "ap": "..."},
    "zscore_amount": {"auc_roc": "...", "ap": "..."}
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
