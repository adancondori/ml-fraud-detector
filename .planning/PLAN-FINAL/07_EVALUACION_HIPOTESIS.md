# Fase 6: Evaluacion y Prueba de Hipotesis

> Responde OE3 y OE4. Cierra formalmente HE1-HE4 con evidencia estadistica, bootstrap CI 95% y estabilidad temporal.

## Archivo principal

`src/fraud_detector/evaluation/metrics.py`

## Modulo de evaluacion: `evaluation.metrics`

El repo actual expone funciones utilitarias en `src/fraud_detector/evaluation/metrics.py`. El plan usa esas funciones como base y, si mas adelante el pipeline lo requiere, puede envolverse en una fachada ligera sin cambiar el contrato estadistico.

### Firma de metodos

```python
def evaluate_scores(labels: np.ndarray, scores: np.ndarray,
                    top_k_percents: list = [0.01, 0.02, 0.05, 0.10]) -> dict:
    """HE2 + HE3: AUC-ROC, AP, Precision@k, EF@k."""

def bootstrap_ci(labels: np.ndarray, scores: np.ndarray,
                 metric_fn: callable, n_iterations: int = 1000,
                 ci: float = 0.95, random_seed: int = 42) -> dict:
    """Intervalo de confianza bootstrap."""

def precision_at_k(labels: np.ndarray, scores: np.ndarray, k_pct: float = 0.05) -> float:
    """Precision@k."""

def enrichment_factor(labels: np.ndarray, scores: np.ndarray, k_pct: float = 0.05) -> float:
    """Enrichment factor."""
```

---

## Nota de diseno: Violaciones SRP e ISP

### Problema SRP

Una implementacion monolitica mezclaria demasiadas responsabilidades: tests estadisticos, metricas de discriminacion, bootstrap, estabilidad temporal, comparacion de modelos y correcciones multiples.

**Descomposicion recomendada:**

| Clase | Metodos | Responsabilidad |
|-------|---------|----------------|
| `StatisticalTester` | `test_mann_whitney`, `ks_test`, `apply_holm_bonferroni` | Tests estadisticos y correcciones |
| `DiscriminationEvaluator` | `compute_discrimination`, `compute_topk`, `per_status_evaluation` | AUC-ROC, AP, top-k, EF, evaluacion por status |
| `BootstrapAnalyzer` | `bootstrap_ci` | Intervalos de confianza bootstrap |
| `TemporalAnalyzer` | `temporal_stability` | Estabilidad mensual de metricas |
| `EvaluationFacade` (opcional) | `full_evaluation`, `compare_models` | **Facade** que compone las funciones anteriores |

### Problema ISP (Interface Segregation)

Un cliente que solo necesita evaluar HE2/HE3 no deberia depender de una clase grande. La descomposicion anterior resuelve esto: el cliente importa solo las funciones necesarias.

### Metodo faltante: `per_status_evaluation`

La Fase 7 (Sensibilidad) describe un analisis per-status que requiere calcular AUC-ROC por cada tipo de reembolso individualmente. Esa logica puede vivir como helper adicional en `evaluation.metrics` o en un modulo de sensibilidad dedicado.

---

## HE1: Separacion Estadistica

### Prueba

Mann-Whitney U unilateral.

```python
from scipy.stats import mannwhitneyu

scores_proxy1 = scores[proxy == 1]
scores_proxy0 = scores[proxy == 0]
U, p = mannwhitneyu(scores_proxy1, scores_proxy0, alternative='greater')
```

### Tamano de efecto (formula CORREGIDA)

```
rank-biserial r = 2 * U / (n1 * n2) - 1
```

- La formula produce `r > 0` cuando los scores de anomalia son mas altos que los normales (orientacion correcta).
- Escala de Cohen: `r > 0.10` pequeno, `r > 0.30` mediano, `r > 0.50` grande.
- **Bug potencial (floating point):** Con N > 2.5M, `2*U/(n1*n2)` puede producir valores ligeramente fuera de `[-1, 1]` por precision de punto flotante. Se debe aplicar clamping: `r = np.clip(2*U/(n1*n2) - 1, -1.0, 1.0)`.

### CLES (Common Language Effect Size)

```
CLES = U / (n1 * n2)
```

Probabilidad de que un caso anomalo aleatorio tenga score mayor que un caso normal aleatorio.

### Criterio de aceptacion HE1

```
p < 0.05 AND rank_biserial_r > 0.10
```

**Nota importante:** Con N > 2.5M, los p-values seran siempre extremadamente pequenos. El foco del analisis debe estar en los tamanos de efecto (`r` y CLES), no en la significancia estadistica.

### Retorno de `test_mann_whitney`

```python
{
    "U_statistic": float,
    "p_value": float,
    "rank_biserial_r": float,
    "cles": float,
    "n_anomaly": int,
    "n_normal": int,
    "he1_pass": bool
}
```

---

## HE2: Capacidad Discriminativa

### Metricas

```python
from sklearn.metrics import roc_auc_score, average_precision_score

auc = roc_auc_score(proxy, scores)
ap  = average_precision_score(proxy, scores)
```

### Criterio de aceptacion HE2

```
AUC-ROC > 0.70 AND AP > 6.33% (tasa base del proxy estricto)
```

Ambas condiciones deben cumplirse simultaneamente.

### Interpretacion de AUC

| Rango | Interpretacion |
|-------|---------------|
| 0.90+ | Excelente |
| 0.80 - 0.90 | Bueno |
| 0.70 - 0.80 | Aceptable |
| < 0.70 | Insuficiente |

### Retorno de `compute_discrimination`

```python
{
    "auc_roc": float,
    "average_precision": float,
    "base_rate": float,
    "ap_over_baseline": float,   # ap / base_rate
    "he2_pass": bool
}
```

---

## HE3: Concentracion Top-K

### Multiples valores de k

Se evalua en `k_values = [0.01, 0.02, 0.05, 0.10]` (1%, 2%, 5%, 10%).

### Metricas por cada k

```
Precision@k  = proxy_in_topk / k_count
Recall@k     = proxy_in_topk / total_anomalies
EF            = Precision@k / base_rate     (Enrichment Factor)
```

### Criterio de aceptacion HE3

```
EF > 1 al top-5%  (criterio primario)
```

Utilidad practica: `EF > 1.5` indica valor operacional real.

### Retorno de `compute_topk`

```python
{
    "precision_at_1pct": float,
    "recall_at_1pct": float,
    "ef_at_1pct": float,
    "precision_at_2pct": float,
    "recall_at_2pct": float,
    "ef_at_2pct": float,
    "precision_at_5pct": float,
    "recall_at_5pct": float,
    "ef_at_5pct": float,
    "precision_at_10pct": float,
    "recall_at_10pct": float,
    "ef_at_10pct": float,
    "precision_at_k": float,        # alias de precision_at_5pct
    "recall_at_k": float,           # alias de recall_at_5pct
    "enrichment_factor": float,     # alias de ef_at_5pct
    "he3_pass": bool
}
```

---

## HE4: Comparacion de Modelos

### Modelos

- Isolation Forest (IF)
- Local Outlier Factor (LOF)
- One-Class SVM (OC-SVM)

### Metricas de comparacion

| # | Metrica |
|---|---------|
| 1 | AUC-ROC |
| 2 | Average Precision |
| 3 | Precision@5% |
| 4 | Enrichment Factor (EF) |

### Criterio de aceptacion HE4

```
IF debe ganar en >= 3 de las 4 metricas (alineado con tesis: "mayoria")
```

### Condiciones de equidad (fairness)

Toda comparacion debe usar exactamente:

- **Mismo snapshot** de datos
- **Mismo test set** temporal (Sep-Dic 2025)
- **Mismo proxy** (estricto)
- **Mismas 31 features** (todos los modelos usan el mismo conjunto completo)
- **Misma politica de nulos** y preprocesamiento
- **Misma orientacion de scores** (alto = mas anomalo)

Sin estas condiciones, HE4 no es defendible.

### Nota sobre variantes de features

- **IF-31** (31 features) es el modelo principal evaluado en HE1-HE4.
- **IF-30** (30 features, sin `user_reversal_ratio_30d`) e **IF-21** (21 features base) se evaluan exclusivamente en la **Fase 7 (Sensibilidad)** para medir el impacto del conjunto expandido de features.
- La comparacion HE4 (IF vs LOF vs OC-SVM) se realiza con los 3 modelos entrenados sobre las mismas 31 features.

### Retorno de `compare_models`

```python
{
    "metrics_comparison": {
        "isolation_forest": {"auc_roc": ..., "ap": ..., "precision_at_5pct": ..., "ef_at_5pct": ...},
        "lof": {...},
        "ocsvm": {...}
    },
    "if_wins": int,       # cuantas metricas gano IF
    "if_wins_on": list,   # nombres de metricas donde gano
    "he4_pass": bool
}
```

---

## Bootstrap CI 95%

### Parametros

- `n_iterations = 1000`
- `ci = 0.95`
- `random_seed = 42`
- Resampleo con reemplazo

### Procedimiento

```python
rng = np.random.RandomState(random_seed)
values = []
for _ in range(n_iterations):
    idx = rng.choice(n, n, replace=True)
    values.append(metric_fn(scores[idx], proxy[idx]))
values = np.array(values)
alpha = (1 - ci) / 2
```

### Retorno de `bootstrap_ci`

```python
{
    "mean": float,
    "lower": float,    # percentil 2.5
    "upper": float,    # percentil 97.5
    "std": float,
    "n_iterations": int
}
```

Se computa para: AUC-ROC y AP (minimo). Opcionalmente para Precision@5% y EF.

---

## Estabilidad Temporal

### Proposito

Verificar que el modelo IF-31 (modelo principal, 31 features) no se degrada a lo largo del periodo de test (Sep, Oct, Nov, Dic).

### Procedimiento

1. Agrupar test set por mes usando `created_at`.
2. Calcular AUC-ROC mensual por modelo.
3. Comparar contra AUC global del test.

### Criterio de alerta

```
Si algun mes tiene AUC que cae > 0.10 respecto al AUC global → flag de inestabilidad
```

No es criterio de rechazo, pero se reporta y discute.

### Retorno de `temporal_stability`

```python
{
    "model_name": str,
    "monthly_auc": {
        "2025-09": {"auc_roc": float, "n_samples": int, "proxy_rate": float},
        "2025-10": {...},
        "2025-11": {...},
        "2025-12": {...}
    }
}
```

---

## Correccion de Holm-Bonferroni

### Proposito

Corregir p-values de HE1-HE4 por pruebas multiples para reducir error tipo I.

### Formula

```
p_ajustado[idx] = p[idx] * (n - rank)
```

Donde `rank` es la posicion (0-indexed) del p-value en el ordenamiento ascendente, y `n` es el numero total de hipotesis.

### Implementacion

```python
def apply_holm_bonferroni(self, p_values: list) -> list:
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = p_values[idx] * (n - rank)
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted.tolist()
```

Se aplica sobre los p-values de HE1 (Mann-Whitney), HE2 (se podria obtener via permutacion), HE3 (test de proporcion), HE4 (test de diferencia).

---

## Test de Kolmogorov-Smirnov

### Proposito

Medir la separacion maxima entre las distribuciones acumuladas de scores de proxy+ y proxy-. Complementario a Mann-Whitney (que mide tendencia central).

### Implementacion

```python
from scipy.stats import ks_2samp

stat, p = ks_2samp(scores[proxy == 1], scores[proxy == 0])
```

### Retorno de `ks_test`

```python
{
    "ks_statistic": float,
    "p_value": float
}
```

---

## Metodo `full_evaluation`

Orquesta la evaluacion completa de un modelo. Invoca:

1. `test_mann_whitney` → HE1
2. `compute_discrimination` → HE2
3. `compute_topk` → HE3
4. `ks_test` → complementario
5. `bootstrap_ci` → para AUC y AP
6. `temporal_stability` → si `dates` no es `None`

### Retorno

```python
{
    "model_name": str,
    "he1": dict,
    "he2": dict,
    "he3": dict,
    "ks": dict,
    "bootstrap_ci_auc": dict,
    "bootstrap_ci_ap": dict,
    "temporal_stability": dict   # solo si dates != None
}
```

Se ejecuta una vez por modelo (IF, LOF, OC-SVM). Luego `compare_models` recibe los tres resultados.

---

## Visualizaciones

### Curvas ROC

- 3 modelos superpuestos (IF, LOF, OC-SVM) con colores distintos.
- Linea diagonal punteada como referencia aleatoria.
- AUC en la leyenda de cada modelo.

### Curvas Precision-Recall

- 3 modelos superpuestos.
- Linea horizontal punteada en `base_rate` (6.33%) como baseline.
- AP en la leyenda.

### Distribuciones de Scores

- 3 subplots (uno por modelo).
- Histograma superpuesto: proxy+ (naranja) vs proxy- (azul).
- Transparencia para ver solapamiento.

### Curva de Enriquecimiento (Lift)

- Eje X: porcentaje del dataset revisado (k = 0.1% a 100%).
- Eje Y: Enrichment Factor.
- Linea horizontal en EF = 1 como referencia.
- 3 modelos superpuestos.

---

## Entregables

| Artefacto | Ruta |
|-----------|------|
| Modulo de evaluacion | `src/fraud_detector/evaluation/metrics.py` |
| Resultados completos | `output/results.json` |
| Scores con metadata | `output/scores/test_scores.parquet` |

### Estructura de `test_scores.parquet`

| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `id` | int | ID de transaccion |
| `created_at` | datetime | Fecha de la transaccion |
| `score_if` | float | Score de Isolation Forest |
| `score_lof` | float | Score de LOF |
| `score_ocsvm` | float | Score de OC-SVM |

### Estructura de `results.json`

```json
{
  "isolation_forest": {
    "he1": {"U_statistic": ..., "p_value": ..., "rank_biserial_r": ..., "cles": ..., "he1_pass": ...},
    "he2": {"auc_roc": ..., "average_precision": ..., "he2_pass": ...},
    "he3": {"ef_at_5pct": ..., "he3_pass": ...},
    "ks": {"ks_statistic": ..., "p_value": ...},
    "bootstrap_ci_auc": {"mean": ..., "lower": ..., "upper": ...},
    "bootstrap_ci_ap": {"mean": ..., "lower": ..., "upper": ...},
    "temporal_stability": {"monthly_auc": {...}}
  },
  "lof": {...},
  "ocsvm": {...},
  "he4": {"if_wins": ..., "he4_pass": ...},
  "holm_bonferroni": {"original_p_values": [...], "adjusted_p_values": [...]}
}
```

---

## Contratos TDD

Tests a escribir **ANTES** de ampliar el modulo `evaluation.metrics` con toda la logica de HE1-HE4. Usan datos sinteticos para verificar comportamiento correcto:

| # | Test | Verifica |
|---|------|----------|
| 1 | `test_mann_whitney_perfect_separation_passes_he1` | Que scores perfectamente separados producen `he1_pass=True` |
| 2 | `test_mann_whitney_random_scores_fails_he1` | Que scores aleatorios producen `he1_pass=False` |
| 3 | `test_auc_roc_perfect_discrimination` | Que labels perfectas producen `AUC=1.0` |
| 4 | `test_auc_roc_random_is_approximately_half` | Que scores aleatorios producen `AUC ~ 0.5` (tolerancia 0.05) |
| 5 | `test_enrichment_factor_perfect_concentration` | Que cuando todas las anomalias estan en el top-k, `EF = 1/base_rate` |
| 6 | `test_bootstrap_ci_lower_leq_mean_leq_upper` | Que `lower <= mean <= upper` siempre se cumple |
| 7 | `test_holm_bonferroni_increases_p_values` | Que los p-values ajustados son >= a los originales |
| 8 | `test_compare_models_counts_wins_correctly` | Que IF con mejores metricas obtiene el conteo correcto de victorias |
| 9 | `test_full_evaluation_returns_all_keys` | Que el dict retornado contiene `he1`, `he2`, `he3`, `ks`, `bootstrap_ci_auc`, `bootstrap_ci_ap` |
| 10 | `test_compare_models_all_use_31_features` | Que los 3 modelos (IF, LOF, OC-SVM) reciben datos con 31 columnas de features |

```python
# Ejemplo de test #1 (separacion perfecta)
def test_mann_whitney_perfect_separation_passes_he1():
    scores = np.concatenate([np.ones(100) * 10, np.zeros(900)])  # anomalias con score alto
    proxy = np.concatenate([np.ones(100), np.zeros(900)])
    result = test_mann_whitney(scores, proxy)
    assert result["he1_pass"] is True
    assert result["rank_biserial_r"] > 0.10

# Ejemplo de test #7 (Holm-Bonferroni nunca reduce p-values)
def test_holm_bonferroni_increases_p_values():
    original = [0.01, 0.04, 0.03, 0.001]
    adjusted = apply_holm_bonferroni(original)
    for orig, adj in zip(original, adjusted):
        assert adj >= orig
```

---

## Gate D

No proceder a Fase 7 hasta cumplir TODOS:

| # | Condicion | Verificacion |
|---|-----------|-------------|
| 1 | Los 3 modelos evaluados en test | `results.json` tiene claves para IF, LOF, OC-SVM |
| 2 | HE1-HE4 computadas con Holm-Bonferroni | `holm_bonferroni.adjusted_p_values` presente |
| 3 | Bootstrap CIs computados | `bootstrap_ci_auc.lower < bootstrap_ci_auc.mean < bootstrap_ci_auc.upper` |
| 4 | Estabilidad temporal | `temporal_stability.monthly_auc` tiene 4 meses |
| 5 | `results.json` guardado | Archivo existe y es JSON valido |
