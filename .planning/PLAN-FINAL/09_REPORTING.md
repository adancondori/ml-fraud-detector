# Fase 8: Reporting

> Genera todas las tablas LaTeX y figuras PDF/PNG para Capitulo 2 y Capitulo 3 de la tesis. Todo artefacto visual debe salir del pipeline; esta prohibido editar `.tex` manualmente.

## Archivos

| Archivo | Clase | Proposito |
|---------|-------|-----------|
| `src/fraud_detector/reporting/__init__.py` | — | Package init |
| `src/fraud_detector/reporting/latex_tables.py` | `ThesisTableGenerator` | Tablas LaTeX con booktabs |
| `src/fraud_detector/reporting/figures.py` | `ThesisFigureGenerator` | Figuras PDF + PNG |

---

## Notas de diseño (SOLID)

### SRP — Funciones puras para generación de tablas

`ThesisTableGenerator` expone 24 metodos, uno por tabla. Esto es aceptable para un generador de reportes, siempre que **cada metodo sea una funcion pura**: recibe datos como entrada, retorna un `str` LaTeX como salida, sin efectos secundarios (no escribe a disco, no muta estado). La escritura a disco la realiza opcionalmente el caller via `output_path`, pero el metodo en si solo construye el string. Esto facilita el testing unitario: se puede verificar la salida sin tocar el filesystem.

### OCP — Patrón de registro para extensibilidad

Agregar una nueva tabla requiere actualmente modificar `ThesisTableGenerator` (añadir un método). Para cumplir mejor con OCP, se puede adoptar un **table registry pattern**:

```python
# Ejemplo conceptual (no obligatorio en v1, pero recomendado si el catálogo crece)
TABLE_REGISTRY: dict[str, Callable[[dict], str]] = {}

def register_table(name: str):
    def decorator(fn):
        TABLE_REGISTRY[name] = fn
        return fn
    return decorator

@register_table("3.10_model_comparison")
def _table_model_comparison(results: dict) -> str:
    ...
```

Esto permite añadir tablas sin modificar la clase existente. Para la v1 del pipeline, la clase monolitica con 24 metodos es aceptable, pero este patron queda documentado para futuras extensiones.

---

## Contratos de test (`test_reporting.py` — tablas)

Antes de implementar, escribir estos tests (TDD red-green-refactor):

```python
def test_table_output_contains_begin_end_table():
    """Toda tabla generada debe abrir con \\begin{table} y cerrar con \\end{table}."""
    gen = ThesisTableGenerator()
    output = gen.table_model_comparison(SAMPLE_RESULTS)
    assert r"\begin{table}" in output
    assert r"\end{table}" in output

def test_table_escapes_underscores_in_feature_names():
    """Feature names como 'txn_amount_log' deben aparecer como 'txn\\_amount\\_log'."""
    gen = ThesisTableGenerator()
    output = gen.table_feature_descriptions()
    assert r"txn\_amount\_log" in output
    assert "txn_amount_log" not in output  # sin escapar NO debe aparecer

def test_figure_saves_both_pdf_and_png(tmp_path):
    """Cada figura debe guardarse en ambos formatos."""
    gen = ThesisFigureGenerator(output_dir=tmp_path)
    gen.roc_curves(SAMPLE_SCORES, SAMPLE_PROXY)
    assert (tmp_path / "roc_curves.pdf").exists()
    assert (tmp_path / "roc_curves.png").exists()

def test_figure_closes_matplotlib_figure_after_save(tmp_path):
    """Después de guardar, la figura matplotlib debe cerrarse (no acumular memoria)."""
    import matplotlib.pyplot as plt
    gen = ThesisFigureGenerator(output_dir=tmp_path)
    figs_before = len(plt.get_fignums())
    gen.roc_curves(SAMPLE_SCORES, SAMPLE_PROXY)
    figs_after = len(plt.get_fignums())
    assert figs_after == figs_before  # no hay figuras abiertas nuevas

def test_empty_results_handled_gracefully():
    """Con resultados vacíos, el generador no debe lanzar excepción."""
    gen = ThesisTableGenerator()
    output = gen.table_hypothesis_summary({})
    assert isinstance(output, str)

def test_missing_grid_csv_skips_heatmap(tmp_path, caplog):
    """Si grid_search_if.csv no existe, se emite warning y se omite la figura."""
    gen = ThesisFigureGenerator(output_dir=tmp_path)
    # No crear el CSV — simular ausencia
    # El orquestador (no el generador) maneja esto, pero el test valida el flujo
    assert not (tmp_path / "grid_search_heatmap.pdf").exists()
```

---

## `ThesisTableGenerator` (`latex_tables.py`)

### Helper: `_escape_latex`

```python
def _escape_latex(self, text: str) -> str:
    special_chars = {
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for char, escaped in special_chars.items():
        text = text.replace(char, escaped)
    return text
```

Critico para feature names con underscores (e.g., `txn_amount_log` → `txn\_amount\_log`).

### Convenciones de salida

- Cada metodo retorna un `str` con el contenido LaTeX completo (`\begin{table}` a `\end{table}`).
- Cada metodo acepta `output_path: Optional[Path]` para guardar a disco.
- Formato: **booktabs** (`\toprule`, `\midrule`, `\bottomrule`).
- Caption y label incluidos en cada tabla.
- Metricas: **4 decimales** (e.g., `0.7234`).
- Porcentajes: **2 decimales** (e.g., `6.33\%`).

### Catalogo de tablas

| Metodo | Tabla # | Descripcion | Input |
|--------|---------|-------------|-------|
| `table_dataset_summary(split_info)` | 3.5 | Resumen del dataset (splits, tasas de proxy, rangos de fecha) | `dict` con info por split |
| `table_feature_descriptions()` | 3.6 | Catalogo de las 31 features (nombre, tipo, descripcion) | Estatico desde `FEATURE_NAMES` |
| `table_feature_statistics(stats_df)` | 3.7 | Estadisticas descriptivas (mean, std, min, max) de las 31 features | `pd.DataFrame` |
| `table_grid_search_top10(grid_df)` | 3.8 | Top 10 configuraciones del grid search de IF | `pd.DataFrame` de CSV |
| `table_model_comparison(results)` | 3.10 | Comparacion de 3 modelos x 4+ metricas | `dict` de resultados |
| `table_he1_results(results)` | 3.11 | Mann-Whitney U, p-value, rank-biserial r, CLES | `dict` por modelo |
| `table_he2_results(results)` | 3.12 | AUC-ROC + AP con intervalos de confianza | `dict` por modelo |
| `table_he3_results(results)` | 3.13 | Enrichment Factor a multiples k (1%, 2%, 5%, 10%) | `dict` por modelo |
| `table_he4_comparison(results)` | 3.14 | Matriz IF vs competidores (4 metricas, ganador marcado) | `dict` de comparacion |
| `table_bootstrap_ci(results)` | 3.15 | Bootstrap CIs (mean, lower, upper) por modelo y metrica | `dict` por modelo |
| `table_sensitivity_proxy(results)` | 3.16 | Proxy estricto vs amplio (AUC, AP, delta) | `dict` de sensibilidad |
| `table_sensitivity_feature18(results)` | 3.17 | 31 vs 30 features (AUC, delta, Jaccard, Spearman) | `dict` de sensibilidad |
| `table_ablation_31vs21(results)` | 3.18 | Ablacion IF-31 vs IF-21: AUC-ROC, AP, P@5%, EF con deltas | `dict` de ablacion |
| `table_temporal_stability(results)` | 3.19 | AUC mensual por modelo (Sep, Oct, Nov, Dic) | `dict` temporal |
| `table_hypothesis_summary(results)` | 3.20 | Resumen HE1-HE4 con veredicto (respaldada/rechazada) | `dict` de resultados |
| `table_metrics_by_role(results)` | 3.21 | AUC-ROC, AP, P@5%, EF por rol de usuario | `dict` de metricas por rol |
| `table_metrics_by_category(results)` | 3.22 | AUC-ROC, AP, P@5%, EF por categoria de pago | `dict` de metricas por categoria |
| `table_anomaly_types(results)` | 3.23 | Tipologia de 9 tipos de anomalia con features dominantes y descripciones | `dict` de tipologia |
| `table_user_risk_profile(results)` | 3.24 | Perfil de riesgo agregado por usuario (metricas de riesgo) | `dict` de perfiles |
| `table_posthoc_facility(results)` | 3.25 | Top 10 centros con mayor concentracion de anomalias | `dict` post-hoc |
| `table_posthoc_manager(results)` | 3.26 | Top 10 actores/usuarios asociados con mayor concentracion de anomalias y descuentos; si no hay identidad validada, exporta agregado anonimo | `dict` post-hoc |
| `table_posthoc_currency(results)` | 3.27 | Distribucion de anomalias por moneda | `dict` post-hoc |
| `table_posthoc_discount_pattern(results)` | 3.28 | Top pares centro-actor con patron de descuento anomalo; degradado a centro si no hay identidad validada | `dict` post-hoc |

### Tablas que requieren datos fuera de `results.json`

Las tablas 3.5, 3.6, 3.7, 3.23 y 3.24 no se derivan de `results.json`. El orquestador debe:

1. **Tabla 3.5:** Computar `split_info` (filas, proxy rate, date range) desde los parquets de features.
2. **Tabla 3.6:** Usar `FEATURE_NAMES` y descripciones estaticas definidas en `engineering.py`.
3. **Tabla 3.7:** Computar `describe()` sobre `train_features.parquet`.

```python
# En step8_generate_reports():
split_info = {}
for name in ["train", "val", "test"]:
    df = pd.read_parquet(settings.processed_dir / f"{name}_features.parquet")
    proxy = DataManager.assign_proxy_labels(df, proxy_type)
    split_info[name] = {
        "rows": len(df),
        "proxy_rate": float(proxy.mean()),
        "date_range": (str(df["created_at"].min()), str(df["created_at"].max())),
    }
    del df

train_df = pd.read_parquet(settings.processed_dir / "train_features.parquet")
stats_df = train_df[FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]]
del train_df
```

4. **Tabla 3.18 (ablacion 31 vs 21):** Requiere ejecutar IF con 31 features y con 21 features (subset base), luego comparar AUC-ROC, AP, P@5%, EF y sus deltas.
5. **Tabla 3.23 (tipologia de anomalias):** Requiere clustering o analisis de los top-5% anomalias para identificar los 9 tipos, con features dominantes por tipo.
6. **Tabla 3.24 (perfil de riesgo por usuario):** Agregar scores de anomalia y metricas por `user_id` desde los resultados de prediccion.

---

## `ThesisFigureGenerator` (`figures.py`)

### Configuracion de estilo

```python
STYLE_CONFIG = {
    "font_family": "serif",
    "figure_width": 6.0,        # pulgadas (ancho \textwidth en tesis)
    "figure_height": 4.0,       # pulgadas
    "dpi": 300,
    "output_formats": ["pdf", "png"],
    "language": "es",
}
```

### Reglas obligatorias

- `plt.rcParams["font.family"] = "serif"` al inicio de cada figura.
- `plt.close()` despues de cada `savefig()` para liberar memoria.
- Guardar **PDF** (para `\includegraphics` en LaTeX) **y PNG** (para visualizacion en notebooks).
- Etiquetas de ejes, titulos y leyendas en **espanol**.

### Helper: `_save_figure`

```python
def _save_figure(self, fig, name: str):
    """Guarda figura en PDF y PNG, luego cierra."""
    for fmt in ["pdf", "png"]:
        path = self.output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Figure saved: {path}")
    plt.close(fig)
```

### Catalogo de figuras

| Metodo | Descripcion | Input |
|--------|-------------|-------|
| `roc_curves(scores_dict, proxy)` | Curvas ROC de 3 modelos + linea diagonal de referencia aleatoria. AUC en leyenda. | `{model: scores_array}` |
| `pr_curves(scores_dict, proxy)` | Curvas Precision-Recall de 3 modelos + linea horizontal en `base_rate`. AP en leyenda. | `{model: scores_array}` |
| `score_distributions(scores_dict, proxy)` | 3 subplots (uno por modelo). Histograma proxy+ (naranja) vs proxy- (azul) superpuesto con transparencia. | `{model: scores_array}` |
| `grid_search_heatmap(grid_df)` | Heatmap 2D: `max_samples` (eje Y) x `n_estimators` (eje X). Valor = media de AUC sobre `max_features`. | `pd.DataFrame` |
| `shap_summary(model, X_sample, feature_names)` | SHAP summary plot con top 15 features (de 31 totales). TreeExplainer con fallback a KernelExplainer. | Modelo IF, array, lista de nombres |
| `enrichment_curve(scores_dict, proxy)` | Curva de enriquecimiento (lift) de k=0.1% a k=100%. Linea horizontal en EF=1. 3 modelos superpuestos. | `{model: scores_array}` |
| `temporal_stability_plot(results)` | Serie temporal de AUC mensual por modelo. Puntos conectados, un color por modelo. | `dict` con resultados temporales |
| `correlation_matrix(X, feature_names)` | Heatmap de correlaciones de Pearson entre las 31 features. Colores divergentes, anotaciones en celdas. | Array + nombres |
| `anomaly_type_distribution(type_results)` | Barplot horizontal: distribucion de 9 tipos de anomalia en el top-5% de scores. Anotacion con porcentaje y count. | `dict` de tipologia |
| `posthoc_facility_anomaly_rate(posthoc_results)` | Barplot horizontal: top 15 centros ordenados por tasa de anomalias. Linea vertical en tasa base (5%). | `dict` post-hoc |
| `posthoc_discount_by_manager(posthoc_results)` | Scatter plot: monto total de descuento (X) vs tasa de anomalias (Y) por actor; si no hay identidad validada, usar version agregada o pseudonimizada. Tamano de punto = n_transacciones. | `dict` post-hoc |
| `posthoc_currency_distribution(posthoc_results)` | Barplot agrupado: anomalias vs normales por moneda, con tasa de anomalias anotada. | `dict` post-hoc |

### Grid search heatmap: manejo de CSV ausente

Si el pipeline se ejecuto con `--skip-grid-search`, el archivo `grid_search_if.csv` no existe. El orquestador debe verificar antes de llamar:

```python
grid_path = settings.output_dir / "grid_search_if.csv"
if grid_path.exists():
    grid_df = pd.read_csv(grid_path)
    fig_gen.grid_search_heatmap(grid_df)
else:
    logger.warning("grid_search_if.csv no encontrado, omitiendo heatmap")
```

La figura se omite gracefully sin error.

---

## Reglas de calidad

| # | Regla | Verificacion |
|---|-------|-------------|
| 1 | Todas las tablas compilan con `pdflatex` sin errores | Test de compilacion |
| 2 | Todas las figuras PDF/PNG tienen tamano > 0 bytes | `os.path.getsize() > 0` |
| 3 | Feature names con underscores escapados en LaTeX | `_escape_latex` aplicado |
| 4 | `plt.close()` despues de cada figura | Sin acumulacion de memoria |
| 5 | Etiquetas en espanol | Revision visual |
| 6 | Grid search heatmap se omite si CSV falta | Log warning, sin excepcion |
| 7 | Metricas a 4 decimales, porcentajes a 2 decimales | Formato consistente |
| 8 | Todas las tablas tienen `\caption` y `\label` | Compilacion LaTeX |

## Gate editorial y de gobernanza

Antes de exportar tablas 3.26 y 3.28 o la figura `posthoc_discount_by_manager`, el orquestador debe revisar `results_posthoc.json`:

- `actor_identity_validated = true`: se permite exportar actores individuales, idealmente pseudonimizados si la tesis sale del ambito interno.
- `actor_identity_validated = false`: queda prohibido exportar identificadores individuales. La tabla 3.26 debe degradarse a un resumen agregado de `paid_by_manager`, y la 3.28 a top centros con descuentos anomalos sin actor individual.

Ademas:

- toda tabla o figura post-hoc debe incluir disclaimer metodologico en caption o manifest;
- la version publica de la tesis no debe incluir ids crudos ni nombres personales;
- si se usan nombres de facility en documento publico, deben pseudonimizarse.

---

## Entregables

### Modulos

| Archivo | Contenido |
|---------|-----------|
| `src/fraud_detector/reporting/__init__.py` | Imports de `ThesisTableGenerator` y `ThesisFigureGenerator` |
| `src/fraud_detector/reporting/latex_tables.py` | Clase con 24 metodos de generacion de tablas |
| `src/fraud_detector/reporting/figures.py` | Clase con 12 metodos de generacion de figuras |

### Tablas LaTeX

| Archivo | Tabla |
|---------|-------|
| `output/tables/table_3_05_dataset_summary.tex` | 3.5 |
| `output/tables/table_3_06_feature_descriptions.tex` | 3.6 |
| `output/tables/table_3_07_feature_statistics.tex` | 3.7 |
| `output/tables/table_3_08_grid_search_top10.tex` | 3.8 |
| `output/tables/table_3_10_model_comparison.tex` | 3.10 |
| `output/tables/table_3_11_he1_results.tex` | 3.11 |
| `output/tables/table_3_12_he2_results.tex` | 3.12 |
| `output/tables/table_3_13_he3_results.tex` | 3.13 |
| `output/tables/table_3_14_he4_comparison.tex` | 3.14 |
| `output/tables/table_3_15_bootstrap_ci.tex` | 3.15 |
| `output/tables/table_3_16_sensitivity_proxy.tex` | 3.16 |
| `output/tables/table_3_17_sensitivity_feature18.tex` | 3.17 |
| `output/tables/table_3_18_ablation_33vs23.tex` | 3.18 |
| `output/tables/table_3_19_temporal_stability.tex` | 3.19 |
| `output/tables/table_3_20_hypothesis_summary.tex` | 3.20 |
| `output/tables/table_3_21_metrics_by_role.tex` | 3.21 |
| `output/tables/table_3_22_metrics_by_category.tex` | 3.22 |
| `output/tables/table_3_23_anomaly_types.tex` | 3.23 |
| `output/tables/table_3_24_user_risk_profile.tex` | 3.24 |
| `output/tables/table_3_25_posthoc_facility.tex` | 3.25 |
| `output/tables/table_3_26_posthoc_manager.tex` | 3.26 |
| `output/tables/table_3_27_posthoc_currency.tex` | 3.27 |
| `output/tables/table_3_28_posthoc_discount_pattern.tex` | 3.28 |

### Figuras

| Archivo (sin extension) | Formatos | Descripcion |
|--------------------------|----------|-------------|
| `output/figures/roc_curves` | `.pdf`, `.png` | Curvas ROC |
| `output/figures/pr_curves` | `.pdf`, `.png` | Curvas Precision-Recall |
| `output/figures/score_distributions` | `.pdf`, `.png` | Histogramas de scores |
| `output/figures/grid_search_heatmap` | `.pdf`, `.png` | Heatmap IF (si CSV existe) |
| `output/figures/shap_summary` | `.pdf`, `.png` | SHAP top 15 features (de 31 totales) |
| `output/figures/enrichment_curve` | `.pdf`, `.png` | Curva de enriquecimiento |
| `output/figures/temporal_stability` | `.pdf`, `.png` | AUC mensual |
| `output/figures/correlation_matrix` | `.pdf`, `.png` | Correlacion entre 31 features |
| `output/figures/anomaly_type_distribution` | `.pdf`, `.png` | Distribucion de 9 tipos de anomalia en top-5% |
| `output/figures/posthoc_facility_anomaly_rate` | `.pdf`, `.png` | Tasa de anomalias por centro (top 15) |
| `output/figures/posthoc_discount_by_manager` | `.pdf`, `.png` | Descuento vs anomalias por manager |
| `output/figures/posthoc_currency_distribution` | `.pdf`, `.png` | Distribucion de anomalias por moneda |
