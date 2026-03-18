# Fase 6: Reporting

## Archivos nuevos
- `src/fraud_detector/reporting/__init__.py`
- `src/fraud_detector/reporting/latex_tables.py`
- `src/fraud_detector/reporting/figures.py`

---

## `ThesisTableGenerator` (latex_tables.py)

### Helper `_escape_latex`

```python
def _escape_latex(self, text: str) -> str:
    """Escapa caracteres especiales de LaTeX en contenido dinamico.

    Critico para feature names con underscores (e.g., txn_amount_log).
    """
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

### Convencion de salida

- Cada metodo retorna un `str` con el contenido LaTeX completo (desde `\begin{table}` hasta `\end{table}`).
- Cada metodo acepta un parametro opcional `output_path: Optional[Path]` para guardar a disco.
- Formato de tabla: `booktabs` (con `\toprule`, `\midrule`, `\bottomrule`).
- Caption y label incluidos en cada tabla.
- Numeros formateados a 4 decimales para metricas, 2 decimales para porcentajes.

### Tablas a generar (con firmas de metodo)

| Metodo | Tabla | Input | Descripcion |
|--------|-------|-------|-------------|
| `table_dataset_summary(split_info: dict)` | 3.5 | `{name: {rows, proxy_rate, date_range}}` | Resumen del dataset |
| `table_feature_descriptions()` | 3.6 | Estatico: `FEATURE_NAMES` + descripciones | Catalogo de features |
| `table_feature_statistics(stats_df: pd.DataFrame)` | 3.7 | DataFrame con mean/std/min/max por feature | Estadisticas descriptivas |
| `table_grid_search_top10(grid_df: pd.DataFrame)` | 3.8 | Grid search CSV top 10 por AUC | Resultados del grid search |
| `table_model_comparison(results: dict)` | 3.10 | `{model: {auc, ap, p@5%, r@5%}}` | Comparacion de 3 modelos |
| `table_he1_results(results: dict)` | 3.11 | `{model: {U, p, r, cles}}` | Resultados Mann-Whitney |
| `table_he2_results(results: dict)` | 3.12 | `{model: {auc, ap} + CI}` | Discriminacion + CI |
| `table_he3_results(results: dict)` | 3.13 | `{model: topk at 1,2,5,10%}` | Enriquecimiento a multiples k |
| `table_he4_comparison(results: dict)` | 3.14 | Matriz de comparacion | IF vs competidores |
| `table_bootstrap_ci(results: dict)` | 3.15 | `{model: {ci_auc, ci_ap}}` | Bootstrap CIs |
| `table_sensitivity_proxy(results: dict)` | 3.16 | strict vs wide AUC/AP | Sensibilidad de proxy |
| `table_sensitivity_feature17(results: dict)` | 3.17 | AUC delta + Jaccard + Spearman | Feature 17 |
| `table_temporal_stability(results: dict)` | 3.18 | AUC mensual por modelo | Estabilidad temporal |
| `table_hypothesis_summary(results: dict)` | 3.19 | HE1-HE4 pass/fail summary | Veredicto final |

### NOTA: Tablas que requieren datos fuera de results.json

Las tablas 3.5, 3.6 y 3.7 requieren datos que NO estan en `results.json`. El orquestador debe:

1. **Tabla 3.5**: Computar `split_info` (conteo de filas, tasas de proxy, rangos de fechas) desde los archivos parquet.
2. **Tabla 3.6**: Usar `FEATURE_NAMES` y descripciones estaticas de `engineering.py`.
3. **Tabla 3.7**: Computar estadisticas descriptivas desde `train_features.parquet`.

El Step 8 del orquestador debe pasar estos datos al generador de tablas por separado:

```python
# En step8_generate_reports():
split_info = {}
for name in ["train", "val", "test"]:
    df = pd.read_parquet(settings.processed_dir / f"{name}_features.parquet")
    proxy = DataManager.assign_proxy_labels(df, proxy_type)
    split_info[name] = {
        "rows": len(df),
        "proxy_rate": proxy.mean(),
        "date_range": (str(df["created_at"].min()), str(df["created_at"].max())),
    }
    del df

train_df = pd.read_parquet(settings.processed_dir / "train_features.parquet")
stats_df = train_df[FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]]
del train_df

tg.table_dataset_summary(split_info)
tg.table_feature_descriptions()
tg.table_feature_statistics(stats_df)
```

---

## `ThesisFigureGenerator` (figures.py)

### Configuracion de estilo

```python
STYLE_CONFIG = {
    "font_family": "serif",
    "figure_width": 6.0,       # pulgadas (ancho \textwidth en tesis)
    "figure_height": 4.0,      # pulgadas (ajustable por figura)
    "dpi": 300,
    "output_formats": ["pdf", "png"],  # PDF para LaTeX, PNG para notebooks
    "language": "es",           # Etiquetas en espanol
}
```

Reglas obligatorias:

- `plt.rcParams["font.family"] = "serif"` al inicio de cada figura.
- `plt.close()` despues de cada `savefig()` para liberar memoria.
- Guardar PDF (para `\includegraphics` en LaTeX) Y PNG (para visualizacion en notebooks) lado a lado.
- Etiquetas de ejes, titulos y leyendas en espanol.

### Figuras a generar (con firmas)

| Metodo | Descripcion | Input |
|--------|-------------|-------|
| `roc_curves(scores_dict, proxy)` | Curvas ROC de 3 modelos + diagonal de referencia | `{model: scores_array}` |
| `pr_curves(scores_dict, proxy)` | Curvas Precision-Recall de 3 modelos + linea base_rate | `{model: scores_array}` |
| `score_distributions(scores_dict, proxy)` | Histogramas superpuestos por modelo (3 subplots, normal vs proxy) | `{model: scores_array}` |
| `grid_search_heatmap(grid_df)` | Heatmap 2D: `max_samples` x `n_estimators` (media de AUC sobre `max_features`) | DataFrame |
| `shap_summary(model, X_sample, feature_names)` | SHAP summary plot (top 10 features) con try/except fallback | Modelo IF |
| `enrichment_curve(scores_dict, proxy)` | Curva de enriquecimiento (lift) de k=0.1% a k=100% | `{model: scores_array}` |
| `temporal_stability_plot(results)` | Serie temporal de AUC mensual por modelo | Dict con resultados temporales |
| `correlation_matrix(X, feature_names)` | Heatmap de correlaciones entre features | Array escalado + nombres |

### SHAP: compatibilidad y fallback

`TreeExplainer` funciona con `IsolationForest` desde SHAP v0.39+. Envolver en try/except con KernelSHAP como fallback:

```python
def shap_summary(self, model, X_sample, feature_names):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            max_display=10,
            show=False,
        )
    except Exception as e:
        logger.warning(f"TreeExplainer fallo ({e}), intentando KernelExplainer...")
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.decision_function, background)
        shap_values = explainer.shap_values(X_sample[:500])
        shap.summary_plot(
            shap_values, X_sample[:500],
            feature_names=feature_names,
            max_display=10,
            show=False,
        )
```

### Salida dual: PDF + PNG

Cada metodo de figura debe guardar ambos formatos:

```python
def _save_figure(self, fig, name: str):
    """Guarda figura en PDF y PNG, luego cierra."""
    for fmt in ["pdf", "png"]:
        path = self.output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Figure saved: {path}")
    plt.close(fig)
```

### Manejo de grid_search_results.csv ausente

Step 8 del orquestador debe verificar que `grid_search_if.csv` exista antes de llamar a `grid_search_heatmap()`. Si el pipeline se ejecuto con `--skip-grid-search`, el archivo no existe y la figura se omite:

```python
grid_path = settings.output_dir / "grid_search_if.csv"
if grid_path.exists():
    grid_df = pd.read_csv(grid_path)
    fig_gen.grid_search_heatmap(grid_df)
else:
    logger.warning("grid_search_if.csv no encontrado, omitiendo heatmap")
```

---

## Dependencias

- `matplotlib` (figuras)
- `seaborn` (heatmaps, paletas)
- `shap` (SHAP summary)
- `pandas` (manipulacion de datos para tablas)
- `numpy` (operaciones numericas)

Todas ya estan en `requirements.txt`.

## Criterios de aceptacion

1. Todas las tablas compilan con `pdflatex` sin errores.
2. Todas las figuras se generan en PDF y PNG con tamano > 0 bytes.
3. Los caracteres especiales de LaTeX estan correctamente escapados (especialmente `_` en feature names).
4. `plt.close()` se llama despues de cada figura (sin acumulacion de memoria).
5. Las etiquetas de figuras estan en espanol.
6. El heatmap de grid search se omite gracefully cuando falta el CSV.
