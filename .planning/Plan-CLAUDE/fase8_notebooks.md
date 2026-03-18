# Fase 8: Notebooks

## Archivos nuevos

| Archivo | Contenido |
|---------|-----------|
| `notebooks/01_eda_capitulo2.ipynb` | EDA para Capitulo 2 de la tesis |
| `notebooks/02_resultados_capitulo3.ipynb` | Visualizacion de resultados Cap 3 |

**Eliminar**: `notebooks/01_exploratory_analysis.ipynb` (scaffolding viejo).

---

## 01_eda_capitulo2.ipynb

### Proposito

Analisis exploratorio de datos para el Capitulo 2 de la tesis. Genera visualizaciones y estadisticas descriptivas del dataset de transacciones de TechSport Inc.

### Correcciones respecto a la version anterior

1. **FIX: No cargar todos los splits simultaneamente** (pico de ~10 GB). Cargar uno a la vez para estadisticas por split, borrar antes de cargar el siguiente.
2. **FIX: Eliminar `sys.path.insert(0, "..")`** -- usar `pip install -e .` en su lugar.
3. **Agregar celdas markdown** explicando cada analisis y que seccion de la tesis respalda.
4. **Usar `settings.figures_dir`** en vez de rutas relativas hardcodeadas.
5. **Guardar figuras** como PDF y PNG.

### Celdas del notebook

#### Celda 1: Setup

```python
# Requires: pip install -e . (from project root)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import settings
from fraud_detector.data.loader import DataManager

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "font.family": "serif",
})

figures_dir = settings.figures_dir
figures_dir.mkdir(parents=True, exist_ok=True)
```

#### Celda 2: Resumen del dataset (carga secuencial para evitar pico de memoria)

```markdown
## 2.1 Resumen del dataset
Seccion de la tesis: Capitulo 2, Descripcion del dataset.
Cargamos cada split por separado para evitar exceder la memoria disponible.
```

```python
dm = DataManager(settings)
split_stats = {}
for name in ["train", "val", "test"]:
    df = dm.load_split(name)
    proxy = DataManager.assign_proxy_labels(df, "strict")
    split_stats[name] = {
        "rows": len(df),
        "proxy_rate": proxy.mean(),
        "date_min": df["created_at"].min(),
        "date_max": df["created_at"].max(),
    }
    print(f"{name}: {len(df):,} rows, proxy rate: {proxy.mean()*100:.2f}%, "
          f"range: {df['created_at'].min()} to {df['created_at'].max()}")
    del df, proxy

total = sum(s["rows"] for s in split_stats.values())
print(f"\nTotal: {total:,} transacciones")
dm.close()
```

#### Celda 3: Distribucion de montos (carga solo train para este analisis)

```markdown
## 2.2 Distribucion de montos
Seccion de la tesis: Capitulo 2, Variables del dataset.
```

```python
train = pd.read_parquet(settings.processed_dir / "train_raw.parquet")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train["amount"].clip(upper=500), bins=100, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Monto (USD)")
axes[0].set_ylabel("Frecuencia")
axes[0].set_title("Distribucion de Montos (truncado a $500)")

axes[1].hist(np.log1p(train["amount"]), bins=100, edgecolor="black", alpha=0.7)
axes[1].set_xlabel("log(1 + Monto)")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title("Distribucion de log(Monto)")

plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(figures_dir / f"eda_amount_distribution.{fmt}")
plt.show()
```

#### Celda 4: Distribucion de status

```markdown
## 2.3 Distribucion de status
Seccion de la tesis: Capitulo 2, Variable criterio (proxy).
```

```python
status_counts = train["status"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
status_counts.plot(kind="barh", ax=ax)
ax.set_xlabel("Cantidad de Transacciones")
ax.set_title("Distribucion de Status de Transacciones")
plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(figures_dir / f"eda_status_distribution.{fmt}")
plt.show()

proxy_strict = DataManager.assign_proxy_labels(train, "strict")
proxy_wide = DataManager.assign_proxy_labels(train, "wide")
print(f"Tasa proxy strict (train): {proxy_strict.mean()*100:.2f}%")
print(f"Tasa proxy wide (train):   {proxy_wide.mean()*100:.2f}%")
```

#### Celda 5: Metodos de pago

```markdown
## 2.4 Metodos de pago
Seccion de la tesis: Capitulo 2, Variables del dataset.
```

```python
method_counts = train["payment_method"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
method_counts.head(10).plot(kind="bar", ax=ax)
ax.set_xlabel("Metodo de Pago")
ax.set_ylabel("Cantidad")
ax.set_title("Top 10 Metodos de Pago")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(figures_dir / f"eda_payment_methods.{fmt}")
plt.show()
```

#### Celda 6: Patrones temporales (hora, dia, mes)

```markdown
## 2.5 Patrones temporales
Seccion de la tesis: Capitulo 2, Analisis temporal.
```

```python
train["created_at"] = pd.to_datetime(train["created_at"])
train["hour"] = train["created_at"].dt.hour
train["day_of_week"] = train["created_at"].dt.dayofweek
train["month"] = train["created_at"].dt.to_period("M")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

train.groupby("hour").size().plot(ax=axes[0])
axes[0].set_title("Transacciones por Hora del Dia")
axes[0].set_xlabel("Hora")

dow_labels = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
dow_counts = train.groupby("day_of_week").size()
axes[1].bar(range(7), dow_counts.values)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(dow_labels)
axes[1].set_title("Transacciones por Dia de Semana")

monthly = train.groupby("month").size()
monthly.plot(ax=axes[2])
axes[2].set_title("Transacciones por Mes")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(figures_dir / f"eda_temporal_patterns.{fmt}")
plt.show()
```

#### Celda 7: Tasa proxy por mes

```markdown
## 2.6 Tasa de proxy por mes
Seccion de la tesis: Capitulo 2, Estabilidad temporal del proxy.
```

```python
train["_proxy"] = DataManager.assign_proxy_labels(train, "strict")
monthly_proxy = train.groupby("month")["_proxy"].mean() * 100

fig, ax = plt.subplots(figsize=(10, 4))
monthly_proxy.plot(ax=ax, marker="o")
ax.set_ylabel("Tasa Proxy (%)")
ax.set_title("Tasa de Proxy Anomalias por Mes")
ax.axhline(y=train["_proxy"].mean()*100, color="red", linestyle="--", label="Promedio")
ax.legend()
plt.tight_layout()
for fmt in ["pdf", "png"]:
    plt.savefig(figures_dir / f"eda_proxy_rate_monthly.{fmt}")
plt.show()
```

#### Celda 8: Estadisticas descriptivas

```markdown
## 2.7 Estadisticas descriptivas
Seccion de la tesis: Capitulo 2, Tabla de variables.
```

```python
numerical_cols = ["amount", "technology_fee", "tax", "tip", "discount"]
desc = train[numerical_cols].describe().round(2)
print(desc.to_string())

del train  # Liberar memoria
```

---

## 02_resultados_capitulo3.ipynb

### Proposito

Carga `output/results.json` y visualiza los resultados del pipeline. Util para iterar rapidamente sobre presentacion sin re-ejecutar el pipeline.

### Correcciones respecto a la version anterior

1. **FIX: Eliminar `sys.path.insert(0, "..")`** -- usar `pip install -e .`.
2. **Usar `IPython.display.Image`** para mostrar PNG inline (PDFs no renderizan inline).
3. **Agregar celdas markdown** con contexto de cada seccion.

### Celdas del notebook

#### Celda 1: Setup

```python
# Requires: pip install -e . (from project root)
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from config.config import settings

with open(settings.output_dir / "results.json") as f:
    results = json.load(f)

print("Modelos evaluados:", [k for k in results.keys()
      if k not in ["he4", "sensitivity_proxy", "sensitivity_feature17", "proxy_type"]])
```

#### Celda 2: Resumen HE1-HE4

```markdown
## 3.1 Resumen de hipotesis especificas
```

```python
for model in ["isolation_forest", "lof", "ocsvm"]:
    r = results[model]
    print(f"\n{'='*50}")
    print(f"Modelo: {model}")
    print(f"  HE1 (Mann-Whitney): p={r['he1']['p_value']:.2e}, "
          f"r={r['he1']['rank_biserial_r']:.4f} "
          f"-> {'PASA' if r['he1']['he1_pass'] else 'NO PASA'}")
    print(f"  HE2 (Discriminacion): AUC={r['he2']['auc_roc']:.4f}, "
          f"AP={r['he2']['average_precision']:.4f} "
          f"-> {'PASA' if r['he2']['he2_pass'] else 'NO PASA'}")
    print(f"  HE3 (Top-K): EF@5%={r['he3']['enrichment_factor_5pct']:.2f} "
          f"-> {'PASA' if r['he3']['he3_pass'] else 'NO PASA'}")
```

#### Celda 3: Tabla comparativa

```python
rows = []
for model in ["isolation_forest", "lof", "ocsvm"]:
    r = results[model]
    rows.append({
        "Modelo": model,
        "AUC-ROC": r["he2"]["auc_roc"],
        "AP": r["he2"]["average_precision"],
        "EF@5%": r["he3"]["enrichment_factor_5pct"],
    })
df_comp = pd.DataFrame(rows).set_index("Modelo")
df_comp.style.highlight_max(axis=0)
```

#### Celda 4: Bootstrap CI

```python
for model in ["isolation_forest", "lof", "ocsvm"]:
    r = results[model]
    ci_auc = r["bootstrap_ci_auc"]
    ci_ap = r["bootstrap_ci_ap"]
    print(f"{model}:")
    print(f"  AUC: {ci_auc['mean']:.4f} [{ci_auc['lower']:.4f}, {ci_auc['upper']:.4f}]")
    print(f"  AP:  {ci_ap['mean']:.4f} [{ci_ap['lower']:.4f}, {ci_ap['upper']:.4f}]")
```

#### Celda 5: HE4 Comparacion

```python
he4 = results["he4"]
print(f"IF gana en {he4['if_wins_count']}/{he4['total_metrics']} metricas")
print(f"HE4: {'PASA' if he4['he4_pass'] else 'NO PASA'}")
```

#### Celda 6: Estabilidad temporal

```python
if "temporal_stability" in results.get("isolation_forest", {}):
    ts = results["isolation_forest"]["temporal_stability"]
    print("Estabilidad temporal (AUC mensual):")
    for month, auc in ts.items():
        print(f"  {month}: {auc:.4f}")
```

#### Celda 7: Sensibilidad proxy

```python
sp = results.get("sensitivity_proxy", {})
if sp:
    print(f"AUC strict: {sp['auc_strict']:.4f}")
    print(f"AUC wide:   {sp['auc_wide']:.4f}")
    print(f"Delta AUC:  {sp['delta_auc']:.4f}")
    print(f"AP strict:  {sp['ap_strict']:.4f}")
    print(f"AP wide:    {sp['ap_wide']:.4f}")
    print(f"Delta AP:   {sp['delta_ap']:.4f}")
```

#### Celda 8: Sensibilidad feature 17

```python
sf = results.get("sensitivity_feature17", {})
if sf:
    print(f"AUC con feature 17:    {sf['auc_with']:.4f}")
    print(f"AUC sin feature 17:    {sf['auc_without']:.4f}")
    print(f"Delta AUC:             {sf['delta_auc']:.4f}")
    print(f"Jaccard similarity:    {sf['jaccard']:.4f}")
    print(f"Spearman correlation:  {sf['spearman']:.4f}")
```

#### Celda 9: Mostrar figuras generadas (PNG para inline)

```python
import os

fig_dir = settings.figures_dir
for f in sorted(os.listdir(fig_dir)):
    if f.endswith(".png"):
        print(f"\n--- {f} ---")
        display(Image(filename=str(fig_dir / f), width=600))
```
