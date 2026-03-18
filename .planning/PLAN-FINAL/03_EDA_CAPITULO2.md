# Fase 2. EDA y Diagnostico — Capitulo 2

> Generar toda la evidencia descriptiva para el OE2 (diagnostico del estado transaccional) y dejar el Capitulo 2 de la tesis completamente soportado por artefactos reproducibles del pipeline.

---

## 1. Proposito

Producir evidencia empirica para el Objetivo Especifico 2 (OE2): diagnosticar el estado actual de las transacciones de pagos digitales de TechSport Inc. Cada tabla, figura y estadistica del Capitulo 2 debe ser generada programaticamente, no editada manualmente.

**Restriccion critica:** Los numeros generados por el EDA deben reproducir exactamente los valores ya presentes en `02_diagnostico.tex` de la tesis. Si hay discrepancias, se corrige el `.tex` desde los artefactos del pipeline (no al reves).

---

## 2. Analisis requeridos

### 2.1 Distribucion estructural

| Dimension | Detalle |
|-----------|---------|
| `status` | Conteo y porcentaje por cada valor. Tabla principal del diagnostico. |
| `source_enum` (canal) | Distribucion por canal de origen (app, web, API, POS, etc.) |
| `gateway` | Distribucion por gateway de pago |
| `payment_method` | Top 10+ metodos de pago con conteos |
| Mes | Volumen mensual (Ene-Dic 2025) — serie temporal |

### 2.2 Estadisticas descriptivas de montos

| Metrica | Agrupacion |
|---------|-----------|
| Media, mediana, P95, P99, min, max | Global |
| Media, mediana, P95, P99 | Por status (captured, totally_refunded, refunded_to_credit, partially_refunded) |
| Histograma de montos (truncado a $500) | Global |
| Histograma de log(1 + amount) | Global |

### 2.3 Comparacion proxy+ vs. proxy-

| Analisis | Proxy+ | Proxy- |
|----------|--------|--------|
| N, porcentaje | `status IN (totally_refunded, refunded_to_credit)` | Resto |
| Montos: media, mediana, P95, desv. estandar | Por grupo | Por grupo |
| Distribucion horaria | Frecuencia por hora del dia | Frecuencia por hora del dia |
| Distribucion por canal | Top canales por grupo | Top canales por grupo |
| Distribucion por gateway | Top gateways por grupo | Top gateways por grupo |
| Comportamiento de usuario | Transacciones promedio por usuario | Transacciones promedio por usuario |

Incluir tasa proxy estricta (~6.33%) y amplia (~7.55%) con sus conteos.

### 2.4 Analisis de outliers y patrones extremos

| Hallazgo | Detalle |
|----------|---------|
| Montos $0 | ~21% de transacciones con `reservation_paid_out = 0`. Documentar distribucion y relacion con status. |
| Montos >$1M | ~50 transacciones con montos extremos. Listar y analizar. |
| Usuarios con 100% tasa de reembolso | Usuarios donde todas sus transacciones fueron reembolsadas. Cuantificar y perfilar. |
| Usuarios con alta velocidad | Hasta 636 transacciones/dia. Distribucion de velocidad transaccional. |
| Facilities con tasa de reembolso desproporcionada | Top facilities por proxy rate. Posible senal contextual. |

### 2.5 Vacio del sistema actual de deteccion

| Dimension | Estado actual |
|-----------|--------------|
| Sistema de deteccion de anomalias | **No existe.** 100% reactivo. |
| Proceso de reembolso | Manual, post-facto. No hay scoring ni priorizacion automatica. |
| Gap identificado | Justifica el OE1 (construir un pipeline de deteccion). |

Este analisis debe alimentar la seccion de "justificacion del problema" del Capitulo 2.

---

## 3. Notebook: `notebooks/01_eda_capitulo2.ipynb`

### Correcciones respecto a versiones previas

1. **No cargar todos los splits simultaneamente.** Cargar uno a la vez para estadisticas por split; liberar memoria antes del siguiente. Pico estimado ~10 GB si se cargan todos; ~3-4 GB cargando secuencialmente.
2. **Usar `pip install -e .`** en lugar de `sys.path.insert(0, "..")`.
3. **Celdas markdown** explicando cada analisis y que seccion de la tesis respalda.
4. **Usar `settings.figures_dir` y `settings.tables_dir`** en vez de rutas hardcodeadas.
5. **Guardar figuras** en ambos formatos: PDF (para LaTeX) y PNG (para preview).

### Estructura de celdas

#### Celda 1: Setup

```python
# Requiere: pip install -e . (desde la raiz del proyecto)
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
tables_dir = settings.tables_dir
figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)
```

#### Celda 2: Resumen del dataset (carga secuencial)

```markdown
## 2.1 Resumen del dataset
Seccion de la tesis: Capitulo 2 — Descripcion del dataset de TechSport Inc.
Se carga cada split por separado para evitar exceder la memoria disponible.
```

```python
dm = DataManager(settings)
split_stats = {}
for name in ["train", "val", "test"]:
    df = dm.load_split(name)
    proxy_strict = DataManager.assign_proxy_labels(df, "strict")
    proxy_wide = DataManager.assign_proxy_labels(df, "wide")
    split_stats[name] = {
        "rows": len(df),
        "proxy_rate_strict": proxy_strict.mean(),
        "proxy_rate_wide": proxy_wide.mean(),
        "date_min": str(df["created_at"].min()),
        "date_max": str(df["created_at"].max()),
    }
    print(f"{name}: {len(df):,} rows, "
          f"proxy strict: {proxy_strict.mean()*100:.2f}%, "
          f"proxy wide: {proxy_wide.mean()*100:.2f}%")
    del df, proxy_strict, proxy_wide

total = sum(s["rows"] for s in split_stats.values())
print(f"\nTotal: {total:,} transacciones")
dm.close()
```

#### Celda 3: Distribucion de status

```markdown
## 2.2 Distribucion de status
Seccion de la tesis: Capitulo 2 — Variable criterio (proxy).
```

Cargar train, calcular conteos por status, generar tabla LaTeX y figura.

#### Celda 4: Distribucion de montos

```markdown
## 2.3 Distribucion de montos
Seccion de la tesis: Capitulo 2 — Variables del dataset.
```

Histograma truncado a $500, histograma log(1+amount), tabla de estadisticas descriptivas.

#### Celda 5: Distribucion por canal, gateway, metodo de pago

```markdown
## 2.4 Canales, gateways y metodos de pago
Seccion de la tesis: Capitulo 2 — Caracterizacion del ecosistema transaccional.
```

#### Celda 6: Patrones temporales (hora, dia, mes)

```markdown
## 2.5 Patrones temporales
Seccion de la tesis: Capitulo 2 — Analisis temporal de transacciones.
```

Serie mensual, distribucion por hora del dia, distribucion por dia de la semana.

#### Celda 7: Tasa proxy por mes

```markdown
## 2.6 Estabilidad temporal del proxy
Seccion de la tesis: Capitulo 2 — Consistencia de la variable criterio.
```

#### Celda 8: Comparacion proxy+ vs. proxy-

```markdown
## 2.7 Perfiles proxy+ vs. proxy-
Seccion de la tesis: Capitulo 2 — Evidencia de diferencias observables.
```

Tabla comparativa de montos, patrones horarios, canales, gateways.

#### Celda 9: Outliers y patrones extremos

```markdown
## 2.8 Casos extremos y outliers
Seccion de la tesis: Capitulo 2 — Anomalias observadas en el dataset.
```

Montos $0 (21%), montos >$1M (~50 txns), usuarios de alta velocidad (636 txns/dia), usuarios con 100% reversal rate.

#### Celda 10: Exportacion de artefactos

```python
import json

# Exportar resumen consolidado
summary = {
    "split_stats": split_stats,
    "proxy_strict_rate": ...,
    "proxy_wide_rate": ...,
    "total_rows": total,
    # ... metricas adicionales
}
with open(settings.output_dir / "metrics" / "cap2_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
```

#### Celda 11: Vacio del sistema actual

```markdown
## 2.9 Estado actual de deteccion de anomalias
Seccion de la tesis: Capitulo 2 — Justificacion del problema.

TechSport Inc. no cuenta con un sistema de deteccion de anomalias transaccionales.
El proceso de reembolso es 100% reactivo y manual. Este vacio justifica el OE1.
```

### Reglas del notebook

- **Memoria:** Cargar un split a la vez; `del df` + `gc.collect()` cuando ya no se necesite.
- **Figuras:** Guardar siempre en PDF y PNG: `for fmt in ["pdf", "png"]: plt.savefig(...)`.
- **Tablas LaTeX:** Usar `df.to_latex()` o similar; nunca escribir `.tex` manualmente.
- **Reproducibilidad:** Todo dato sale del snapshot congelado (Parquets de Fase 1).
- **Instalacion:** `pip install -e .` en vez de `sys.path.insert`.

---

## 4. Artefactos de salida

### Tablas LaTeX

| Archivo | Contenido |
|---------|-----------|
| `output/tables/cap2_status_distribution.tex` | Distribucion de status con N y % |
| `output/tables/cap2_channel_distribution.tex` | Distribucion por source_enum |
| `output/tables/cap2_gateway_distribution.tex` | Distribucion por gateway |
| `output/tables/cap2_payment_method_distribution.tex` | Distribucion por metodo de pago |
| `output/tables/cap2_amount_descriptive.tex` | Estadisticas descriptivas de montos |
| `output/tables/cap2_proxy_profiles.tex` | Comparativa proxy+ vs. proxy- |
| `output/tables/cap2_monthly_volume.tex` | Volumen mensual |

### Figuras

| Archivo | Contenido |
|---------|-----------|
| `output/figures/cap2_amount_distribution.{pdf,png}` | Histogramas de montos (original y log) |
| `output/figures/cap2_status_distribution.{pdf,png}` | Barras horizontales de status |
| `output/figures/cap2_proxy_amounts.{pdf,png}` | Box plot de montos proxy+ vs. proxy- |
| `output/figures/cap2_temporal_patterns.{pdf,png}` | Patrones por hora, dia, mes |
| `output/figures/cap2_proxy_rate_monthly.{pdf,png}` | Tasa proxy mensual con linea promedio |
| `output/figures/cap2_payment_methods.{pdf,png}` | Top metodos de pago |
| `output/figures/cap2_hourly_proxy.{pdf,png}` | Patron horario proxy+ vs. proxy- |

### Metricas

| Archivo | Contenido |
|---------|-----------|
| `output/metrics/cap2_summary.json` | Resumen consolidado: conteos, tasas, estadisticas, por split y global |

---

## 5. Consistencia con 02_diagnostico.tex

El EDA **debe reproducir** los mismos numeros que ya estan en `Tesis-Latex/capitulos/02_diagnostico.tex`. Esto incluye:

- N total del universo
- Tasa de proxy estricto y amplio
- Conteos por status
- Estadisticas descriptivas de montos
- Porcentaje de transacciones con monto $0
- Conteos de outliers extremos

**Si hay discrepancia:** Se corrige el `.tex` desde los artefactos del pipeline. Los numeros del pipeline son la fuente de verdad.

**Workflow:** Generar artefactos -> comparar con `.tex` -> si difieren, actualizar `.tex` con `\input{output/tables/cap2_*.tex}` donde sea posible.

---

## 6. Definition of Done

- [ ] Notebook `01_eda_capitulo2.ipynb` ejecutable de principio a fin sin errores
- [ ] Todas las tablas LaTeX generadas en `output/tables/cap2_*.tex`
- [ ] Todas las figuras generadas en `output/figures/cap2_*.{pdf,png}`
- [ ] `cap2_summary.json` creado con metricas consolidadas
- [ ] Los numeros del EDA coinciden con (o corrigen) los de `02_diagnostico.tex`
- [ ] Ningun numero del Capitulo 2 depende de edicion manual
- [ ] Capitulo 2 completable unicamente con artefactos del pipeline
- [ ] Analisis de vacio del sistema actual documentado
- [ ] Memoria pico del notebook no excede ~4 GB (carga secuencial verificada)
- [ ] Los hallazgos del EDA identifican patrones que alimentan el feature engineering (Fase 3)

---

## 7. Relacion con otras fases

| Fase | Relacion |
|------|----------|
| Fase 0 (Contrato) | El EDA opera sobre el universo definido en el contrato. Mismos filtros, mismo proxy. |
| Fase 1 (Snapshot) | El EDA usa los Parquets congelados. No re-extrae de ClickHouse. |
| Fase 3 (Features) | Los hallazgos del EDA (outliers, patrones temporales, usuarios de alta velocidad) informan el catalogo de features. |
| Fase 8 (Reporting) | Las tablas y figuras del EDA se integran directamente en la tesis via `\input{}` o `\includegraphics{}`. |

---

## 8. Gate de salida — Fase 2

**No avanzar a modelado final sin:**

1. Tablas y figuras base del Capitulo 2 generadas y verificadas.
2. Confirmacion de que OE2 queda respondido con la evidencia producida.
3. Inventario de hallazgos que alimentan el feature engineering documentado.
4. `cap2_summary.json` como referencia numerica unica.
5. Ningun artefacto del Capitulo 2 requiere intervencion manual para regenerarse.
