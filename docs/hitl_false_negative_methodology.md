# Metodología de estimación de falsos negativos — Cola HITL

**Contexto:** HITL-03 / Fase 5 (cola HITL y captura de etiquetas) · `hitl_queue_builder.py`

---

## 1. Motivación

El modelo frame-v1 asigna a cada transacción un percentil de anomalía relativo a su contexto de facility (moneda, escala operativa, hora local). La cola de revisión humana top-k recoge las transacciones con percentil más elevado, que son precisamente las que el modelo señala como más alejadas del comportamiento habitual.

Revisar exclusivamente el top-k permite estimar la **precisión** del modelo —qué fracción de los casos señalados resultan ser instancias efectivamente inusuales según el criterio del revisor— pero no dice nada sobre el **recall**: cuántas transacciones anómalas el modelo no ha señalado, porque les ha asignado un percentil bajo. Esta zona no alertada es, por definición, opaca a cualquier proceso que solo audite el extremo superior de la distribución.

Sin muestrear la zona no alertada, la tasa de falsos negativos del proxy de reembolso permanece inobservable. El equipo operativo no puede distinguir entre un modelo con recall alto (pocos falsos negativos) y uno con recall bajo que simplemente no alerta sobre una fracción relevante del comportamiento inusual.

---

## 2. Estrategia de muestreo defensivo

La cola HITL combina dos estratos:

| Estrato | Selección | Criterio | Fuente |
|---------|-----------|----------|--------|
| **top_k** | Order by `percentile DESC` | Lo que el modelo señala con mayor confianza | Capacidad discriminativa alta |
| **below_p50** | `percentile < p50`, orden aleatorio (`ORDER BY rand()`) | Muestra de la zona no alertada | Estimación de falsos negativos |

El percentil mediano (`p50`) de la distribución `shadow_new` actúa como frontera entre "zona alertada" y "zona no alertada". Utilizar el p50 como corte garantiza que la muestra defensiva provenga de transacciones que el modelo considera por debajo del punto central de su escala de anomalía.

### Criterio del 20 %

El 20 % de la capacidad de revisión como mínimo asignado al estrato below-p50 es el umbral base de HITL-03. El razonamiento es conservador: si la tasa de anomalias reales en la zona below-p50 es del orden del 1-2 %, una muestra de 20 casos sobre un lote de 100 permite detectar esa tasa con un intervalo de confianza al 95 % de anchura razonable sin saturar la capacidad de revisión.

El parámetro `--below-p50-pct` (o `HITL_BELOW_P50_PCT`) permite ajustar este porcentaje según la capacidad real del equipo revisor. Hasta que se confirme la capacidad HITL disponible (bloqueante conocido registrado en STATE.md), el valor predeterminado es 0.20.

---

## 3. Estimación de falsos negativos como cota inferior

Una vez que los revisores etiquetan las filas del estrato below-p50, la fracción de filas clasificadas como `sospecha_fraude` o `anomalia_operativa` proporciona una **estimación de la tasa de anomalias reales entre las transacciones no alertadas**. Extrapolada al volumen total de transacciones `shadow_new` con `percentile < p50`, esa fracción constituye una cota inferior del número de falsos negativos del modelo.

Conviene precisar el alcance de esta estimación:

- Es una **estimación correlacional sobre el proxy**, no una medida de fraude verdadero. El proxy de reembolso (`status IN ('totally_refunded', 'refunded_to_credit')`) captura comportamiento anómalo asociado a devoluciones, pero no recoge insider fraud ni otras anomalias sin consecuencia visible en el estado del pago.

- La extrapolación asume que la muestra below-p50 es representativa del estrato completo. La selección aleatoria uniforme (`ORDER BY rand()`) apoya este supuesto; no obstante, si la distribución de anomalias reales within the estrato below-p50 fuese muy heterogénea por facility o moneda, la cota podría subestimar el FN real.

- Al tratarse de un modelo no supervisado que entrena sin etiquetas, no existe una definición operacional de "anomalía verdadera" independiente del criterio humano. La tasa de FN estimada es, por tanto, condicional a la definición de anomalía que los revisores apliquen durante la sesión de etiquetado.

---

## 4. Vocabulario de etiquetas

La coherencia del proceso de estimación depende de que todos los revisores apliquen las mismas categorías. Las cuatro categorías válidas son las mismas que `hitl_ingest_labels.py` y el formulario `HitlLabelForm` (05-02):

| Categoría | Significado |
|-----------|-------------|
| `sospecha_fraude` | La transacción presenta indicios de comportamiento fraudulento o abusivo |
| `anomalia_operativa` | Inusual pero explicable por circunstancias operativas (error, prueba, evento puntual) |
| `falso_positivo` | El modelo señaló la transacción pero el revisor no identifica nada inusual |
| `indeterminado` | Información insuficiente para decidir; requiere seguimiento |

Para el objetivo de estimación de falsos negativos, las filas del estrato below-p50 que reciben `sospecha_fraude` o `anomalia_operativa` son las que contribuyen al numerador de la tasa de FN.

---

## 5. Trazabilidad por fila

Cada fila del archivo exportado por `hitl_queue_builder.py` incluye los campos necesarios para reconstruir sin ambigüedad el contexto de la etiqueta posterior:

| Campo | Contenido |
|-------|-----------|
| `hitl_queue_source` | `top_k` o `below_p50` — identifica el estrato de procedencia |
| `model_version` | Versión del modelo que generó el score en el momento del scoring (trazabilidad de artefacto) |
| `scored_at` | Marca temporal UTC del momento en que se calculó el score |
| `percentile` | Score percentilizado en el momento del scoring |
| `top_factors` | Array JSON de factores (`feature`, `value`, `z_score`, `direction`) que más contribuyeron al score |

Cuando la fila se ingiere en el registro de etiquetas (`hitl_ingest_labels.py`), el campo `scored_at` actúa como `score_at_label_time` implícito, y `model_version` como `model_version_at_label`. Esto permite que, en versiones futuras del modelo, se pueda recalcular el score con el modelo actualizado y compararlo con el score que motivó la revisión.

---

## 6. Estado operacional diferido

La metodología, el exportador y los tests están disponibles y verificados con datos sintéticos. El poblado real de la cola queda diferido hasta que:

1. El modo `SCORING_MODE=shadow_dual` acumule al menos **2 semanas de datos** en `anomaly_scores` (umbral de datos insuficientes: `MIN_SHADOW_DAYS=14`, `MIN_SHADOW_ROWS=500` — `shadow_gate.py`).
2. Se confirme la **capacidad de revisión del equipo HITL** (bloqueante registrado en STATE.md como Pre-Fase 5).

Una vez alcanzadas ambas condiciones, ejecutar:

```bash
python scripts/hitl_queue_builder.py \
    --capacity <N> \
    --below-p50-pct 0.20 \
    --output output/hitl_queue_$(date +%Y%m%d).csv
```

La capacidad `N` la determina el coordinador del equipo revisor en función de la disponibilidad semanal.
