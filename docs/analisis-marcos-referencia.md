# Análisis: marcos de referencia heterogéneos en la detección de anomalías

**Fecha:** 2026-07-02
**Ámbito:** uso operativo del pipeline `ml-fraud-detector` (no tesis)
**Estado:** hallazgo confirmado con experimento controlado

---

## Resumen ejecutivo

El modelo de detección de anomalías rinde hoy **a nivel de azar** (AUC-ROC ≈ 0.51 en features
limpias). La investigación encontró **dos patologías simultáneas**:

1. **Marcos de referencia heterogéneos.** Las features se definen mezclando poblaciones que no son
   comparables entre sí. Hay tres ejes de contaminación: **moneda** (magnitud), **escala de
   facility** (magnitud) y **zona horaria** (tiempo). El resultado es que el Isolation Forest aísla
   *artefactos* (montos grandes, facilities grandes, "noche" en UTC) en lugar de comportamiento.

2. **Fuga de etiqueta (leakage).** Las dos únicas features que hacían subir el AUC se derivan de
   `status`, la misma columna que define el proxy de evaluación. Ese "rendimiento" no era detección.

Un experimento controlado sobre la cohorte USD demuestra el daño y el efecto de corregirlo:

| Evidencia | Marco contaminado (V0) | Marco normalizado (V1) |
|---|---|---|
| Monto medio del top-5% marcado / promedio | **16.9×** | **1.8×** |
| Off-hours (madrugada) detectada | 26.4% (UTC) | 4.2% (local) |
| Enriquecimiento @ top-5% | 1.03× | 1.22× |
| AUC-ROC vs proxy reembolso | 0.478 | 0.500 |

**Conclusión honesta:** corregir los marcos es **necesario** —convierte un "detector de montos
grandes" en un detector de comportamiento y mejora el enriquecimiento +18%— pero **no es
suficiente** para levantar el AUC frente al proxy de reembolso, porque **el reembolso no es una
anomalía estadística**. El techo de ~0.50 lo pone el proxy, no las features. Esto redefine el
objetivo operativo (ver §5).

---

## 1. El problema

Resultados actuales del pipeline (`output/revision/tables/`, test = 2.5M transacciones, gestión 2025):

| Modelo | Feature set | AUC-ROC |
|---|---|---|
| Isolation Forest | FS-clean-A-29 (sin reversal) | **0.508** ← azar |
| Isolation Forest | FS-operational-B | 0.503 ← azar |
| Isolation Forest | FS-baseline-31 (**con** reversal) | 0.576 |
| LOF | FS-clean-A-29 | 0.536 |
| OC-SVM | FS-clean-A-29 | 0.506 |

Con features limpias el modelo no distingue nada. El único set que sube (baseline-31) es el que
incluye las features de reversal, que son leakage (§2.4).

---

## 2. Análisis: tres marcos + leakage

El principio unificador: **una anomalía solo tiene sentido relativa a una población homogénea de
referencia.** Cada feature debe definirse dentro del marco de su propia facility (su moneda, su
escala, su reloj local). El pipeline actual viola esto en tres ejes.

### 2.1 Marco de moneda (magnitud)

Los montos se convierten a USD con tipos de cambio (`utils/currency.py`) y luego las features de
monto se comparan contra una media **global** que mezcla 21 monedas. Una transacción normal para su
mercado se ve "enorme" o "diminuta" contra ese promedio mezclado. Además la conversión FX inyecta
ruido temporal: el `amount` en USD se mueve con el tipo de cambio macro aunque el comportamiento no
cambie.

Evidencia complementaria (ClickHouse, gestión 2025): la tasa de reembolso varía **4×** entre
monedas (MYR/ILS ~9% vs USD ~2%), señal de que las poblaciones no son comparables.

### 2.2 Marco de escala de facility (magnitud)

Aun dentro de una sola moneda, un club grande y uno chico tienen escalas de monto distintas. El
modelo trata el monto absoluto como señal, así que un club grande "parece" anómalo solo por ser
grande. Este es el eje que el experimento aísla (cohorte USD, sin confound de moneda).

### 2.3 Marco temporal (zona horaria)

`created_at` está en **UTC** (`DateTime` sin zona). Las features temporales se calculan directo
sobre la hora UTC, sin convertir a la hora local de la facility (`facilities.time_zone` existe pero
no se usa). La cohorte USD se reparte en ~9 husos (Eastern, Central, Mountain, Pacific, Arizona sin
DST, Hawaii, etc.). Resultado medido para facilities Eastern:

- `is_off_hours` en **UTC = 18.5%** vs en **hora local = 4.6%** → la feature infla la "madrugada"
  ~4× (para toda la cohorte USD: 26.4% → 4.2%). Mide *qué tan al oeste está la facility*, no si la
  transacción fue realmente nocturna.

**Nota:** agrupar por offset estático (UTC−N) no basta: el horario de verano corre Eastern entre
−5/−4 y Arizona/Hawaii no lo aplican. Hay que convertir con la zona IANA real (respeta DST).

### 2.4 Fuga de etiqueta (leakage)

El proxy de evaluación es `status IN ('totally_refunded','refunded_to_credit')`. Dos features se
derivan de esa misma columna `status`:

- `user_reversal_ratio_30d` (F17)
- `user_reversal_count_30d` (F30)

Correlacionan con el proxy por construcción. La diferencia baseline-31 (0.576) vs clean-29 (0.508)
es **exactamente** el efecto de estas features: ~0.07 de AUC que no es detección, es leakage.

### Catálogo de features afectadas

| # | Feature (código) | Eje | Problema |
|---|---|---|---|
| 1 | `amount`, `log_amount` (F1, F2) | Moneda + escala | Magnitud absoluta cruda; codifica moneda/tamaño + drift FX |
| 2 | `amount_usd_ratio` (F3) | Moneda + escala | Divide por **una** media global de 21 monedas y todas las facilities |
| 3 | `facility_avg_amount` (F20) | Moneda + escala | Media absoluta por facility = proxy de tamaño, no comportamiento |
| 4 | `user_amount_24h` (F14) | Moneda + escala | Suma absoluta en USD |
| 5 | `user_debit_amount_30d` (F24) | Moneda + escala | Suma absoluta en USD |
| 6 | `hour_sin/cos`, `is_off_hours`, `day_of_week`, `is_weekend` (F6–F10) | Zona horaria | Calculadas en UTC, no en hora local |
| 7 | `user_reversal_ratio_30d` (F17) | Leakage | Derivada de `status` (define el proxy) |
| 8 | `user_reversal_count_30d` (F30) | Leakage | Ídem (conteo) |
| 9 | `day_of_week` (F8) | Codificación | Entero lineal 1–7; la hora sí es cíclica (sin/cos), el día no |

Features **seguras** (invariantes al marco): velocidad (`user_txn_count_1h/24h`,
`time_since_last_txn`) son *deltas* temporales; no requieren corrección.

### Calidad de datos (hallazgos laterales)

- **Moneda vacía** en 577 facilities (117K registros) → requiere fallback / limpieza.
- Facilities con `time_zone` vacío/nulo → fallback temporal.

---

## 3. Propuesta de solución

**Principio: normalizar cada feature al marco homogéneo de su facility.**

1. **Magnitud → relativa a la facility.** Reemplazar `amount`/`log_amount`/`amount_usd_ratio`
   absolutos por:
   - `amount_facility_ratio` = monto / media(train) de la facility *(ya existe, F21)*
   - `amount_facility_z` = (monto − mediana) / IQR de la facility *(robusto a colas)*
   - Idem para sumas de ventana (`user_amount_24h`, `user_debit_amount_30d`) → normalizar por la
     escala de la facility del usuario.
   - Con features relativas **no se necesitan tipos de cambio**: menos ruido, menos dependencia de
     la tabla FX, y las monedas nuevas entran sin reentrenar.

2. **Tiempo → hora local.** Convertir `created_at` (UTC) a la zona de la facility vía
   `facilities.time_zone` (mapeo Rails→IANA) **antes** de calcular las features temporales.
   `toTimeZone` respeta DST. Corregir también `day_of_week` a codificación cíclica.

3. **Eliminar leakage.** Sacar `user_reversal_ratio_30d` y `user_reversal_count_30d` del set
   operativo. No pueden usarse para evaluar contra un proxy que sale de la misma columna.

4. **Umbral por segmento.** La distribución del score se corre por facility/segmento; un
   `contamination` global sobre-marca a unos y sub-marca a otros. Umbral por facility (o por tier).

5. **Calidad de datos.** Fallback explícito para moneda vacía y `time_zone` nulo.

---

## 4. Test / experimento

**Script:** `scripts/exp_reference_frames.py` — **Resultados:** `output/revision/frames_experiment.json`

**Diseño (A/B controlado).** Se compara el **mismo** Isolation Forest (n_estimators=200,
max_samples=512, contamination=auto, StandardScaler, seed=42) con dos sets de features que difieren
**únicamente en el marco**:

- **V0 (contaminado):** magnitud absoluta (`amount`, `log_amount`, `amount/media_global`) + hora UTC.
- **V1 (normalizado):** magnitud relativa a la facility (ratio + z-score robusto) + hora local.

Ambos excluyen features de leakage. Cohorte `currency='USD'` (controla el eje moneda por
construcción, aísla escala + tiempo). Split temporal train Ene–Ago / test Sep–Dic 2025
(anti-leakage). Muestra 1/25 (~129K train, ~72K test). Proxy Tipo A (reembolso), solo evaluación.

**Resultados:**

| Métrica | V0 contaminado | V1 normalizado | Δ |
|---|---|---|---|
| AUC-ROC | 0.478 | 0.500 | +0.022 |
| AUC-PR | 0.059 | 0.062 | +0.003 |
| Precisión @ top-5% | 6.2% | 7.3% | +1.1 pp |
| Enriquecimiento @ top-5% | 1.03× | 1.22× | **+18%** |
| **Monto medio top-5% / promedio** | **16.9×** | **1.8×** | — |
| Off-hours detectada | 26.4% (UTC) | 4.2% (local) | −6.3× |

**Interpretación:**

- El diagnóstico `top5_amount_x_avg = 16.9×` es la prueba directa del confound: V0 marca como
  anómalas las transacciones más grandes (17× el promedio). No detecta comportamiento, detecta
  **tamaño**. V1 lo baja a 1.8× → el score pasa a ser conductual.
- Corregir el marco mejora el enriquecimiento +18% y el AUC +0.022, de forma consistente.
- Pero **el AUC sigue ~0.50**. El fix es real y necesario, no drástico frente a este proxy.

---

## 5. Conclusiones y próximos pasos

1. **Los marcos son un bug real y hay que corregirlo** — no por el AUC, sino porque hoy el sistema
   marca "transacciones grandes" y "noche mal calculada". Operativamente eso es inservible; V1 lo
   arregla.

2. **El near-random es en gran parte un problema de proxy, no solo de features.** El reembolso no
   es una anomalía estadística: aun con el marco limpio, IF no lo separa (~0.50). El objetivo de
   evaluación debe repensarse — el reembolso es un sustituto débil.

3. **El "rendimiento" reportado era leakage.** Cualquier resultado que dependa de las features de
   reversal debe descartarse para uso operativo.

**Recomendaciones operativas:**

- **Implementar la normalización de marco** (magnitud relativa + hora local + sin leakage) en el
  pipeline de features. Es el prerequisito para cualquier detección con sentido.
- **Redefinir el objetivo:** en vez de "predecir reembolsos", producir **anomalías tipificadas**
  (por velocidad, por descuentos, por circuito de crédito, por horario, etc.) vía atribución SHAP
  sobre el top-k. El valor operativo es la anomalía interpretable, no un AUC contra reembolsos.
- **Umbral por facility/segmento** para el disparo de alertas.
- **Corregir calidad de datos:** moneda vacía (577 facilities) y `time_zone` nulo.

**Reproducir:** `./venv/bin/python scripts/exp_reference_frames.py`
