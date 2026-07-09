# Plan operativo: marco de referencia (Plan A) + anomalías tipificadas (Plan B)

**Fecha:** 2026-07-07
**Ámbito:** herramienta operativa `ml-fraud-detector` (fuera de alcance de tesis)
**Base:** `docs/analisis-marcos-referencia.md`, `output/revision/frames_improvement_results.json`,
`output/frame_v1_bias_report.json`

---

## 0. Estado real del sistema (corrección importante)

Una auditoría del código reveló que **la normalización de marco ya existe y está desplegada**, no
hay que construirla. El estado actual es:

| Componente | Estado | Evidencia |
|---|---|---|
| **IF-40-v1** (campeón en producción) | Activo. Features **base-31 contaminadas**, proxy `pure_fraud`, AUC ~0.80 | `scorer/main.py:140`, `output/models/` |
| **frame-v1** (retador) | Entrenado + en **shadow** (no decide) | `scoring/features_frame_v1.py`, `scorer/main.py:118-152` |
| FS-frame-v1 (30 features, marco limpio) | Completo: magnitud relativa a facility + hora local IANA con DST, sin leakage | `FRAME_V1_FEATURE_NAMES` |
| Artefactos frame-v1 | Modelo, `RobustScaler`, `facility_stats_v1.json`, metadata | `output/models/*_frame_v1.*` |
| Paridad batch↔real-time | **PASS** (maxdiff 0.0) | `model_metadata_frame_v1.json` |
| Bias report frame-v1 | Hecho | `output/frame_v1_bias_report.json` |
| Umbral segmentado (facility→currency→global) | Artefacto **construido** (452 facilities, 17 monedas) **y conectado** al scorer | `thresholds_segmented_v1.json`; `scoring/scorer.py:55-67,142-156` |
| Tipificación SHAP por transacción | **No existe** (solo `FactorItem` z-score/direction) | `scorer/schemas.py:42` |

**Conclusión:** Plan A se redefine de "construir" a **"promover frame-v1 de shadow a campeón y activar
umbrales segmentados"**. Plan B (tipificación SHAP) es trabajo nuevo, construido sobre el marco limpio.

> Nota: el experimento `scripts/exp_reference_frames.py` y `docs/analisis-marcos-referencia.md`
> (creados el 2026-07-02) re-derivaron de forma cruda lo que `frames_improvement_results.json` ya
> medía con 3 semillas y ambos proxies. Se conservan como material de respaldo del hallazgo, pero la
> fuente canónica de métricas es `output/revision/`.

---

# PLAN A — Promoción de frame-v1 a producción + umbrales segmentados

## A.1 Objetivo y criterio de éxito

Reemplazar el campeón IF-40-v1 (features contaminadas) por frame-v1 (marco limpio) como modelo de
decisión, y activar el umbral por segmento. El resultado: alertas que reflejan **comportamiento
relativo al contexto de la facility**, no tamaño ni artefactos de huso, con disparo calibrado por
facility.

**Gates de aceptación (decisión shadow→champion):**

| Gate | Fuente | Umbral de aprobación |
|---|---|---|
| Sesgo de tamaño (monto medio top-5% / promedio) | backtest frame-v1 | **< 3×** (IF-40 base-31 ≈ 15.6×) |
| Off-hours calculada | backtest | ~4–6% (local), no ~26% (UTC) |
| Discriminación vs `pure_fraud` no inferior al campeón | shadow backtest | AUC frame-v1 **≥ IF-40 − 0.02** |
| Enriquecimiento @ top-5% | shadow backtest | **≥ 1.2×** y ≥ IF-40 |
| Estabilidad multi-semilla | 3 semillas | rango AUC < 0.01 |
| Cobertura de datos (tz + currency presentes) | `FrameFlags` | **> 98%** de transacciones sin fallback crítico |
| Paridad batch↔real-time | test de paridad | maxdiff < 1e-8 (ya PASS) |

## A.2 Alcance

**Dentro:** decisión shadow, construcción del artefacto de umbrales segmentados, conexión del
`SegmentedThresholdClassifier` al scorer en vivo, cutover champion, monitoreo de calidad de datos,
deprecación ordenada del path base-31.

**Fuera:** re-arquitectura del feature engineering (ya hecho en frame-v1), tipificación SHAP (Plan B).

## A.3 Trabajo pendiente (detallado)

### A.3.1 Evaluación shadow y decisión (champion vs challenger) — ✅ EJECUTADO (Paso 1)

**Script:** `scripts/backtest_shadow.py` · **Resultado:** `output/revision/shadow_decision_frame_v1.json`
(test Sep–Dic 2025, muestra 1/10 = 251.448 filas, cobertura de datos 100%).

| Gate | frame-v1 | campeón IF-40 | Veredicto |
|---|---|---|---|
| Sesgo de tamaño top-5% (winsor. p99.9) < 3× | **1.57×** | 4.06× | ✅ PASS |
| Off-hours local en banda 3–7% | **4.4%** | 30.1% (UTC) | ✅ PASS |
| Cobertura de datos > 98% | 100% | — | ✅ PASS |
| AUC `pure_fraud` ≥ campeón − 0.02 | 0.609 | 0.840 | ❌ FAIL |
| EF@5% `pure_fraud` ≥ 1.2 y ≥ campeón | 1.51 | 6.02 | ❌ FAIL |

**Decisión automática: HOLD** (3/5). **Pero los dos gates que fallan son inválidos por
circularidad:**

- El proxy `pure_fraud` se define con 9 variables (`same_amount_count_1h`, `user_account_age_days`,
  `user_txn_count_1h`, `is_third_party_payment`, `is_new_user`, `rapid_burst`, …).
- El **campeón IF-40 contiene las 9**; **frame-v1 contiene 0**. Verificado programáticamente.
- Por tanto el AUC 0.84 del campeón es **autovalidación** (sus features *son* la definición del
  proxy). Comparar frame-v1 (honesto, 0.61) contra esa línea base castiga al modelo correcto.
- En el **único proxy no circular** (reembolso/`tipo_a`) **ambos rinden ~0.49 (azar)**. Es decir, la
  superioridad del campeón desaparece por completo cuando se quita la circularidad.

**Lectura honesta:** frame-v1 gana decisivamente en lo que importa metodológicamente (elimina el
sesgo de tamaño y de huso, y no es circular). Su "pérdida" de AUC es un artefacto de comparar contra
un campeón circular. Ningún modelo predice el proxy no circular → **el valor de frame-v1 no es el
AUC-vs-proxy, sino producir anomalías conductuales limpias e interpretables** (→ Plan B).

**Criterio de decisión corregido (reemplaza los gates 4–5 circulares):**
- Gates de sesgo (1–3): **PASS obligatorio** — cumplidos.
- Discriminación: evaluar contra un proxy **disjunto de las features de ambos modelos**; si ambos
  son ~azar, la promoción NO se decide por AUC.
- **Decisión de negocio (requiere sign-off):** promover frame-v1 hará caer la métrica de titular
  (AUC 0.84 → 0.61) porque se elimina la circularidad. **Nunca cutover silencioso**; documentar que
  la caída es corrección de un artefacto, no regresión real.

### A.3.2 Artefacto de umbrales segmentados — ✅ YA EXISTE
- `output/models/thresholds_segmented_v1.json` está construido para frame-v1: **452 facilities,
  17 monedas**, fallback facility→currency→global, con `min_n_threshold` (anti-sparsidad) y
  `percentile` de calibración. No hay que reconstruirlo.
- Pendiente menor: verificar que la calibración corresponde al modelo frame-v1 vigente (re-generar
  si se reentrena el modelo).

### A.3.3 Conexión del clasificador segmentado al scorer — ✅ YA HECHO
- `SingleTransactionScorer` (`scoring/scorer.py:55-67`) ya despacha al path frame-v1
  (`FrameV1FeatureCalculator` + `SegmentedThresholdClassifier`) cuando los artefactos cargados
  incluyen `facility_stats` + `thresholds_segmented`, y puebla `calibration_segment`,
  `fallback_level` y `frame_flags` en la respuesta (`scorer.py:142-156`).
- El comentario "NOT connected (Phase 3)" en `classifier.py:129` está **obsoleto**; corregirlo.
- Selección efectiva = qué artefactos carga el scorer (campeón vs retador), es decir, la decisión de
  promoción (A.3.5).

### A.3.4 Monitoreo de calidad de datos
- Explotar `FrameFlags` (`timezone_missing`, `currency_missing`, `currency_unknown`): emitir métrica
  de cobertura por lote y alertar si el fallback crítico supera un umbral.
- Corregir en la fuente las 577 facilities con moneda vacía y las facilities con `time_zone` nulo
  (ver `docs/analisis-marcos-referencia.md`).

### A.3.5 Cutover y deprecación
- Promover frame-v1 a `scoring_mode='active'` como campeón; degradar IF-40 a fallback temporal.
- Marcar el path base-31 (`scoring/features.py`, 31 features) y `thresholds.json` legacy como
  deprecados; plan de retiro tras período de observación.

## A.4 TDD

1. **Umbral segmentado — resolución de fallback:** facility conocida → usa su umbral; moneda conocida
   sin facility → usa moneda; desconocidas → global. (Regresión de `classifier.py:146-191`.)
2. **Construcción de artefacto — anti-sparsidad:** facility con < N txns no entra en `by_facility`.
3. **Cobertura de flags:** tz/currency ausentes generan el flag correcto y caen al fallback sin romper.
4. **Cutover no rompe contrato:** `ScoreResponse` conserva forma; `calibration_segment`/`fallback_level`
   poblados en modo segmentado, `None` en global.
5. **Paridad:** regresión del test de paridad batch↔real-time.

## A.5 Pasos atómicos (un commit por paso)

1. ✅ `scripts/backtest_shadow.py` + correr shadow → `shadow_decision_frame_v1.json` (HECHO; ver A.3.1).
2. ✅ Artefacto de umbrales segmentados → `thresholds_segmented_v1.json` (YA EXISTÍA; ver A.3.2).
3. ✅ Clasificador segmentado conectado al scorer (`scorer.py:55-67`; YA HECHO; ver A.3.3).
4. Métrica de cobertura de `FrameFlags` + alerta (mejora menor de observabilidad).
5. **Decisión de promoción con sign-off** (no cutover silencioso; ver criterio corregido en A.3.1)
   + deprecación documentada del path base-31 + corregir comentario obsoleto en `classifier.py:129`.

## A.6 Estado de Plan A: **código completo**; decisión de arquitectura tomada

Toda la ingeniería de Plan A ya existe (features frame-v1, modelo, scaler, umbrales segmentados,
clasificador cableado, frame_flags, script de retrain, bias report).

### Decisión de arquitectura (2026-07-07): **detección por capas**

El objetivo es la mejor detección de anomalías lanzable a producción, no elegir entre AUC 0.84 y
0.61. El 0.84 de IF-40 es circular (reimplementa reglas). Arquitectura elegida:

1. **Capa de reglas explícitas** — lo que IF-40 "detectaba" circularmente (card-testing,
   ráfaga de usuario nuevo, pago de tercero) se declara como reglas/flags. No requiere ML; son
   precisas, baratas e interpretables.
2. **Capa de anomalía no supervisada = frame-v1** — para comportamiento atípico que las reglas NO
   capturan. Sin artefactos de tamaño/huso, sin circularidad. Es el valor real del ML.
3. **Capa de tipificación (Plan B)** — arquetipos SHAP para que cada alerta sea accionable.

**Lanzamiento por fases:** promover frame-v1 a campeón del *score de anomalía*; retener IF-40 y las
reglas como señales secundarias durante una ventana de observación (el `ShadowDualRunner` ya lo
permite); luego deprecar el path base-31. La caída de AUC de titular se documenta como
de-circularización, no regresión.

El trabajo de ingeniería productiva pasa a **Plan B**, que se construye offline contra el artefacto
frame-v1 (ya limpio) sin esperar la promoción a producción.

---

# PLAN B — Anomalías tipificadas (SHAP) sobre marco limpio

## B.1 Objetivo y criterio de éxito

Convertir el score continuo en **alertas accionables tipificadas**: para cada transacción marcada,
atribuir *por qué* es anómala (arquetipo dominante + features contribuyentes), de modo que un
operador sepa si actuar y cómo. El score dice "cuánto"; la tipificación dice "de qué tipo".

**Prerrequisito duro:** Plan A cerrado (frame-v1 en producción). SHAP sobre features contaminadas
tiparía artefactos ("anomalía de tamaño"), dando alertas confiadamente equivocadas.

**Criterio de éxito:**
- Cada alerta del top-k lleva `dominant_archetype` + `top_contributions` (feature, contribución, dirección).
- Cobertura de tipificación 100% del top-k (con clase `mixed` como residual explícito).
- Estabilidad: la asignación de arquetipo es determinista y estable entre semillas del explainer.
- Validez aparente (face validity): muestreo revisado por operador → el arquetipo coincide con la
  lectura humana en la mayoría de los casos.

## B.2 Diseño detallado

### B.2.1 Explainer
- `shap.TreeExplainer` (SHAP 0.49.1 ya instalado) sobre el `IsolationForest` frame-v1. TreeSHAP
  soporta IsolationForest y atribuye la contribución de cada feature al score de anomalía.
- Dos superficies, igual que el feature calculator:
  - **Batch:** SHAP sobre el top-k del backtest (denso, para calibrar arquetipos).
  - **Real-time:** SHAP por transacción en el scorer (solo si `is_anomaly`, para no gravar latencia).

### B.2.2 Mapeo feature → arquetipo (`FRAME_V1_FEATURE_NAMES` → grupos)

| Arquetipo | Features frame-v1 |
|---|---|
| **magnitude** | `log_amount_fac`, `amount_facility_ratio`, `user_amount_24h_fac`, `small_amount_at_facility`, `very_small_amount_at_facility` |
| **temporal** | `hour_sin_loc`, `hour_cos_loc`, `dow_sin_loc`, `dow_cos_loc`, `is_weekend_loc`, `is_off_hours_loc`, `off_hours_high_value_loc` |
| **discount** | `discount_ratio` |
| **credit_flow** | `is_club_credit`, `user_debit_count_30d`, `user_debit_amount_30d_fac`, `credit_flow_ratio` |
| **diversity** | `user_distinct_facilities_30d`, `user_distinct_methods`, `category_entropy_30d`, `user_merchandise_ratio_30d` |
| **staff_role** | `is_staff`, `paid_by_manager`, `staff_amount_zscore` |
| **gateway_channel** | `gateway_change_recent`, `is_main_gateway`, `is_first_gateway_for_user`, `source_change_recent` |
| **tip** | `has_tip` |
| **velocity** | `time_since_last_txn` |

### B.2.3 Asignación de arquetipo por transacción
- Sumar la contribución SHAP (positiva hacia anomalía) por grupo.
- **Dominante** = grupo con mayor contribución **si** supera un umbral de concentración (p. ej.
  ≥ 50% de la contribución total positiva); si no, `mixed`.
- Guardar el ranking completo de grupos + top-3 features individuales.

### B.2.4 Descubrimiento empírico (complemento, no reemplazo)
- Además del mapeo por reglas, clusterizar los vectores SHAP del top-k (KMeans/HDBSCAN) para
  **descubrir** arquetipos no anticipados y validar que el mapeo por reglas cubre la estructura real.
  Reportar en `output/revision/archetype_clusters.json`.

### B.2.5 Esquema de alerta (extiende `scorer/schemas.py`)
- Añadir a `ScoreResponse`: `dominant_archetype: str`, `archetype_scores: Dict[str, float]`,
  y reutilizar/enriquecer `factors: List[FactorItem]` con la contribución SHAP (no solo z-score).
- Salida de operador: `transaction_id, facility, score, percentile_in_facility, risk_level,
  dominant_archetype, top_contributions`.

### B.2.6 Evaluación de la tipificación (honesta)
- SHAP explica el modelo; **no** es ground truth. Validación:
  - **Face validity:** muestreo estratificado por arquetipo revisado manualmente.
  - **Estabilidad:** misma transacción → mismo arquetipo entre corridas/semillas.
  - **Descriptivo (no objetivo):** tabla cruzada arquetipo × proxy (`pure_fraud`, refund) para ver
    con qué se asocia cada tipo — reportado como descriptivo, nunca como métrica de entrenamiento.
- Precedente reutilizable: `reporting/latex_tables.py:table_anomaly_types` ya tipificó anomalías por
  SHAP a nivel agregado; Plan B lo lleva a nivel transacción y al scorer en vivo.

## B.3 TDD

1. **Mapeo determinista:** un vector SHAP sintético con toda la masa en `discount_ratio` → arquetipo
   `discount`.
2. **Umbral de concentración:** contribución repartida sin dominante → `mixed`.
3. **Solo-anomalías en real-time:** transacción normal no dispara SHAP (control de latencia).
4. **Cobertura:** todo el top-k recibe un arquetipo (incluido `mixed`).
5. **Contrato de esquema:** `ScoreResponse` con los campos nuevos, retrocompatible (default vacío).

## B.4 Validación / gate
- 100% del top-k tipificado; estabilidad de arquetipo ≥ 99% entre dos corridas.
- Face validity documentada sobre muestra ≥ 50 alertas por arquetipo mayoritario.
- Reporte `output/revision/archetype_report.json` + tabla cruzada descriptiva.

## B.5 Pasos atómicos (un commit por paso)
1. ✅ Prototipo SHAP + arquetipos offline → `scripts/exp_archetypes.py` (HECHO; ver B.6).
2. ✅ `scoring/archetypes.py` (mapeo canónico + `assign_archetype` dominante/`mixed`/top-k) +
   `tests/test_archetypes.py` (11 tests PASS) + umbral ajustado a 0.35 (HECHO; ver B.6).
3. **(siguiente)** Clustering empírico de validación → `archetype_clusters.json`.
4. Extender `ScoreResponse`/`FactorItem` con arquetipo + contribuciones + tests B.3.5.
5. Cablear SHAP en el scorer (solo si `is_anomaly`) + control de latencia + tests B.3.3.
6. Reporte de evaluación (face validity + cruzada descriptiva) → gate B.4.

## B.6 Resultado del prototipo (Paso 1) — ✅ EJECUTADO

`scripts/exp_archetypes.py` → `output/revision/archetype_report.json` (SHAP TreeExplainer sobre
frame-v1, top-5% de anomalías del test, muestra 1/10). Distribución de arquetipo dominante:

| Arquetipo | % del top-5% | tasa tipo_a | tasa pure_fraud |
|---|---|---|---|
| **mixed** | 65.7% | 4.7% | 3.2% |
| **credit_flow** | 27.2% | **14.3%** | 5.9% |
| diversity | 3.9% | 1.2% | **16.4%** |
| gateway_channel | 1.8% | 4.8% | 0.4% |
| **magnitude** | **1.1%** | 8.1% | 16.2% |
| temporal | 0.3% | 12.5% | 0.0% |

**Hallazgos (umbral inicial 0.50):**
- **Validación del marco limpio:** `magnitude` es solo **1.1%** de los arquetipos dominantes — el
  modelo frame-v1 **ya no es un detector de tamaño** (en IF-40 contaminado dominaría). El fix funciona
  a nivel de tipificación, no solo de métrica de sesgo.
- **65.7% `mixed`** con umbral 0.50 → las anomalías del marco limpio son genuinamente multifactor.

**Ajuste aplicado (B.5.2, umbral 0.35 en `scoring/archetypes.py`):** `mixed` baja a **27.2%** y
emergen los tipos dominantes:

| Arquetipo | % top-5% | tasa tipo_a | tasa pure_fraud |
|---|---|---|---|
| **credit_flow** | 42.5% | **12.0%** | 5.0% |
| mixed | 27.2% | 3.4% | 3.2% |
| **gateway_channel** | 17.6% | 4.0% | 0.7% |
| diversity | 6.7% | 1.9% | **15.2%** |
| magnitude | 3.7% | 6.5% | 9.5% |
| temporal | 2.2% | 7.7% | 1.1% |
| staff_role | 0.2% | 0.0% | 10.0% |

`credit_flow` (42.5%) concentra 2.5× la tasa de reembolso; `diversity` se asocia a `pure_fraud`
(15.2%); `magnitude` sigue marginal (3.7%). Tipos accionables y revisables por operador.

---

## Secuencia global y dependencias

```
Plan A (shadow→champion + umbral segmentado)  ──►  Plan B (tipificación SHAP)
   necesario para que SHAP explique comportamiento y no artefactos
```

Plan B **no debe empezar** hasta que frame-v1 sea el campeón (Plan A, paso 5). La tipificación sobre
el modelo contaminado IF-40 produciría arquetipos falsos.
