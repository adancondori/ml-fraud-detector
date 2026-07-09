# Plan de refactor — Alternativa 1: detección de anomalías por capas

**Fecha:** 2026-07-08
**Ámbito:** herramienta operativa `ml-fraud-detector` + integración con `platform` (fuera de alcance de tesis)
**Base:** `docs/plan-normalizacion-marcos.md`, `docs/analisis-marcos-referencia.md`,
`output/revision/alt1_viability.json` (pruebas de viabilidad, 2026-07-08)
**Decisión del usuario:** Alternativa 1 (capas), IF como base, abandonar AUC-vs-reembolso como métrica titular, **reentrenar el modelo**.

---

## 1. Problema y decisión de paradigma

El score actual no discrimina el proxy de reembolso bajo controles estrictos (AUC 0.508 ≈ azar) y
el "campeón" IF-40 (AUC 0.84) es circular: sus features definen el proxy `pure_fraud`. Conclusión ya
validada experimentalmente: **el reembolso no es una anomalía estadística**; ningún modelo honesto lo
discrimina del azar (~0.50 para todos). El techo lo pone el proxy, no el algoritmo.

**Decisiones (2026-07-08, ratificadas 2026-07-09):**

1. **El proxy de reembolso sale por completo de la evaluación.** No es titular, no es gate, no es
   columna de referencia en el scoreboard. Su único rol es **experiencia documentada**: la lección
   de que un proxy administrativo (reembolso) no es una anomalía estadística y de cómo detectar
   leakage/circularidad (queda en `docs/analisis-marcos-referencia.md` y §2 de este plan).
   **La base de evaluación son anomalías operativas reales**, en dos niveles:
   - *Automático (CI/backtest):* proxies tipificados no circulares — card_testing, new_user_burst,
     velocity_extreme, third_party — verificando programáticamente la disyunción feature↔proxy.
   - *Verdad de terreno (creciente):* alertas revisadas por operador vía el circuito HITL existente
     (`hitl_queue_builder.py`, `hitl_export_alerts.py`, `hitl_ingest_labels.py`,
     `golden_set_v0.parquet`). La precisión confirmada por humanos es la métrica de valor real y
     gana peso sobre los proxies a medida que acumula etiquetas.
2. **Isolation Forest se mantiene como base** del score no supervisado, sobre el feature set limpio
   `frame-v1` (30 features: magnitud relativa a facility, hora local IANA, sin leakage).
3. **Arquitectura por capas:** reglas explícitas (tipos conocidos) + IF frame-v1 (comportamiento
   atípico) + tipificación SHAP (accionabilidad).
4. **El modelo se reentrena** con ventana de datos actualizada antes del cutover (§6).

## 2. Evidencia de viabilidad (pruebas ejecutadas 2026-07-08)

Script: `scripts/verify_alt1_viability.py` → `output/revision/alt1_viability.json`
(muestra 1/10 del test Sep–Dic 2025, n=251,448).

| Prueba | Resultado | Veredicto |
|---|---|---|
| T1 — Artefactos frame-v1 cargan y puntúan (recipe del scorer en vivo) | 30 features, 100% scores finitos | ✅ |
| T2 — IF frame-v1 enriquece anomalías tipificadas SIN contener sus features | card_testing EF@1% = **5.08×**; velocity_extreme EF@1% = **5.05×** (AUC 0.81) | ✅ |
| T2 — Reembolso confirmado como no-objetivo | AUC 0.498, EF@1% 1.19 | ✅ (justifica decisión 1) |
| T3 — Volúmenes de reglas operables | ver tabla §4.1 (21–255 alertas/día según umbral) | ✅ |
| T4 — Complementariedad capas | solo **4.9%** del top-5% de IF coincide con reglas → 95.1% es cobertura conductual neta | ✅ |
| Sesgo estructural del top-5% | monto 1.57× (gate < 3×) | ✅ |
| Datos para regla de card-testing por fallos | `failed_payment_logs` en ClickHouse (`user_token_id`, `facility_id`, `created_at`): con `FINAL` + `_peerdb_is_deleted = 0`, **~436.5K filas** totales y **~631 eventos/día** en 90 días *(snapshot 2026-07-09 — tabla viva, las cifras derivan)* | ✅ |

Debilidades detectadas (las cubren las reglas, no el ML):
- `new_user_burst`: EF@1% 1.36 — IF casi no lo ve → regla.
- `third_party_multi`: EF@1% 0.60 — IF no lo ve → regla (volumen alto, requiere umbral estricto).
- Off-hours: la **definición** ya es canónica y consistente (`OFF_HOURS = {23,0..6}` en
  `features_frame_v1.py:226`, `retrain_frame_v1.py:54`, `backtest_shadow.py:42`); lo que difiere es
  la **superficie de medición**: shadow 4.43% = población completa del sample, metadata 6.46% =
  población de val, viabilidad 12.44% = top-5%. El gate §5 fija superficie explícita.

## 3. Alcance

**Universo de scoring (explícito):** pagos con `payment_method NOT IN ('reversal','free')`,
`user_id != 0`, `_peerdb_is_deleted = 0`. **Inconsistencia a corregir (paso 4b):** el loader de
entrenamiento ya aplica las tres exclusiones (`src/fraud_detector/data/loader.py:101`), pero el
`_FETCH_SQL` del batch (`scorer/batch/scorer.py:151`) NO filtra `user_id != 0` — en 2025 eso son
**47,299 pagos** con `user_id = 0` que el batch puntuaría con historial de usuario vacío. Alinear
ambas superficies al mismo universo. En 2025 eso cubre
**14 gateways activos** (más registros con gateway vacío; card_connect 2.7M, stripe 1.6M, …) y
20 monedas no vacías (21 categorías contando vacío).
**Dentro:** scoring batch primero (Sidekiq/cron), reentrenamiento frame-v1, capa de reglas,
tipificación SHAP, scoreboard nuevo de métricas, promoción champion con sign-off.
**Fuera:** bloqueo en tiempo real (requiere ventana de observación), deep learning, enriquecimiento
externo (device fingerprinting), cambios de tesis (la tesis conserva su marco propio).

## 4. Arquitectura por capas

```
Pago → [Capa 1: Reglas] → flags tipificados (interpretabilidad por construcción)
     → [Capa 2: IF frame-v1] → anomaly score + percentil por segmento (facility→currency→global)
     → [Capa 3: SHAP] → arquetipo dominante + top contribuciones
     → Alerta unificada: {flags[], score, percentile, risk_level, archetype, factors[]}
```

### 4.1 Capa 1 — Reglas explícitas (nuevo módulo `src/fraud_detector/scoring/rules.py`)

Umbral inicial elegido por volumen operable (estimaciones a población completa, T3):

| Regla | Definición | Alertas/día est. | Estado |
|---|---|---|---|
| `card_testing_burst` | `same_amount_count_1h >= 5` | ~103 | Activa |
| `card_testing_failed` | ≥5 fallos mismo `user_token_id` en 1h (`failed_payment_logs`) | por medir | Activa (fuente nueva; ver nota CH) |
| `new_user_burst` | edad < 14d AND `user_txn_count_1h >= 3` | ~24 | Activa |
| `velocity_extreme` | `user_txn_count_24h > 100` | ~86 | Activa |
| `third_party_multi` | tercero AND `user_txn_count_1h >= 2` | ~426 | Shadow (volumen alto; calibrar antes de activar) |
| `discount_extreme` | `user_discount_ratio_30d > 100%` | ~830 | Solo señal agregada por facility (no alerta por txn) |

Cada regla emite un flag tipificado e interpretable. Umbrales en config, no hardcodeados.

**Nota ClickHouse (`card_testing_failed`):** `failed_payment_logs` ordena por
`(facility_id, user_token_id, created_at, id)`; consultar solo por `user_token_id` salta el prefijo
del sorting key. La tabla es pequeña (~436.5K filas con `FINAL` + `_peerdb_is_deleted = 0`;
~631 eventos/día en los últimos 90 días — snapshot 2026-07-09), por lo que un escaneo es tolerable hoy, pero la query
canónica debe incluir `facility_id` en el filtro/agrupación cuando esté disponible. Un skip index
solo se justificaría con `EXPLAIN indexes = 1` sobre datos reales si el volumen crece.

### 4.1b Taxonomía extendida — 16 categorías evaluadas contra datos reales (2026-07-09)

Evaluación de la taxonomía ampliada propuesta, con consultas ejecutadas sobre ClickHouse
(ventana Sep–Nov 2025 salvo indicación; universo estándar §3). Script reproducible:
`scripts/verify_rule_taxonomy_viability.py` → `output/revision/rule_taxonomy_viability.json`.

**Veredictos posibles:** `existente` (ya en §4.1) · `factible_regla` (regla nueva) ·
`factible_senal_agregada` (señal por usuario/facility, no alerta por txn) · `cubierto_por_if`
(feature central de frame-v1; regla redundante y proxy circular) · `diferido` (datos existen,
falta definición de negocio) · `descartado` (sin datos, con evidencia).

| # | Categoría | Regla/señal | Veredicto | Evidencia (datos reales) | ¿Proxy scoreboard? |
|---|-----------|-------------|-----------|--------------------------|--------------------|
| 1 | Velocidad | `velocity_extreme` | existente | ~86 alertas/día (§4.1) | Sí (ya titular) |
| 2 | Descuentos | `discount_extreme` | existente | ~830/día → solo agregada por facility (§4.1) | No (circular: `discount_ratio` ∈ frame-v1) |
| 3 | Card testing | `card_testing_burst` + `card_testing_failed` | existente | ~103/día + fuente `failed_payment_logs` (§4.1) | Sí (ya titular) |
| 4 | Refunds | `refund_extreme` | factible_senal_agregada | ≥5 refunds/30d: ~1,025 user-meses/mes ≈ 34/día | **No — prohibido.** El reembolso salió de la evaluación (§1.1); además es post-hoc (el refund ocurre después del pago) |
| 5 | Montos | `high_amount` | cubierto_por_if | regla >10× promedio facility ≈ 181/día (demasiado); `amount_facility_ratio` + `small/very_small_amount_at_facility` son features centrales | No (circular) |
| 6 | Horarios | `odd_hours` | cubierto_por_if | 4.43% de la población es off-hours local → regla standalone inviable; `is_off_hours_loc` + `off_hours_high_value_loc` ∈ frame-v1; ya existe como gate de sesgo (§5) | No (circular) |
| 7 | Ubicación | `geo_anomaly` | **descartado** | Sin datos: `billing_address_id = 0` en el **100%** del universo 2025 (6,784,670 pagos); `audits.remote_address` existe pero con **0** registros `auditable_type='Payment'` (Sep–Nov: 1.2M audits, ninguno de pagos). No hay IP/país/ciudad por pago | No |
| 8 | Dispositivo | `device_change` | **descartado** | Sin datos: no existe device fingerprint (`jwt_tokens` solo trae `refresh_hash`/`signer`/`session_reference`). Ya estaba fuera de alcance (§3) | No |
| 9 | Cliente nuevo | `new_customer_risk` | existente (= `new_user_burst`) | ~24/día (§4.1) | Sí (ya titular) |
| 10 | Método de pago | `payment_method_switch` | factible_regla | ≥4 métodos distintos/30d: ~630 user-meses/mes ≈ **21/día** (≥3 da ~196/día, inviable) | **No** (circular: `user_distinct_methods` ∈ frame-v1) |
| 11 | Fallos | `failed_payment_burst` | existente (= `card_testing_failed`) | ~631 eventos/día en la fuente (§4.1) | Sí (vía card_testing) |
| 12 | Membresías | `membership_abuse` | diferido (Fase 3) | Datos existen: 4,658 membership_payments con ≥10 guest passes (máx 195) sobre 109,842; pero "abuso" exige join con límites del plan (`membership_plan_types`/`facility_membership_plans`) — definición de negocio pendiente | No (por ahora) |
| 13 | Reservas | `booking_burst` | factible_regla (shadow) | Nueva superficie (`reservations`, 30.7M): ≥10 reservas/h no-admin (`admin_booked=0`, `generated_by_court=0`, `recurring_event_id=0`) ≈ **136/día** → calibrar umbral (~20) en shadow | Candidato (disjunto de features de pago; requiere construir etiqueta a nivel pago) |
| 14 | Cancelaciones | `cancel_after_booking` | factible_regla (shadow) | ≥3 cancelaciones rápidas (<1h tras crear, `deleted_at`)/semana ≈ **49 usuarios/día**. Ojo: 30.15% de reservas 2025 tienen `deleted_at` → mucho ruido benigno, shadow obligatorio | Candidato (disjunto; misma nota que #13) |
| 15 | Multi-cuenta | `multi_account_token` | factible_regla | Vía token de gateway (`user_tokens.token`): 45,063 tokens en ≥2 cuentas, **6,423 en ≥3** (máx 133 cuentas/token). **`last4+card_brand` descartado como identificador**: solo 3,569 firmas para 6.7M pagos (espacio de colisión, máx 1,859 usuarios/firma) | Candidato (disjunto; controlar falsos positivos por familias — `users_relations`, `user_children`) |
| 16 | Comercio | `merchant_outlier` | factible_senal_agregada | Refund rate por facility > μ+3σ (cohorte ≥200 txns): **10 de 549** facilities en Sep–Nov → señal mensual operable | No (granularidad facility, no txn; y usa refund → solo monitoreo) |

**Reglas de decisión aplicadas:**

1. **Circularidad:** si la señal de la categoría es (o deriva de) una feature de frame-v1
   (`user_distinct_methods`, `amount_facility_ratio`, `is_off_hours_loc`, `discount_ratio`),
   puede ser regla operativa pero **nunca** proxy del scoreboard §5. La disyunción se verifica
   programáticamente (mapa señal↔feature en el script de viabilidad y en `eval_scoreboard.py`).
2. **Refund:** cualquier señal basada en estados de reembolso (`refund_extreme`,
   `merchant_outlier`) es solo operativa/descriptiva — jamás entra a evaluación (lección §1.1).
3. **Descartes con evidencia:** `geo_anomaly` y `device_change` se descartan por ausencia de
   datos verificada con queries, no por supuesto. Si platform algún día replica IP/device a
   ClickHouse, se reevalúan (registrado en §11).
4. **Nuevos proxies tipificados candidatos** para EF@k (§5): `multi_account_token`,
   `booking_burst`, `cancel_after_booking` — señales disjuntas de las features de pago. Antes de
   entrar al scoreboard requieren: construcción de la etiqueta a nivel pago (usuario con la
   condición activa en ventana), verificación de disyunción en CI y face validity HITL.

**Decisión de alcance pre-cutover (2026-07-09):** la tabla anterior es el **mapa completo**
(su valor es la evidencia, en especial los descartes); la **implementación antes del cutover se
recorta a una sola regla nueva** para proteger la ruta crítica (pasos 2–10) y la calidad del
primer ciclo HITL (menos tipos de regla = mejores etiquetas por tipo):

| Ítem | Decisión | Justificación costo/beneficio |
|---|---|---|
| `multi_account_token` | **MANTENER (paso 4c)** | Única señal genuinamente nueva que ni reglas ni IF ven (el token no es feature). Como regla de *flujo* — cuenta nueva que se une a token ya compartido por ≥2 — el volumen medido es **~3.1 alertas/día** (280 eventos Sep–Nov), no el stock de 6,423 |
| `payment_method_switch` | **QUITAR** | La capa de reglas existe para lo que IF no ve; `user_distinct_methods` ES feature de frame-v1, así que IF ya ve esta señal y el arquetipo SHAP la tipificaría. Regla redundante (+21 alertas/día sin cobertura neta). Reevaluar solo si HITL muestra un hueco |
| `refund_extreme` | **QUITAR del alcance ml-fraud-detector** | Post-hoc por naturaleza y reintroduce el reembolso justo después de sacarlo del paradigma — riesgo conceptual mayor que el beneficio. Si operaciones lo quiere, es un reporte de platform, no del detector |
| `booking_burst` | **DIFERIR a Fase 3** | Exige nueva superficie de ingesta (`reservations`, 30.7M filas) solo para una regla en shadow; `velocity_extreme` sobre pagos ya correlaciona con el patrón (reserva→pago) |
| `cancel_after_booking` | **DIFERIR a Fase 3** | Misma superficie nueva + 30% de `deleted_at` benigno; el patrón es más abuso de inventario/reservas que anomalía de pago |
| `merchant_outlier` | **DIFERIR a Fase 3** | Barato (1 query mensual) pero es monitoreo refund-based a nivel facility, no detección por transacción; agrupa naturalmente con `membership_abuse` en la fase de señales agregadas |

Presupuesto de alertas resultante: reglas activas ~213/día + `multi_account_token` ~3/día
(+1.4%), sin superficies de datos nuevas antes del cutover (solo `user_tokens`, misma base
ClickHouse ya consultada por el batch).

### 4.2 Capa 2 — IF frame-v1 (reentrenado, §6)

Score no supervisado sobre las 30 features de marco limpio. Umbrales segmentados
(`thresholds_segmented_v1.json`, 452 facilities / 17 monedas) ya conectados al scorer
(`src/fraud_detector/scoring/scorer.py:55-67`). Cubre lo que las reglas no anticipan (95.1% del
top-5% es neto).

### 4.3 Capa 3 — Tipificación SHAP (cierra Plan B existente)

`src/fraud_detector/scoring/archetypes.py` + TreeExplainer sobre frame-v1, solo para transacciones anómalas
(control de latencia). Estado actual del prototipo: 100% del top-5% tipificado
(`output/revision/archetype_report.json`: credit_flow 42%, mixed 27%, gateway_channel 18%).
Pendiente: umbral de concentración final, estabilidad entre corridas ≥99%, face validity ≥50
alertas por arquetipo mayoritario, integración del campo `dominant_archetype` en `ScoreResponse`.

**Cambio de contrato API (no es detalle menor):** `ScoreResponse` (`scorer/schemas.py:64`) hoy solo
tiene los campos de calibración frame-v1; añadir `flags: List[str] = Field(default_factory=list)`
(capa 1 — `default_factory`, no `= []`, para evitar el antipatrón de default mutable) y
`dominant_archetype: Optional[str] = None` + `archetype_scores` (capa 3) como campos opcionales con
default, siguiendo el mismo patrón retrocompatible de `calibration_segment`/`frame_flags`.
Coordinar con el consumidor Rails antes del cutover.

### 4.4 Integración con platform (decisión cerrada, ya no pregunta abierta)

Estado verificado en platform: `FraudAnalyzers::PaymentAnalyzer` NO existe — es un comentario
"Future" en `app/services/fraud_orchestrator.rb:13` — y `Dynamo::FraudScorecard` espera scores
enteros 0–100 (`add_fraud_event` clampa en `fraud_scorecard.rb:54`), mientras el scorer expone
`raw_score` / `percentile` / `risk_level`. **Contrato decidido:**

- Score del evento payment: **`round(percentile)` en escala 0–100** (el percentil por segmento ya
  vive en esa escala; no se inventa otra normalización). `details` lleva `risk_level`, `flags[]`
  (capa 1), `dominant_archetype` (capa 3), `model_version`.
- **Shadow NO escribe en `FraudScorecard`.** Verificado: `add_fraud_event` recalcula
  `overall_risk_score` y ejecuta `determine_blocking_action` al guardar
  (`fraud_scorecard.rb:52-90`) — aun sin llamar `process_with_blocking`, un score shadow podría
  dejar el scorecard en `soft_block`/`hard_block`. Durante shadow, `PaymentAnalyzer` persiste solo
  en la superficie propia ya existente (`anomaly_scores` en ClickHouse local, donde el batch ya
  inserta) y/o logs estructurados. `add_fraud_event('payment', ...)` se conecta ÚNICAMENTE al
  activar (fin de shadow, con sign-off), o antes si platform expone un modo shadow que registre sin
  recalcular blocking.
- Implementar `PaymentAnalyzer` es trabajo en platform, DESPUÉS del cutover del scorer (paso 10),
  siguiendo el pack `anomaly_detection` (Layered Design + TDD).

## 5. Métricas titulares nuevas (reemplazan AUC-vs-reembolso)

| Métrica | Definición | Gate |
|---|---|---|
| **P@k operativo (base real)** | Precisión de alertas revisadas por operador (HITL) — la fuente de verdad del sistema | **Gate de fase, NO de cutover**: el primer ciclo (paso 9) CREA el baseline (hoy no es medible: `golden_set_v0.parquet` tiene 14,831 filas de features/metadata sin etiquetas humanas, y `output/hitl/` no existe). El cutover (paso 10) se decide con los gates automáticos; ciclos HITL siguientes exigen ≥ baseline documentado |
| **EF@1% / EF@5% tipificado (automático)** | Enriquecimiento del score vs cada proxy operativo NO circular (card_testing, velocity_extreme, new_user_burst) | EF@1% ≥ 2× en ≥2 tipos |
| **Sesgo de tamaño** | monto medio top-5% / promedio (winsorizado p99.9) | < 3× |
| **Off-hours — sanidad de datos** | % off-hours local en la **población completa** del test (verifica conversión tz) | banda 3–7% |
| **Off-hours — sesgo de selección** | % off-hours local en el **top-5%** ÷ % poblacional | ≤ 3× (viabilidad actual: 12.44/4.43 ≈ 2.8×) |
| **Estabilidad multi-semilla** | rango relativo de EF@1% entre 3 semillas (rango/mediana), sobre los proxies del gate EF (card_testing, velocity_extreme) y la unión tipificada, población completa del test | < 15% |
| **Estabilidad multi-semilla — AUC (secundaria, sin gate)** | rango de AUC entre semillas, solo diagnóstico | reportar |
| **Cobertura de tipificación** | % del top-k con arquetipo asignado | 100% (incl. `mixed`) |

**Nota de re-especificación del gate de estabilidad (2026-07-09, validada empíricamente):** el gate
anterior ("rango AUC/EF < 0.01") estaba mal especificado: (a) un rango absoluto de 0.01 sobre EF
(escala 1–5×) es inalcanzable por construcción; (b) reproducción in-memory con semillas 42/43/44
(muestra 1/10, receta exacta de `retrain_frame_v1.py`; script reproducible
`scripts/verify_multiseed_stability.py` → `output/revision/multiseed_stability.json`, gate PASS)
mostró AUC rango 0.027 en velocity_extreme —
rompía el gate sin indicar inestabilidad real. Los rangos relativos de EF@1% en los proxies del gate
fueron: card_testing 2.0%, unión tipificada 5.4%, velocity_extreme 13.3% (→ umbral 15% con margen;
población completa será más estable que la muestra). `new_user_burst` y `third_party_multi` se
EXCLUYEN del gate de estabilidad: IF es casi ciego a ellos (EF ~1–2 con base rate mínima → varianza
alta estructural) y su cobertura es responsabilidad de las reglas, no del modelo. AUC deja de ser
gate — coherente con su abandono como métrica titular (§1). Sobre la "estabilidad de volumen": el
volumen de alertas es estable **por construcción** (umbral por percentil segmentado fija el top-k);
lo que varía entre semillas es *qué* pagos caen en el top-k, y eso lo mide el solapamiento top-1%
entre semillas (Jaccard 0.66–0.69 en la reproducción 1/10 — diagnóstico reportado, sin gate).

El reembolso (`tipo_a`) **no aparece en el scoreboard** — ni como gate ni como referencia. Cualquier
script de evaluación nuevo (`eval_scoreboard.py`) no lo calcula; la experiencia que dejó está
documentada en §2 y en `docs/analisis-marcos-referencia.md`.

Todo reporte que hoy encabeza con AUC-vs-reembolso (README, tablas, dashboards, `results*.json`
nuevos) pasa a encabezar con este scoreboard.

## 6. Reentrenamiento del modelo (requisito explícito)

El frame-v1 vigente se entrenó con train Ene–Jun 2025; hay datos hasta jul 2026 en ClickHouse.

1. **Re-extracción:** extender dataset vía `DataManager.extract_from_clickhouse`
   (`src/fraud_detector/data/loader.py:148`) / `run_pipeline.py step1_extract` (`run_pipeline.py:132`)
   con ventana rodante: warm 1 mes + train 6–12 meses recientes; corregir en la fuente facilities
   con moneda/tz vacía. *(No existe `scripts/extract_full_dataset.py`; el plan anterior lo
   referenciaba por error.)*
2. **Regenerar `facility_stats_v1.json`** con `scripts/build_facility_stats.py` sobre el train nuevo
   (stats de magnitud por facility deben reflejar la población actual).
3. **Reentrenar IF frame-v1** con `scripts/retrain_frame_v1.py` (saneo p99.99 por moneda).
   **Trabajo nuevo requerido:** el script hoy solo entrena `random_state=42`
   (`retrain_frame_v1.py:440`); ampliarlo con `--seeds 42,43,44` (o runner) para poder cumplir el
   gate de estabilidad multi-semilla.
4. **Recalibrar umbrales segmentados** con `scripts/calibrate_segmented_thresholds.py`.
5. **Gates post-reentrenamiento:** paridad batch↔real-time (maxdiff < 1e-8), sesgo de tamaño < 3×,
   off-hours local en banda, estabilidad multi-semilla, cobertura tz/currency > 98%, y scoreboard §5.
6. **Regenerar explainer/arquetipos** sobre el modelo nuevo (SHAP se invalida al reentrenar).
7. **Política de reentrenamiento periódico:** documentar cadencia (trimestral o por drift de
   `FrameFlags`/cobertura) — nuevo `docs/retraining-policy.md`.

## 7. TDD (tests antes de código)

1. `rules.py`: cada regla con casos borde (umbral exacto, campos nulos, usuario sin historial).
2. Off-hours: helper único compartido (scorer/backtest/eval) que reciba la superficie (población |
   top-k) como parámetro explícito; test de que ambas métricas del §5 se calculan sobre la
   superficie correcta.
3. Alerta unificada: `ScoreResponse` retrocompatible con `flags[]` + `dominant_archetype` (default vacío).
4. Scoreboard: cálculo EF@k/P@k determinista con fixture sintético (proxy conocido).
5. Regresión de paridad batch↔real-time tras reentrenar (test existente, correr con artefactos nuevos).
6. Arquetipos: mapeo determinista, umbral de concentración → `mixed`, cobertura 100% del top-k.
7. `card_testing_failed`: agregación por `user_token_id` ventana 1h excluye el evento actual (anti-leakage).
8. Taxonomía extendida — disyunción programática: test de que ninguna categoría marcada
   `cubierto_por_if`/circular (`payment_method_switch`, `high_amount`, `odd_hours`,
   `discount_extreme`) aparece como proxy del scoreboard; el mapa señal↔feature se valida contra
   `FRAME_V1_FEATURE_NAMES` real (no contra una copia).
9. Taxonomía extendida — política refund: test de que `refund_extreme` y `merchant_outlier`
   quedan excluidos del scoreboard aunque su señal sea disjunta (política §1.1, no disyunción).
10. `multi_account_token`: umbral exacto (3 cuentas = flag, 2 = no), token vacío no cuenta,
    identificador es `user_tokens.token` (test que rechaza `last4+brand` como clave).
11. *(Fase 3, cuando se implementen)* Reglas de reservas (`booking_burst`, `cancel_after_booking`):
    exclusión de `admin_booked`/`generated_by_court`/`recurring_event_id`; cancelación rápida usa
    `deleted_at > created_at` con ventana <1h y umbral exacto; ambas nacen en shadow.
12. Clasificación de volumen (`classify_volume`): bordes exactos operable/shadow/agregada.

## 8. Pasos atómicos (un commit por paso)

| # | Paso | Depende de |
|---|---|---|
| 0 | Versionar en git este plan + `scripts/verify_alt1_viability.py` (hoy untracked) | — |
| 1 | Gate off-hours con superficie explícita (población = sanidad; top-5%/población = sesgo) + test | — |
| 2 | Re-extracción de datos + regenerar `facility_stats_v1` | — |
| 3 | Reentrenar frame-v1 (3 semillas) + recalibrar umbrales + gates §6.5 | 2, **6** (los gates del reentrenamiento se juzgan con el scoreboard — construir el juez antes que lo juzgado) |
| 4 | `src/fraud_detector/scoring/rules.py` — 6 reglas de §4.1: 4 activas (`card_testing_burst`, `card_testing_failed`, `new_user_burst`, `velocity_extreme`), 1 en shadow (`third_party_multi`), 1 señal agregada por facility (`discount_extreme`) + config de umbrales + tests | 1 |
| 4b | Alinear universo batch↔loader: añadir `user_id != 0` a `_FETCH_SQL` (`scorer/batch/scorer.py:151`) **y a `_CURSOR_END_SQL` (`scorer/batch/scorer.py:161`)** — si el cursor no filtra, puede avanzar apoyado en pagos fuera del universo — + tests de ambos | — |
| 4c | `multi_account_token` (única regla nueva de la taxonomía extendida que entra pre-cutover, decisión de alcance §4.1b): regla de flujo ~3/día, guard de familias (`users_relations`/`user_children`), fuente `user_tokens` + tests 8–10 §7 | 4 |
| 5 | Fuente `failed_payment_logs`: query canónica + regla `card_testing_failed` + medición de volumen | 4 |
| 6 | Scoreboard nuevo: `scripts/eval_scoreboard.py` (EF@k tipificado, sesgos, estabilidad) **con verificación programática de disyunción feature↔proxy integrada** (reutilizar `scripts/validate_if40_pivot_disjoint.py`) y corriendo en CI; reemplaza eval centrado en refund-AUC. Se construye y valida contra los artefactos frame-v1 vigentes + fixture sintético (§7.4); corre sobre el campeón reentrenado en los pasos 3 y 9 | 1 |

**Secuencia de ejecución (2026-07-09, revisada):** 0 → 4b → 1 → 6 → 2 → 3 → 4 → 4c → 5 → 7 → 8 → 9 → 10.
No se reentrena antes de cerrar la superficie de scoring (4b) ni de tener el juez automático (6).
El paso 2 (re-extracción) es I/O-bound e independiente: puede lanzarse en paralelo con 1/6 sin
violar la secuencia — lo único que exige el orden es que 3 no arranque sin 6 terminado.
| 7 | Cerrar Plan B: arquetipos sobre modelo reentrenado + campos en `ScoreResponse` | 3 |
| 8 | Alerta unificada (flags + score + arquetipo) en el scorer batch | 4,7 |
| 9 | Backtest de capas completo (reglas + IF + arquetipos) sobre test reciente → reporte + cola HITL para validación humana (base real del scoreboard) | **4b (bloqueante)**, 6, 8 |
| 10 | **Decisión de promoción con sign-off** + deprecación path base-31 + actualizar README/docs | 9 |

## 9. Rollout

1. **Shadow (2 semanas):** capas corren en paralelo al campeón IF-40; comparar volúmenes y face validity.
2. **Sign-off explícito:** la métrica de titular cambia (desaparece AUC 0.84 circular). Documentar como
   **de-circularización, no regresión**. Nunca cutover silencioso.
3. **Champion:** frame-v1 reentrenado decide; IF-40 degradado a señal secundaria durante observación.
4. **Retiro:** deprecar path base-31 (`src/fraud_detector/scoring/features.py` 31f, `thresholds.json` legacy).
5. **Post-cutover (platform):** `PaymentAnalyzer` en shadow según contrato §4.4.

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Caída de la métrica de titular percibida como regresión | Sign-off + documento de de-circularización (§9.2) |
| Volumen de alertas inmanejable (union rate 8% en backtest) | El top-5% es artefacto de backtest; producción usa umbrales segmentados calibrados por percentil; reglas `third_party`/`discount` en shadow |
| Reentrenar con ventana nueva cambia sesgos | Gates §6.5 obligatorios post-retrain |
| `failed_payment_logs.description` sin parsear (sin monto/tarjeta explícitos) | Regla v1 solo por `user_token_id`+ventana; parsing de description como mejora posterior |
| Proxies tipificados comparten ancestros con features (disyunción imperfecta) | Mantener verificación programática de disyunción feature↔proxy en CI (ya existe patrón en pivot IF-40) |
| `multi_account_token` penaliza tarjetas familiares legítimas | Guard con `users_relations`/`user_children` antes de activar; umbral ≥3 cuentas; shadow si el volumen diario supera lo operable |
| `cancel_after_booking` sobre superficie con 30% de `deleted_at` benigno *(diferida a Fase 3)* | Shadow obligatorio cuando se implemente; umbral por cancelación *rápida* (<1h), no por cancelación a secas |
| Señales de reservas contaminan el scoreboard sin construcción correcta de etiqueta | Los candidatos (§4.1b.4) no entran a EF@k hasta tener etiqueta a nivel pago + disyunción en CI + face validity HITL |

## 11. Preguntas abiertas

1. ¿Umbral final de `third_party_multi` para salir de shadow (426/día es mucho)?
2. ¿Cadencia de reentrenamiento: trimestral fija o disparada por drift?
3. ¿Definición de negocio de `membership_abuse` (límites de guest passes por plan)? — bloquea el veredicto diferido de §4.1b #12.
4. *(Fase 3)* ¿Umbral de `booking_burst` para salir de shadow (≥10/h da 136/día; probar ≥20)?
5. Si platform replica IP o device fingerprint a ClickHouse, reevaluar `geo_anomaly` y `device_change` (hoy descartados por ausencia de datos, §4.1b #7–8).

*(Resuelta 2026-07-09: la alerta unificada se consume vía `Dynamo::FraudScorecard` con
`round(percentile)` como score 0–100 — ver contrato §4.4.)*
