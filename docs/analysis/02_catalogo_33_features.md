# DOCUMENTO DEPRECADO (2026-03-24)
> **Supersedido por:** `.planning/PLAN-FINAL/01_CONTRATO_ALCANCE.md` (seccion 5) y `04_FEATURE_ENGINEERING.md`
> Este documento se mantiene como referencia historica. La fuente canonica de features es el PLAN-FINAL.

---

# Catálogo de 33 Features para Detección de Anomalías

> Documento de referencia para el modelo Isolation Forest.
> Incluye los 23 features originales de la tesis (Fase 0) más 10 features
> nuevos propuestos para robustecer la detección.
>
> Cada feature se calcula **a nivel transacción** respetando separación
> temporal estricta para prevenir data leakage.

---

## Resumen de Composición

| Categoría | Features Originales | Features Nuevos | Total |
|-----------|:-------------------:|:---------------:|:-----:|
| Transaccionales | 6 (F01–F06) | — | 6 |
| Temporales | 5 (F07–F11) | — | 5 |
| Velocidad | 4 (F12–F15) | — | 4 |
| Comportamiento del Usuario | 6 (F16–F21) | — | 6 |
| Contextuales | 2 (F22–F23) | — | 2 |
| **Crédito / Flujo** | — | **4 (F24–F27)** | **4** |
| **Rol / Staff** | — | **3 (F28–F30)** | **3** |
| **Diversidad Operacional** | — | **3 (F31–F33)** | **3** |
| **Total** | **23** | **10** | **33** |

---

## ALERTA CRÍTICA: Normalización Multi-Moneda

### El Problema

Los montos en `payments.reservation_paid_out` están almacenados en **moneda local**,
NO normalizados a USD. El dataset contiene **21 monedas distintas** a través de
**13 gateways** en ~1,039 facilities.

### Distribución de Monedas

| Moneda | Gateway(s) | Txns | Mediana local | ≈ USD | Distorsión vs USD |
|--------|-----------|-----:|-------------:|------:|:-----------------:|
| USD | CardConnect, Stripe, Azul | 4,905,529 | 27 | $27 | 1x (base) |
| CAD | Stripe, CardConnect | 411,137 | 21 | $15 | 0.8x |
| MYR | Stripe | 342,830 | 70 | $15 | 2.6x |
| HNL | PixelPay, BAC | 215,118 | 261 | $10 | **10x** |
| AUD | Stripe | 187,700 | 25 | $16 | 0.9x |
| NIO | PixelPay, BAC | 175,542 | 328 | $9 | **12x** |
| ILS | iCount, OnePay | 137,530 | 51 | $14 | 1.9x |
| GTQ | BAC | 86,883 | 120 | $16 | **4.4x** |
| PKR | PayFast | 62,796 | 7,200 | $26 | **267x** |
| HKD | Stripe | 54,479 | 250 | $33 | 9x |
| SGD | Stripe | 51,366 | 39 | $29 | 1.4x |
| COP | OnePay | 22,992 | 33,750 | $8 | **1,250x** |
| BWP | DPO | 16,030 | 128 | $9 | 4.7x |
| AED | Stripe | 15,927 | 110 | $30 | 4x |
| EUR | Stripe | 12,965 | 9 | $10 | 0.3x |
| RWF | Stripe | 2,365 | 15,000 | $11 | **556x** |
| JPY | Stripe | 2,292 | 6,000 | $40 | **222x** |
| MXN | Stripe, PayCode, Mitec | 861 | 800 | $39 | **30x** |
| INR | RazorPay | 64 | 618 | $7 | 23x |
| NZD | Stripe | 2 | 37 | $22 | 1.4x |

### Hallazgo Clave

- **1,035 facilities** son mono-moneda → normalización per-facility es viable
- Solo **4 facilities** manejan 2 monedas
- En USD equivalente, las medianas globales convergen a **$8–$40** (deportes similares)
- Sin normalización, una transacción COP de Q33,750 (~$8 USD) se confunde con
  una anomalía de $33,750 USD

### Impacto por Feature

| Feature | Impacto | Severidad | Solución |
|---------|---------|:---------:|----------|
| F01 `reservation_paid_out` | COP/PKR/JPY distorsionan completamente | **CRÍTICO** | Normalizar a USD o usar z-score per-facility |
| F02 `log_amount` | Log reduce pero no elimina (log(33750)=10.4 vs log(27)=3.3) | **ALTO** | Normalizar antes de log |
| F03 `amount_usd_ratio` | Ratio contra promedio global mezclando monedas | **CRÍTICO** | Calcular ratio per-currency o per-facility |
| F04 `discount_ratio` | Ratio interno (misma moneda) | OK | Sin cambio |
| F15 `user_amount_24h` | Acumulación en moneda local | **MEDIO** | Normalizar a USD |
| F22 `facility_avg_amount` | Per-facility = misma moneda | OK | Sin cambio |
| F23 `amount_facility_ratio` | Ratio dentro de facility | OK | Sin cambio |
| F26 `user_debit_amount_30d` | Si usuario opera en una sola moneda (mayoría) | **BAJO** | Normalizar a USD para usuarios multi-facility multi-moneda |
| F27 `credit_flow_ratio` | Ratio interno (misma moneda) | OK | Sin cambio |
| F30 `staff_amount_zscore` | Cohort de rol cruza monedas | **CRÍTICO** | Calcular z-score per-currency-AND-role |

### Estrategia de Normalización Recomendada

**Enfoque híbrido (2 capas):**

1. **Capa 1 — Normalización a USD:** Agregar campo `amount_usd` usando tabla
   de tipos de cambio mensuales. Esto normaliza F01, F02, F15, F26.
   ```
   amount_usd = reservation_paid_out * exchange_rate(currency, month)
   ```

2. **Capa 2 — Features relativos per-facility:** Para F03, F30, usar
   normalización dentro del contexto de la facility (que ya es mono-moneda
   en 99.6% de los casos). Esto evita depender de tipos de cambio exactos.
   ```
   amount_facility_zscore = (amount - facility_avg) / facility_std
   ```

3. **Feature adicional:** Agregar `currency_group` como feature categórico
   (one-hot o target encoding) para capturar patrones por mercado.

### Implicancia para los Documentos de Fase 0 y Fase 1

- **Fase 0 (03_fase_0_outputs.md):** El pipeline de feature engineering debe
  incluir un paso de normalización monetaria ANTES del cálculo de features.
- **Fase 1 (04_fase_1_diseno.md):** Los features de usuario deben normalizarse
  per-currency para comparaciones de cohort.
- **Evaluación:** Las métricas (AUC-ROC, AP) deben reportarse tanto globales
  como segmentadas por moneda para verificar que el modelo no tiene sesgo.

---

## Categoría A: Features Transaccionales (F01–F06)

### F01 — `reservation_paid_out`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Transaccional (original) |
| **Fórmula** | Monto en USD de la transacción |
| **Fuente** | `payments.reservation_paid_out` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Magnitud absoluta de la transacción. Las transacciones reembolsadas promedian 6.2x más que las capturadas ($1,236.72 vs $198.97). Es el indicador más directo de riesgo por monto. |
| **Comportamiento esperado** | Normal: concentrado entre $0–$200. Anómalo: > $1,000 o exactamente $0 en contexto inusual. |
| **Patrón para caso insider** | Montos individuales pueden ser normales ($0–$500), pero el volumen acumulado es alto. Feature insuficiente por sí solo para detectar insider. |
| **Resultado que aporta** | Score base por magnitud. Transacciones de monto extremo reciben scores altos. |

### F02 — `log_amount`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Transaccional (original) |
| **Fórmula** | `log(reservation_paid_out + 1)` |
| **Fuente** | Derivado de `payments.reservation_paid_out` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | La distribución de montos es extremadamente sesgada a la derecha (mediana ~$30, media ~$199, max >$1M). La transformación logarítmica comprime la cola derecha, permitiendo que el modelo no se sature con outliers extremos y capture variaciones en rangos bajos-medios. |
| **Comportamiento esperado** | Distribución más simétrica que F01. Valores > 7 (≈$1,096) son inusuales. |
| **Patrón para caso insider** | Similar a F01 — valores moderados individualmente. |
| **Resultado que aporta** | Mejora la sensibilidad del modelo a variaciones de monto en rangos normales ($10–$500). |

### F03 — `amount_usd_ratio`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Transaccional (original) |
| **Fórmula** | `reservation_paid_out / AVG(reservation_paid_out)_global` |
| **Fuente** | Derivado de `payments.reservation_paid_out` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Normaliza el monto contra el promedio global. Permite identificar transacciones que se desvían del baseline general, independientemente de la escala absoluta. Un ratio > 5 indica una transacción 5x mayor al promedio. |
| **Comportamiento esperado** | Normal: 0.1–2.0. Anómalo: > 5.0 o = 0 (transacción gratuita). |
| **Patrón para caso insider** | Mixto — muchas txns con ratio ≈ 0 (free/discount), algunas con ratio moderado. |
| **Resultado que aporta** | Contexto relativo global del monto. |

### F04 — `discount_ratio`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Transaccional (original) |
| **Fórmula** | `discount / (reservation_paid_out + 0.01)` |
| **Fuente** | `payments.discount`, `payments.reservation_paid_out` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Descuentos desproporcionados son un indicador de abuso operacional. Un descuento > monto (ratio > 1.0) es particularmente sospechoso. En el dataset, muchas transacciones de insider tienen discount ≈ monto total. |
| **Comportamiento esperado** | Normal: 0 (sin descuento) o < 0.5. Anómalo: ≥ 1.0 (descuento cubre el monto completo). |
| **Patrón para caso insider** | **MUY ALTO** — Pablo tiene cientos de txns donde discount ≈ monto. Este feature SÍ señala muchas de sus transacciones. |
| **Resultado que aporta** | Señal directa de descuentos inapropiados. Uno de los features más discriminativos para fraude operacional. |

### F05 — `has_tip`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Transaccional (original) |
| **Fórmula** | `1 si tip > 0, 0 si no` (binario) |
| **Fuente** | `payments.tip` |
| **Tipo de dato** | Binario (0/1) |
| **Justificación** | Las transacciones con propina tienen un perfil de riesgo diferente: implican una interacción de servicio real (lección, clínica). Transacciones anómalas raramente incluyen propina. |
| **Comportamiento esperado** | ~5-10% de las txns tienen propina. |
| **Patrón para caso insider** | Bajo — los pagos de manager rara vez incluyen propina. |
| **Resultado que aporta** | Indicador de autenticidad de la transacción. |

### F06 — `is_free`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Transaccional (original) |
| **Fórmula** | `1 si payment_method = 'free', 0 si no` (binario) |
| **Fuente** | `payments.payment_method` |
| **Tipo de dato** | Binario (0/1) |
| **Justificación** | El 21.15% de las transacciones del dataset tienen monto $0. Transacciones "free" son comunes para cortesías de manager, pero un patrón excesivo indica abuso de privilegios. |
| **Comportamiento esperado** | Jugadores normales: < 5% free. Managers: 15-30% free. |
| **Patrón para caso insider** | **ALTO** — Pablo tiene un % elevado de txns free, especialmente reservas. |
| **Resultado que aporta** | Señal de transacciones sin cargo. En combinación con velocidad y rol, indica abuso. |

---

## Categoría B: Features Temporales (F07–F11)

### F07 — `hour_sin`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Temporal (original) |
| **Fórmula** | `sin(2π × hour / 24)` |
| **Fuente** | `payments.created_at` (hora extraída) |
| **Tipo de dato** | Continuo, [-1, 1] |
| **Justificación** | Codificación cíclica de la hora del día. Permite que el modelo capture que las 23:00 y las 01:00 están temporalmente cerca (cosa que una codificación lineal 0-23 no logra). Las transacciones fraudulentas tienden a ocurrir en horarios atípicos. |
| **Comportamiento esperado** | Distribución según horario de operación de cada facility. |
| **Patrón para caso insider** | Pablo tiene cargas de crédito a las 02:00-03:43 AM — horarios atípicos. |
| **Resultado que aporta** | Componente de par trigonométrico para capturar periodicidad horaria. |

### F08 — `hour_cos`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Temporal (original) |
| **Fórmula** | `cos(2π × hour / 24)` |
| **Fuente** | `payments.created_at` (hora extraída) |
| **Tipo de dato** | Continuo, [-1, 1] |
| **Justificación** | Complemento de F07. Juntos (sin + cos) forman una representación completa de la hora en coordenadas polares, preservando la distancia real entre cualquier par de horas. |
| **Comportamiento esperado** | Complementario a F07. |
| **Patrón para caso insider** | Igual que F07. |
| **Resultado que aporta** | Completa la codificación cíclica temporal. |

### F09 — `day_of_week`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Temporal (original) |
| **Fórmula** | Día de la semana (1=Lunes, 7=Domingo) |
| **Fuente** | `payments.created_at` |
| **Tipo de dato** | Ordinal, 1–7 |
| **Justificación** | Los patrones de pago varían significativamente por día. Fines de semana tienen más reservas de jugadores; entre semana hay más operaciones administrativas. Transacciones en días inusuales para su tipo son sospechosas. |
| **Comportamiento esperado** | Distribución bimodal: pico miércoles-jueves (admin) + sábado-domingo (juego). |
| **Patrón para caso insider** | Pablo opera todos los días — no tiene patrón semanal claro. |
| **Resultado que aporta** | Contexto de día laboral vs. fin de semana. |

### F10 — `is_weekend`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Temporal (original) |
| **Fórmula** | `1 si day_of_week ∈ {6, 7}, 0 si no` (binario) |
| **Fuente** | Derivado de `payments.created_at` |
| **Tipo de dato** | Binario (0/1) |
| **Justificación** | Simplificación binaria del patrón semanal. Operaciones administrativas en fin de semana son menos comunes y potencialmente sospechosas. |
| **Comportamiento esperado** | ~30% de txns en fin de semana para un club deportivo. |
| **Patrón para caso insider** | Pablo opera tanto entre semana como fines de semana — lo cual es normal para un manager. |
| **Resultado que aporta** | Flag rápido de temporalidad semanal. |

### F11 — `is_off_hours`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Temporal (original) |
| **Fórmula** | `1 si hour ∈ [23, 0, 1, 2, 3, 4, 5, 6], 0 si no` (binario) |
| **Fuente** | `payments.created_at` |
| **Tipo de dato** | Binario (0/1) |
| **Justificación** | Transacciones entre 23:00–06:00 son fuera del horario operativo normal de un club deportivo. El pico de refunds ocurre a las 10:00 AM (24.9% de reversals), pero las transacciones de madrugada son un indicador diferente de actividad sospechosa. |
| **Comportamiento esperado** | < 5% de txns en off-hours. |
| **Patrón para caso insider** | Pablo tiene cargas de crédito a las 02:00 y 03:43 — off-hours. |
| **Resultado que aporta** | Señal de actividad fuera de horario. |

---

## Categoría C: Features de Velocidad (F12–F15)

### F12 — `user_txn_count_1h`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Velocidad (original) |
| **Fórmula** | Cantidad de transacciones del mismo usuario en la hora previa a esta transacción |
| **Fuente** | Ventana temporal sobre `payments` por `user_id` |
| **Tipo de dato** | Entero, ≥ 0 |
| **Justificación** | Ráfagas de actividad (burst activity) son indicadoras de automatización, operaciones batch, o abuso. Un usuario con 10+ transacciones en una hora es atípico para un jugador, pero puede ser normal para un manager haciendo reservas masivas. |
| **Comportamiento esperado** | Jugadores: 0–3. Managers: 0–15. Anómalo: > 20. |
| **Patrón para caso insider** | **MUY ALTO** — Pablo crea lotes de 10-20 reservas en minutos. Este feature lo señalaría. |
| **Resultado que aporta** | Detección de bursts transaccionales. Uno de los features más señalizadores para Pablo. |

### F13 — `user_txn_count_24h`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Velocidad (original) |
| **Fórmula** | Cantidad de transacciones del mismo usuario en las 24 horas previas |
| **Fuente** | Ventana temporal sobre `payments` por `user_id` |
| **Tipo de dato** | Entero, ≥ 0 |
| **Justificación** | Volumen diario excesivo. El dataset tiene un usuario con 636 transacciones en un día. Incluso managers normales rara vez superan 50-100. |
| **Comportamiento esperado** | Jugadores: 0–5. Managers: 5–50. Anómalo: > 100. |
| **Patrón para caso insider** | **ALTO** — Pablo alcanza picos de cientos de transacciones en un día. |
| **Resultado que aporta** | Detección de volumen diario excesivo. |

### F14 — `time_since_last_txn`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Velocidad (original) |
| **Fórmula** | Segundos desde la transacción anterior del mismo usuario |
| **Fuente** | Diferencia temporal entre `payments.created_at` consecutivos por `user_id` |
| **Tipo de dato** | Continuo, ≥ 0 (NULL para primera txn → se imputa con mediana) |
| **Justificación** | Transacciones en rápida sucesión (< 60 segundos) son indicadoras de operación masiva, posible automatización, o abuso de sistema batch. |
| **Comportamiento esperado** | Normal: > 300 segundos (5 min). Sospechoso: < 30 segundos entre txns. |
| **Patrón para caso insider** | **MUY BAJO** — Pablo tiene transacciones separadas por segundos (ej: 2025-01-04 crea 20+ reservas en 10 minutos). |
| **Resultado que aporta** | Detección de velocidad transaccional excesiva. |

### F15 — `user_amount_24h`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Velocidad (original) |
| **Fórmula** | Suma de montos de transacciones del usuario en las 24 horas previas |
| **Fuente** | Ventana temporal sobre `payments` por `user_id` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Gasto acumulado diario. Un usuario que acumula >$5,000 en un día es atípico. Captura tanto transacciones individuales altas como acumulación de muchas pequeñas. |
| **Comportamiento esperado** | Jugadores: $0–$500. Managers: varía. Anómalo: > $5,000 sin justificación. |
| **Patrón para caso insider** | Moderado — los montos individuales de Pablo son modestos, pero acumula volumen. |
| **Resultado que aporta** | Detección de acumulación de gasto diario. |

---

## Categoría D: Features de Comportamiento del Usuario (F16–F21)

### F16 — `user_distinct_facilities_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Comportamiento (original) |
| **Fórmula** | Cantidad de facilities distintas donde el usuario transacciona en 30 días |
| **Fuente** | Ventana 30d sobre `payments.facility_id` por `user_id` |
| **Tipo de dato** | Entero, ≥ 1 |
| **Justificación** | La mayoría de usuarios opera en 1-2 facilities. Diversificación anómala (4+) puede indicar actividad distribuida para evitar detección o acceso administrativo amplio. |
| **Comportamiento esperado** | Jugadores: 1–2. Multi-club players: 2–3. Managers multi-sede: 3–5. |
| **Patrón para caso insider** | **4 facilities** — Pablo opera en Williamsburg, Dumbo, Nashville, Atlanta. Alto para cualquier rol. |
| **Resultado que aporta** | Detección de diversificación geográfica anómala. |

### F17 — `user_distinct_methods`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Comportamiento (original) |
| **Fórmula** | Cantidad de métodos de pago distintos usados por el usuario |
| **Fuente** | `payments.payment_method` por `user_id` |
| **Tipo de dato** | Entero, ≥ 1 |
| **Justificación** | Usuarios normales usan 1-2 métodos (tarjeta + eventualmente cash). Diversidad alta (3+) puede indicar manipulación de métodos para ocultar patrones. |
| **Comportamiento esperado** | Normal: 1–2. Sospechoso: 4+. |
| **Patrón para caso insider** | 3 métodos (cash, free, prepaid) — moderadamente diverso. |
| **Resultado que aporta** | Señal de diversidad de métodos de pago. |

### F18 — `user_reversal_ratio_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Comportamiento (original) |
| **Fórmula** | `reversals_30d / total_txns_30d` |
| **Fuente** | Ventana 30d sobre `payments` filtrado por status o categoría reversal |
| **Tipo de dato** | Continuo, [0, 1] |
| **Justificación** | Ratio de transacciones revertidas. Es el feature más directamente correlacionado con el proxy. **ADVERTENCIA**: Tiene correlación mecánica con la variable proxy (refunds), ya que reversals se generan automáticamente después de refunds. |
| **Comportamiento esperado** | Normal: 0–0.05. Alto: > 0.10. Existen usuarios con 100% reversal rate. |
| **Patrón para caso insider** | Elevado — 204 refunds en 8 meses, pero ratio moderado por alto volumen total. |
| **Nota metodológica** | Si la diferencia en AUC-ROC entre modelo de 33 features y modelo de 32 features (sin F18) es ≥ 0.02, se reporta el modelo de 32 features como primario para evitar inflación artificial de métricas. |
| **Resultado que aporta** | Concentración de reversals por usuario. Feature de referencia con advertencia de circularidad. |

### F19 — `user_account_age_days`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Comportamiento (original) |
| **Fórmula** | `(fecha_transacción - user.created_at).days` |
| **Fuente** | `payments.created_at` - `users.created_at` |
| **Tipo de dato** | Entero, ≥ 0 |
| **Justificación** | Cuentas nuevas (< 30 días) tienen mayor riesgo de fraude. Cuentas establecidas con cambio repentino de comportamiento también son señal de compromiso. |
| **Comportamiento esperado** | Distribución amplia. Nuevas (< 30d): mayor riesgo. Establecidas (> 365d): riesgo base menor. |
| **Patrón para caso insider** | ~850 días — cuenta muy establecida. No es señal de alerta por sí sola. |
| **Resultado que aporta** | Contexto de madurez de la cuenta. |

### F20 — `user_discount_ratio_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Comportamiento (original) |
| **Fórmula** | `SUM(discount_30d) / SUM(reservation_paid_out_30d)` |
| **Fuente** | Ventana 30d sobre `payments` por `user_id` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Patrón acumulado de descuentos. Un usuario que consistentemente recibe descuentos > 50% del monto total es sospechoso, especialmente si no tiene un cupón o membresía que lo justifique. |
| **Comportamiento esperado** | Normal: 0–0.15. Sospechoso: > 0.50. |
| **Patrón para caso insider** | **MUY ALTO** — Pablo aplica descuentos sistemáticos. Muchas txns con discount ≈ monto total. |
| **Resultado que aporta** | Detección de patrones de descuento acumulado. |

### F21 — `user_free_pct_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Comportamiento (original) |
| **Fórmula** | `count(payment_method='free')_30d / total_txns_30d` |
| **Fuente** | Ventana 30d sobre `payments` por `user_id` |
| **Tipo de dato** | Continuo, [0, 1] |
| **Justificación** | Propensión a transacciones gratuitas. Managers tienen legítimamente más txns free (cortesías, demos), pero un % excesivo indica abuso de privilegios. |
| **Comportamiento esperado** | Jugadores: 0–0.05. Managers legítimos: 0.10–0.25. Abusivo: > 0.35. |
| **Patrón para caso insider** | **ALTO** — Pablo tiene gran % de reservas "free". |
| **Resultado que aporta** | Señal de abuso de gratuidades. |

---

## Categoría E: Features Contextuales (F22–F23)

### F22 — `facility_avg_amount`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Contextual (original) |
| **Fórmula** | Promedio de `reservation_paid_out` para la facility de esta transacción |
| **Fuente** | Agregado de `payments` por `facility_id` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Cada facility tiene un baseline de monto diferente según su mercado, precios, y clientela. Lo que es normal en una facility premium ($200/reserva) es anómalo en una económica ($30/reserva). |
| **Comportamiento esperado** | Varía por facility. Range típico: $20–$500. |
| **Patrón para caso insider** | Las facilities de Pablo (Padel Haus) tienen promedios moderados. |
| **Resultado que aporta** | Baseline contextual por sede para normalizar montos. |

### F23 — `amount_facility_ratio`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Contextual (original) |
| **Fórmula** | `reservation_paid_out / facility_avg_amount` |
| **Fuente** | Derivado de F01 y F22 |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Qué tan inusual es el monto de esta transacción para esta facility específica. Ratio > 5 = transacción 5x mayor al promedio de la sede. Más informativo que F03 (ratio global) porque elimina el sesgo inter-facility. |
| **Comportamiento esperado** | Normal: 0.2–3.0. Anómalo: > 5.0 o = 0. |
| **Patrón para caso insider** | Mixto — muchas txns de Pablo son $0 (ratio = 0) o moderadas. |
| **Resultado que aporta** | Anomalía de monto contextualizada por sede. |

---

## Categoría F: Features de Crédito / Flujo (F24–F27) — NUEVOS

### F24 — `is_club_credit`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Crédito/Flujo (**NUEVO**) |
| **Fórmula** | `1 si club_credit_flag = true, 0 si no` (binario) |
| **Fuente** | `payments.club_credit_flag` |
| **Tipo de dato** | Binario (0/1) |
| **Justificación** | El 5.4% del dataset (~1.99M transacciones) usa club credits. Actualmente todas las transacciones se tratan homogéneamente sin diferenciar el método de financiamiento. Las transacciones con club credit tienen un perfil de riesgo diferente: están pre-pagadas con fondos que pueden provenir de cargas de cash no verificadas. |
| **Comportamiento esperado** | Jugadores con membresía: 10-30% club credit. Managers: varía. |
| **Patrón para caso insider** | **ALTO** — La mayoría de las compras de merchandise y muchas reservas de Pablo usan prepaid/club_credit. Es el mecanismo de consumo del crédito cargado. |
| **Resultado que aporta** | Diferencia transacciones financiadas con crédito preexistente vs. pago directo. Habilita el análisis de flujo de créditos. |
| **Dato del dataset** | 1,992,639 transacciones con club_credit_flag = true en 2025. |

### F25 — `user_debit_count_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Crédito/Flujo (**NUEVO**) |
| **Fórmula** | Cantidad de transacciones con `category = 'debit'` del usuario en 30 días |
| **Fuente** | Ventana 30d sobre `payments` filtrado por `category = 'debit'` y `user_id` |
| **Tipo de dato** | Entero, ≥ 0 |
| **Justificación** | Los "debits" son cargas de cash al balance de club credits del usuario. La frecuencia de estas cargas es un indicador de patrón: un usuario que carga créditos cada 2-4 semanas (como Pablo) tiene un perfil diferente a uno que carga una vez al año. |
| **Comportamiento esperado** | Mayoría de usuarios: 0. Usuarios con créditos: 0–1 por mes. Sospechoso: > 2 por mes recurrente. |
| **Patrón para caso insider** | **ELEVADO** — Pablo: ~2.5 cargas/mes en promedio, 20 en 8 meses. |
| **Resultado que aporta** | Frecuencia de cargas de crédito. En combinación con F26 y F27, captura el patrón de ciclo cerrado. |
| **Dato del dataset** | 176,366 transacciones debit en 2025, por 68,499 usuarios distintos. Mediana: $44.39, promedio: $3,511 (altamente sesgado). |

### F26 — `user_debit_amount_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Crédito/Flujo (**NUEVO**) |
| **Fórmula** | `SUM(reservation_paid_out) WHERE category = 'debit'` del usuario en 30 días |
| **Fuente** | Ventana 30d sobre `payments` por `user_id` y `category` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Volumen de cash convertido a crédito en el período. Complementa F25 (frecuencia) con magnitud. Un usuario que carga $500+ mensualmente en cash necesita justificación operativa. |
| **Comportamiento esperado** | Mayoría: $0. Usuarios normales con créditos: $20–$200. Managers con acceso: $200–$1,000. Sospechoso: > $500/mes recurrente sin justificación. |
| **Patrón para caso insider** | **ALTO** — Pablo carga $400–$700 mensuales, totalizando ~$5,487 en 8 meses. |
| **Resultado que aporta** | Magnitud de cargas de crédito. |

### F27 — `credit_flow_ratio`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Crédito/Flujo (**NUEVO**) |
| **Fórmula** | `user_debit_amount_30d / (user_prepaid_spend_30d + 0.01)` donde `prepaid_spend_30d = SUM(reservation_paid_out) WHERE club_credit_flag = true AND category != 'debit'` |
| **Fuente** | Derivado de F26 y ventana sobre `payments` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | **Feature clave para esquemas cerrados.** Mide el ratio entre lo que entra (cargas de cash) y lo que sale (gasto con créditos). Un ratio ≈ 1.0 indica ciclo cerrado: todo lo cargado se consume. Un ratio >> 1 indica acumulación (no sospechoso). Un ratio << 1 indica que el usuario consume más de lo que carga (créditos de membresía o cortesía — normal). |
| **Comportamiento esperado** | Mayoría: 0 (no cargan créditos). Usuarios normales: < 0.5 o > 2.0. **Sospechoso: 0.8–1.2** (ciclo cerrado perfecto). |
| **Patrón para caso insider** | **≈ 1.0** — Pablo carga ~$5,487 y consume hasta dejar balance $0.00 en todas las sedes. Ciclo cerrado casi perfecto. |
| **Resultado que aporta** | Detección directa de esquemas de ciclo cerrado de créditos. Este feature es el más discriminativo para el tipo de fraude que representa el caso Pablo. No existe en el modelo original y su ausencia es la brecha más significativa. |

---

## Categoría G: Features de Rol / Staff (F28–F30) — NUEVOS

### F28 — `is_staff`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Rol/Staff (**NUEVO**) |
| **Fórmula** | `1 si user.role ∈ ('court_manager', 'court_operator', 'teacher'), 0 si no` |
| **Fuente** | `facilities_users.role` (JOIN con `payments` por `user_id` y `facility_id`) |
| **Tipo de dato** | Binario (0/1) |
| **Justificación** | El 47.6% de todos los pagos son procesados por managers (`paid_by_manager = true`). Sin este feature, el modelo no distingue entre un jugador con velocidad anómala y un manager operando normalmente. Esto genera masivos falsos positivos (todos los managers) o falsos negativos (insiders que se camuflan con operaciones legítimas). |
| **Comportamiento esperado** | ~11,509 usuarios son staff (court_manager + court_operator + teacher). |
| **Patrón para caso insider** | `is_staff = 1` — Pablo es court_manager. Este feature permite que el modelo contextualice sus demás features contra pares de su mismo rol. |
| **Resultado que aporta** | Segmentación staff vs. jugador. Reduce falsos positivos en managers y mejora detección de insiders que abusan del rol. |
| **Dato del dataset** | court_manager: 3,323 usuarios / 1,540,546 pagos. court_operator: 4,235 / 1,513,610. teacher: 3,951 / 965,186. |

### F29 — `paid_by_manager`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Rol/Staff (**NUEVO**) |
| **Fórmula** | Campo existente en tabla (binario) |
| **Fuente** | `payments.paid_by_manager` |
| **Tipo de dato** | Binario (0/1) |
| **Justificación** | Indica si un manager procesó el pago (vs. auto-servicio del usuario). El 47.6% de pagos tienen `paid_by_manager = true`. Una transacción procesada por manager tiene un perfil diferente: puede tener descuentos manuales, cortesías, ajustes que no son posibles en self-service. |
| **Comportamiento esperado** | Reservas de jugadores: mayormente false. Clínicas/lessons: mayormente true. |
| **Patrón para caso insider** | **ALTO** — Pablo procesa sus propios pagos como manager. |
| **Resultado que aporta** | Contexto de quién procesó el pago. |

### F30 — `staff_amount_zscore`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Rol/Staff (**NUEVO**) |
| **Fórmula** | `(reservation_paid_out - AVG_role_cohort) / STD_role_cohort` donde el cohort se define por `role` |
| **Fuente** | Derivado de `payments.reservation_paid_out` agrupado por `facilities_users.role` |
| **Tipo de dato** | Continuo (z-score estandarizado) |
| **Justificación** | **Feature crítico para detección de insiders.** Compara el monto de la transacción contra el promedio y desviación estándar de su cohort de rol. Un court_manager con monto inusual PARA un court_manager se detecta, mientras que F03 (amount_usd_ratio) compara contra todo el universo donde los managers siempre lucen diferentes. |
| **Comportamiento esperado** | Normal: z ∈ [-2, 2]. Anómalo: abs(z) > 3. |
| **Patrón para caso insider** | Mixto — depende de la distribución del cohort de managers. Court_managers promedian $1,406/txn, así que los montos de Pablo ($0–$700) podrían estar por debajo del promedio de su cohort. Lo interesante es que combinar este feature con los de crédito/flujo expone el patrón. |
| **Resultado que aporta** | Detección de anomalías relativas al rol. Resuelve el problema de que F03 no es informativo para staff. |
| **Dato del dataset** | Promedios por rol: court_manager $1,406, court_operator $195, teacher $43, guest $498. |

---

## Categoría H: Features de Diversidad Operacional (F31–F33) — NUEVOS

### F31 — `category_entropy_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Diversidad (**NUEVO**) |
| **Fórmula** | Entropía de Shannon: `H = -Σ p(c) × log₂(p(c))` donde `p(c) = count(category=c) / total` calculado sobre transacciones del usuario en 30 días |
| **Fuente** | Ventana 30d sobre `payments.category` por `user_id` |
| **Tipo de dato** | Continuo, ≥ 0 |
| **Justificación** | Mide la diversidad de tipos de transacción. Un jugador normal usa 1-2 categorías (reservation, merchandise). Un manager con acceso amplio usa 5+ (reservation, merchandise, debit, reversal, lesson, clinic, special program). Alta entropía indica acceso operacional diverso — legítimo para un manager, pero combinado con otros indicadores señala riesgo. |
| **Comportamiento esperado** | Jugadores: H < 1.0 (1-2 categorías). Managers normales: H ≈ 1.5–2.0. Sospechoso: H > 2.5 (muchas categorías distintas con distribución uniforme). |
| **Patrón para caso insider** | **ALTO** — Pablo usa reservation, merchandise, debit, reversal, lesson, clinic, special program = 7 categorías → H ≈ 2.3–2.8. |
| **Resultado que aporta** | Detección de diversidad operacional. Complementa F28 (is_staff) cuantificando el ALCANCE del acceso utilizado. |
| **Dato del dataset** | 1,279 categorías distintas en el dataset. Las top 10 cubren ~85% del volumen. |

### F32 — `user_reversal_count_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Diversidad (**NUEVO**) |
| **Fórmula** | `COUNT(*) WHERE category = 'reversal'` del usuario en 30 días |
| **Fuente** | Ventana 30d sobre `payments` por `user_id` y `category` |
| **Tipo de dato** | Entero, ≥ 0 |
| **Justificación** | Complementa F18 (`user_reversal_ratio_30d`). El ratio puede ser bajo si el volumen total es alto, pero el count absoluto revela la magnitud real de actividad de reversión. Un usuario con 15+ reversals en un mes merece escrutinio independientemente del ratio. |
| **Comportamiento esperado** | Jugadores: 0–1. Managers: 0–5. Sospechoso: > 10 por mes. |
| **Patrón para caso insider** | **ALTO** — Pablo: ~8 reversals/mes en promedio, con pico de 17 en febrero. |
| **Resultado que aporta** | Magnitud absoluta de actividad de reversión. Evita el enmascaramiento por alto volumen que afecta al ratio (F18). |
| **Dato del dataset** | 1,662,923 reversals en 2025. |

### F33 — `user_merchandise_ratio_30d`

| Aspecto | Detalle |
|---------|--------|
| **Categoría** | Diversidad (**NUEVO**) |
| **Fórmula** | `COUNT(category='merchandise')_30d / total_txns_30d` |
| **Fuente** | Ventana 30d sobre `payments` por `user_id` y `category` |
| **Tipo de dato** | Continuo, [0, 1] |
| **Justificación** | El consumo de merchandise con club credits es una vía común de extracción de valor en esquemas de fraude interno. Un usuario cuyo % de transacciones de merchandise es desproporcionado indica un patrón de consumo retail inusual, especialmente si combina con alto F24 (is_club_credit). |
| **Comportamiento esperado** | Jugadores: 0–0.10. Managers con acceso a bar/tienda: 0.10–0.25. Sospechoso: > 0.30 sostenido. |
| **Patrón para caso insider** | **ALTO** — Merchandise representa ~39% de las txns de Pablo (280 de ~710). La mayoría pagada con prepaid/club_credit. |
| **Resultado que aporta** | Detección de consumo retail desproporcionado. En combinación con F24 y F27, captura la vía de extracción del esquema cerrado. |
| **Dato del dataset** | 956,391 transacciones de merchandise en 2025. |

---

## Matriz de Cobertura: Features vs. Patrones de Anomalía

| Patrón de anomalía | Features que lo detectan | Cobertura |
|---------------------|--------------------------|:---------:|
| Monto atípico (alto/bajo) | F01, F02, F03, F22, F23, F30 | Original |
| Descuento excesivo | F04, F20 | Original |
| Velocidad/burst | F12, F13, F14 | Original |
| Horario atípico | F07, F08, F11 | Original |
| Diversificación geográfica | F16 | Original |
| Ratio de reversals | F18, F32 | Original + Nuevo |
| Transacciones gratuitas excesivas | F06, F21 | Original |
| **Uso de club credits** | **F24** | **Nuevo** |
| **Cargas de crédito recurrentes** | **F25, F26** | **Nuevo** |
| **Ciclo cerrado de créditos** | **F27** | **Nuevo** |
| **Contexto de rol/staff** | **F28, F29, F30** | **Nuevo** |
| **Diversidad operacional** | **F31** | **Nuevo** |
| **Consumo retail excesivo** | **F33** | **Nuevo** |

---

## Consideraciones Metodológicas

### Separación temporal

Todos los features de ventana (30d, 24h, 1h) se calculan con datos **estrictamente
anteriores** a la transacción evaluada. Nunca se incluye información futura.

### Tratamiento de features nuevos con variable proxy

Los 10 features nuevos NO utilizan la variable proxy (status de refund) como
input. No hay riesgo de circularidad adicional. La misma advertencia de F18
(`user_reversal_ratio_30d`) aplica: si incluirlo infla métricas ≥ 0.02 AUC-ROC,
se reporta el modelo sin ese feature como primario.

### Compatibilidad con modelo existente

Los 10 features nuevos se extraen de las mismas tablas (`payments`,
`facilities_users`) y usan las mismas ventanas temporales que los originales.
No requieren fuentes de datos adicionales.

### Impacto esperado

- **Reducción de falsos positivos**: F28, F29, F30 contextualizan las
  operaciones de staff, evitando marcar a todos los managers como anómalos.
- **Detección de insider fraud**: F24–F27 capturan el patrón de ciclo cerrado
  de créditos invisible al modelo original.
- **Mejora en ranking**: F31, F32, F33 agregan dimensiones de diversidad
  operacional que diferencian el insider del manager legítimo.
