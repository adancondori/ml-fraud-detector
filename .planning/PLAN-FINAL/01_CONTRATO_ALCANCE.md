# Fase 0. Contrato y Alcance

> Traducir la tesis a un contrato tecnico verificable para que ninguna fase posterior trabaje con supuestos ambiguos.
> **Version:** 3.0 — Verificada contra ClickHouse (2026-03-24). Ver `A6_VERIFICACION_CLICKHOUSE.md`.

---

## 1. Universo del estudio

### SQL canonico (con FINAL obligatorio + JOINs verificados)

```sql
SELECT
    p.id, p.user_id, p.facility_id, p.facility_name,
    p.created_at, p.captured_at,
    p.payment_method, p.gateway, p.source_enum, p.status,
    p.reservation_paid_out, p.discount, p.tax, p.tip,
    p.card_brand, p.currency,
    p.paid_by_manager, p.reversed_id, p.debit_refund,
    p.category, p.club_credit_flag,
    p._peerdb_version,
    -- Campos derivados via JOIN
    CASE WHEN fu.role IN ('court_manager', 'court_operator', 'teacher')
         THEN 1 ELSE 0 END AS is_staff,
    coalesce(fu.role, 'player') AS user_role,
    u.created_at AS user_created_at
FROM pbp_productionDB_optimized.payments p FINAL
LEFT JOIN pbp_productionDB_optimized.facilities_users fu FINAL
    ON p.user_id = fu.user_id AND p.facility_id = fu.facility_id
    AND fu._peerdb_is_deleted = 0
LEFT JOIN pbp_productionDB_optimized.users u FINAL
    ON p.user_id = u.id
    AND u._peerdb_is_deleted = 0
WHERE p.created_at >= %(start)s
  AND p.created_at < %(end)s
  AND p.payment_method != 'reversal'
  AND p.payment_method != 'free'
  AND p.user_id != 0
  AND p._peerdb_is_deleted = 0
ORDER BY p.created_at, p.id
```

**Columnas verificadas en ClickHouse (2026-03-24):**

| Columna | Tabla fuente | Tipo | Features |
|---------|-------------|------|----------|
| `category` | payments | String | F25, F31, F33 |
| `club_credit_flag` | payments | Bool | F24 |
| `currency` | payments | LowCardinality(String) | Normalizacion |
| `paid_by_manager` | payments | Bool | F29 |
| `is_staff` | **DERIVADO** (facilities_users.role) | Bool | F28, F30 |
| `user_role` | **DERIVADO** (facilities_users.role) | String | Segmentacion |
| `user_created_at` | users.created_at | DateTime | F19 |

**Roles en `facilities_users`:** court_manager (9,109), teacher (5,943), court_operator (4,909), guest (1,306), rental_user (251). Usuarios sin registro = 'player' (default).

**Monedas verificadas: 20 distintas** (no 21). USD = 74.1% (5M txns). Ver detalle en `A6_VERIFICACION_CLICKHOUSE.md`.

**Tabla fuente:** `SharedReplacingMergeTree`, `ORDER BY (facility_id, created_at, id)`, sin `PARTITION BY`.
`FINAL` es obligatorio porque la tabla contiene multiples versiones de filas; sin el, los conteos quedan inflados.

### Exclusiones

| Filtro | Razon |
|--------|-------|
| `payment_method != 'reversal'` | Movimientos contables internos, no transacciones de cliente |
| `payment_method != 'free'` | Reservaciones sin cobro |
| `user_id != 0` | Transacciones de sistema/anonimas sin perfil conductual |
| `_peerdb_is_deleted = 0` | Filas soft-deleted por replicacion PeerDB |

---

## 2. Proxy de anomalia

El estudio es **no supervisado**: los modelos entrenan sin etiquetas. El proxy se usa **exclusivamente para evaluacion** (nunca para entrenamiento).

### Proxy estricto (principal)

```
status IN ('totally_refunded', 'refunded_to_credit')
```

- N estimado: **~429,418** (6.33% del universo)
- Uso: evaluacion principal de HE1-HE4

### Proxy amplio (sensibilidad)

```
status IN ('totally_refunded', 'refunded_to_credit', 'partially_refunded')
```

- N estimado: **~512,582** (7.55% del universo)
- Uso: analisis de sensibilidad (verificar que conclusiones no dependen de la definicion del proxy)

> **Proxy != fraude.** Nunca afirmar que las anomalias detectadas equivalen a fraude. El proxy es una aproximacion basada en el estado transaccional. Usar lenguaje de asociacion, no causal.

---

## 3. Temporal splits

| Split | Inicio (inclusive) | Fin (exclusive) | N estimado | Proporcion | Proxy estricto |
|-------|--------------------|-----------------|------------|------------|----------------|
| Train | 2025-01-01 | 2025-07-01 | ~3,137,086 | 46.24% | 6.45% |
| Validation | 2025-07-01 | 2025-09-01 | ~1,130,118 | 16.66% | 6.07% |
| Test | 2025-09-01 | 2026-01-01 | ~2,517,492 | 37.10% | 6.30% |
| **Total** | | | **~6,784,696** | 100% | **6.33%** |

Ademas:
- **Warm history:** 2024-12-01 a 2024-12-31 (para inicializar ventanas rolling de 30 dias en enero 2025)

Tolerancia de verificacion: los conteos reales no deben divergir mas de **+-1%** de los valores estimados.

La estabilidad de la tasa de proxy entre conjuntos (~6.07%–6.45%) indica que el fenomeno es estructural, no estacional.

---

## 4. Normalizacion monetaria (PREREQUISITO CRITICO)

El dataset contiene **20 monedas distintas** (verificado 2026-03-24). El campo `reservation_paid_out` registra montos en **moneda local**, no USD. Sin normalizacion, features de monto se distorsionan hasta 1,250x (COP).

### Monedas por volumen (verificado en ClickHouse)

| Moneda | Txns | % | Moneda | Txns | % |
|--------|-----:|--:|--------|-----:|--:|
| USD | 5,024,639 | 74.1% | SGD | 44,484 | 0.7% |
| CAD | 408,709 | 6.0% | COP | 24,245 | 0.4% |
| MYR | 316,312 | 4.7% | AED | 21,463 | 0.3% |
| HNL | 222,769 | 3.3% | BWP | 14,697 | 0.2% |
| NIO | 183,329 | 2.7% | EUR | 12,026 | 0.2% |
| AUD | 183,108 | 2.7% | JPY | 2,492 | <0.1% |
| ILS | 122,470 | 1.8% | RWF | 2,312 | <0.1% |
| GTQ | 83,257 | 1.2% | MXN | 1,299 | <0.1% |
| PKR | 60,716 | 0.9% | INR | 65 | <0.1% |
| HKD | 56,294 | 0.8% | NZD | 4 | <0.1% |

**Nota:** CRC, DOP, VES mencionados en versiones anteriores de la tesis NO estan presentes en el dataset depurado 2025. Actualizar Cap 2 con esta lista.

### Paso obligatorio (antes de feature engineering)

```python
# FORMULA CANONICA (unica fuente de verdad)
# ClickHouse: base_currency='USD', target_currency='COP', conversion_rate=3679
# Significa: 1 USD = 3679 COP
# Internamente: rate_to_usd = 1 / conversion_rate = 0.000272
# Aplicacion:   amount_usd = reservation_paid_out * rate_to_usd
# Equivalente:  amount_usd = reservation_paid_out / conversion_rate
# Ejemplo: 31500 COP / 3679 = $8.56 USD ✓
```

**Fuente de tasas:** `default.exchange_rates` en ClickHouse (snapshot 2026-03-20, 172 monedas).

**Decisión (2026-03-24):** Usar snapshot actual como aproximacion. Justificacion:
- Variacion tasas 2025 vs snapshot: < 10% volatiles (COP, PKR), < 3% estables (CAD, AUD, EUR)
- Irrelevante para anomaly detection: anomalias difieren en ordenes de magnitud
- Documentar como limitacion metodologica en tesis

**Implementacion:** `CurrencyNormalizer` en `src/fraud_detector/utils/currency.py`:
1. Lee tasas: `SELECT target_currency, conversion_rate FROM default.exchange_rates WHERE base_currency = 'USD'`
2. Convierte internamente: `rate_to_usd = 1 / conversion_rate`
3. USD (74.1%) → `amount_usd = reservation_paid_out` (rate_to_usd = 1.0)
4. Non-USD (25.9%) → `amount_usd = reservation_paid_out * rate_to_usd`

### Impacto en features

| Feature | Usar `amount_usd` en lugar de `reservation_paid_out` |
|---------|------------------------------------------------------|
| F01 `reservation_paid_out` | Si — monto normalizado a USD |
| F02 `log_amount` | Si — log(amount_usd + 1) |
| F03 `amount_usd_ratio` | Si — ratio sobre promedio global en USD |
| F15 `user_amount_24h` | Si — suma acumulada en USD |
| F22, F23 | Sin cambio (per-facility = mono-moneda en 99.6%) |
| F26 `user_debit_amount_30d` | Si — normalizar para usuarios multi-moneda |
| F30 `staff_amount_zscore` | Calcular z-score per-currency + per-role |

---

## 5. Catalogo oficial de 31 features (8 grupos, F06 y F21 eliminadas)

> Alineado con `03_propuesta_validacion.tex`, Tabla `tab:catalogo-33-features`.
> **Nota:** Se conserva la numeracion original F01-F33 para evitar confusion con referencias existentes. F06 y F21 estan marcadas como ELIMINADA.
>
> F06 y F21 eliminados: el universo excluye payment_method='free', por lo tanto estas features serian constantes (=0) y no aportarian varianza al modelo.

### Grupo A: Transaccionales (5 activas, 1 eliminada)

| # | Feature | Descripcion |
|---|---------|-------------|
| 1 | `reservation_paid_out` | Monto pagado en USD (normalizado) |
| 2 | `log_amount` | `log(amount_usd + 1)` — compresion de cola derecha |
| 3 | `amount_usd_ratio` | `amount_usd / global_avg_amount` (fitted en train) |
| 4 | `discount_ratio` | `discount / (amount_usd + 0.01)` |
| 5 | `has_tip` | Indicador binario: `tip > 0` |
| ~~6~~ | ~~`is_free`~~ | **ELIMINADA** — `payment_method='free'` excluido del universo; seria constante (=0) |

### Grupo B: Temporales (5)

| # | Feature | Descripcion |
|---|---------|-------------|
| 7 | `hour_sin` | `sin(2*pi*hour/24)` — codificacion ciclica |
| 8 | `hour_cos` | `cos(2*pi*hour/24)` — codificacion ciclica |
| 9 | `day_of_week` | Dia de la semana (1=Lun, 7=Dom) |
| 10 | `is_weekend` | Indicador binario: Sab o Dom |
| 11 | `is_off_hours` | Indicador binario: horas 23-06 |

### Grupo C: Velocidad (4)

| # | Feature | Descripcion |
|---|---------|-------------|
| 12 | `user_txn_count_1h` | Txns del usuario en ultima hora (excluye txn actual) |
| 13 | `user_txn_count_24h` | Txns del usuario en ultimas 24h (excluye txn actual) |
| 14 | `time_since_last_txn` | Segundos desde la transaccion anterior del usuario |
| 15 | `user_amount_24h` | Monto acumulado del usuario en 24h (excluye txn actual) |

### Grupo D: Comportamiento del Usuario (5 activas, 1 eliminada)

| # | Feature | Descripcion |
|---|---------|-------------|
| 16 | `user_distinct_facilities_30d` | Facilities distintas en 30 dias |
| 17 | `user_distinct_methods` | Metodos de pago distintos acumulados (expanding, excluye txn actual) |
| 18 | `user_reversal_ratio_30d` | Reversiones / Total en 30d (shift per-group) — **SENSITIVITY: correlacion mecanica con proxy** |
| 19 | `user_account_age_days` | Dias desde creacion de cuenta (`users.created_at`); fallback: primera txn en train |
| 20 | `user_discount_ratio_30d` | Sum(discount) / Sum(amount) en 30 dias (shift per-group) |
| ~~21~~ | ~~`user_free_pct_30d`~~ | **ELIMINADA** — `payment_method='free'` excluido del universo; seria constante (=0) |

### Grupo E: Contextuales (2)

| # | Feature | Descripcion |
|---|---------|-------------|
| 22 | `facility_avg_amount` | Monto promedio de la facility (fitted en train) |
| 23 | `amount_facility_ratio` | `amount / (facility_avg_amount + 1e-8)` |

### Grupo F: Credito / Flujo (4) — NUEVO

| # | Feature | Descripcion |
|---|---------|-------------|
| 24 | `is_club_credit` | `club_credit_flag = true` (binaria) — financiamiento con credito prepagado |
| 25 | `user_debit_count_30d` | Cargas de credito (`category = 'debit'`) del usuario en 30d |
| 26 | `user_debit_amount_30d` | Σ monto de cargas de credito en 30d (en USD) |
| 27 | `credit_flow_ratio` | Cargas / (gasto prepaid + 0.01) — deteccion de ciclo cerrado |

### Grupo G: Rol / Staff (3) — NUEVO

| # | Feature | Descripcion |
|---|---------|-------------|
| 28 | `is_staff` | Rol ∈ {manager, operator, teacher} (binaria) |
| 29 | `paid_by_manager` | Flag de pago por manager (binaria) — ya existe en SQL |
| 30 | `staff_amount_zscore` | `(amount - μ_rol) / σ_rol` — anomalia relativa al rol |

### Grupo H: Diversidad Operacional (3) — NUEVO

| # | Feature | Descripcion |
|---|---------|-------------|
| 31 | `category_entropy_30d` | Entropia de Shannon de categorias de pago en 30d |
| 32 | `user_reversal_count_30d` | Reversiones absolutas del usuario en 30d |
| 33 | `user_merchandise_ratio_30d` | % transacciones merchandise del usuario en 30d |

### Variantes de features

| Variante | Features | Proposito |
|----------|----------|-----------|
| **IF-31** | F01-F33 sin F06/F21 (31 features oficiales) | Modelo principal completo |
| **IF-30** | IF-31 sin F18 (30 features) | Sensibilidad: eliminar correlacion mecanica con proxy |
| **IF-21** | F01-F23 sin F06/F21 (21 features base) | Ablacion: cuantificar aporte de grupos F, G, H |

> **Feature #18 (`user_reversal_ratio_30d`):** Tiene correlacion mecanica con el proxy. Analisis de sensibilidad obligatorio. Si delta AUC >= 0.02, el modelo de 30 features se reporta como resultado principal y el de 31 como referencia.
>
> **Feature #32 (`user_reversal_count_30d`):** Tambien derivada de reversiones pero es conteo absoluto, no ratio. Monitorear correlacion pero no requiere eliminacion automatica.

---

## 6. Algoritmos y grid search

### Modelos

| Rol | Algoritmo | Features | Grid combos | Justificacion |
|-----|-----------|----------|-------------|---------------|
| **Principal** | Isolation Forest (IF) | 31 (+ variantes 30 y 21) | 240 | Tesis propone IF como metodo principal |
| **Comparador** | Local Outlier Factor (LOF) | 31 | 3 | Metodo basado en densidad local |
| **Comparador** | One-Class SVM (OC-SVM) | 31 | 6 | Metodo basado en frontera; subsample de 100K |

**Excluido:** DBSCAN — no produce ranking continuo de anomalia, incompatible con metricas de la tesis.

### Grid search de Isolation Forest (240 combinaciones)

> Alineado con `03_propuesta_validacion.tex`, Tabla `tab:grid-search-espacio`.

| Hiperparametro | Valores | N |
|----------------|---------|---|
| `n_estimators` | [100, 200, 300, 500] | 4 |
| `max_samples` | [256, 512, 1024, 2048] | 4 |
| `contamination` | [0.01, 0.03, 0.05, 0.06, 0.08] | 5 |
| `max_features` | [0.5, 0.75, 1.0] | 3 |

Total: 4 × 4 × 5 × 3 = **240 combinaciones**

> **Nota sobre contamination:** A diferencia del plan anterior que excluia contamination del grid (por no afectar el ranking en AUC-ROC), la tesis la incluye explicitamente. Si bien el ranking de `-decision_function(X)` no cambia con contamination, el offset si afecta al threshold binario, y la tesis evalua metricas basadas en threshold (Precision@k). Se mantiene en el grid para alineacion con la tesis.

**Metrica de seleccion:** AUC-ROC en validacion. **Checkpoint cada 10 combos** para resume.

### Grid search de LOF (3 combinaciones)

| Hiperparametro | Valores |
|----------------|---------|
| `n_neighbors` | [20, 50, 100] |
| `novelty` | `True` (fijo) |

### Grid search de OC-SVM (6 combinaciones)

| Hiperparametro | Valores |
|----------------|---------|
| `nu` | [0.01, 0.05, 0.10] |
| `gamma` | [scale, auto] |
| Subsample | 100K (estratificacion temporal, no aleatoria) |

**Score convention:** score alto = mas anomalo. Se usa `-decision_function(X)` para IF y OC-SVM.

---

## 7. Hipotesis especificas (HE1-HE4) con umbrales cuantificados

### HE1 — Separacion estadistica de scores

> Los scores de anomalia de IF asignan valores significativamente mas altos a transacciones proxy+ que a transacciones proxy-.

| Metrica | Umbral | Detalle |
|---------|--------|---------|
| Mann-Whitney U | p < 0.05 | Test unilateral (alternative="greater") |
| Rank-biserial r | r_rb > 0.10 | `r = 2*U/(n1*n2) - 1`; positivo cuando anomaly > normal |

**HE1 pasa si:** p < 0.05 **Y** r_rb > 0.10 (ambos despues de correccion Holm-Bonferroni).

Metricas complementarias: CLES (Common Language Effect Size), KS statistic.

### HE2 — Capacidad discriminativa

> IF logra una capacidad discriminativa superior a la asignacion aleatoria y al baseline de tasa base.

| Metrica | Umbral |
|---------|--------|
| AUC-ROC | > 0.70 |
| Average Precision (AP) | > 6.33% (tasa base del proxy estricto) |

**HE2 pasa si:** AUC > 0.70 **Y** AP > tasa base proxy estricto.

Bootstrap CI 95% (N=1000) para ambas metricas.

### HE3 — Concentracion en top-K

> Las transacciones proxy+ se concentran desproporcionadamente en los percentiles mas altos de anomalia segun IF.

| Metrica | Umbral |
|---------|--------|
| Enrichment Factor (EF) | > 1 en top-5% |

**HE3 pasa si:** EF > 1 en top-5%.

Se evalua ademas en top-1%, top-2%, top-5%, top-10% (multiples k).

EF = (proporcion de proxy+ en top-k%) / (proporcion global de proxy+).

### HE4 — Comparacion IF vs. competidores

> IF presenta desempeno discriminativo comparable o superior a LOF y OC-SVM en la mayoria de las metricas principales.

| Metricas de comparacion |
|------------------------|
| AUC-ROC |
| Average Precision (AP) |
| Precision@5% |
| Enrichment Factor@5% |

**HE4 pasa si:** IF >= competidores en **>=3 de 4** metricas.

> **Nota:** La tesis tabla `tab:resumen-hipotesis` indica "≥ 2/4" pero la tabla `tab:he4-validacion` y el cuerpo del texto usan "≥ 3/4". Este plan usa **≥ 3/4** como criterio oficial; la tabla resumen de la tesis debe corregirse.

Comparacion justa: mismo snapshot, mismas filas, mismo set de 31 features (F06 y F21 eliminadas por exclusion de free), mismo proxy, misma ventana temporal, misma orientacion del score.

### Correcciones estadisticas

- **Holm-Bonferroni** aplicada al conjunto de tests HE1-HE4 (**m = 4** comparaciones).
  - Ordenar p-values de menor a mayor: p(1) ≤ p(2) ≤ p(3) ≤ p(4)
  - Thresholds ajustados: α/(4), α/(3), α/(2), α/(1) = 0.0125, 0.0167, 0.025, 0.05
  - Nota: la correccion aplica a las 4 hipotesis, NO a los 3 modelos por separado (m=4, no m=12)
- **Estabilidad temporal:** AUC mensual en test set (Sep, Oct, Nov, Dic 2025).

---

## 7.1. Analisis adicionales (alineados con tesis Cap. 3)

### Analisis por segmentos

Metricas (AUC-ROC, AP, P@5%, EF) estratificadas por:

| Segmento | Justificacion |
|----------|---------------|
| **Por rol de usuario** (Players, Court Managers, Court Operators, Teachers) | Verificar que el modelo no discrimina exclusivamente por perfil operativo |
| **Por categoria de pago** (Reservation, Merchandise, Lesson/Clinic, Debit) | Verificar robustez entre categorias |

### Tipologia de anomalias (derivada de SHAP)

Para cada transaccion en top-5%, identificar la feature SHAP dominante y asignar uno de 9 tipos:

| Tipo | Features dominantes |
|------|---------------------|
| `amount` | F01, F02, F03, F23 |
| `velocity` | F12, F13, F14, F15 |
| `discount` | F04, F20 |
| `temporal` | F07, F08, F11 |
| `credit_flow` | F24, F25, F26, F27 |
| `role_deviation` | F28, F30 |
| `diversity` | F16, F31, F33 |
| `reversal` | F18, F32 |
| `mixed` | Multiples categorias |

### Perfil de riesgo agregado por usuario

| Metrica | Descripcion |
|---------|-------------|
| Score promedio | Promedio de anomaly scores del usuario |
| Score maximo | Score de la transaccion mas anomala |
| Percentil 95 | Robustez ante outliers |
| n_top5% | Transacciones en el top-5% global |
| Concentracion (c) | n_top5% / n_total (c > 0.10 = patron sistematico) |
| Tipo dominante | Categoria de anomalia mas frecuente |

### Decisiones tecnicas consolidadas (2026-03-24)

| Tema | Decision | Justificacion |
|------|----------|---------------|
| **Estabilidad temporal** | AUC mensual en test set (Sep-Dic); criterio: max - min < 0.05 | Drift > 5pp indica estacionalidad no capturada |
| **Multi-seed** | `random_state=42` para grid search. Si varianza > 1% entre runs, reportar con [42, 52, 62] | Reproducibilidad primero, robustez si es necesario |
| **SHAP sampling** | TreeExplainer sobre top-5% test (~125K) + 5K normales aleatorios | Full-universe inviable (2.5M × 31 features); top-5% captura anomalias relevantes |
| **OC-SVM subsample** | 100K filas con estratificacion temporal: `step = len(X) // 100_000; indices = range(0, len(X), step)[:100_000]` | Preserva distribucion temporal; no random para reproducibilidad |
| **Contamination en grid IF** | Se mantiene (240 combos) pero no afecta ranking — documentar | `score_samples()` invariante; refuerza tesis |
| **LOF novelty** | `novelty=True` obligatorio | Sin novelty, LOF no puede scorear val/test |
| **Bootstrap N** | Default: 1000 iter. `--fast`: 100 iter (dev only) | Config param `bootstrap_n` |
| **Tipo anomalia: umbral dominancia** | top_shap >= 2x segundo_shap → tipo puro; sino → `mixed` | Pragmatico; parametrizable si necesario |
| **Post-hoc anonimizacion** | Solo revelar identidad si `is_staff=True AND n_txns >= 100` | Privacidad; tesis usa pseudonimo TechSport |

---

## 8. Contratos de test (TDD red-green-refactor)

Cada clase del pipeline tiene un contrato de comportamiento que DEBE expresarse como tests **antes** de escribir la implementacion.

### Contratos por clase

| Clase | Test contract (escribir PRIMERO) | Comportamiento que valida |
|-------|----------------------------------|--------------------------|
| `DataManager` | `test_extract_returns_correct_columns` | El DataFrame extraido contiene exactamente las columnas del SQL canonico (incluyendo `category`, `club_credit_flag`) |
| `DataManager` | `test_extract_validates_row_counts` | Los conteos por split estan dentro de +-1% de los valores esperados |
| `DataManager` | `test_proxy_labels_strict` | `assign_proxy_labels(df, "strict")` marca exactamente los status `totally_refunded` y `refunded_to_credit` |
| `DataManager` | `test_proxy_labels_wide` | `assign_proxy_labels(df, "wide")` incluye ademas `partially_refunded` |
| `DataManager` | `test_downcast_preserves_large_ids` | `id` y `reversed_id` no se truncan a int32 (valores > 2^31 sobreviven) |
| `DataManager` | `test_manifest_contains_required_fields` | El JSON sidecar contiene: name, start_date, end_date, row_count, extracted_at, checksum_sha256 |
| `DataManager` | `test_atomic_write_survives_interruption` | Si el proceso muere durante la escritura, no queda un Parquet corrupto en la ruta final |
| `CurrencyNormalizer` | `test_usd_gateways_unchanged` | Txns de CardConnect/Stripe/PixelPay/AzulPay mantienen su monto original |
| `CurrencyNormalizer` | `test_non_usd_normalized` | Txns en COP/PKR/NIO se convierten correctamente a USD |
| `FeatureEngineer` | `test_first_txn_counts_zero` | La primera transaccion de cada usuario tiene velocity features == 0 |
| `FeatureEngineer` | `test_no_nans_after_transform` | Cero NaN en las 31 features oficiales despues de transform |
| `FeatureEngineer` | `test_fit_transform_produces_31_features` | El output contiene exactamente las 31 columnas de `FEATURE_NAMES` |
| `FeatureEngineer` | `test_f18_shift_within_group` | F18 primera fila de cada usuario = 0; nunca ve su propio status de reversal |
| `FeatureEngineer` | `test_f17_cumulative_first_is_zero` | F17 primera ocurrencia de usuario = 0 (no ha visto ningun metodo) |
| `FeatureEngineer` | `test_f17_state_persists_cross_split` | Estado acumulado de F17 de train se hereda en val/test (fit-only en train) |
| `FeatureEngineer` | `test_credit_flow_features` | F24-F27 calculados correctamente para usuarios con y sin debits |
| `FeatureEngineer` | `test_staff_features` | F28-F30 calculados correctamente; zscore per-role |
| `UnsupervisedPreprocessor` | `test_scaler_fitted_on_train_only` | Los parametros del scaler provienen exclusivamente del train set |
| `UnsupervisedPreprocessor` | `test_output_shape_preserved` | El numero de filas y features no cambia tras preprocesamiento |
| `ModelTrainer` | `test_score_orientation` | Scores mas altos corresponden a mayor anomalia (convencion `-score_samples`) |
| `ModelTrainer` | `test_models_train_and_score` | IF, LOF y OC-SVM entrenan y generan scores finitos |
| `ModelTrainer` | `test_lof_novelty_true` | LOF se instancia con `novelty=True` (obligatorio para scoring en val/test) |
| `ModelTrainer` | `test_ocsvm_subsample_temporal` | OC-SVM usa subsample de 100K filas con estratificacion temporal |
| `ModelTrainer` | `test_grid_search_recovers_from_failure` | Si un combo falla, se registra con AUC=NaN y continua al siguiente |
| `ModelTrainer` | `test_all_variants_trained` | IF-31, IF-30, IF-21, LOF-31, OC-SVM-31 definidos en el flujo |
| `evaluation.metrics` | `test_he1_requires_both_conditions` | HE1 solo pasa si p < 0.05 Y r_rb > 0.10 |
| `evaluation.metrics` | `test_he4_threshold_gte_3_of_4` | HE4 pasa con 3/4 metricas, falla con 2/4 |
| `evaluation.metrics` | `test_bootstrap_ci_coherent` | `lower <= mean <= upper` en bootstrap |

> **Regla:** Ningun modulo pasa a implementacion hasta que sus tests de contrato esten escritos y fallen (fase Red).

---

## 9. Fuera de alcance

| Excluido | Razon |
|----------|-------|
| Despliegue en produccion | Tesis metodologica, no operativa |
| Scoring en tiempo real | Requiere infraestructura de streaming |
| API de inferencia | Pipeline offline y reproducible |
| Algoritmos adicionales (deep learning, DBSCAN, etc.) | No requeridos por los objetivos de la tesis |
| Dashboards operativos | Fuera del alcance academico |
| Deteccion legal de fraude | El proxy no equivale a fraude |
| MLflow como dependencia central | Se elimina; reproducibilidad via manifests y seeds |

---

## 10. Matriz de trazabilidad: Objetivo - Hipotesis - Modulo - Output

| Objetivo | Hipotesis | Modulo / Clase | Output |
|----------|-----------|----------------|--------|
| OE1: Fundamentar referentes teoricos | — | `Tesis-Latex/capitulos/01_marco_teorico.tex` | Marco teorico en tesis (Cap. 1) |
| OE2: Diagnosticar el estado transaccional actual | — | `03_EDA_CAPITULO2.md`, reporting pendiente | `output/tables/cap2_*.tex`, `output/figures/cap2_*.pdf` |
| OE3: Construir pipeline y evaluar capacidad discriminativa de IF | HE1, HE2, HE3 | `CurrencyNormalizer`, `DataManager`, `FeatureEngineer`, `UnsupervisedPreprocessor`, `ModelTrainer`, `evaluation.metrics` | `data/processed/*.parquet`, `output/models/*.joblib`, `output/results.json` |
| OE4: Comparar IF contra LOF y OC-SVM | HE4 | `evaluation.metrics` | `output/results.json` (seccion he4) |
| Sensibilidad | Gate D | `evaluation.metrics` + modulo de sensibilidad pendiente | `output/results_sensitivity.json` |
| Analisis por segmentos | — | modulo pendiente | Metricas por rol y categoria en `results.json` o artefacto equivalente |
| Interpretabilidad | — | SHAP + clasificador de tipologia pendiente | `output/figures/shap_summary.pdf`, tipologia en `results.json` |
| Post-hoc | — | modulo pendiente | `output/results_posthoc.json` |

---

## 11. Entregables de la Fase 0

| Entregable | Descripcion |
|------------|-------------|
| `config/config.py` | `Settings` reescrito: splits temporales, proxy labels, grids de tuning, exchange rates |
| `src/fraud_detector/utils/currency.py` | `CurrencyNormalizer` con soporte para CSV mensual directo o snapshot ClickHouse |
| `scripts/verify_counts.py` | Script que verifica conteos contra ClickHouse |
| Este documento (`01_CONTRATO_ALCANCE.md`) | Contrato tesis-codigo congelado v2.0 |

### Cambios en config.py

**Eliminar:** `model_type`, `test_size`, `validation_size`, `mlflow_*`, `use_gpu`, `fraud_threshold`, `high_risk_threshold`, `auto_decline_threshold`, `api_host`, `api_port`, `database_url`, `use_smote`, `smote_sampling_strategy`, `use_class_weights`, `use_temporal_split`, `temporal_split_date`, `embargo_days`, `enable_drift_detection`, todo parametro supervisado.

**Agregar:** `train_start`, `train_end`, `val_end`, `test_end`, `strict_proxy_statuses`, `wide_proxy_statuses`, grids de IF/LOF/OC-SVM, `bootstrap_n`, `top_k_percents`, `shap_sample_size`, `exchange_rates` o ruta a tabla de tasas.

---

## 12. Definition of Done

- [ ] `config/config.py` reescrito y cargable sin errores con `.env` actual
- [ ] No existe ninguna referencia a `is_fraud` en el codigo fuente
- [ ] No existe ninguna referencia a parametros supervisados (`SMOTE`, `fraud_threshold`, etc.)
- [ ] `verify_counts.py` ejecutable contra ClickHouse
- [ ] Los filtros del universo (SQL canonico) estan centralizados en una unica fuente de verdad
- [ ] `CurrencyNormalizer` implementado y testeado
- [ ] La matriz de trazabilidad Objetivo-Hipotesis-Modulo-Output esta documentada
- [ ] Los umbrales de HE1-HE4 estan codificados como constantes
- [ ] README del proyecto describe correctamente la tesis actual (no supervisado, no fraude)
- [ ] Las columnas `category` y `club_credit_flag` verificadas en ClickHouse

---

## 13. Gate de salida — Fase 0

**No pasar a Fase 1 hasta que:**

1. Exista una unica fuente de verdad para filtros SQL, proxy y splits temporales (`config.py`).
2. Las columnas nuevas (`category`, `club_credit_flag`) esten verificadas en ClickHouse.
3. `CurrencyNormalizer` normalice correctamente las 20 monedas a USD.
4. El README describa correctamente la tesis actual.
5. No haya ninguna ruta critica dependiente de `is_fraud`.
6. Los archivos a eliminar esten identificados: `balancing.py`, `run_simple_rf.py`, `check_git_status.py`, `verify_setup.py`, `QUICKSTART.md`, `EXPERT_AUDIT.md`, `01_exploratory_analysis.ipynb`.
7. Las dependencias supervisadas esten marcadas para remocion: `imbalanced-learn`, `xgboost`, `lightgbm`, `mlflow`, `optuna`.
8. Este documento este revisado y aprobado como contrato congelado.
