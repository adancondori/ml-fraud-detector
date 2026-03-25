# Fase 0 — Outputs del Modelo

> Define qué produce el modelo de Isolation Forest (Fase 0) y en qué formato,
> de modo que los resultados sean directamente consumibles por Fase 1.
>
> Fase 0 es el alcance de la tesis: detección de anomalías a nivel transacción.

---

## 0. Pre-requisito Crítico: Normalización Multi-Moneda

El dataset contiene **20 monedas distintas** a través de 13 gateways.
Los montos en `reservation_paid_out` están en **moneda local** (no USD).
Sin normalización, features de monto se distorsionan hasta 1,250x
(ver sección "ALERTA CRÍTICA" en `02_catalogo_33_features.md`).

### Paso obligatorio en el pipeline (antes de feature engineering)

```
PASO 0: NORMALIZACIÓN MONETARIA
  ├─ Input: payments con reservation_paid_out en moneda local
  ├─ Tabla de referencia: exchange_rates (currency, month, rate_to_usd)
  ├─ Cálculo: amount_usd = reservation_paid_out / conversion_rate
  ├─ Output: payments + campo amount_usd
  └─ Alternativa: z-score per-facility (no requiere tasas de cambio)
```

| Moneda | Txns | Factor vs USD | Ejemplo: mediana local → USD |
|--------|-----:|:-------------:|------------------------------|
| USD | 4.9M | 1x | $27 → $27 |
| COP | 23K | 1,250x | 33,750 → $8 |
| PKR | 63K | 267x | 7,200 → $26 |
| JPY | 2.3K | 222x | 6,000 → $40 |
| HNL | 215K | 10x | 261 → $10 |
| NIO | 176K | 12x | 328 → $9 |

### Campos adicionales en el esquema de salida

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `currency` | VARCHAR | Moneda original de la transacción |
| `amount_usd` | FLOAT | Monto normalizado a USD |
| `exchange_rate` | FLOAT | Tasa de cambio aplicada |

### Impacto en features

- **F01, F02, F03, F15**: Usar `amount_usd` en lugar de `reservation_paid_out`
- **F22, F23**: Sin cambio (ya son per-facility = mono-moneda en 99.6%)
- **F26, F27**: Normalizar a USD para usuarios multi-facility/multi-moneda
- **F30**: Calcular z-score per-currency + per-role (no global)
- **Evaluación**: Reportar métricas segmentadas por moneda

---

## 1. Outputs Primarios (Nivel Transacción)

### 1.1 Tabla de Scores por Transacción

Cada transacción del dataset recibe un **anomaly score** continuo.

**Esquema de salida:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `payment_id` | INT | ID único de la transacción (PK) |
| `user_id` | INT | Usuario que generó la transacción |
| `facility_id` | INT | Sede donde ocurrió |
| `created_at` | DATETIME | Timestamp de la transacción |
| `category` | VARCHAR | Categoría del pago (reservation, merchandise, debit, etc.) |
| `payment_method` | VARCHAR | Método de pago (card, cash, free, prepaid, etc.) |
| `currency` | VARCHAR | Moneda original (USD, COP, PKR, etc.) |
| `amount_local` | DECIMAL | Monto en moneda local (`reservation_paid_out`) |
| `amount_usd` | DECIMAL | Monto normalizado a USD |
| `anomaly_score` | FLOAT | Score continuo de anomalía (mayor = más anómalo) |
| `anomaly_rank_pct` | FLOAT | Percentil del score (0.0–1.0, donde 1.0 = más anómalo) |
| `is_anomaly_top1pct` | BOOL | ¿Está en el top 1% de anomalías? |
| `is_anomaly_top5pct` | BOOL | ¿Está en el top 5% de anomalías? |
| `is_anomaly_top10pct` | BOOL | ¿Está en el top 10% de anomalías? |
| `proxy_label` | BOOL | ¿Es refund? (solo para evaluación, no input) |

**Volumen esperado:** SHAP muestreado: top-5% anomalias test (~125K txns) + 5K normales para contraste.

**Formato de archivo:** Parquet o CSV comprimido.

**Ejemplo de registros:**

```
payment_id | user_id | facility_id | created_at          | category    | amount  | anomaly_score | anomaly_rank_pct | top1 | top5 | proxy
18643668   | 221113  | 171         | 2025-01-02 22:02:11 | debit       | 204.84  | 0.7823        | 0.9412           | F    | T    | F
18742536   | 221113  | 171         | 2025-01-04 20:49:41 | reservation | 82.00   | 0.8156        | 0.9678           | T    | T    | T
19974071   | 221113  | 171         | 2025-02-01 20:52:47 | debit       | 375.00  | 0.7234        | 0.9201           | F    | T    | F
```

### 1.2 Feature Values por Transacción

Los 31 valores de features calculados para cada transacción (F06 y F21 eliminadas).

**Esquema de salida:**

| Campo | Tipo |
|-------|------|
| `payment_id` | INT (PK) |
| `f01_reservation_paid_out` | FLOAT |
| `f02_log_amount` | FLOAT |
| `f03_amount_usd_ratio` | FLOAT |
| `f04_discount_ratio` | FLOAT |
| `f05_has_tip` | INT (0/1) |
| ~~`f06_is_free`~~ | ~~INT (0/1)~~ | **ELIMINADA** — `payment_method='free'` excluido del universo |
| `f07_hour_sin` | FLOAT |
| `f08_hour_cos` | FLOAT |
| `f09_day_of_week` | INT |
| `f10_is_weekend` | INT (0/1) |
| `f11_is_off_hours` | INT (0/1) |
| `f12_user_txn_count_1h` | INT |
| `f13_user_txn_count_24h` | INT |
| `f14_time_since_last_txn` | FLOAT |
| `f15_user_amount_24h` | FLOAT |
| `f16_user_distinct_facilities_30d` | INT |
| `f17_user_distinct_methods` | INT |
| `f18_user_reversal_ratio_30d` | FLOAT |
| `f19_user_account_age_days` | INT |
| `f20_user_discount_ratio_30d` | FLOAT |
| ~~`f21_user_free_pct_30d`~~ | ~~FLOAT~~ | **ELIMINADA** — `payment_method='free'` excluido del universo |
| `f22_facility_avg_amount` | FLOAT |
| `f23_amount_facility_ratio` | FLOAT |
| `f24_is_club_credit` | INT (0/1) |
| `f25_user_debit_count_30d` | INT |
| `f26_user_debit_amount_30d` | FLOAT |
| `f27_credit_flow_ratio` | FLOAT |
| `f28_is_staff` | INT (0/1) |
| `f29_paid_by_manager` | INT (0/1) |
| `f30_staff_amount_zscore` | FLOAT |
| `f31_category_entropy_30d` | FLOAT |
| `f32_user_reversal_count_30d` | INT |
| `f33_user_merchandise_ratio_30d` | FLOAT |

**Volumen:** ~6.78M filas × 32 columnas (31 features + payment_id).

**Uso:** Interpretabilidad (SHAP), debugging, y como input directo para
agregaciones de Fase 1.

---

## 2. Outputs de Interpretabilidad (SHAP)

### 2.1 SHAP Values por Transacción

Para cada transacción, la contribución de cada feature al anomaly score.

**Esquema de salida:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `payment_id` | INT | PK |
| `shap_f01` … `shap_f33` | FLOAT | Contribución SHAP de cada feature (excl. F06, F21) |
| `shap_base_value` | FLOAT | Valor base del modelo |
| `top_feature_1` | VARCHAR | Feature con mayor contribución (nombre) |
| `top_feature_1_value` | FLOAT | Valor SHAP del top feature |
| `top_feature_2` | VARCHAR | Segundo feature más contribuyente |
| `top_feature_2_value` | FLOAT | Valor SHAP del segundo feature |
| `top_feature_3` | VARCHAR | Tercer feature más contribuyente |
| `top_feature_3_value` | FLOAT | Valor SHAP del tercer feature |

**Uso en Fase 1:** Los top features por transacción permiten agrupar
transacciones anómalas por TIPO de anomalía (ej: "anomalías por velocidad",
"anomalías por crédito", "anomalías por monto").

### 2.2 Feature Importance Global

Importancia relativa de cada feature en el modelo completo.

**Esquema:**

| Campo | Tipo |
|-------|------|
| `feature_name` | VARCHAR |
| `feature_number` | INT |
| `mean_abs_shap` | FLOAT |
| `rank` | INT |
| `category` | VARCHAR |

**Ejemplo esperado:**

```
feature_name               | mean_abs_shap | rank | category
f14_time_since_last_txn    | 0.0834        | 1    | Velocidad
f04_discount_ratio         | 0.0712        | 2    | Transaccional
f12_user_txn_count_1h      | 0.0698        | 3    | Velocidad
f27_credit_flow_ratio      | 0.0651        | 4    | Crédito/Flujo
f13_user_txn_count_24h     | 0.0623        | 5    | Velocidad
...
```

---

## 3. Outputs Agregados (Nivel Usuario) — Bridge a Fase 1

### 3.1 User Risk Profile

Agregación de scores por usuario para el período evaluado.

**Esquema de salida:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `user_id` | INT | PK |
| `email` | VARCHAR | Referencia |
| `role` | VARCHAR | Rol principal del usuario |
| `n_transactions` | INT | Total de transacciones en el período |
| `n_facilities` | INT | Facilities distintas |
| `avg_anomaly_score` | FLOAT | Promedio de anomaly scores |
| `max_anomaly_score` | FLOAT | Score máximo (peor transacción) |
| `std_anomaly_score` | FLOAT | Desviación estándar de scores |
| `median_anomaly_score` | FLOAT | Mediana de scores |
| `p95_anomaly_score` | FLOAT | Percentil 95 de scores |
| `n_top1pct` | INT | Transacciones en top 1% de anomalías |
| `n_top5pct` | INT | Transacciones en top 5% de anomalías |
| `anomaly_concentration` | FLOAT | `n_top5pct / n_transactions` |
| `total_amount` | DECIMAL | Monto total del período |
| `total_debits` | DECIMAL | Total cargado en créditos |
| `total_reversals` | INT | Total de reversals |
| `dominant_anomaly_type` | VARCHAR | Tipo de anomalía predominante (basado en top SHAP features) |

**Ejemplo para caso Pablo:**

```
user_id: 221113
email: pablo@padel.haus
role: court_manager
n_transactions: 710
n_facilities: 4
avg_anomaly_score: 0.62
max_anomaly_score: 0.87
std_anomaly_score: 0.18
p95_anomaly_score: 0.81
n_top1pct: 12
n_top5pct: 89
anomaly_concentration: 0.125
total_amount: $8,234.56
total_debits: $5,487.45
total_reversals: 61
dominant_anomaly_type: velocity+credit_flow
```

### 3.2 Anomaly Type Distribution por Usuario

Clasificación de las transacciones anómalas de cada usuario por tipo de anomalía.

**Esquema:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `user_id` | INT | FK |
| `anomaly_type` | VARCHAR | Tipo derivado del top SHAP feature |
| `n_anomalous_txns` | INT | Cantidad de txns anómalas de este tipo |
| `pct_of_user_anomalies` | FLOAT | % de las anomalías del usuario |
| `avg_score` | FLOAT | Score promedio de este tipo |

**Tipos posibles de anomalía (derivados de SHAP):**

| Tipo | Features dominantes | Descripción |
|------|---------------------|-------------|
| `amount` | F01, F02, F03, F23 | Monto atípico |
| `velocity` | F12, F13, F14, F15 | Velocidad excesiva |
| `discount` | F04, F20 | Descuento desproporcionado |
| `temporal` | F07, F08, F11 | Horario atípico |
| `credit_flow` | F24, F25, F26, F27 | Patrón de créditos sospechoso |
| `role_deviation` | F28, F30 | Comportamiento atípico para su rol |
| `diversity` | F16, F31, F33 | Diversidad operacional excesiva |
| `reversal` | F18, F32 | Reversals excesivos |
| `mixed` | Múltiples categorías | Sin dominante claro |

---

## 4. Métricas de Evaluación del Modelo

### 4.1 Métricas Primarias

| Métrica | Umbral mínimo | Descripción |
|---------|:-------------:|-------------|
| **AUC-ROC** | > 0.70 | Capacidad discriminativa general |
| **Average Precision** | > 6.33% | Calidad de ranking contra base rate |
| **Precision@5%** | Reportar | Concentración de proxy-positivos en top-5% |
| **Mann-Whitney U** | p < 0.05 | Significancia de diferencia de distribuciones |

### 4.2 Métricas por Segmento (nuevo)

Las mismas métricas calculadas por segmentos:

| Segmento | Justificación |
|----------|---------------|
| Por rol (manager, operator, teacher, player) | ¿El modelo discrimina igual en todos los roles? |
| Por facility | ¿Hay sesgo por sede? |
| Por categoría de pago | ¿El modelo favorece alguna categoría? |
| Con y sin F18 (reversal_ratio) | Control de inflación por correlación mecánica |
| Con 21 vs 31 features | ¿Los features nuevos mejoran la discriminación? |

### 4.3 Comparación de Modelos

| Modelo | Features | Propósito |
|--------|:--------:|-----------|
| IF-21 | F01–F23 (sin F06, F21) | Modelo base para ablacion |
| IF-31 | F01–F33 (sin F06, F21) | Modelo robusto con features nuevos |
| IF-30 | F01–F33 sin F06, F18, F21 | Control sin reversal_ratio |
| LOF-31 | F01–F33 (sin F06, F21) | Baseline comparativo |
| OC-SVM-31 | F01–F33 (sin F06, F21) | Baseline comparativo |

---

## 5. Artefactos Persistidos

### 5.1 Modelos Entrenados

| Artefacto | Formato | Contenido |
|-----------|---------|-----------|
| `model_if_31.joblib` | scikit-learn joblib | Isolation Forest entrenado (31 features) |
| `model_if_21.joblib` | scikit-learn joblib | Isolation Forest entrenado (21 features) |
| `scaler_31.joblib` | joblib | StandardScaler ajustado al train set |
| `feature_config.json` | JSON | Nombres, tipos, y orden de features |

### 5.2 Datasets de Scores

| Archivo | Contenido | Tamaño estimado |
|---------|-----------|:---------------:|
| `transaction_scores.parquet` | Scores por transacción (§1.1) | ~500 MB |
| `transaction_features.parquet` | Features por transacción (§1.2) | ~1.5 GB |
| `transaction_shap.parquet` | SHAP values por transacción (§2.1) | ~1.5 GB |
| `user_risk_profiles.parquet` | Perfiles de riesgo por usuario (§3.1) | ~50 MB |
| `user_anomaly_types.parquet` | Distribución de tipos por usuario (§3.2) | ~20 MB |
| `feature_importance.csv` | Importancia global (§2.2) | < 1 KB |
| `evaluation_metrics.json` | Métricas de evaluación (§4) | < 10 KB |

### 5.3 Registro en MLflow (OPCIONAL — no requerido por la tesis)

Todos los experimentos, hiperparámetros, métricas, y artefactos se registran
en MLflow para trazabilidad y reproducibilidad.

| Elemento | Registrado |
|----------|:----------:|
| Hiperparámetros (n_estimators, max_samples, contamination, max_features) | Sí |
| Métricas (AUC-ROC, AP, Precision@k) | Sí |
| Artefactos (modelos, scalers, configs) | Sí |
| Feature importance | Sí |
| Dataset versions (hash) | Sí |

---

## 6. Pipeline de Generación de Outputs

```
┌─────────────────────────────────────────────────────────────────┐
│                      FASE 0 — PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. EXTRACCIÓN (ClickHouse)                                     │
│     └─ payments + facilities_users + users → raw dataset        │
│                                                                 │
│  1.5 NORMALIZACIÓN MONETARIA (NUEVO — CRÍTICO)                  │
│     ├─ JOIN con exchange_rates (currency, month → rate_to_usd)  │
│     ├─ amount_usd = reservation_paid_out / conversion_rate          │
│     ├─ Validar: 20 monedas, 1,039 facilities                   │
│     └─ Alternativa: z-score per-facility (sin tasas)            │
│                                                                 │
│  2. FEATURE ENGINEERING                                         │
│     ├─ Calcular F01–F23 usando amount_usd (originales)          │
│     ├─ Calcular F24–F33 (nuevos)                                │
│     └─ Separación temporal: train / validation / test           │
│                                                                 │
│  3. ENTRENAMIENTO                                               │
│     ├─ IF-31 (Isolation Forest con 31 features)                 │
│     ├─ IF-21 (modelo base para ablacion)                        │
│     ├─ LOF-31 (baseline)                                        │
│     └─ OC-SVM-31 (baseline, subsample)                          │
│                                                                 │
│  4. SCORING                                                     │
│     ├─ Score cada transacción → transaction_scores.parquet       │
│     └─ Generar SHAP values → transaction_shap.parquet           │
│                                                                 │
│  5. AGREGACIÓN                                                  │
│     ├─ User Risk Profiles → user_risk_profiles.parquet          │
│     └─ Anomaly Type Distribution → user_anomaly_types.parquet   │
│                                                                 │
│  6. EVALUACIÓN                                                  │
│     ├─ Métricas globales y por segmento                         │
│     ├─ Comparación IF-31 vs IF-21                               │
│     └─ Bootstrap CI (1,000 iteraciones)                         │
│                                                                 │
│  7. PERSISTENCIA                                                │
│     ├─ MLflow (tracking)                                        │
│     └─ Parquet/CSV (datasets)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    FASE 1       │
                    │  (ver doc 04)   │
                    └─────────────────┘
```

---

## 7. Formato de Entrega para la Tesis

### En el documento (Capítulo 3 — Propuesta y Validación)

- Tabla de métricas comparativas (IF-21 vs IF-31 vs LOF vs OC-SVM)
- Gráficos: distribución de scores, curvas ROC, feature importance
- SHAP summary plots (global y por segmento)
- Análisis de un caso anonimizado (referencia a doc 01)

### En apéndices

- Código fuente de feature engineering (Apéndice A)
- Tablas completas de métricas por segmento (Apéndice B)
- Configuración de hiperparámetros y MLflow (Apéndice C)

### Para Fase 1 (fuera de tesis)

- Archivos parquet con scores y profiles (§5.2)
- Modelos serializados (§5.1)
- Feature config JSON para reproducir pipeline
