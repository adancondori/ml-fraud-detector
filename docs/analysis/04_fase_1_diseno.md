# Fase 1 — Diseño de Detección a Nivel Usuario

> **Este documento NO es parte de la tesis.**
> Fase 1 es el siguiente paso natural después de la tesis (Fase 0).
> Utiliza los outputs de Fase 0 como inputs y agrega capas de detección
> que el modelo transaccional no puede cubrir.

### Dependencia Crítica: Multi-Moneda

Fase 1 hereda la normalización monetaria de Fase 0. Los scores y features
de Fase 0 ya deben estar calculados sobre `amount_usd` (normalizado).
Sin embargo, Fase 1 agrega consideraciones adicionales:

- **Comparación de cohorts por rol (§3.2 Bloque B):** Los percentiles de
  rol deben calcularse **per-currency** o sobre montos normalizados. Un
  court_manager en Colombia (COP) no es comparable con uno en EEUU (USD)
  sin normalización.
- **Reglas expertas (§4.2):** Los umbrales monetarios en las reglas (ej:
  `total_debits > $1,000` en R07) deben expresarse en USD o ajustarse
  por moneda.
- **Clustering (§4.1 Capa 3):** HDBSCAN es sensible a escala. Los features
  de monto deben estar normalizados antes del clustering.

---

## 1. Posición en la Arquitectura

```
┌──────────────────────────────────────────────────────────┐
│                  ARQUITECTURA DE DETECCIÓN               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  FASE 0 (Tesis)                                          │
│  ├─ Nivel: Transacción individual                        │
│  ├─ Modelo: Isolation Forest (33 features)               │
│  ├─ Output: anomaly_score por transacción                │
│  └─ Detecta: anomalías puntuales                         │
│       │                                                  │
│       ▼ scores + features + SHAP                         │
│                                                          │
│  FASE 1 (Este documento)                                 │
│  ├─ Nivel: Usuario / Comportamiento agregado             │
│  ├─ Modelo: Ensemble (IF + reglas + clustering)          │
│  ├─ Input: outputs de Fase 0 + features de usuario       │
│  └─ Detecta: patrones sostenidos, insider fraud,         │
│              esquemas secuenciales                        │
│       │                                                  │
│       ▼ user risk scores + alertas                       │
│                                                          │
│  FASE 2 (Futuro)                                         │
│  ├─ Nivel: Red / Grafo                                   │
│  ├─ Modelo: Graph Neural Networks / Link Analysis        │
│  └─ Detecta: colusión, redes de fraude                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Qué Inputs Consume de Fase 0

| Input de Fase 0 | Archivo | Uso en Fase 1 |
|------------------|---------|---------------|
| User Risk Profiles | `user_risk_profiles.parquet` | Base del modelo de usuario. Contiene avg/max/std de anomaly scores, concentración de anomalías, tipo dominante |
| Transaction Scores | `transaction_scores.parquet` | Para calcular features secuenciales (tendencias, clusters temporales) |
| Transaction Features | `transaction_features.parquet` | Para agregar features a nivel usuario (promedios, percentiles de F01-F33) |
| SHAP Values | `transaction_shap.parquet` | Para clasificar anomalías por tipo y construir perfiles de comportamiento |
| Feature Importance | `feature_importance.csv` | Para ponderar features en el modelo de usuario |
| Anomaly Types | `user_anomaly_types.parquet` | Distribución de tipos de anomalía por usuario |

---

## 3. Features de Nivel Usuario (Fase 1)

### 3.1 Features Heredados de Fase 0 (Agregados)

Cada feature transaccional (F01-F33) se agrega por usuario para crear un perfil:

| Feature de Fase 1 | Derivación | Descripción |
|--------------------|------------|-------------|
| `f0_avg_score` | `mean(anomaly_score)` | Tendencia general de anomalía |
| `f0_max_score` | `max(anomaly_score)` | Peor transacción individual |
| `f0_std_score` | `std(anomaly_score)` | Variabilidad de comportamiento |
| `f0_p95_score` | `percentile(anomaly_score, 95)` | Cola superior persistente |
| `f0_anomaly_concentration` | `n_top5pct / n_total` | Densidad de anomalías |
| `f0_dominant_type` | Moda de top SHAP features | Tipo de anomalía predominante |
| `f0_type_entropy` | Entropía de distribución de tipos | Diversidad de anomalías |
| `f0_avg_f04` | `mean(f04_discount_ratio)` | Promedio de discount ratio |
| `f0_avg_f27` | `mean(f27_credit_flow_ratio)` | Promedio de credit flow ratio |
| `f0_max_f12` | `max(f12_user_txn_count_1h)` | Pico de velocidad horaria |
| `f0_avg_f31` | `mean(f31_category_entropy_30d)` | Diversidad operacional promedio |

### 3.2 Features Exclusivos de Fase 1

Features que solo se pueden calcular a nivel usuario y no existen en Fase 0:

#### Bloque A: Perfil Comportamental

| Feature | Cálculo | Justificación |
|---------|---------|---------------|
| `score_trend_slope` | Pendiente de regresión lineal del anomaly_score vs. tiempo | ¿El comportamiento del usuario se vuelve más anómalo con el tiempo? Un slope positivo indica escalación. |
| `score_volatility` | Coeficiente de variación del anomaly_score | Comportamiento errático vs. consistente. Insiders consistentes son más peligrosos que errores esporádicos. |
| `anomaly_burst_count` | Número de clusters temporales con ≥3 txns en top-5% dentro de 24h | Bursts de anomalías concentradas en tiempo. |
| `days_active_ratio` | Días con actividad / días del período | ¿Qué tan frecuentemente opera? Un manager que opera 7/7 días es diferente de uno que opera 3/7. |
| `weekend_activity_ratio` | Txns en weekend / total | Patrón semanal del usuario. |

#### Bloque B: Comparación con Cohort de Rol

| Feature | Cálculo | Justificación |
|---------|---------|---------------|
| `role_score_percentile` | Percentil del avg_anomaly_score del usuario dentro de su cohort de rol | ¿Qué tan anómalo es ESTE manager comparado con OTROS managers? Pablo en el P95 de managers es más significativo que Pablo en el P80 de todos los usuarios. |
| `role_volume_percentile` | Percentil del volumen de txns dentro de su cohort | ¿Opera más o menos que sus pares? |
| `role_debit_percentile` | Percentil del total de débitos dentro de su cohort | ¿Carga más créditos que sus pares? |
| `role_reversal_percentile` | Percentil de reversals dentro de su cohort | ¿Revierte más que sus pares? |
| `role_deviation_composite` | Promedio ponderado de los 4 percentiles anteriores | Score compuesto de desviación del rol. |

#### Bloque C: Patrones Secuenciales

| Feature | Cálculo | Justificación |
|---------|---------|---------------|
| `debit_then_spend_sequences` | Count de secuencias: debit → ≥3 prepaid txns dentro de 72h | Patrón de "cargar y gastar rápido" — indicador de ciclo cerrado. |
| `spend_then_reversal_sequences` | Count de secuencias: prepaid txn → reversal de esa txn dentro de 7d | Patrón de "gastar y revertir" — extracción de créditos. |
| `credit_exhaustion_events` | Count de veces que el balance del usuario llega a $0 después de estar > $100 | ¿Cuántas veces agota completamente sus créditos? Recurrencia indica patrón sistemático. |
| `avg_days_to_exhaust_credit` | Promedio de días desde debit hasta balance $0 | ¿Qué tan rápido consume los créditos cargados? Rápido = sospechoso. |
| `monthly_debit_regularity` | Coeficiente de variación del monto de débitos mensuales | ¿Los montos de carga son regulares (como un salario) o erráticos? Regularidad alta + agotamiento rápido = patrón. |

#### Bloque D: Análisis de Red (preparación para Fase 2)

| Feature | Cálculo | Justificación |
|---------|---------|---------------|
| `unique_users_affected` | Count de usuarios distintos cuyas membresías/reservas fueron modificadas por este usuario | ¿A cuántos usuarios impacta? Managers legítimos afectan muchos; insiders pueden focalizarse en pocos (o en sí mismos). |
| `self_transaction_ratio` | Txns donde user_id = beneficiario / total | ¿Cuántas transacciones benefician al propio usuario? Alto ratio = auto-beneficio. |
| `facility_concentration` | Índice Herfindahl de distribución de txns por facility | ¿Concentra operaciones en una sede (HHI alto) o las distribuye (HHI bajo)? Distribución puede indicar evasión de controles. |

---

## 4. Modelo de Detección de Fase 1

### 4.1 Enfoque: Ensemble de 3 Capas

```
┌─────────────────────────────────────────────────────┐
│              MODELO FASE 1 — ENSEMBLE               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Capa 1: ISOLATION FOREST (a nivel usuario)         │
│  ├─ Input: features de §3.1 + §3.2                  │
│  ├─ Output: user_anomaly_score_unsupervised         │
│  └─ Detecta: outliers multidimensionales            │
│                                                     │
│  Capa 2: REGLAS EXPERTAS                            │
│  ├─ Input: features clave con umbrales definidos    │
│  ├─ Output: rule_flags (binario por regla)          │
│  └─ Detecta: patrones conocidos (caso Pablo, etc.)  │
│                                                     │
│  Capa 3: CLUSTERING (HDBSCAN)                      │
│  ├─ Input: features de comportamiento               │
│  ├─ Output: cluster_id, cluster_outlier_score       │
│  └─ Detecta: grupos de comportamiento anómalo       │
│                                                     │
│  COMBINACIÓN                                        │
│  ├─ user_risk_score = weighted_avg(3 capas)         │
│  ├─ risk_category = High / Medium / Low             │
│  └─ alert_reasons = [lista de motivos]              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 4.2 Capa 2: Reglas Expertas (basadas en caso Pablo)

| Regla | Condición | Severidad | Tipo de fraude |
|-------|-----------|:---------:|----------------|
| R01 | `credit_flow_ratio ∈ [0.8, 1.2]` AND `user_debit_count > 3` en 90d | Alta | Ciclo cerrado de créditos |
| R02 | `credit_exhaustion_events ≥ 3` en 6 meses | Alta | Agotamiento recurrente |
| R03 | `role = 'court_manager'` AND `role_score_percentile > 0.90` | Alta | Insider con score extremo |
| R04 | `self_transaction_ratio > 0.30` AND `is_staff = true` | Media | Auto-beneficio excesivo |
| R05 | `category_entropy_30d > 2.5` AND `user_reversal_count_30d > 10` | Media | Diversidad + reversals |
| R06 | `debit_then_spend_sequences > 5` en 90d | Alta | Patrón debit-spend recurrente |
| R07 | `avg_days_to_exhaust_credit < 14` AND `total_debits > $1,000` | Alta | Consumo rápido de crédito cargado |
| R08 | `n_facilities ≥ 4` AND `is_staff = true` | Media | Distribución multi-sede |
| R09 | `f0_avg_f04 > 0.50` AND `is_staff = true` | Media | Descuento sistemático por staff |
| R10 | `anomaly_burst_count ≥ 3` en 30d | Media | Bursts anómalos recurrentes |

### 4.3 Calibración de Umbrales

Los umbrales de las reglas se calibran con:
1. **Distribución empírica** del cohort de managers (percentiles P90, P95, P99)
2. **Casos conocidos** (Pablo + otros casos documentados por la empresa)
3. **Feedback loop**: Ajuste basado en investigaciones reales (falsos positivos confirmados → suben umbral, falsos negativos → bajan umbral)

---

## 5. Output de Fase 1

### 5.1 User Risk Dashboard

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `user_id` | INT | PK |
| `email` | VARCHAR | Referencia |
| `role` | VARCHAR | Rol del usuario |
| `risk_score` | FLOAT [0, 1] | Score combinado de riesgo (0 = normal, 1 = máximo riesgo) |
| `risk_category` | ENUM | `critical` / `high` / `medium` / `low` / `normal` |
| `alert_reasons` | JSON | Lista de motivos (ej: `["R01: ciclo cerrado", "R06: debit-spend x7"]`) |
| `n_rules_triggered` | INT | Cantidad de reglas activadas |
| `phase0_avg_score` | FLOAT | Score promedio de Fase 0 |
| `role_percentile` | FLOAT | Percentil dentro de su cohort de rol |
| `dominant_pattern` | VARCHAR | Patrón predominante (ej: "credit_cycle", "discount_abuse") |
| `recommended_action` | VARCHAR | `investigate` / `monitor` / `no_action` |
| `last_anomalous_txn` | DATETIME | Fecha de la última transacción anómala |
| `investigation_priority` | INT | 1 = más urgente |

### 5.2 Alert Feed

Para integración con sistemas de monitoreo (ej: Slack, email, dashboard interno).

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `alert_id` | UUID | Identificador único |
| `created_at` | DATETIME | Timestamp de generación |
| `user_id` | INT | Usuario en cuestión |
| `severity` | ENUM | `critical` / `high` / `medium` |
| `pattern_type` | VARCHAR | Tipo de patrón detectado |
| `description` | TEXT | Descripción legible del hallazgo |
| `evidence` | JSON | Datos de soporte (txn IDs, montos, fechas) |
| `status` | ENUM | `new` / `investigating` / `confirmed` / `dismissed` |

### 5.3 Ejemplo de Output para Caso Pablo

```json
{
  "user_id": 221113,
  "email": "pablo@padel.haus",
  "role": "court_manager",
  "risk_score": 0.91,
  "risk_category": "critical",
  "alert_reasons": [
    "R01: Credit flow ratio = 0.98 (ciclo cerrado) con 20 debits en 8 meses",
    "R02: 3 eventos de agotamiento de crédito (balance → $0)",
    "R03: Score P97 dentro de cohort court_manager",
    "R06: 12 secuencias debit→spend detectadas",
    "R07: Promedio 18 días para agotar crédito cargado",
    "R08: Opera en 4 facilities",
    "R09: Discount ratio promedio 0.62 (staff)"
  ],
  "n_rules_triggered": 7,
  "phase0_avg_score": 0.62,
  "role_percentile": 0.97,
  "dominant_pattern": "credit_cycle + discount_abuse",
  "recommended_action": "investigate",
  "investigation_priority": 1,
  "evidence": {
    "total_debits_loaded": 5487.45,
    "current_balance_all_facilities": 0.00,
    "total_reversals": 61,
    "total_merchandise_prepaid": 2434.30,
    "facilities": ["Williamsburg", "Dumbo", "Nashville", "Atlanta"],
    "top_anomalous_transactions": [18742536, 19974071, 26839875]
  }
}
```

---

## 6. Cobertura Comparativa: Fase 0 vs Fase 1

### Matriz de Detección por Tipo de Anomalía

| Tipo de anomalía | Fase 0 | Fase 1 | Ejemplo |
|------------------|:------:|:------:|---------|
| Transacción de monto extremo | **Sí** | Sí | Pago de $50,000 |
| Burst de velocidad (individual) | **Sí** | Sí | 20 txns en 10 min |
| Descuento excesivo (individual) | **Sí** | Sí | Discount = 100% del monto |
| Horario atípico (individual) | **Sí** | Sí | Transacción a las 3 AM |
| Ciclo cerrado de créditos | Parcial | **Sí** | Cash → credit → spend → $0 |
| Insider con scores persistentes | No | **Sí** | Manager P97 sostenido |
| Patrón secuencial debit→spend | No | **Sí** | Carga → gasta en 72h |
| Agotamiento recurrente de crédito | No | **Sí** | Balance → $0 tres veces |
| Desviación de cohort de rol | No | **Sí** | Manager peor que 97% de pares |
| Auto-beneficio excesivo | No | **Sí** | >30% txns propias |
| Diversidad operacional extrema | Parcial | **Sí** | 7+ categorías + reversals |
| Escalación temporal | No | **Sí** | Score creciente mes a mes |
| Distribución multi-sede evasiva | Parcial | **Sí** | 4+ facilities con HHI bajo |
| Colusión entre usuarios | No | No | → Fase 2 (grafos) |
| Red de fraude organizado | No | No | → Fase 2 (grafos) |

### Resumen de Cobertura

| Métrica | Fase 0 | Fase 0 + 1 |
|---------|:------:|:----------:|
| Anomalías puntuales | 100% | 100% |
| Fraude transaccional simple | ~70% | ~90% |
| Fraude insider/ocupacional | ~30% | ~85% |
| Esquemas secuenciales | ~10% | ~75% |
| Fraude en red/colusión | ~5% | ~15% |

---

## 7. Implementación Técnica

### 7.1 Stack Tecnológico

| Componente | Tecnología | Justificación |
|------------|-----------|---------------|
| Feature engineering | Python (pandas, polars) | Consistencia con Fase 0 |
| Modelo IF usuario | scikit-learn | Consistencia con Fase 0 |
| Reglas expertas | Python (custom engine) | Flexibilidad y auditabilidad |
| Clustering | hdbscan | No requiere k predefinido, maneja ruido |
| Pipeline | Airflow o Prefect | Orquestación de ETL + scoring |
| Storage | ClickHouse | Ya existe la infraestructura |
| Dashboard | Metabase o Grafana | Visualización de alertas |
| Alertas | Slack webhooks | Integración con operaciones |

### 7.2 Frecuencia de Ejecución

| Componente | Frecuencia | Justificación |
|------------|-----------|---------------|
| Fase 0 scoring | Diario (batch nocturno) | Nuevas transacciones del día |
| Fase 1 agregación | Semanal | Los patrones de usuario cambian lento |
| Reglas expertas | Diario (sobre datos agregados semanales) | Detección oportuna |
| Recalibración de modelos | Trimestral | Drift de distribución |
| Revisión de umbrales | Mensual (con feedback) | Ajuste basado en investigaciones |

### 7.3 Pipeline de Ejecución

```
DIARIO (batch nocturno):
  1. Extraer nuevas transacciones de ClickHouse
  2. Calcular 33 features para nuevas txns
  3. Scoring con modelo Fase 0 → transaction_scores
  4. Generar SHAP para txns con score > P90

SEMANAL:
  5. Agregar scores → user_risk_profiles
  6. Calcular features Fase 1 (§3.2)
  7. Scoring con modelo Fase 1 (IF usuario)
  8. Evaluar reglas expertas (§4.2)
  9. Clustering (HDBSCAN)
  10. Combinar scores → user_risk_dashboard
  11. Generar alertas → Alert Feed
  12. Notificar por Slack/email si severity ≥ high
```

---

## 8. Validación de Fase 1

### 8.1 Desafío: Sin Ground Truth

Al igual que Fase 0, no existe un dataset etiquetado de fraude.
Estrategias de validación:

| Estrategia | Descripción |
|------------|-------------|
| **Casos conocidos** | Validar que el modelo detecta casos ya investigados (Pablo, otros) |
| **Expert review** | Presentar top-50 usuarios de mayor riesgo a operaciones para validación manual |
| **Backtesting** | Ejecutar modelo sobre datos históricos y verificar si los usuarios investigados retroactivamente habrían sido señalados |
| **A/B temporal** | Comparar tasa de fraude confirmado antes/después de implementar el sistema |

### 8.2 Métricas Operativas

| Métrica | Definición | Meta |
|---------|------------|:----:|
| **Precision@50** | De los top-50 alertados, ¿cuántos son confirmados? | > 40% |
| **Time to detect** | Días desde primera transacción sospechosa hasta alerta | < 30d |
| **False positive rate** | % de alertas descartadas como no sospechosas | < 60% |
| **Coverage** | % de casos investigados que habrían sido detectados | > 70% |
| **Investigation efficiency** | Horas de investigación por caso alertado | < 4h |

---

## 9. Roadmap de Implementación

| Etapa | Duración estimada | Entregable |
|-------|:-----------------:|------------|
| **E1**: Consumir outputs de Fase 0 | 1 semana | Pipeline de lectura de parquets |
| **E2**: Features de Fase 1 (§3.2) | 2 semanas | Feature engineering de usuario |
| **E3**: Modelo IF usuario + reglas | 2 semanas | Modelo entrenado + motor de reglas |
| **E4**: Dashboard + alertas | 2 semanas | Metabase/Grafana + Slack |
| **E5**: Validación con casos conocidos | 1 semana | Reporte de validación |
| **E6**: Ajuste de umbrales (feedback loop) | Continuo | Mejora iterativa |

**Total estimado hasta primera alerta operativa: ~8 semanas.**

---

## 10. Consideraciones de Privacidad y Compliance

| Aspecto | Tratamiento |
|---------|-------------|
| **PII** | Scores y alertas no contienen datos personales. Referencia por user_id. |
| **Acceso** | Dashboard restringido a equipo de compliance/fraud. |
| **Audit trail** | Cada alerta tiene evidence JSON trazable a transacciones. |
| **Right to explanation** | SHAP values proveen explicación por transacción (PCI DSS 4.0). |
| **Data retention** | Scores se retienen 2 años. Alertas investigadas, indefinidamente. |
| **Bias** | Monitorear que el modelo no discrimine por demografía. Revisar distribución de alertas por segmento. |
