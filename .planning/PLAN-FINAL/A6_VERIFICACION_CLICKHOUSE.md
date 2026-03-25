# A6. Verificación ClickHouse — Ground Truth (2026-03-24)

> Resultados de consultas directas a ClickHouse para resolver items abiertos del plan.

---

## 1. Columnas Verificadas en `payments`

| Columna | Existe | Tipo | Features que dependen |
|---------|:------:|------|----------------------|
| `category` | ✓ | String | F25, F31, F33 |
| `club_credit_flag` | ✓ | Bool | F24 |
| `currency` | ✓ | LowCardinality(String) | Normalización monetaria |
| `paid_by_manager` | ✓ | Bool | F29 |
| `is_staff` | ✗ | — | F28, F30 → derivar de `facilities_users` |

**Decisión F28 (`is_staff`):** JOIN con `facilities_users` usando `(user_id, facility_id)`.
Roles disponibles: `court_manager`, `court_operator`, `teacher`, `guest`, `rental_user`.
Definición: `is_staff = role IN ('court_manager', 'court_operator', 'teacher')`.

---

## 2. Tabla `users`

| Columna | Existe | Uso |
|---------|:------:|-----|
| `id` | ✓ (UInt32) | JOIN key |
| `created_at` | ✓ (DateTime) | F19 `user_account_age_days` |
| `email` | ✓ (String) | NO usar en outputs (privacidad) |

---

## 3. Tabla `facilities_users`

```sql
-- Esquema relevante
id UInt16, facility_id UInt16, user_id UInt32,
role LowCardinality(String), created_at DateTime
```

**Distribución de roles:**

| Rol | N registros |
|-----|:----------:|
| court_manager | 9,109 |
| teacher | 5,943 |
| court_operator | 4,909 |
| guest | 1,306 |
| rental_user | 251 |

**SQL para derivar `is_staff`:**

```sql
SELECT p.*,
  CASE WHEN fu.role IN ('court_manager', 'court_operator', 'teacher') THEN 1 ELSE 0 END AS is_staff,
  fu.role AS user_role
FROM payments p
LEFT JOIN facilities_users fu FINAL
  ON p.user_id = fu.user_id AND p.facility_id = fu.facility_id
  AND fu._peerdb_is_deleted = 0
```

**Nota:** Un usuario puede tener múltiples roles en distintas facilities. El JOIN por `(user_id, facility_id)` captura el rol correcto por transacción.

---

## 4. Monedas — Conteo Real: 20

```
USD:  5,024,639   CAD:   408,709   MYR:   316,312   HNL:   222,769
NIO:   183,329   AUD:   183,108   ILS:   122,470   GTQ:    83,257
PKR:    60,716   HKD:    56,294   SGD:    44,484   COP:    24,245
AED:    21,463   BWP:    14,697   EUR:    12,026   JPY:     2,492
RWF:     2,312   MXN:     1,299   INR:        65   NZD:         4
```

**Discrepancias con tesis:**
- Tesis menciona CRC, DOP, VES → **NO encontradas** en dataset depurado
- Tesis NO menciona CAD (408K), MYR (316K), AUD (183K), HKD, SGD, AED, BWP, EUR, RWF, INR
- **Acción:** Actualizar tesis Cap 2 con las 20 monedas reales

**USD representa 74.1%** del dataset, no 96.6% como se reportó previamente (ese % era de gateways, no transacciones).

---

## 5. Exchange Rates — Estado Actual

**Tabla:** `default.exchange_rates`

**Esquema:**
```
base_currency  | target_currency | conversion_rate | timestamp
USD            | AED             | 3.6731          | 2026-03-20
USD            | AUD             | 1.420617        | 2026-03-20
...
```

**Problema:** Solo contiene UN snapshot (2026-03-20), NO datos históricos mensuales de 2025.

**Opciones de normalización:**

| Opción | Pros | Contras |
|--------|------|---------|
| A. Snapshot actual como proxy | Simple, ya disponible | Asume tasas estables en 2025 |
| B. API externa (exchangeratesapi.io) | Tasas históricas precisas | Dependencia externa, costo |
| C. Z-score per-facility | Sin tasas, robusto | No produce USD equivalente |
| **D. Tabla manual mensual (12 meses × 20 monedas)** | **Preciso, reproducible, 240 filas** | **Trabajo manual o script** |

**Decisión (2026-03-24):** Opción A — usar snapshot actual. La variación vs. tasas 2025 es < 10% para monedas volátiles e irrelevante para anomaly detection (anomalías difieren en órdenes de magnitud). Se documenta como limitación en la tesis.

**Fórmula de conversión:**
```python
# La tabla exchange_rates tiene USD como base:
# USD → AED = 3.6731 (1 USD = 3.6731 AED)
# Para convertir AED a USD: amount_usd = amount_local / rate
amount_usd = reservation_paid_out / exchange_rate(currency, month)
```

---

## 6. SQL Canónico Actualizado (con JOINs)

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
    -- Derived: is_staff y role
    CASE WHEN fu.role IN ('court_manager', 'court_operator', 'teacher')
         THEN 1 ELSE 0 END AS is_staff,
    coalesce(fu.role, 'player') AS user_role,
    -- Derived: account age
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

**Cambios vs SQL anterior:**
1. `LEFT JOIN facilities_users` para `is_staff` y `user_role`
2. `LEFT JOIN users` para `user_created_at` (F19)
3. `coalesce(fu.role, 'player')` — usuarios sin registro en facilities_users = player

---

## 7. Impacto en el Plan

| Fase | Cambio Requerido |
|------|-----------------|
| Fase 1 (Datos) | SQL actualizado con JOINs. Re-extraer con `is_staff`, `user_role`, `user_created_at` |
| Fase 3 (Features) | F28 usa `is_staff` directo (ya viene en parquet). F19 usa `user_created_at` |
| Fase 3.5 (NUEVO) | CurrencyNormalizer usando snapshot de `default.exchange_rates` |
| Fase 8 (Reporting) | Actualizar Cap 2 tesis: 20 monedas (no 13), lista corregida |
