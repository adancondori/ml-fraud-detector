# Prompt: Etiquetador de Fraude para PlaybyPoint

## Rol

Actúa como un **Analista de Fraude Senior** especializado en plataformas de reservas deportivas y pagos digitales. Tu experiencia incluye:
- Detección de fraude en pagos CNP (Card Not Present)
- Análisis de patrones en plataformas multi-tenant (facilities/clubs)
- Economía del fraude: balance entre pérdidas por fraude vs. fricción al usuario

---

## Objetivo

Diseñar un **etiquetador rule-based** que genere pseudo-labels para entrenar un modelo de ML de detección de fraude. Este sistema:
- NO es el modelo ML final
- Debe ser explicable, auditable y determinista
- Debe generar señales a nivel transacción, usuario y facility

---

## Contexto del Negocio: PlaybyPoint

PlaybyPoint es una plataforma SaaS para gestión de instalaciones deportivas (canchas de tenis, padel, squash, etc.). Flujo típico:

1. Usuario crea cuenta (o es invitado por facility)
2. Usuario reserva cancha/clase y paga online
3. Pago se procesa (tarjeta, ACH, Terminal, crédito del club, otros)
4. Usuario asiste (o no) a la reserva
5. Si hay problema: refund, reversión, o disputa

### Tipos de fraude relevantes

| Tipo | Descripción | Señales típicas |
|------|-------------|-----------------|
| **Tarjeta robada** | Uso de tarjeta comprometida para pagar reservas | Usuario nuevo, monto alto, múltiples facilities |
| **Friendly fraud** | Usuario legítimo disputa cargo después de usar servicio | Historial normal, luego refund |
| **Abuso de refunds** | Patrón sistemático de reservar y pedir reembolso | Ratio alto de refunds, timing sospechoso |
| **Account takeover** | Cuenta comprometida usada para fraude | Cambio súbito de comportamiento |
| **Testing de tarjetas** | Probar tarjetas con montos pequeños | Montos bajos, múltiples intentos rápidos |
| Fraude por reintentos forzados | Manipulación de fallos de pago para forzar estados inconsistentes |  Múltiples retries, webhooks inconsistentes, estados processing prolongados | 
| Fraude interno (insider misuse) | Uso indebido del sistema por personal o admins | Acciones fuera de horario, overrides frecuentes, patrones manuales | 

---

## Datos Disponibles (Esquema Real)

```sql
-- Tabla: payments (ClickHouse)
SELECT
    id,                    -- ID único de transacción
    user_id,               -- ID del usuario
    facility_id,           -- ID de la instalación (tenant)
    facility_name,         -- Nombre de la instalación
    created_at,            -- Timestamp de creación
    updated_at,            -- Timestamp de actualización

    -- Pago
    payment_method,        -- 'credit_card', 'cash', 'club_credit', etc.
    card_brand,            -- 'visa', 'mastercard', 'amex', etc.
    status,                -- 'completed', 'totally_refunded', 'partially_refunded', etc.
    category,              -- Categoría del pago

    -- Montos
    reservation_paid_out,  -- Monto principal (amount)
    technology_fee,        -- Fee de la plataforma
    tax,                   -- Impuestos
    tip,                   -- Propina
    discount,              -- Descuento aplicado

    -- Metadata
    payment_source,        -- Origen del pago
    source_enum,           -- Enum de fuente
    paid,                  -- Boolean: pagado
    paid_by_manager,       -- Boolean: pagado por manager
    club_credit_flag,      -- Boolean: usó crédito del club

    -- Indicadores de fraude (labels conocidos)
    reversed_id,           -- ID de reversión (>0 si fue revertido)
    debit_refund           -- Boolean: fue un refund débito
FROM payments
```
### Datos Disponibles (Esquema Real PlaybyPoint)

**Tabla `payments` (principal de transacciones):**
- id, user_id, facility_id, facility_name, facility_group_id, facility_group_name
- created_at, updated_at
- payment_method, card_brand, status, category
- reservation_paid_out, technology_fee, tax, tip, discount,
  original_amount_paid_out, original_tax
- paid, paid_by_manager, club_credit_flag
- payment_source (código numérico), source_enum (pbp_web, pbp_app, pbp_onsite, ...)
- reversed_id, debit_refund
- reservation_id (para unir con `reservations`)
- card_connect_token, stripe_charge_id, stripe_customer_id, stripe_token_id
- otros IDs financieros (accounts_receivable_id, balance_user_id, etc.)

**Tabla `reservations`:**
- id, facility_id, facility_name, court_id
- date, time_start, time_end, hour_start, hour_end
- status, total, total_paid_out
- incident_enum (no_show, cancel_on_time, cancel_out_of_time, full_refunded, ...)
- kind_enum (reservation, clinic, lesson, rental, ...)

**Tabla `users`:**
- id, created_at, updated_at
- email, first_name, last_name
- sign_in_count, last_sign_in_at, current_sign_in_ip, last_sign_in_ip
- otros metadatos (rating, zip_code, etc.)

**Tabla `failed_payment_logs`:**
- id, facility_id, user_token_id, created_at, description

(Otros: `membership_payments`, `user_balances`, `sales_details`, etc. para enriquecer contexto.)


### Datos que NO tenemos a nivel transacción (y NO debemos asumir)

- IP address específica del pago (solo tenemos IPs de sign-in en `users`)
- Device fingerprint por transacción
- Resultados AVS/CVV
- País de la tarjeta
- Código detallado de chargeback del procesador (motivo exacto)

---

## Output Esperado del Etiquetador

Para cada transacción, generar:

```python
{
    "transaction_id": "123456",
    "label": "FRAUD" | "SUSPICIOUS" | "LEGIT",
    "risk_score": 0-100,
    "risk_band": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
    "reasons": ["refund_after_completion", "high_amount_new_user"],
    "triggered_rules": ["R001", "R015"],
    "evidence": {
        "amount": 250.00,
        "user_transaction_count_30d": 1,
        "refund_ratio_user": 0.50
    },
    "recommended_action": "ALLOW" | "REVIEW" | "HOLD" | "BLOCK",
    "confidence": "low" | "medium" | "high"
}
```

---

## Entregables Solicitados

### A) Definiciones Operativas

Define claramente qué significa cada label en términos observables:

| Label | Definición | Criterio |
|-------|------------|----------|
| **FRAUD** | Alta confianza de fraude | Reglas críticas activadas |
| **SUSPICIOUS** | Requiere revisión | Patrones anómalos |
| **LEGIT** | Bajo riesgo | Comportamiento normal |

**Unidad de análisis**: ¿Transacción? ¿Usuario? ¿Facility?

**Ventanas de tiempo recomendadas** para features de agregación.

---

### B) Mapa de Patrones

Tabla con patrones detectables con los datos disponibles:

| Pattern | Señales Disponibles | Umbral Inicial | Riesgo FP | Acción |
|---------|---------------------|----------------|-----------|--------|
| Card testing | Múltiples txns pequeñas, mismo user, < 1 hora | > 3 txns < $10 en 1h | Bajo | HOLD |
| ... | ... | ... | ... | ... |

---

### C) Reglas Heurísticas (IF/THEN)

Formato:

```yaml
- rule_id: R001
  name: high_refund_ratio_new_user
  condition: |
    user_transaction_count_90d < 5
    AND refund_ratio_user > 0.5
  score_delta: +40
  label_override: SUSPICIOUS
  explanation: "Usuario nuevo con >50% de transacciones reembolsadas"
  anti_fp_exceptions:
    - "Si facility tiene política de reembolso generosa conocida"
    - "Si monto total < $50"
```

Proporcionar **15-20 reglas** priorizadas por impacto.

---

### D) Sistema de Scoring

Scoring aditivo con:

| Band | Score | Label Default | Acción |
|------|-------|---------------|--------|
| LOW | 0-29 | LEGIT | ALLOW |
| MEDIUM | 30-59 | SUSPICIOUS | REVIEW |
| HIGH | 60-79 | SUSPICIOUS | HOLD |
| CRITICAL | 80-100 | FRAUD | BLOCK |

**Score base**: 10 (todos empiezan con algo de riesgo inherente)

Indicar cómo calibrar por facility (algunos tienen más refunds legítimos).

---

### E) Features Calculables

Top 15 features que puedes calcular con los datos disponibles:

| Feature | Definición | Cálculo | Ventana |
|---------|------------|---------|---------|
| `user_txn_count_7d` | Transacciones del usuario en 7 días | COUNT(*) WHERE user_id = X AND created_at > NOW() - 7d | 7 días |
| `user_refund_ratio_30d` | Ratio de refunds del usuario | SUM(is_refund) / COUNT(*) | 30 días |
| ... | ... | ... | ... |

---

### F) Ejemplos Concretos

Proporcionar 3 ejemplos con datos ficticios pero realistas:

**Ejemplo 1: LEGIT**
```json
{
  "user_id": 1001,
  "facility_id": 50,
  "amount": 45.00,
  "payment_method": "credit_card",
  "card_brand": "visa",
  "status": "completed",
  "user_txn_count_30d": 12,
  "user_refund_ratio_30d": 0.0
}
// Resultado: score=15, label=LEGIT, action=ALLOW
```

**Ejemplo 2: SUSPICIOUS**
(incluir datos y reglas activadas)

**Ejemplo 3: FRAUD**
(incluir datos y reglas activadas)

---

### G) Estrategia de Pseudo-Labeling

Cómo convertir las reglas en labels para ML:

1. **Labels duros vs. blandos**: ¿Usar probabilidad?
2. **Sampling balanceado**: Cómo manejar el desbalance (~10% fraude)
3. **Casos ambiguos**: Marcar con `do_not_train=True`
4. **Target leakage**: ¿`reversed_id` y `status` son target leakage?
   - Sí: No usar como features, solo como label
   - Qué hacer con `debit_refund`

---

### H) Limitaciones y Riesgos

1. **Datos faltantes**: Sin IP/device, perdemos señales de comportamiento de red
2. **Label noise**: `reversed_id` puede incluir refunds legítimos
3. **Drift**: Patrones cambian por temporada (verano vs. invierno)
4. **Gaming**: Fraudeadores pueden aprender las reglas
5. **Sesgo por facility**: Algunos clubs tienen más refunds legítimos
6. **Recomendaciones de mantenimiento**:
   - Backtesting mensual
   - Monitoreo de tasa de FP/FN
   - Versionado de reglas

---

## Restricciones Importantes

1. **Solo usa datos que existen**: No asumas IP, device, AVS/CVV
2. **Señales medibles**: Todo debe mapear a columnas reales
3. **Anti-FP**: Siempre mencionar qué NO debe marcarse
4. **Multi-tenant**: El "normal" varía por facility
5. **Temporalidad**: Considera eventos legítimos en ráfaga (torneos, clases grupales)

---

## Preguntas Clave a Responder

1. ¿Qué patrones indican fraude con **alta confianza** dado los datos disponibles?
2. ¿Qué patrones son solo **sospechosos** y requieren más contexto?
3. ¿Qué **combinaciones** de señales aumentan riesgo de forma no lineal?
4. ¿Cuáles reglas típicas generan **falsos positivos** en este dominio?
5. ¿Cómo manejar la distinción entre **refund legítimo** vs. **fraude**?
6. ¿`reversed_id > 0` es buen proxy de fraude o incluye demasiado ruido?
