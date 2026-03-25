# Caso de Estudio: Pablo Muñoz-Hernández

> Análisis de auditoría completo para el período enero–agosto 2025.
> Este caso ejemplifica un patrón de **fraude ocupacional/insider** que el modelo
> de Fase 0 (Isolation Forest a nivel transacción) detecta parcialmente, y que
> motiva el diseño de la Fase 1 (detección a nivel usuario).

---

## 1. Identificación del Usuario

| Campo | Valor |
|-------|-------|
| **User ID** | 221113 |
| **Nombre** | Pablo Munoz (apellido en sistema: "DEACTIVATED") |
| **Email** | pablo@padel.haus |
| **Cuenta creada** | 2022-09-21 |
| **Último login** | 2025-08-03 |
| **Estado** | Activo |
| **Roles** | court_manager en 4 sedes |

### Sedes donde opera

| Facility ID | Nombre | Rol |
|:-----------:|--------|-----|
| 171 | Williamsburg | court_manager |
| 520 | Dumbo | court_manager |
| 836 | Nashville | court_manager |
| 1402 | Atlanta | court_manager |

### Balances de Club Credits (al cierre del período)

| Facility | Balance | Última actualización |
|----------|--------:|---------------------|
| Williamsburg (171) | $0.00 | 2025-08-04 |
| Dumbo (520) | $0.00 | 2025-08-04 |
| Nashville (836) | $0.00 | 2025-08-05 |

> Los tres balances están en **$0.00** a pesar de haber recibido ~$5,487 en
> cargas de crédito durante el período. Todo fue consumido.

---

## 2. Fuentes de Datos Consultadas

| Fuente | Tabla | Registros encontrados | Notas |
|--------|-------|:---------------------:|-------|
| Auditoría (Audited gem) | `audits` | 102 | 52 MembershipPayment + 50 Membership |
| Audit Logs (DynamoDB) | `audit_logs` | 0 | Sin registros para este usuario |
| Logs de actividad | `logs` | ~2,000+ | Reservas, pagos, clínicas |
| Pagos | `payments` | ~710 (distintos) | Todas las categorías |
| Balances de crédito | `user_balances` | 3 | Facilities 171, 520, 836 |

---

## 3. Resumen de Pagos por Mes y Categoría

| Mes | Categoría | Txns | Monto ($) | Impuesto ($) |
|-----|-----------|:----:|----------:|-----------:|
| **Ene** | Reservations | 57 | 392.00 | 17.10 |
| | Merchandise | 42 | 260.66 | 21.74 |
| | Debit (carga crédito) | 2 | 704.84 | 0.00 |
| | Reversals | 9 | 82.00 | 3.15 |
| | Lessons | 3 | 0.00 | 0.00 |
| | Clinic | 1 | 40.00 | 1.80 |
| **Feb** | Reservations | 62 | 40.00 | 1.80 |
| | Merchandise | 55 | 344.35 | 22.41 |
| | Debit (carga crédito) | 4 | 687.67 | 0.00 |
| | Reversals | 17 | 85.00 | 7.48 |
| | Lessons | 12 | 0.00 | 0.00 |
| **Mar** | Reservations | 48 | 703.87 | 39.67 |
| | Merchandise | 32 | 213.30 | 17.25 |
| | Debit | 2 | 651.14 | 0.00 |
| | Reversals | 9 | 35.21 | 1.59 |
| **Abr** | Reservations | 48 | 276.50 | 18.77 |
| | Merchandise | 21 | 280.57 | 15.11 |
| | Debit | 5 | 406.00 | 0.00 |
| | Reversals | 15 | 66.50 | 6.15 |
| **May** | Reservations | 36 | 130.00 | 5.86 |
| | Merchandise | 30 | 294.60 | 24.30 |
| | Debit | 3 | 597.68 | 0.00 |
| | Reversals | 5 | 0.00 | 0.00 |
| **Jun** | Reservations | 26 | 105.00 | 4.73 |
| | Merchandise | 45 | 255.72 | 21.80 |
| | Debit | 2 | 569.48 | 0.00 |
| **Jul** | Reservations | 16 | 55.00 | 2.07 |
| | Merchandise | 25 | 132.30 | 10.91 |
| | Debit | 1 | 704.99 | 0.00 |
| **Ago** | Reservations | 2 | 40.00 | 1.80 |
| | Merchandise | 8 | 638.81 | 8.92 |
| | Debit | 1 | 400.00 | 0.00 |

---

## 4. Movimientos de Club Credits (Débitos / Cargas)

Total de cargas en el período: **~$5,487.45**

| Fecha | Payment ID | Facility | Monto ($) | Método | Notas |
|-------|-----------|----------|----------:|--------|-------|
| 2025-01-02 22:02 | 18643668 | Williamsburg | 204.84 | cash | |
| 2025-01-02 22:02 | 18643678 | Dumbo | 500.00 | cash | |
| 2025-02-01 02:02 | 19939219 | Dumbo | 62.25 | cash | Registrado como discount |
| 2025-02-01 20:52 | 19974047 | Dumbo | 312.61 | cash | |
| 2025-02-01 20:52 | 19974058 | Dumbo | 0.06 | cash | Micro-ajuste |
| 2025-02-01 20:52 | 19974071 | Williamsburg | 375.00 | cash | |
| 2025-03-03 15:19 | 21277857 | Williamsburg | 360.53 | cash | |
| 2025-03-03 15:19 | 21277900 | Dumbo | 290.61 | cash | |
| 2025-04-01 03:43 | 22563637 | Dumbo | 56.00 | cash | |
| 2025-04-03 20:26 | 22694108 | Williamsburg | 200.00 | cash | Registrado como discount |
| 2025-04-05 21:40 | 22786507 | Williamsburg | 104.41 | cash | Registrado como discount |
| 2025-04-11 19:10 | 23053689 | Nashville | 400.00 | cash | Registrado como discount |
| 2025-04-18 14:18 | 23374099 | Dumbo | 350.00 | cash | |
| 2025-05-01 18:44 | 23976296 | Dumbo | 156.99 | cash | |
| 2025-05-01 18:44 | 23976301 | Williamsburg | 340.69 | cash | |
| 2025-05-27 17:37 | 25149350 | Dumbo | 100.00 | cash | |
| 2025-06-02 19:25 | 25443782 | Williamsburg | 270.13 | cash | |
| 2025-06-02 19:25 | 25443817 | Dumbo | 299.35 | cash | |
| 2025-07-02 01:27 | 26839875 | Dumbo | 704.99 | cash | |
| 2025-08-05 19:14 | 28511893 | Nashville | 400.00 | prepaid | Único pago con prepaid |

### Observaciones sobre los débitos

- Patrón mensual recurrente: cargas de $200–$700 cada 2–4 semanas
- Generalmente carga en pares (una por Williamsburg, otra por Dumbo)
- Todas en cash excepto la última (prepaid)
- Algunos registrados como "discount" en lugar de "amount" directo
- Balance final $0.00 en todas las sedes = consumo total

---

## 5. Reversals (Devoluciones)

61 reversals en el período. Las que involucran montos significativos:

| Fecha | Payment ID | Facility | Monto + Tax | Reversed Payment |
|-------|-----------|----------|------------:|:----------------:|
| 2025-01-04 | 18742560 | Williamsburg | $85.15 | 18742536 |
| 2025-02-20 | 20806900 | Dumbo | $92.48 | 20412866 |
| 2025-03-14 | 21784516 | Williamsburg | $36.80 | 21784491 |
| 2025-04-11 | 23053753 | Nashville | $30.04 | 23053679 |
| 2025-04-11 | 23053755 | Nashville | $42.61 | 23053721 |
| 2025-04-11 | 23053757 | Nashville | $42.61 | 23053730 |
| 2025-04-11 | 23053763 | Nashville | $42.61 | 23053736 |
| 2025-04-11 | 23053966 | Nashville | $30.04 | 23053829 |

> Las 48 reversals restantes son de monto $0 (cancelaciones de reservas sin
> cargo). El cluster de Nashville el 11 de abril sugiere una operación de
> limpieza masiva.

---

## 6. Actividad en Logs

### Resumen mensual por tipo de acción principal

| Mes | Reservas Creadas | Eliminadas (soft) | Movidas | Refunds | Clínicas (add/remove) |
|-----|:---:|:---:|:---:|:---:|:---:|
| Ene | 130 | 66 | 48 | 14 | 3 |
| Feb | 151 | 114 | 65 | 50 | 15 |
| Mar | 44 | 79 | 48 | 48 | 9 |
| Abr | 138 | 69 | 37 | 32 | 12 |
| May | 67 | 37 | 13 | 32 | — |
| Jun | 26 | 46 | 20 | 26 | 3 |
| Jul | 21 | 30 | 23 | — | 8 |

### Tipos de acciones en logs (top)

| Acción | Tipo | Conteo |
|--------|------|:------:|
| create | Reservation | 577 |
| soft_delete_with_no_incident | Reservation | 441 |
| move | Reservation | 256 |
| refund | Payment | 204 |
| removed PH Guest | Reservation | 66 |
| added Pablo Munoz | Reservation | 60 |
| resize | Reservation | 58 |
| delete | Reservation | 39 |
| added PH Guest | Reservation | 29 |
| added Pablo Munoz | ClinicLesson | 27 |

---

## 7. Auditoría de Membresías (tabla `audits`)

102 registros de auditoría para acciones realizadas por user_id 221113.
Todas sobre Membership y MembershipPayment.

### Tipos de operaciones registradas

| Operación | Ejemplos |
|-----------|----------|
| **Creación de membresías** | Membership 394600 (2025-01-08), 445185 (2025-03-14) |
| **Cambios de estado** | active → paused, active → cancelled, failed → active, paused → active, cancelled → active |
| **Ajustes de períodos** | Cambios en `current_period_end_at`, `termination_date` |
| **Procesamiento de pagos** | MembershipPayment: pending → payment_success |
| **Eliminación de registros** | destroy de MembershipPayment (ej: 540615, 952014) |
| **Reactivaciones** | Membership 340482: cancelled → active (2025-08-01) |

### Operaciones notables

1. **2025-01-08**: Crea membership 394600, cancela membership 268912 y la archiva
   en la misma sesión (request_uuid `b8b641d0...`)
2. **2025-01-11**: Destruye MembershipPayment 540615, recrea como 938181, ajusta
   fechas retroactivas, procesa pago y reactiva membership pausada — todo en
   secuencia rápida
3. **2025-03-14**: Crea membership 445185 con `acquired_at` futuro (2025-06-30),
   luego lo modifica retroactivamente a 2025-03-13

---

## 8. Tipificación del Caso

### Tipo de fraude potencial

**Fraude ocupacional / insider threat** con características de:

| Categoría ACFE | Descripción | Evidencia |
|----------------|-------------|-----------|
| **Asset misappropriation** | Uso de posición de manager para manipular créditos | Cargas de cash, consumo total, balance $0 |
| **Esquema cerrado de crédito** | Cash in → créditos → consumo → balance exhausto | Ciclo repetido mensualmente |
| **Descuentos sistemáticos** | Aplicación de descuentos/gratuidades inapropiadas | Alto % de pagos "free" y con discount = monto |
| **Abuso operacional de volumen** | Procesamiento excesivo de refunds | 204 refunds en 8 meses |

### Indicadores de alerta (red flags)

1. **Ciclo cerrado de créditos**: Carga cash → consume como prepaid → balance $0
2. **Operación multi-sede**: 4 facilities, distribuye operaciones
3. **Volumen desproporcionado**: ~710 pagos en 8 meses como manager
4. **Reversals masivas**: 61 reversals, incluyendo cluster de 5 en Nashville
5. **Manipulación de membresías**: Crea, destruye, y recrea registros de pago
6. **Ajustes retroactivos**: Modifica fechas de membresía hacia el pasado
7. **Horarios atípicos**: Cargas de crédito a las 02:00, 03:43

---

## 9. Análisis de Detectabilidad por Fase

### Fase 0 (Isolation Forest — transacción individual)

| Feature | Valor en txns de Pablo | ¿Señalaría? |
|---------|----------------------|:-----------:|
| `user_txn_count_1h` | Muy alto (lotes de 10-20) | Sí |
| `user_txn_count_24h` | Muy alto | Sí |
| `time_since_last_txn` | Muy bajo (segundos) | Sí |
| `discount_ratio` | Alto (discount ≈ monto) | Sí |
| `user_free_pct_30d` | Alto | Sí |
| `is_free` | Frecuente = 1 | Sí |
| `user_distinct_facilities_30d` | 4 | Sí |
| `user_reversal_ratio_30d` | Elevado | Sí |

**Resultado:** Señalaría ~40-60% de las transacciones individuales, pero **no
capturaría el esquema completo** (cadena debit → prepaid → reversal).

### Fase 1 (Modelo usuario — necesario)

| Capacidad requerida | ¿Disponible en Fase 0? |
|---------------------|:----------------------:|
| Score agregado por usuario | No (requiere post-proceso) |
| Comparación vs. cohort de managers | No |
| Detección de ciclo cerrado de créditos | No |
| Análisis de secuencias temporales | No |
| Red de relaciones entre transacciones | No |

---

## 10. Implicancias para la Investigación

Este caso demuestra con evidencia real que:

1. La detección a nivel transacción (Fase 0) es **necesaria pero insuficiente**
   para fraude ocupacional
2. Se requiere **agregación por usuario** con segmentación por rol
3. Los **features de flujo de créditos** (cargas vs. consumo) son críticos
   para detectar esquemas cerrados
4. La **diversidad de categorías** (category entropy) es un indicador clave
   de acceso operacional amplio
5. El **proxy de refunds** no captura el 100% del caso — muchas transacciones
   sospechosas están marcadas como "captured" (especialmente merchandise y debits)

> Este caso se puede referenciar en la tesis como ejemplo anonimizado para
> justificar las limitaciones del enfoque puramente transaccional y la
> recomendación de una Fase 1 a nivel usuario.
