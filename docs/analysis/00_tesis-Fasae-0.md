Análisis: ¿Puede la Tesis Detectar el Caso Pablo Muñoz-Hernández?

  El Tipo de Caso que Representa Pablo

  El caso de Pablo es un potencial fraude interno/ocupacional (insider threat) con estas características:

  ┌─────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │         Patrón          │                                Evidencia en los datos                                 │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Esquema cerrado de      │ Carga ~$5,487 en cash → convierte a club credits → consume en merchandise/reservas →  │
  │ créditos                │ balance final $0.00 en todas las sedes                                                │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Descuentos sistemáticos │ Cientos de pagos "free" y "cash" con discount = monto total (ej: discount $322.50 en  │
  │                         │ reserva "free")                                                                       │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Volumen operacional     │ 577 reservas creadas, 441 eliminadas, 204 refunds en 8 meses como court_manager       │
  │ excesivo                │                                                                                       │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Operación multi-sede    │ Actividad en 4 facilities (Williamsburg, Dumbo, Nashville, Atlanta)                   │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Abuso de privilegios de │ Gestión de membresías de otros usuarios, refunds masivos, cargas de crédito en cash   │
  │  manager                │                                                                                       │
  └─────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Qué Tipo de Fraude/Anomalía Detecta la Tesis

  La tesis detecta anomalías puntuales a nivel de transacción individual usando 31 features en 8 categorias:

  1. Transaccionales (monto, log-amount, discount_ratio, has_tip)
  2. Temporales (hora ciclica, dia de semana, fin de semana, horario nocturno)
  3. Velocidad (txn_count_1h/24h, time_since_last, amount_24h)
  4. Comportamiento del usuario (facilities distintas, metodos de pago, reversals, account_age, discount_ratio_30d)
  5. Contextuales (facility_avg_amount, amount_facility_ratio)
  6. Credito/Flujo (club_credit, debit_count/amount_30d, credit_flow_ratio)
  7. Rol/Staff (is_staff, paid_by_manager, staff_amount_zscore)
  8. Diversidad Operacional (category_entropy_30d, reversal_count_30d, merchandise_ratio_30d)

  El proxy de evaluación es: status IN ('totally_refunded', 'refunded_to_credit') — es decir, transacciones que terminan
   en reembolso.

  ---
  ¿Qué SÍ Detectaría la Tesis de las Transacciones de Pablo?

  Varias features de Pablo serían señaladas como anómalas a nivel de transacción individual:

  ┌──────────────────────────────┬────────────────────────────────────────────────────┬───────────┐
  │           Feature            │             Valor esperado para Pablo              │ ¿Anómalo? │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ user_txn_count_1h            │ MUY ALTO (crea lotes de 10-20 reservas en minutos) │    Sí     │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ user_txn_count_24h           │ MUY ALTO (cientos por día)                         │    Sí     │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ time_since_last_txn          │ MUY BAJO (segundos entre transacciones)            │    Sí     │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ discount_ratio               │ ALTO (muchos pagos donde discount ≈ monto total)   │    Sí     │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ ~~user_free_pct_30d~~ (F21)  │ ELIMINADA — excluida del universo depurado         │    N/A    │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ ~~is_free~~ (F06)            │ ELIMINADA — excluida del universo depurado         │    N/A    │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ user_distinct_facilities_30d │ 4 (multi-sede)                                     │    Sí     │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ user_reversal_ratio_30d      │ Elevado (204 refunds)                              │    Sí     │
  ├──────────────────────────────┼────────────────────────────────────────────────────┼───────────┤
  │ user_discount_ratio_30d      │ ALTO                                               │    Sí     │
  └──────────────────────────────┴────────────────────────────────────────────────────┴───────────┘

  Individualmente, algunas transacciones de Pablo recibirían anomaly scores altos. El modelo sí capturaría señales
  parciales.

  ---
  ¿Qué NO Detectaría la Tesis?

  Aquí están las brechas fundamentales:

  1. El esquema es secuencial, no puntual

  La tesis detecta anomalías de punto (transacciones individuales). Pero el patrón de Pablo es una cadena:

  cash deposit → club credit → merchandise/reserva → refund → recarga

  Ninguna transacción individual de esta cadena es necesariamente anómala por sí sola. Un depósito de $500 en cash puede
   ser legítimo. Una compra de merchandise de $6 con prepaid es normal. Un refund de una reserva es operación estándar.
  Es la secuencia la que es sospechosa.

  2. No hay segmentación por rol

  La tesis trata a todos los usuarios igual. Pero Pablo es court_manager — sus patrones de velocidad y descuentos son
  normales para un manager y anómalos para un jugador. Sin segmentación por rol, el modelo podría:
  - Tratar sus patrones como normales (si hay muchos managers con patrones similares)
  - O marcar a TODOS los managers como anómalos (falsos positivos masivos)

  3. La categoría "debit" (cargas de crédito) no es visible como anomalía

  Los débitos de cash → club credit no producen refunds, así que el proxy no los marca. Son transacciones "captured" que
   pasan inadvertidas para el modelo.

  4. El merchandise con prepaid es invisible

  Las compras de merchandise usando club credits (prepaid) no son refunds. Son transacciones normales de consumo. El
  modelo no las cuestiona.

  5. La propia tesis reconoce esto

  En la sección de limitaciones, la tesis documenta exactamente estos patrones:
  ▎ "Fraude interno documentado: descuentos sistemáticos no autorizados aplicados por personal operativo"
  ▎ "Esquemas cerrados de crédito: depósitos de efectivo sin recepción real, consumidos como crédito prepagado"
  ▎ "Estos evaden el proxy basado en reembolsos (transacciones marcadas como 'captured')"

  ---
  La Tesis como "Fase 0" — Análisis de Viabilidad

  Sí, la tesis funciona como Fase 0. Aquí está por qué y qué se necesita después:

  Fase 0 (La Tesis Actual)

  - Establece la infraestructura de scoring — cada transacción recibe un anomaly score
  - Identifica señales parciales en transacciones individuales de Pablo
  - Crea el framework de evaluación (AUC-ROC, Average Precision)
  - Define las 31 features (8 categorias) que capturan comportamiento transaccional
  - Valida que Isolation Forest tiene capacidad discriminativa sobre datos reales

  Fase 1 (Capa de Agregación por Usuario) — Lo que falta para Pablo

  ┌─────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │       Componente        │                                      Descripción                                      │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Score agregado por      │ Promedio/percentil del anomaly score de todas las transacciones de un usuario en      │
  │ usuario                 │ ventana temporal                                                                      │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Perfil de rol           │ Comparar managers vs. managers, no managers vs. jugadores                             │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Patrón de créditos      │ Features: total_debits_30d, credit_turnover_ratio, cash_deposit_frequency             │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Análisis de flujo       │ cash_in vs credits_consumed vs refunds_generated                                      │
  ├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Red de transacciones    │ ¿Un manager opera siempre en las mismas facilities? ¿Con los mismos usuarios?         │
  └─────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

  Fase 2 (Detección de Patrones Secuenciales)

  ┌─────────────────────┬───────────────────────────────────────────────────────────┐
  │     Componente      │                        Descripción                        │
  ├─────────────────────┼───────────────────────────────────────────────────────────┤
  │ Sequence mining     │ Detectar cadenas: debit → prepaid_purchase → refund       │
  ├─────────────────────┼───────────────────────────────────────────────────────────┤
  │ Graph analysis      │ Relaciones entre usuarios, facilities, y flujos de dinero │
  ├─────────────────────┼───────────────────────────────────────────────────────────┤
  │ Temporal clustering │ Agrupar transacciones relacionadas en sesiones            │
  └─────────────────────┴───────────────────────────────────────────────────────────┘

  ---
  Resumen Ejecutivo

  ┌───────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────┐
  │           Pregunta            │                                    Respuesta                                    │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ ¿La tesis detecta el caso     │ Parcialmente — señalaría algunas transacciones individuales, pero no el esquema │
  │ Pablo?                        │  completo                                                                       │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ ¿Qué tipo de fraude analiza   │ Anomalías puntuales a nivel transacción, evaluadas contra proxy de refunds      │
  │ la tesis?                     │                                                                                 │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ ¿Qué tipo de fraude es Pablo? │ Fraude ocupacional/insider con esquema cerrado de créditos (secuencial,         │
  │                               │ multi-transacción)                                                              │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ ¿La tesis sirve como Fase 0?  │ Sí — establece scoring base, features, e infraestructura sobre la cual          │
  │                               │ construir                                                                       │
  ├───────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ ¿Qué falta para detectar a    │ Agregación por usuario, segmentación por rol, análisis de flujo de créditos,    │
  │ Pablo?                        │ detección de secuencias                                                         │
  └───────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────┘

  La tesis es consciente de esta brecha y la documenta como limitación. Eso es metodológicamente correcto — no
  sobre-promete. Pero los datos del caso Pablo son un excelente ejemplo real para justificar una "Fase 1" en las
  recomendaciones de la tesis (Capítulo 4), demostrando con datos concretos por qué la detección a nivel transacción es
  necesaria pero insuficiente para fraude interno.