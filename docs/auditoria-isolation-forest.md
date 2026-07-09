# Auditoría crítica: Isolation Forest para detección de anomalías en pagos

**Fecha:** 2026-07-07 · **Auditor:** revisión técnica interna (Claude Code)
**Alcance:** `ml-fraud-detector` completo — formulación, leakage, split, features, IF, métricas,
robustez — con validación sobre código, artefactos entrenados, `output/revision/` y ClickHouse.
**Nota sobre el objetivo "subir AUC-ROC > 0.80":** ver sección G — se explica por qué ese objetivo,
contra los proxies disponibles, solo se alcanza con circularidad o leakage, qué AUC>0.80 ya existe y
por qué no es creditable, y cuál es la ruta honesta.

---

## A. Veredicto general: **PARCIALMENTE CORRECTO**

La ruta confirmatoria (FS-clean-A-29, split temporal, proxy solo-evaluación, multi-semilla,
bootstrap) es metodológicamente **sólida y defendible**. La ruta operativa (campeón IF-40 con AUC
0.84) contiene **circularidad total** entre features y proxy de evaluación, y ese número no puede
reportarse como capacidad discriminativa.

## B. Riesgo metodológico: **MEDIO**

Era ALTO antes de FS-clean-A-29; la exclusión de las features de reversal y el split temporal lo
bajaron. El riesgo residual está en (1) reportar el 0.84 circular como logro, y (2) perseguir
AUC>0.80 contra proxies débiles, lo que presiona hacia re-introducir leakage.

---

## C. Problemas críticos (invalidan resultados si no se corrigen)

**C1. Circularidad total del campeón operativo IF-40 (AUC 0.84).**
El proxy `pure_fraud` se define como OR de reglas sobre `same_amount_count_1h`,
`user_account_age_days`, `user_txn_count_1h`, `is_third_party_payment`. El feature set del campeón
(`final_feature_list.json`, 40 features) **contiene las 9 variables que definen el proxy**
(incluidas recodificaciones: `is_new_user`, `is_very_new_user`, `new_user_first_facility`,
`rapid_burst`, `same_amount_count_24h`). Verificado programáticamente (2026-07-07): intersección =
9/9 en el campeón, 0/9 en frame-v1. **El AUC 0.84 mide qué tan bien el modelo reconstruye su propia
definición** — autovalidación, no detección. Evidencia del contrafactual: frame-v1 (sin esas
variables) cae a 0.609 y, contra el único proxy independiente de features (`tipo_a`), **ambos
modelos rinden ~0.49** (`output/revision/shadow_decision_frame_v1.json`).

**C2. El objetivo "AUC > 0.80" contra los proxies disponibles es estructuralmente inalcanzable de
forma honesta** (ver G). Los caminos que llegan a 0.80+ hoy son: features circulares (C1) o
features derivadas de `status` (leakage mecánico, +0.068 medido). Aceptar ese objetivo tal cual
equivale a exigir el error que la propia auditoría debe impedir.

## D. Problemas importantes (corregir, no invalidan todo)

**D1. `capture_delay_seconds` en el feature set IF-40** — derivada de `captured_at`, timestamp
posterior a la creación de la transacción. Información post-evento: en scoring real-time no existe
aún. Correctamente excluida de DISJOINT30/frame-v1; debe salir de cualquier set operativo.

**D2. Proxy `tipo_a` (reembolso) es señal débil, no anomalía estadística.** Refund rate 2–9% según
moneda; la mayoría son operaciones legítimas de servicio al cliente. Todos los modelos no
supervisados rinden ~0.50 contra él — coherente con que el reembolso no es separable por
comportamiento transaccional. Esto NO es un fallo del modelo; es un techo del proxy. Documentarlo.

**D3. Features contaminadas por marcos heterogéneos en base-31/IF-40** (magnitud absoluta
multi-moneda/multi-escala; hora en UTC sobre ~9 husos). Medido: top-5% de anomalías con monto 15.6×
el promedio (detector de tamaño) y off-hours inflada 26–30% vs 4.4% real local. Resuelto en
frame-v1; pendiente promoverlo (decisión de arquitectura por capas ya tomada, ver
`plan-normalizacion-marcos.md` §A.6).

**D4. Calidad de datos:** 117K transacciones con `currency` vacía (577 facilities); facilities sin
`time_zone`. Fallbacks existen (`FrameFlags`), pero corregir en la fuente.

**D5. No existe ground truth de fraude en ClickHouse** — no hay tablas de chargebacks/disputas
(verificado 2026-07-07: solo `failed_payment_logs` y `user_penalties`). Sin ingesta de disputas de
los gateways, ninguna evaluación futura podrá superar el nivel "proxy operativo".

## E. Tabla de leakage

| Variable | Riesgo | Explicación | Acción |
|---|---|---|---|
| `user_reversal_ratio_30d`, `user_reversal_count_30d` | **Crítico (mecánico)** | Derivadas de `status`, la columna que define el proxy tipo_a. Inflan AUC +0.068 (0.508→0.576, medido) | ✅ Ya excluidas en FS-clean-A-29; mantener FS-baseline-31 solo como análisis de sensibilidad |
| 9 vars que definen `pure_fraud` (en IF-40) | **Crítico (circular)** | El modelo contiene la definición del proxy contra el que se evalúa | Excluir del set operativo (frame-v1 ya lo hace) o cambiar de proxy; nunca reportar 0.84 como discriminación |
| `capture_delay_seconds` (en IF-40) | Alto (post-evento) | `captured_at` ocurre después de la transacción; no disponible en scoring real-time | Excluir (frame-v1 ya lo excluye) |
| `status`, `reversed_id`, `debit_refund`, `ach_refund_transfer_id` | Bajo (bien manejado) | Post-outcome puro | ✅ Verificado: solo metadata, nunca features |
| Ventanas rolling 30d/24h/1h | Bajo (bien manejado) | Podrían incluir la fila actual | ✅ `_rolling_shifted_stat` usa `shift(1)`; velocity resta la fila actual (`engineering.py:85-101,178,188`) |
| Stats de facility/rol (fit) | Bajo (bien manejado) | Podrían ajustarse con test | ✅ `fit()` solo en train; test intocable para tuning |

## F. Revisión de métricas

- **Orientación del score: correcta.** `-score_samples()` / `-decision_function()` (mayor = más
  anómalo) consistente en grid search, evaluación y scorer (`trainer.py:88-108`,
  `scoring/scorer.py:187-190`). AUC calculado sobre el score invertido — bien.
- **AUC-ROC, AP, EF@k, precision@k: bien calculados** (`evaluation/metrics.py`), con tasa base
  reportada, Mann-Whitney U + rank-biserial y bootstrap CI (1000 iter). Correcto que AP se compare
  contra la tasa base (6.3%) y no contra 0.5.
- **`contamination='auto'` + ranking:** correcto — las decisiones no usan `predict()` binario sino
  umbrales calibrados por percentil (y segmentados por facility/moneda en frame-v1).
- **Interpretación:** AUC 0.508 vs tipo_a = sin capacidad discriminativa (los IC bootstrap lo
  confirman); AUC 0.84 vs pure_fraud = circular (C1). **Un AUC > 0.80 con estas variables es
  sospechoso por construcción y debe presumirse leakage/circularidad hasta demostrar lo contrario** —
  exactamente lo que ocurrió aquí.
- **Split temporal: correcto.** Warm (Dic 2024) / Train Ene–Jun / Val Jul–Ago / Test Sep–Dic 2025,
  con warm-history para ventanas en fronteras (`transform_with_warm_history`). Un split aleatorio
  habría filtrado historial futuro del mismo usuario a train e inflado métricas; se evitó bien.
- **Robustez: adecuada.** Multi-semilla (rango AUC < 0.01), grid search solo en val, test intocable,
  estabilidad temporal reportada, comparación vs LOF/OC-SVM (LOF 0.536 > IF 0.508 en clean — se
  reporta honestamente, HE4 rechazada).

## G. Recomendaciones concretas (incluye el veredicto sobre "subir AUC > 0.80")

**Sobre el objetivo AUC>0.80 — reencuadre obligatorio.** Ya existe un AUC 0.84 en el repo; es
circular y no vale. Las rutas reales:

1. **Contra tipo_a (reembolso): inalcanzable honestamente.** Techo empírico ~0.51 con features
   limpias (IF/LOF/OC-SVM, 3 semillas, 2.5M test). El reembolso no es una anomalía de
   comportamiento; ningún feature engineering legítimo lo llevará a 0.80.
2. **Contra pure_fraud: alcanzable solo con features disjuntas del proxy.** frame-v1 (0/9 vars del
   proxy) da 0.609. Se puede subir con features nuevas NO usadas en la definición (device/IP si
   existieran, BIN/token — hoy no disponibles), pero cada mejora debe pasar el test de disyunción.
3. **Ruta recomendada para un AUC>0.80 legítimo: crear ground truth real.** Ingerir
   chargebacks/disputas desde los gateways (CardConnect/Stripe los exponen; hoy NO están en
   ClickHouse) y evaluar contra fraude confirmado. Mientras tanto, usar las señales operacionales a
   nivel usuario (`player_blocked`, `user_penalties`, `fraud_like_fail`) como evaluación
   retrospectiva no circular (ya medido: AUC 0.556, EF@1% 5.36 a nivel usuario).
4. **Redefinir el KPI operativo:** en detección no supervisada con proxy débil, el KPI honesto no es
   AUC global sino **EF@k con presupuesto de revisión + tipificación accionable** (arquetipos SHAP ya
   implementados: credit_flow 42.5% del top-5% con 2.5× la tasa de reembolso).

**Acciones de código (orden):** (1) promover frame-v1 según arquitectura por capas ya decidida;
(2) declarar las 9 reglas de pure_fraud como capa de reglas explícitas; (3) excluir
`capture_delay_seconds` de cualquier set vivo; (4) pipeline de ingesta de disputas de gateway;
(5) corregir currency vacía / time_zone nula en fuente; (6) mantener el test de disyunción
feature-proxy como gate de CI para todo feature nuevo.

## H. Párrafos sugeridos (tesis)

**Metodología.** «Se implementó un enfoque no supervisado basado en Isolation Forest, entrenado sin
etiquetas sobre 3,14 millones de transacciones (enero–junio 2025) con partición temporal estricta
(validación: julio–agosto; prueba: septiembre–diciembre). La condición proxy de anomalía —derivada
del estado de reembolso— se empleó exclusivamente en la fase de evaluación, nunca en el
entrenamiento ni en la selección de variables; las variables con dependencia mecánica del proxy
(razón y conteo de reversiones) se excluyeron del catálogo confirmatorio, y su efecto inflacionario
(ΔAUC = +0,068) se cuantificó como análisis de sensibilidad de circularidad.»

**Resultados.** «El puntaje de anomalía de Isolation Forest no mostró capacidad discriminativa
respecto de la condición proxy de reembolso (AUC-ROC = 0,508; IC 95% bootstrap incluye 0,50;
EF@5% = 0,92), resultado consistente entre semillas y modelos alternativos (LOF = 0,536;
OC-SVM = 0,506). Este hallazgo no debe interpretarse como ausencia de anomalías en los datos, sino
como evidencia de que el reembolso —mayoritariamente una operación legítima de servicio— no
constituye una anomalía estadística separable mediante el comportamiento transaccional observado.
Las anomalías identificadas por el modelo no se afirman como fraude confirmado; caracterizan
transacciones atípicas respecto del patrón de su instalación, cuya verificación requiere revisión
operativa.»

## I. Preguntas probables del tribunal / revisor técnico

1. **«¿Su AUC de 0,50 significa que el trabajo fracasó?»** No: el objetivo era *evaluar* la
   capacidad discriminativa, y el resultado negativo es un hallazgo empírico riguroso; además se
   explica el mecanismo (el proxy no es anomalía estadística) y se cuantificó cuánto "mejoraría"
   artificialmente con leakage (+0,068), demostrando control metodológico.
2. **«¿Cómo garantiza que el proxy no contaminó el entrenamiento?»** El modelo entrena sin
   etiquetas; el proxy solo entra en evaluación; las features derivadas de `status` fueron excluidas
   del set confirmatorio; las ventanas rolling excluyen la fila actual; `fit()` solo en train.
3. **«¿Y el AUC 0,84 que aparece en el sistema operativo?»** Es circular: el feature set contiene
   las 9 variables que definen ese proxy; se documenta como autovalidación y no se reporta como
   capacidad discriminativa. Contra el proxy independiente, ese mismo modelo rinde 0,49.
4. **«¿Por qué split temporal y no aleatorio?»** Un split aleatorio mezclaría historial futuro del
   mismo usuario entre train y test (fuga temporal) e inflaría métricas; el fenómeno además exhibe
   drift estacional.
5. **«¿Por qué el score se invierte?»** En sklearn, `score_samples`/`decision_function` devuelven
   valores mayores para puntos normales; se niega el score para que mayor = más anómalo, y el AUC se
   calcula sobre esa orientación de forma consistente en todo el pipeline.
6. **«¿Cómo interpreta los casos top sin afirmar fraude?»** Mediante tipificación descriptiva
   (SHAP → arquetipos: flujo de crédito, canal/gateway, diversidad), lenguaje de asociación y
   revisión operativa; nunca "fraude confirmado".
7. **«¿LOF superó a IF; por qué mantiene IF?»** Se reporta honestamente (HE4 rechazada); IF se
   retiene operativamente por costo computacional y estabilidad, decisión declarada como operativa,
   no como superioridad empírica.

---

**Dictamen final:** la base confirmatoria es defendible tal como está; el sistema operativo requiere
(a) retirar el número circular, (b) promover frame-v1 con la arquitectura por capas, y (c) si se
desea un AUC>0,80 *legítimo*, invertir en ground truth (disputas de gateway) — no en más features
contra proxies débiles.
