# A1 Data Mining 2025

## Hallazgos confirmados

### Baseline Tipo A Sep-Dic 2025

- n: 2,517,473.
- positivos Tipo A: 158,893.
- tasa: 6.3116%.

### Tasa mensual 2025

| Mes | Tasa Tipo A |
|---|---:|
| Ene | 7.0565% |
| Feb | 6.6893% |
| Mar | 6.5123% |
| Abr | 6.1352% |
| May | 6.1993% |
| Jun | 6.1942% |
| Jul | 5.9553% |
| Ago | 6.1989% |
| Sep | 6.6080% |
| Oct | 6.2937% |
| Nov | 6.3210% |
| Dic | 6.0662% |

Septiembre no es extremo, pero se uso para discovery. Por eso test final principal debe ser octubre-diciembre.

## Senales fuertes

| Fuente | Senal | Lift aproximado |
|---|---|---:|
| payments | `source_enum=white_label_app/pbp_app` | ~2.0 |
| payments | `payment_method=prepaid/user_package` | ~2.0 |
| reservations | lead time 8-30d o 31d+ | ~4.0 |
| reservations | `admin_booked=false` | ~2.6 |
| reservations_users | un solo participante | ~1.8, requiere as-of estricto |
| coupons | cupon de reserva/fijo | ~1.5-1.8 |
| user_tokens | gateway mismatch | ~1.1 |
| user_tokens | token 8-30d | ~1.2 |

## Senales debiles

| Fuente | Resultado |
|---|---|
| failed_payment_logs | lift cercano a 1, baja cobertura |
| comments previos | lift <= 1 o negativo |
| audit_logs | cobertura solo Nov-Dic |
| user_penalties | baja cobertura |

## Senales prohibidas por leakage

| Columna | Motivo |
|---|---|
| `incident_enum` | refleja cancelacion/reembolso posterior |
| `reservation_status` | estado posterior |
| `payment_completed` | cambia con el desenlace |
| `membership_state=payment_refunded` | etiqueta directa |
| `payments.comments_count` | puede incrementarse despues del pago |

## Recomendacion

Priorizar:

1. `source_enum/payment_source`.
2. `payment_method/gateway/card_brand`.
3. reserva point-in-time.
4. token/card.
5. cupon.

No gastar primero en failed logs/audits/comentarios.

## Revision de circularidad

El resultado principal debe evitar:

- proxy unificado como metrica principal, porque Tipos C y D comparten variables con el modelo;
- `user_reversal_ratio_30d`, `user_reversal_count_30d` y `user_refund_count_90d` en el modelo central;
- target encoding en la conclusion principal.

Benchmark limpio legacy medido con data inicial disponible:

- sin historial directo de refund;
- sin target encoding;
- con frequency encoding categorico train-only;
- AUC Sep-Dic 2025: ~0.796;
- AUC Oct-Dic 2025: ~0.795;
- P@1% Sep-Dic 2025: ~45.5%.

Esto supera 70% sin las fuentes objetadas por circularidad.

Benchmark Gate A0 v4 ejecutado con `scripts/run_hgb_benchmark.py`:

| Variante | Val AUC | Test Sep-Dic AUC | Test Oct-Dic AUC | P@1% Oct-Dic | AP/base Oct-Dic |
|---|---:|---:|---:|---:|---:|
| V4-CLEAN-NO-RU | 0.8420 | 0.8301 | 0.8285 | 71.3% | 5.64 |
| V4-CLEAN strict as-of RU | 0.8460 | 0.8352 | 0.8339 | 73.4% | 5.78 |
| SIMPLE-RULE | 0.7000 | 0.7049 | 0.7052 | 35.8% | 2.29 |

Conclusion: el resultado no depende de `reservations_users` en estado actual. Incluso excluyendo participantes, V4-CLEAN-NO-RU supera AUC 0.80 y P@1% 65% contra Tipo A.

Metricas mensuales Gate A0 strict:

| Mes | AUC | AP/base | P@1% |
|---|---:|---:|---:|
| Sep | 0.8395 | 5.39 | 70.7% |
| Oct | 0.8367 | 5.73 | 73.1% |
| Nov | 0.8399 | 5.79 | 73.7% |
| Dic | 0.8251 | 5.80 | 73.5% |

Diciembre es el mes mas debil por AUC. Cumple gate, pero debe aparecer en sensibilidad temporal.

## Validaciones adicionales v4

### Soft leakage en `reservations_users`

Consulta de muestra 1% sobre pagos Oct-Dic con reserva:

- pagos muestreados: 9,176;
- pagos con filas `reservations_users.created_at > payment.created_at`: 1,208;
- proporcion afectada: 13.1648%;
- diferencia media entre conteo actual y conteo as-of: 0.223 filas;
- tasa Tipo A con filas futuras: 9.85%;
- tasa Tipo A sin filas futuras: 8.89%.

Conclusion: el riesgo no es masivo, pero es real. Las features de participantes solo son V4-CLEAN si se calculan as-of.

### Mutabilidad de `reservation.date`

Sobre versiones fisicas disponibles 2025:

- reservas evaluadas: 9,476,955;
- reservas con multiples versiones fisicas: 1.1972%;
- reservas con `date` cambiado: 0;
- cambios de `admin_booked`: 40;
- cambios de `reservation_type`: 1,748.

Conclusion: `reservation_lead_days` parece defendible, pero debe quedar test automatizado.

### Regla manual simple

Regla:

```text
score = app_source + prepaid_or_user_package + lead_days>=8 + not_admin_booked
```

Oct-Dic 2025:

- AUC: 0.7052;
- P@1%: 35.8%.

Conclusion: HGB debe compararse contra esta regla para demostrar aporte real de ML.
