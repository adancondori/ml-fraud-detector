# CHANGELOG v4

## Motivo

v3 queda descartado porque intenta mejorar la validez metodologica del modelo no supervisado, pero no resuelve el objetivo solicitado: alcanzar al menos 70% de AUC para detectar anomalias/riesgo de reembolso en pagos transaccionales.

La auditoria empirica mostro que:

- IF-31 sobre Tipo A tiene AUC cercano a 0.576.
- Reducir features a un catalogo lean no cambia el problema de fondo.
- Las 31 features actuales contienen senal suficiente si se entrena un ranker supervisado contra Tipo A.
- La evidencia preliminar indica que, aun removiendo historial directo de reembolso, un ranker supervisado supera el umbral AUC >= 0.70.
- Las cifras previas con target encoding o agregados de participantes en estado actual no son cifra central de tesis; deben tratarse como sensibilidad hasta validar point-in-time estricto.

## Cambios principales vs v3

| Area | v3 | v4 |
|---|---|---|
| Modelo principal | IF no supervisado lean | Ranker supervisado HGB-CLEAN |
| Objetivo | Reducir circularidad | Alcanzar AUC >= 0.70 y utilidad operativa |
| Proxy principal | Tipo A | Tipo A |
| Uso del proxy | Evaluacion/tuning IF | Etiqueta para entrenamiento del ranker |
| IF/LOF/OC-SVM | Principal/comparacion | Baselines academicos y control |
| Dataset final | Sep-Dic 2025 | Oct-Dic 2025 si septiembre se usa para discovery |
| Features | 18 lean | 62 features limpias principales; 65 candidatas |
| Gate principal | HE2 reformulado | AUC >= 0.70, AP/base >= 3, P@1% >= 40% |

## Evidencia preliminar que justifica v4

Benchmarks locales exploratorios con datos 2025. Esta tabla explica por que v4 es viable, pero no reemplaza el benchmark final. Antes de reportar resultados debe ejecutarse `clean-strict`, con `reservations_users.created_at <= payment.created_at` o exclusion de esas features, y debe compararse contra SIMPLE-RULE.

ADVERTENCIA: tabla preliminar, no validada strict. No usar como resultado final de tesis.

| Variante | Val AUC | Test Sep-Dic AUC | Test AP/base | Top 1% precision |
|---|---:|---:|---:|---:|
| HGB 31 features | 0.804 | 0.795 | 4.18 | 55.2% |
| HGB clean sin refund-history ni target encoding | 0.810 | 0.796 | 3.74 | 45.5% |
| HGB 31 + target encoding | 0.829 | 0.818 | 4.51 | 55.2% |
| HGB 31 + categoricos extra | 0.832 | 0.822 | 4.59 | 55.5% |
| HGB sin historial directo de refund | 0.818 | 0.806 | 3.86 | 45.8% |

POC adicional no estricto con features de reserva/token/cupon alcanzo AUC ~0.83 y P@1% ~81% en test, pero usaba agregados actuales de `reservations_users`. Esa cifra queda reemplazada por el benchmark Gate A0 strict as-of y por el control NO-RU.

Benchmark Gate A0 posterior a esta advertencia:

| Variante | Val AUC | Test Oct-Dic AUC | P@1% Oct-Dic | AP/base Oct-Dic |
|---|---:|---:|---:|---:|
| V4-CLEAN-NO-RU | 0.8420 | 0.8285 | 71.3% | 5.64 |
| V4-CLEAN strict as-of RU | 0.8460 | 0.8339 | 73.4% | 5.78 |
| SIMPLE-RULE | 0.7000 | 0.7052 | 35.8% | 2.29 |

Conclusion: el salto no depende de soft leakage en participantes. La variante sin `reservations_users` ya supera AUC 0.80 y P@1% 65% contra Tipo A.

## Datos nuevos priorizados

1. `source_enum` / `payment_source`: app vs web/onsite separa fuertemente el riesgo.
2. `payment_method`: `prepaid` y `user_package` tienen lift cercano a 2.
3. `reservation` point-in-time: lead time, admin booking y tipo; participantes solo si se calculan as-of.
4. `user_tokens`: edad del token, default, gateway mismatch, tipo de tarjeta.
5. `coupons`: edad, tipo fijo/porcentual, aplicacion a reserva.
6. `audit_logs`, `failed_payment_logs`, `comments`, `user_penalties`: quedan como sensibilidad por baja cobertura o lift limitado.

## Datos descartados o solo sensibilidad

- `reservation_status`, `incident_enum`, `payment_completed`: leakage probable.
- `membership_state=payment_refunded`: leakage directo.
- `payments.comments_count`: puede cambiar despues del pago.
- `audit_logs`: cobertura incompleta en 2025, concentrada en noviembre-diciembre.

## Decision metodologica

v4 separa dos entregables:

1. **Linea academica:** IF/LOF/OC-SVM se mantienen como baselines no supervisados para comparacion y trazabilidad.
2. **Linea operativa principal:** HGB-CLEAN supervisado se convierte en modelo principal para ranking de riesgo Tipo A, sin historial directo de reembolso ni target encoding.
3. **Linea operativa boosted:** target encoding e historial de refund se reportan solo como sensibilidad.

El texto de tesis debe decir claramente que Tipo A es proxy de reembolso, no fraude confirmado.
