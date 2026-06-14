# 08 Sensibilidad v4

## Ablaciones obligatorias

| Ablacion | Objetivo |
|---|---|
| V4-CLEAN vs V4-CLEAN-BOOSTED | Medir aporte de target encoding |
| V4-CLEAN vs V4-FULL-SENS | Medir aporte y riesgo de historial de reembolso |
| V4-CLEAN vs LEGACY-31 | Medir ganancia de data nueva |
| V4-CLEAN vs T0 | Separar datos disponibles al crear pago |
| V4-CLEAN vs V4-CLEAN-NO-RU | Medir impacto de participantes de reserva |
| V4-CLEAN strict as-of vs current-state RU | Cuantificar soft leakage de `reservations_users` |
| V4-CLEAN vs sin reserva | Medir aporte de contexto de reserva |
| V4-CLEAN vs sin token/card | Medir aporte de tarjeta |
| V4-CLEAN vs sin cupon | Medir aporte de cupones |
| V4-CLEAN vs SIMPLE-RULE | Medir aporte real de ML frente a regla manual |

## Sensibilidad de labels

Evaluar, no entrenar como principal:

- Tipo A.
- Proxy unificado v2.
- Proxy amplio.
- Per-type A, C, D.

B y E siguen limitados por falta de datos/exclusion del universo.

## Sensibilidad temporal

Reportar:

- Val Jul-Ago.
- Discovery Sep.
- Final Oct-Dic.
- Mes a mes Oct, Nov, Dic.

Gate A0 mostro diciembre como el mes mas debil:

- Oct AUC aproximado: 0.8367.
- Nov AUC aproximado: 0.8399.
- Dic AUC aproximado: 0.8251.

Cumple el gate mensual, pero debe discutirse como drift o estacionalidad si se repite en el pipeline final.

## Sensibilidad operacional

Alertas esperadas:

- top 0.1%.
- top 0.2%.
- top 0.5%.
- top 1%.
- top 2%.
- top 5%.

Calcular horas operador con supuestos:

- 4 min por alerta.
- 8 h por jornada.

## Artefactos

```text
output/v4/sensitivity/
├── ablation_metrics.csv
├── temporal_metrics.csv
├── proxy_sensitivity.csv
├── operational_capacity.csv
└── leakage_stress_tests.csv
```

## Gate

La narrativa final debe usar V4-CLEAN como resultado central aunque V4-CLEAN-BOOSTED o V4-FULL-SENS sean mejores.

Si V4-CLEAN no supera SIMPLE-RULE en AUC y P@1% simultaneamente, no afirmar aporte util de ML; replantear como regla operativa explicita.
