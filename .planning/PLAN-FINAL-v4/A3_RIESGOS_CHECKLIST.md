# A3 Riesgos y Checklist v4

## Riesgos

| Riesgo | Mitigacion |
|---|---|
| Leakage por estados posteriores | Lista de columnas prohibidas y tests |
| Sobreajuste o circularidad percibida por target encoding | Excluirlo del modelo principal; usarlo solo en sensibilidad |
| Septiembre contaminado por discovery | Test principal Oct-Dic |
| Cambio metodologico frente a tesis no supervisada | Presentar HGB como extension operativa |
| Proxy no es fraude | Usar lenguaje de reembolso/riesgo |
| Baja capacidad operativa | Reportar top-k y horas operador |
| Scorer single distinto a batch | Test de equivalencia |
| Benchmark HGB no reproducible | Crear `scripts/run_hgb_benchmark.py` antes de Fase 1 |
| Cupon mal cableado | Usar `payment_discounts` + `coupons`, no `payments.coupon_id` |
| `captured_at` incompleto | V4-CLEAN no depende de captura; T1 solo sensibilidad |
| Soft leakage en participantes de reserva | Agregados as-of o excluir de V4-CLEAN |
| HE3 trivial contra IF | Comparar tambien contra SIMPLE-RULE |
| Proxy Tipo A no equivale a fraude | Revision humana ciega de muestra |
| Confundir Gate A0 con resultado final | Solo declarar exito tras pipeline final USD + catalogo v4 |
| Implementacion `src/` pendiente | No iniciar defensa tecnica solo con `scripts/run_hgb_benchmark.py` |
| SIMPLE-RULE ya supera AUC 0.70 | Defender aporte por top-k, AP/base y operacion |
| ROI estrategico incompleto | Agregar costo-beneficio en reporting |
| Diciembre mas debil | Reportar sensibilidad temporal y posible estacionalidad |

## Checklist por fase

### Datos

- [ ] Benchmark HGB-CLEAN reproducido y guardado en `output/v4/benchmarks/`.
- [ ] Gate A0 tratado como viabilidad, no resultado final.
- [ ] Snapshot solo 2025.
- [ ] Conteos por split.
- [ ] Query guardada.
- [ ] `FINAL` aplicado donde corresponde.
- [ ] `amount` mapeado explicitamente desde `reservation_paid_out` o fallback documentado.
- [ ] Cupones extraidos via `payment_discounts` + `coupons`.
- [ ] `captured_at` no imputado como `created_at` en el modelo principal.
- [ ] `reservations_users` agregado con `created_at <= payment.created_at` o excluido.
- [ ] Mutabilidad de `reservation.date` verificada.

### Features

- [ ] Catalogo con timestamp de disponibilidad.
- [ ] No hay columnas prohibidas.
- [ ] Rolling excluye fila actual.
- [ ] Encoders fit solo train.
- [ ] V4-CLEAN no usa target encoding.
- [ ] V4-CLEAN no usa historial directo de reembolso.

### Modelo

- [ ] Hiperparametros desde validation.
- [ ] Gate C validation AUC >= 0.78 antes de mirar test.
- [ ] Baselines entrenados.
- [ ] SIMPLE-RULE baseline calculado.
- [ ] V4-CLEAN corrida.
- [ ] V4-CLEAN-BOOSTED y V4-FULL-SENS corridas solo como sensibilidad.

### Evaluacion

- [ ] Test final Oct-Dic.
- [ ] Legacy Sep-Dic separado.
- [ ] Bootstrap CI.
- [ ] AUC mensual Oct/Nov/Dic y rango mensual.
- [ ] Top-k y costo operativo.
- [ ] Diciembre documentado si vuelve a ser el mes mas debil.
- [ ] Validacion humana de muestra Tipo A planificada o ejecutada.

### Reporting

- [ ] Tablas reproducibles.
- [ ] Figuras reproducibles.
- [ ] Narrativa proxy != fraude.
- [ ] Tabla HGB-CLEAN vs SIMPLE-RULE.
- [ ] Costo-beneficio/ROI proxy incluido.
- [ ] Alineacion Tesis-Latex actualizada: IF original vs HGB extension.
- [ ] Tabla de mezcla manual Tipo A si la muestra fue ejecutada.

## Kill Switches

- [ ] Detener si Test Oct-Dic AUC < 0.70 en V4-CLEAN.
- [ ] Detener si V4-CLEAN queda por debajo de SIMPLE-RULE en AUC y P@1% simultaneamente.
- [ ] Detener si hay leakage principal sin correccion.
- [ ] Detener si Val AUC < 0.78 y no mejora tras revisar features.

### Scorer

- [ ] Threshold desde validation.
- [ ] Contexto ClickHouse activo.
- [ ] Batch/single equivalente.
