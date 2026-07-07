# Fase 2 — Contexto de entrada del usuario

**Fecha:** 2026-07-06

## Directiva del usuario (arrastrada desde la consulta de cobertura por moneda)

Extender `currency_fallbacks` más allá de las 5 monedas actuales del artefacto `facility_stats_v1.json`. Añadir **al menos AUD, ILS, PKR, GTQ, SGD** para que las facilities de bajo volumen en esas monedas caigan a su moneda y no a `global`.

### Evidencia que motiva la directiva (verificada 2026-07-06)

- `facility_stats_v1.json` (Fase 1) solo construyó `currency_fallbacks` para **5 monedas**: USD, CAD, HNL, MYR, NIO (las que cubren ~90% del volumen).
- Distribución de fallback actual: 580 facility / 1289 currency / 7 global.
- Volumen 2025 (ClickHouse, `payment_method NOT IN ('reversal','free')`) de monedas SIN fallback de moneda hoy:
  - AUD 183.149 (38 fac), ILS 122.470 (5 fac), GTQ 83.275 (10 fac), PKR 60.716 (9 fac), SGD 44.486 (5 fac), HKD 56.297 (5 fac), COP 24.362 (3 fac), AED 21.466 (3 fac), BWP 14.697 (4 fac), EUR 12.041 (2 fac).
  - Marginales (no calibrables de forma fiable): JPY 2.492, RWF 2.312, MXN 1.299, INR 65, NZD 4. `EMPTY` 4.766 en 357 facilities (saneada a USD, calidad de datos).
- En la práctica el hueco casi no muerde hoy (esas monedas tienen alto volumen por facility → stats propias), pero el fallback de moneda debe existir para facilities nuevas/de bajo volumen en esas monedas.

## Cómo encaja en la Fase 2

- La calibración segmentada (facility → currency_group → global) comparte la estructura de moneda con el artefacto. Extender `currency_fallbacks` en el builder de Fase 1 (`src/fraud_detector/stats/builder.py`) y regenerar `facility_stats_v1.json` es un prerequisito natural de la segmentación.
- Criterio: construir `currency_fallbacks` para toda moneda con volumen suficiente (p. ej. n ≥ umbral de moneda, análogo a la guarda de n≥200 por segmento de umbral). Derivar `currency_group` de los datos (distancia KS entre distribuciones de score), NO de geografía (research global).
- Regenerar el artefacto NO debe romper la paridad de Fase 1 (el `FrameV1FeatureCalculator` consume el artefacto por `facility_id`; añadir más `currency_fallbacks` solo cambia el fallback de facilities de bajo-n, no la aritmética). Re-correr `tests/test_parity_phase1.py` tras regenerar.
