# PLAN-FINAL-v4

Plan ejecutable v4 para elevar la capacidad discriminativa del sistema de pagos transaccionales a AUC >= 0.70 sobre proxy Tipo A, usando exclusivamente datos de la gestion 2025 y evitando circularidad metodologica en el resultado principal.

## Estado

- `PLAN-FINAL/` queda como baseline historico v2.
- `PLAN-FINAL-v3/` queda descartado: reduce circularidad, pero no resuelve el objetivo operativo de AUC >= 0.70 con el enfoque no supervisado.
- `PLAN-FINAL-v4/` reemplaza a v3 como guia de implementacion.
- Estado tecnico actual: Gate A0 esta cerrado con `scripts/run_hgb_benchmark.py`; el pipeline productivo v4 en `src/` y `run_pipeline_v4.py` todavia no esta implementado.

## Decision central

La evidencia empirica muestra que el cuello principal no es la falta total de senal, sino el paradigma del modelo:

| Enfoque | AUC test Sep-Dic 2025 | Observacion |
|---|---:|---|
| Isolation Forest actual sobre Tipo A | ~0.576 | Detecta anomalia generica, no riesgo de reembolso |
| HGB supervisado con 31 features actuales | ~0.795 | Supera el objetivo minimo |
| HGB clean sin historial directo de reembolso ni target encoding | ~0.796 | Variante principal defendible contra circularidad |
| HGB V4-CLEAN-NO-RU | 0.8285 Oct-Dic | Strict por exclusion de participantes |
| HGB V4-CLEAN strict as-of RU | 0.8339 Oct-Dic | Gate A0 cerrado con participantes point-in-time |
| SIMPLE-RULE | 0.7052 Oct-Dic | Baseline manual interpretable |
| HGB + target encoding train-only | ~0.818 | Variante boosted, solo sensibilidad/operativa |
| HGB + categoricos adicionales | ~0.822 | Aprovecha `source_enum`, `gateway`, `payment_method`, `card_brand` |
| HGB sin historial directo de reembolso + categoricos extra | ~0.806 | Sigue superando 0.70 sin depender de historial de refunds |

## Indice

| Doc | Proposito |
|---|---|
| `CHANGELOG_V4.md` | Por que se descarta v3 y que cambia |
| `00_PLAN_MAESTRO.md` | Vision, fases, gates y arquitectura objetivo |
| `01_CONTRATO_ALCANCE.md` | Contrato metodologico v4 |
| `02_DATOS_SNAPSHOT.md` | Extraccion 2025, splits y nuevas tablas fuente |
| `03_EDA_CAPITULO2.md` | EDA orientado a senales de reembolso |
| `04_FEATURE_ENGINEERING.md` | Catalogo v4 de features point-in-time |
| `05_PREPROCESAMIENTO.md` | Encoding, escalado y anti-leakage |
| `06_MODELADO_TUNING.md` | Ranker supervisado + baselines no supervisados |
| `07_EVALUACION_HIPOTESIS.md` | Evaluacion temporal, HE1-HE4 y gates |
| `08_SENSIBILIDAD.md` | Ablaciones, leakage audit y variantes |
| `09_REPORTING.md` | Tablas, figuras y narrativa tesis/sistema |
| `10_ORQUESTADOR.md` | Pipeline reproducible v4 |
| `11_TESTS_CLEANUP_INTEGRACION.md` | Tests obligatorios e integracion |
| `12_SINGLE_TRANSACTION_SCORER.md` | Scorer individual v4 |

## Anexos

| Doc | Proposito |
|---|---|
| `A1_DATA_MINING_2025.md` | Hallazgos de ClickHouse/MySQL y data candidata |
| `A2_PROTOCOLO_RUNBOOK.md` | Runbook de ejecucion |
| `A3_RIESGOS_CHECKLIST.md` | Riesgos y checklist |
| `A4_GOBERNANZA_PRIVACIDAD_CONTRATOS.md` | Privacidad y contratos de datos |
| `A5_AUDITORIA_END_TO_END.md` | Auditoria final |
| `A6_VERIFICACION_CLICKHOUSE.md` | Reglas ClickHouse y SQL patterns |
| `A7_VALIDACION_NEGOCIO.md` | Validacion humana del proxy Tipo A |
| `A8_ALINEACION_TESIS_LATEX.md` | Alineacion academica IF vs HGB supervisado |

## Orden de uso

1. Leer `CHANGELOG_V4.md`.
2. Ejecutar el plan desde `00_PLAN_MAESTRO.md`.
3. Implementar fases 1 a 7.
4. Solo despues cerrar reporting y scorer.

No usar septiembre 2025 como test final si se usa para descubrir features. En v4, septiembre queda como ventana de diagnostico; el holdout final principal es octubre-diciembre 2025.

## Precondiciones de viabilidad

Antes de ejecutar Fase 1 deben cerrarse estos 5 gaps materiales ya absorbidos por el plan:

1. Reproducir el benchmark HGB en codigo versionado y guardar metricas en `output/v4/benchmarks/`.
2. Documentar el mapeo real `amount = reservation_paid_out` o fallback equivalente.
3. Construir features de cupon desde `payment_discounts` + `coupons`, no desde `payments.coupon_id`.
4. Ampliar columnas prohibidas por mutabilidad/leakage.
5. Definir tratamiento de `captured_at` faltante.

Adicionalmente, el benchmark aceptado para tesis debe ser **strict point-in-time**:

- `reservations_users` debe agregarse con `created_at <= payment.created_at`; si no se puede implementar eficientemente, esas features se excluyen de V4-CLEAN.
- El benchmark debe comparar HGB-CLEAN contra una regla manual simple, no solo contra Isolation Forest.
- La cifra operativa principal no puede usar POC con participantes en estado actual.

Estado: Gate A0 ya fue ejecutado. `baseline_hgb_clean_strict.json` reporta AUC Oct-Dic 0.8339, AP/base 5.78 y P@1% 73.4%. `baseline_hgb_clean_no_ru.json` reporta AUC Oct-Dic 0.8285 y P@1% 71.3%. `simple_rule_baseline.json` reporta AUC Oct-Dic 0.7052 y P@1% 35.8%.

Limitacion: estos numeros prueban viabilidad, no son el resultado final de tesis. El resultado defendible exige ejecutar el pipeline completo con USD normalizado, catalogo v4 final y `run_pipeline_v4.py all --final`.

## Conteo de features

- Catalogo candidato: 65 features logicas.
- Modelo principal limpio: 62 features logicas.
- Las 3 features excluidas del principal son historial directo de reembolso: `user_reversal_ratio_30d`, `user_reversal_count_30d`, `user_refund_count_90d`.
- Target encoding queda fuera de la conclusion principal; solo se usa como sensibilidad/techo operativo.
