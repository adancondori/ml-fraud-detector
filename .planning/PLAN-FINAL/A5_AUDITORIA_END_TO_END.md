# A5 — Auditoría End-to-End

## Objetivo

Verificar si `PLAN-FINAL` ya es suficientemente completo para ejecutar sin improvisación crítica.

## Estado por perspectiva

### Datos

- `FINAL` definido
- warm history definida
- manifests y lineage definidos
- edge cases principales cubiertos
- normalización monetaria multi-moneda (USD vía `rate_to_usd`) definida como Capa 1.5

### Feature engineering

- catálogo oficial de 31 features (F1-F33)
- variante de 30 features (sin `user_reversal_ratio_30d`)
- features F24-F33 (nuevas): `is_club_credit`, `user_debit_count_30d`, `user_debit_amount_30d`, `credit_flow_ratio`, `is_staff`, `paid_by_manager`, `staff_amount_zscore`, `category_entropy_30d`, `user_reversal_count_30d`, `user_merchandise_ratio_30d`
- F06 (`is_free`) y F21 (`user_free_pct_30d`) **ELIMINADAS** — `payment_method='free'` excluido del universo depurado
- tests TDD y anti-leakage (incluyendo F24-F33)
- bordes temporales explícitos

### Modelado

- IF, LOF y OC-SVM definidos
- comparación justa
- baselines internos
- bootstrap y estabilidad temporal

### Reporting post-hoc

- análisis por centro, moneda y descuentos
- gate explícito para identidad del actor

### Gobernanza

- política de privacidad
- contratos de entrada/salida
- regla de documento interno vs público

## Juicio final

El plan queda **apto para ejecución end-to-end** una vez propagadas las convenciones canonicas:
- 31/30/21 features;
- normalización monetaria vía `rate_to_usd`;
- nombres de features F24-F33 sincronizados en sensibilidad y reporting.

Lo único que permanece abierto por naturaleza y debe resolverse en discovery de datos:

- si la identidad del actor manager es validable;
- ~~si `currency` requiere normalización adicional~~ **RESUELTO**: normalización a USD vía `rate_to_usd` incorporada como Capa 1.5;
- si aparece una anomalía estructural no prevista que obligue a volver a Fase 0.

## Siguiente paso

La siguiente acción correcta ya no es seguir planificando, sino comenzar la implementación desde:

1. `01_CONTRATO_ALCANCE.md`
2. `02_DATOS_SNAPSHOT.md`
3. `10_ORQUESTADOR.md`
