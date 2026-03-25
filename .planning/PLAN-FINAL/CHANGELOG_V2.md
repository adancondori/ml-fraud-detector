# CHANGELOG HISTORICO — Plan v2.0 (2026-03-22)

> Resumen de correcciones aplicadas al PLAN-FINAL para alinearlo con la tesis actualizada (`03_propuesta_validacion.tex`).
> Documento historico: **no usar como fuente operativa de implementacion**. Si hay discrepancias con la version actual, prevalece `01_CONTRATO_ALCANCE.md` y las variantes canonicas `IF-31 / IF-30 / IF-21`.

## Discrepancias Detectadas y Corregidas

### CRITICAS

| # | Discrepancia | Correccion |
|---|-------------|------------|
| 1 | **Features: 23 (plan) vs 33 (tesis)** — Faltaban 10 features en 3 grupos nuevos (F, G, H) | Actualizado a 33 features / 8 grupos en TODOS los documentos |
| 2 | **SQL canonico incompleto** — Faltaban `category`, `club_credit_flag` | Agregadas al SQL en `01_CONTRATO_ALCANCE.md` |
| 3 | **Sensibilidad desalineada** — Plan: 23 vs 22; Tesis: 33 vs 32 + ablacion 33 vs 23 | Actualizado con 3 variantes: IF-33, IF-32, IF-23 |
| 4 | **Normalizacion monetaria ausente** — 21 monedas sin normalizar | Agregada como prerequisito obligatorio (Capa 1.5 en ETL) |

### IMPORTANTES

| # | Discrepancia | Correccion |
|---|-------------|------------|
| 5 | **Grid search IF: 64 vs 240 combos** — Plan excluia contamination | Actualizado a 240 combos (4x4x5x3) incluyendo contamination |
| 6 | **OC-SVM nu: 0.02 vs 0.01** | Corregido a [0.01, 0.05, 0.10] |
| 7 | **Variantes de modelo** — Plan solo tenia IF, LOF, OC-SVM | Agregados IF-33, IF-32, IF-23 como variantes explicitas |
| 8 | **Nuevos analisis ausentes** — Metricas por rol, por categoria, tipologia SHAP, perfil usuario | Agregados en sensibilidad y reporting |

### MENORES

| # | Discrepancia | Correccion |
|---|-------------|------------|
| 9 | F#1: `amount` vs `reservation_paid_out` | Alineado con tesis: `reservation_paid_out` |
| 10 | F#4 epsilon: 1e-8 vs 0.01 | Alineado con tesis: 0.01 |
| 11 | F#16: `_cumul` vs `_30d` | Alineado con tesis: `user_distinct_facilities_30d` |
| 12 | HE4 threshold: tesis tabla resumen dice ">=2/4" | Plan mantiene >=3/4 (correcto); flagged para correccion en tesis |

## Documentos Modificados

| Documento | Cambios principales |
|-----------|--------------------|
| `00_PLAN_MAESTRO.md` | 33 features, 240 combos, nuevos modulos, normalizacion USD |
| `01_CONTRATO_ALCANCE.md` | Reescrito completo: SQL, 33 features, variantes, normalizacion, nuevos analisis |
| `04_FEATURE_ENGINEERING.md` | Reescrito completo: 33 features, 8 grupos, implementaciones F24-F33 |
| `05_PREPROCESAMIENTO.md` | CurrencyNormalizer, 33 features, memorias actualizadas |
| `06_MODELADO_TUNING.md` | 240 combos, IF-32/IF-23 variantes, nu corregido |
| `07_EVALUACION_HIPOTESIS.md` | 33 features en HE4, nota sobre variantes |
| `08_SENSIBILIDAD.md` | 33/32, ablacion 33 vs 23, metricas por segmento, tipologia SHAP, perfil usuario |
| `09_REPORTING.md` | 24 tablas (6 nuevas), 12 figuras (1 nueva) |
| `A1_ETL_LINEAGE_EDGE_CASES.md` | Capa 1.5 (currency), 33 features |
| `A2_PROTOCOLO_RUNBOOK.md` | Paso 4.5 (normalizacion), 33 features |
| `A3_RIESGOS_CHECKLIST.md` | R10 multi-moneda, 32 features, 240 combos |
| `A5_AUDITORIA_END_TO_END.md` | 33 features, F24-F33, normalizacion |
| `CLAUDE.md` (proyecto) | 33 features, normalizacion USD |

## Pendientes para Correccion en Tesis

1. `tab:resumen-hipotesis` (linea ~925 de `03_propuesta_validacion.tex`): dice ">=2/4" para HE4, deberia ser ">=3/4"
2. Verificar disponibilidad de columna `role` en ClickHouse para F28 (`is_staff`); alternativa: derivar de `facilities_users` o usar `paid_by_manager` como proxy

---

# CHANGELOG — Sync v2.1 (2026-03-24)

> Sincronizacion del plan con la tesis actualizada (perfil, cronograma, progress.json).

## Cambios aplicados

| # | Cambio | Detalle |
|---|--------|---------|
| 1 | **progress.json Fase 3**: "20 features" → "33 features, 8 grupos" | El nombre reflejaba una versión anterior |
| 2 | **progress.json Fase 5**: grid search IF "64 combos" → "240 combos" | Alineado con plan maestro (4x4x5x3) |
| 3 | **progress.json Fase 6**: agregado `segment_analysis.py` como artefacto | Documentado en tesis Cap 3 (métricas por rol/categoría) |
| 4 | **progress.json blockers**: resueltos 2 bloqueantes | amount_zero (8.93% correcto), sin_estado (documentado en Cap 2) |
| 5 | **Gate A**: actualizado con cifras duales (tesis + pipeline) | Diferencias despreciables confirmadas |
| 6 | **Cronograma plan maestro**: alineado con 9 actividades del perfil UAGRM | Mapeo actividad → fase del pipeline |
| 7 | **Cronograma técnico**: replanteado desde posición actual (Fase 3 parcial) | 8 semanas restantes para Fases 3-10 |
| 8 | **thesis_alignment**: nueva sección en progress.json | Estado de cada capítulo, mapeo cronograma, discrepancias numéricas |

## Estado actual tras sync v2.1

| Métrica | Valor |
|---------|-------|
| Fases completadas | 3 de 13 (23%) |
| Fases parciales | 1 (Fase 4: Feature Engineering) |
| Fases pendientes | 9 |
| [POR COMPLETAR] en tesis | 61 (57 Cap 3 + 4 Cap 4) |
| Bloqueantes activos | 2 (re-extracción + tasas de cambio) |
| Próxima acción | Re-extraer con JOINs, luego tasas de cambio, luego features |

---

# CHANGELOG — Sync v3.0 (2026-03-24)

> Verificación contra ClickHouse + nuevas fases + correcciones de datos.

## Verificaciones ClickHouse (ground truth)

| Verificación | Resultado | Impacto |
|---|---|---|
| `category` en payments | ✓ String | F25, F31, F33 viables |
| `club_credit_flag` en payments | ✓ Bool | F24 viable |
| `paid_by_manager` en payments | ✓ Bool | F29 viable |
| `is_staff` en payments | ✗ NO existe | Derivar via JOIN facilities_users |
| `users.created_at` | ✓ DateTime | F19 viable |
| Monedas distintas | **20** (no 21) | Actualizar docs y tesis |
| `exchange_rates` tabla | Solo snapshot 2026-03-20 | Necesita historial → Fase 3a |
| `facilities_users.role` | 5 roles verificados | Base para F28 derivado |

## Cambios aplicados

| # | Cambio | Archivos |
|---|--------|----------|
| 1 | **SQL canónico reescrito** con JOINs (facilities_users + users) | 01_CONTRATO_ALCANCE.md, A6 |
| 2 | **2 fases nuevas** (3a: tasas de cambio, 3b: normalización) | 00_PLAN_MAESTRO.md, progress.json |
| 3 | **20 monedas** listadas con volúmenes reales | 01_CONTRATO_ALCANCE.md |
| 4 | **is_staff** derivación documentada | A6_VERIFICACION_CLICKHOUSE.md |
| 5 | **Roles verificados**: court_manager, teacher, court_operator, guest, rental_user | A6 |
| 6 | **exchange_rates** limitación documentada + solución (script + CSV) | 01_CONTRATO_ALCANCE.md, A6 |
| 7 | **Arquitectura repositorio** actualizada con nuevos archivos | 00_PLAN_MAESTRO.md |
| 8 | **progress.json**: total_phases 11→13, nuevos blockers | progress.json |
| 9 | **Flujo de datos** actualizado con Fases 3a/3b | 00_PLAN_MAESTRO.md |
| 10 | **A6_VERIFICACION_CLICKHOUSE.md** creado | NUEVO |

## Correcciones pendientes en tesis

| # | Corrección | Archivo |
|---|-----------|--------|
| 1 | HE4 threshold: >=2/4 → >=3/4 | 03_propuesta_validacion.tex ~linea 925 |
| 2 | Monedas: actualizar lista a 20 reales (eliminar CRC, DOP, VES; agregar CAD, MYR, AUD, etc.) | 02_diagnostico.tex |
| 3 | USD %: 74.1% de txns (no 96.6% que era de gateways) | 02_diagnostico.tex |
| 4 | Maestría: "Dirección Estratégica en Ingeniería de Software" | Ya corregido en informacion.tex |

## Estado final

| Métrica | Valor |
|---------|-------|
| Fases completadas | 3 de 13 |
| Fases parciales | 1 |
| Fases pendientes | 9 (incl. 2 nuevas) |
| Bloqueantes | 2 (re-extracción con JOINs, tasas de cambio históricas) |
| Total phases | 13 (era 11) |
| Documentos actualizados | 5 + 1 nuevo |
| Próxima acción | Fase 1 (re-extracción con JOINs) → Fase 3a (tasas) → Fase 3b (normalización) → Fase 4 (33 features) |
