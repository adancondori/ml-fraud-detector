# Contrato API frame-v1 (feature_frame_version = "frame-v1")

Fecha: 2026-07-09. Change SDD: `frame-normalization-v1` (store en el workspace, `sdd/changes/frame-normalization-v1/`).

## Fuente única del contrato

Los fixtures canónicos viven en **este repo** (dueño del contrato):

```
tests/fixtures/contract/frame_v1/
  request_full.json     # payload Rails completo post-change (incluye legacy reservation_paid_out)
  request_legacy.json   # payload pre-change: sin amount_local ni timezone — DEBE seguir parseando
  response_rt.json      # respuesta real-time: SIN amount_usd_display (no hay tasas en memoria)
  response_batch.json   # entrada de critical_alerts batch: CON amount_usd_display (conversión SQL)
```

Cada fixture lleva `feature_frame_version: "frame-v1"` embebido — ese campo es el gate de versión: **cambiar el contrato sin bump de versión es violación de proceso**. Platform vendoriza copia byte-exacta en `packs/anomaly_detection/spec/fixtures/contract/frame_v1/` con `ORIGIN.md` (commit de origen + SHA-256 por archivo). Al modificar un fixture aquí: bump de `feature_frame_version`, re-vendorizar en platform y actualizar `ORIGIN.md`.

Nota: `.gitignore` cubre `*.json`; los fixtures están trackeados con `git add -f`. Si agregas un fixture nuevo, fuérzalo igual.

## Semántica de la respuesta

- El endpoint `/score` serializa con `response_model_exclude_none=True`: las llaves con `None` se **omiten** (no viajan como `null`). Consumidores deben tolerar llaves ausentes — el contract spec de platform lo fija.
- `amount_usd_display` es OPCIONAL: presente solo cuando existe conversión (batch, SQL). El path real-time lo omite; convertir USD en RT requeriría un artefacto versionado de tasas (change futuro, decisión humana 2026-07-09).
- La respuesta ecoa `amount_local` y `currency` tal como llegaron; `amount_local` = `reservation_paid_out` original, sin conversión ni suma de `technology_fee`/`tax`/`tip`.
- Precedencia de timezone para features temporales: `facility_time_zone_iana` del payload (validada con `ZoneInfo`) → `iana_tz` del artefacto de stats → `Etc/UTC`. Nivel observable en `frame_flags`: `timezone_from_artifact` (nivel 2), `timezone_missing` (nivel 3).

## Desviación monitoreada: features USD-relativas

El feature set frame-v1 usa magnitudes **USD-relativas** (`amount_usd / facility_mean`), no el marco moneda-local puro del plan original (`plans/payment-anomaly-detection.md` §5.1 / `docs/plan-normalizacion-marcos.md`). Desviación aceptada el 2026-07-09 (proposal, riesgo 1): equivalente bajo tipo de cambio estable; bajo devaluación rápida el ratio absorbe deriva cambiaria.

**Gate de monitoreo**: el shadow monitor (`scripts/shadow_monitor.py`, `compute_currency_breakdown`) emite por moneda `top5_amount_x_avg` (winsorizado p99.9), tasa de alertas y distribución de `fallback_level`; NIO/HNL/PKR obligatorias cuando tengan volumen; ninguna moneda con >1.000 pagos en ventana se agrega bajo "otros". **Si una moneda excede 4x**, queda señalada como excedida: el cierre correspondiente lleva warning documentado o se abre un change de reentrenamiento con features moneda-local — nunca cierre limpio silencioso.

## Artefactos acoplados

`facility_stats_v1.json` y `thresholds_segmented_v1.json` se adoptan como par: todo refresco corre la recalibración candidata SIEMPRE y produce reporte comparativo con veredicto `material_change` (umbral declarado en el reporte); adopción sin evidencia de procedencia (snapshot, query, conteos, hashes old/new) está bloqueada. Evidencia del último refresco: `output/reports/stats_refresh_20260709/`.
