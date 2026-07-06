---
phase: 04-shadow-dual-run-y-validacion-de-sesgo
verified: 2026-07-06T18:35:00Z
status: passed
score: 6/6 must-haves verified
re_verification: false
---

# Phase 04: Shadow Dual-Run & Validación de Sesgo — Verification Report

**Phase Goal:** Dual-run (champion + frame-v1) persiste 2 filas por pago; monitoreo shadow reporta reducción de sesgo; gate go/no-go cuantitativo con evaluación real diferida (PENDING_DATA).
**Verified:** 2026-07-06T18:35:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Cada pago produce 2 filas (shadow_old/shadow_new) con `model_version` distinto y dedup token prefijado `shadow-old-`/`shadow-new-` | VERIFIED | `_score_all_dual` + `_insert_chunks_dual` en `scorer/batch/scorer.py`; tokens `shadow-old-{cursor}-IF40v1-chunk-{i}` y `shadow-new-{cursor}-framev1-chunk-{i}` (lines 700/704) |
| 2 | Fallo parcial de un modelo no impide la escritura del otro | VERIFIED | `ShadowDualRunner.score_pair` envuelve cada `scorer.score()` en try/except independiente; devuelve `_error_result` sin lanzar; `_insert_chunks_dual` tiene try/except independiente por chunk INSERT (lines 709-744) |
| 3 | `_INSERT_COLUMNS` = 26 columnas (3 frame-v1 con DEFAULT '') | VERIFIED | Runtime: `len(_INSERT_COLUMNS) == 26`; últimas 3: `calibration_segment`, `fallback_level`, `frame_flags`; modo active pasa `""` para esas 3 columnas |
| 4 | `shadow_monitor.py` computa las 4 métricas SHAD-02 con winsorización robusta p99.9 | VERIFIED | `compute_top5_bias` usa `np.percentile(amounts, 99.9)` + `np.clip`; `compute_jaccard_at_k`, `compute_alert_rate_by_segment`, `compute_off_hours` presentes; 386 líneas |
| 5 | `shadow_gate.py` con guard `INSUFFICIENT_DATA` (exit 2) si `days_span<14 OR n_rows<500` y 4 criterios SHAD-03 | VERIFIED | `evaluate_gate` verifica `days_span < MIN_SHADOW_DAYS or n_rows < MIN_SHADOW_ROWS` como primera lógica de negocio; exit codes 0/1/2 presentes; 310 líneas |
| 6 | Path IF-40 (active mode) y guardrail `assert_write_target_is_safe` intactos | VERIFIED | `scorer_shadow=None` → `_score_all` / `_insert_chunks` (single-model path); `assert_write_target_is_safe` es la primera llamada en `_insert_chunks` (line 514) y en `_insert_chunks_dual` (line 679); 63 tests pasan incluyendo `test_if40_artifacts.py` |

**Score:** 6/6 truths verified

---

### Required Artifacts

| Artifact | Lines | Status | Details |
|----------|-------|--------|---------|
| `docker/clickhouse/migrations/02_anomaly_scores_frame_v1.sql` | 4 | VERIFIED | `ADD COLUMN IF NOT EXISTS` para `calibration_segment`, `fallback_level`, `frame_flags` con `DEFAULT ''`; idempotente |
| `scorer/shadow/dual_runner.py` | 113 | VERIFIED | `ShadowDualRunner.score_pair` con aislamiento de fallo parcial; nunca lanza; delta logging; 16 tests verdes |
| `scorer/artifact_loader.py` | — | VERIFIED | `_load_metadata(metadata_filename)` override presente; legacy fallbacks condicionados a `filename == "model_metadata.json"` (line 101) |
| `scorer/batch/scorer.py` | 745 | VERIFIED | `_INSERT_COLUMNS` = 26; `scorer_shadow=None` default; `_score_all_dual` + `_insert_chunks_dual`; tokens `shadow-old-`/`shadow-new-`; guardrail primera llamada en ambos INSERT paths |
| `scorer/main.py` | — | VERIFIED | `ScorerSettings` con `scoring_mode`, `shadow_champion_metadata`, `shadow_new_metadata`; lifespan carga dual cuando `scoring_mode='shadow_dual'` con assert de `model_version` |
| `scripts/shadow_monitor.py` | 386 | VERIFIED | 5 funciones: `load_shadow_df`, `compute_alert_rate_by_segment`, `compute_top5_bias` (wins p99.9), `compute_off_hours`, `compute_jaccard_at_k`; nunca toca READ prod |
| `scripts/shadow_gate.py` | 310 | VERIFIED | `evaluate_gate` con guard `INSUFFICIENT_DATA` como primera lógica; `compute_spearman` con scipy (NaN si <30 pares); exit 0/1/2; reutiliza funciones de shadow_monitor |
| `tests/test_shadow_dual_runner.py` | 611 | VERIFIED | 16 tests: aislamiento parcial, 2 filas/pago, 26 columnas, dedup tokens, guardrail |
| `tests/test_shadow_monitor.py` | 365 | VERIFIED | 17 tests: winsorizado < raw en heavy-tail, Jaccard overlap, alert rate por segmento |
| `tests/test_shadow_gate.py` | 386 | VERIFIED | 14 tests: INSUFFICIENT_DATA con `days_span<14`, con `n_rows<500`, con df vacío; PASS/FAIL sintético; Spearman NaN con <30 pares |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scorer/main.py` lifespan | `load_artifacts(model_dir, metadata_filename='model_metadata_frame_v1.json')` | carga explícita del challenger | WIRED | lines 118-158 en main.py; assert `scorer_new._model_version == 'frame-v1'` |
| `scorer/batch/scorer.py _insert_chunks_dual` | ClickHouse insert con `insert_deduplication_token` | tokens `shadow-old-*` / `shadow-new-*` distintos | WIRED | tokens construidos en lines 700/704; pasan a `settings={"insert_deduplication_token": token_old/token_new}` |
| `ShadowDualRunner.score_pair` | `scorer_champion.score` y `scorer_new.score` con el mismo `UserContext` | contexto factual compartido, sin duplicar queries | WIRED | `runner.score_pair(payment, context, context)` en `_score_all_dual` line 589; comentario explícito en docstring |
| `scripts/shadow_gate.py evaluate_gate` | guard `days_span<14 or n_rows<500` → INSUFFICIENT_DATA | separación temporal código/evaluación | WIRED | `if days_span < MIN_SHADOW_DAYS or n_rows < MIN_SHADOW_ROWS` como primera instrucción condicional de evaluate_gate |
| `scripts/shadow_monitor.py compute_top5_bias` | `np.percentile(amounts, 99.9)` + `np.clip` | winsorización robusta | WIRED | lines 151-162 de shadow_monitor.py |
| `scripts/shadow_gate.py` | `scipy.stats.spearmanr(pct_old, pct_new)` | correlación de ranking | WIRED | `from scipy.stats import spearmanr` lazy import en `compute_spearman`; `rho, _ = spearmanr(pct_old, pct_new)` |

---

### Test Suite Results

```
63 passed in 7.40s
  tests/test_shadow_dual_runner.py  ........ (16 tests)
  tests/test_shadow_monitor.py      ......... (17 tests)
  tests/test_shadow_gate.py         .............. (14 tests)
  tests/test_batch_scorer.py        .............. (14 tests)
  tests/test_if40_artifacts.py      .. (2 tests)
```

Todos los tests pasan sin datos shadow reales ni ClickHouse activo.

---

### Anti-Patterns Found

Ninguno relevante. No hay TODOs, placeholders ni implementaciones vacías en los archivos de la fase.

---

### Checkpoint PENDING_DATA — Evaluación diferida documentada

SHAD-03 (`shadow_gate.py`) se trata como **cubierto a nivel de infraestructura**. El código del gate existe, es sustantivo (310 líneas), tiene tests (14 tests incluyendo el invariante INSUFFICIENT_DATA), y el guard temporal/volumen funciona correctamente. El veredicto go/no-go **real** está intencionalmente diferido: requiere ≥14 días de datos shadow y ≥500 filas en ClickHouse local. Este estado PENDING_DATA está documentado formalmente en `04-02-SUMMARY.md` con el runbook operativo completo.

Correr `python scripts/shadow_gate.py` hoy imprimirá `INSUFFICIENT_DATA` (exit 2) si Docker/ClickHouse está activo con pocos datos — eso es el comportamiento CORRECTO de la infraestructura, no un fallo.

---

### Gaps Summary

No hay gaps. Los 6 must-haves están verificados con evidencia en código real:

1. DDL idempotente con 3 columnas frame-v1 — archivo SQL de 4 líneas, todas `ADD COLUMN IF NOT EXISTS`.
2. `ShadowDualRunner.score_pair` — 113 líneas, try/except independiente por modelo, nunca lanza.
3. `BatchScorer` modo dual — `_INSERT_COLUMNS` = 26 (verificado en runtime), 2 filas/pago, tokens prefijados distintos, guardrail como primera llamada en ambos INSERT paths.
4. `shadow_monitor.py` — 4 métricas SHAD-02 con winsorización p99.9 real.
5. `shadow_gate.py` — guard INSUFFICIENT_DATA, 4 criterios SHAD-03, exit codes 0/1/2.
6. Retrocompat IF-40 y modo active — `scorer_shadow=None` preserva el path existente de 26 columnas con `""` para las 3 frame-v1.

---

*Verified: 2026-07-06T18:35:00Z*
*Verifier: Claude (so-verifier)*
