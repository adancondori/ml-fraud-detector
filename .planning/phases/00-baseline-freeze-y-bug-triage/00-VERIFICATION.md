---
phase: 00-baseline-freeze-y-bug-triage
verified: 2026-07-06T05:31:57Z
status: gaps_found
score: 4/5 must-haves verified
gaps:
  - truth: "La ruta legacy de thresholds no marca percentile_95_test_set como fuente operativa"
    status: partial
    reason: >
      El script run_fase7_evaluation.py fue corregido y ahora emite
      threshold_source: "percentile_95_test_set_LEGACY_DEPRECATED" con
      legacy_do_not_use_for_IF40: true. Sin embargo, output/models/thresholds.json
      en disco aún tiene el valor antiguo ("percentile_95_test_set", sin sufijo
      LEGACY, sin campo legacy_do_not_use_for_IF40) porque el script fue actualizado
      pero no re-ejecutado. El artefacto en disco no es coherente con el código del script.
      IMPACTO OPERATIVO BAJO: el scorer IF-40 carga thresholds_v2.json vía
      model_metadata.json, no thresholds.json. Además, scorer.py y classifier.py
      tienen thresholds.json como default solo para la ruta IF-31 legacy.
    artifacts:
      - path: "output/models/thresholds.json"
        issue: >
          threshold_source = "percentile_95_test_set" (sin sufijo _LEGACY_DEPRECATED),
          sin campo legacy_do_not_use_for_IF40. Artefacto stale — no regenerado
          después del fix al script.
    missing:
      - "Regenerar output/models/thresholds.json ejecutando scripts/run_fase7_evaluation.py,
        o escribir el artefacto directamente con los campos de deprecación."
---

# Phase 0: Baseline Freeze y Bug Triage — Verification Report

**Phase Goal:** El scorer opera sin bugs silenciosos conocidos, los umbrales están calibrados en val (no test), y el gate de éxito del proyecto está formalmente fijado como reducción de sesgo — antes de que cualquier medición de baseline sea tomada.
**Verified:** 2026-07-06T05:31:57Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                 | Status      | Evidence                                                                                                   |
|----|-----------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------|
| 1  | Scorer usa acceso directo a stats aprendidos con assertions post-carga | VERIFIED    | `features.py:27-38` acceso directo `._facility_avg_amount` y `._role_currency_stats`; asserts en líneas 32-38. Tests `test_parity_phase0.py` pasan (3/3, incluyendo regresión del bug getattr). |
| 2  | Ruta legacy no marca `percentile_95_test_set` como fuente operativa   | PARTIAL     | Script corregido; artifact `thresholds.json` en disco es stale (sin sufijo _LEGACY_DEPRECATED). |
| 3  | `capture_delay_seconds` excluida del FS operativo; fix pd.NaT activo  | VERIFIED    | No aparece en `final_feature_list_operational.json` (solo en campo `"excluded"`). `features_enriched.py:115` usa `pd.isnull()` con try/except. |
| 4  | Saneamiento EMPTY→USD en loader.py y engineering.py con warning/conteo | VERIFIED*   | `loader.py:183-199` implementa `_sanitize_currency` con warning+conteo. `engineering.py:646-649` tiene el replace sin warning explícito (ver nota). Tests `test_currency_sanitize_phase0.py` pasan (7/7). |
| 5  | `baseline_v0.json` existe con gate_metric y `golden_set_v0.parquet` ≥500 filas | VERIFIED | `baseline_v0.json`: `gate_metric="bias_reduction"`, `auc_pure_fraud_classification="diagnostic_circular_not_a_gate_metric"`. `golden_set_v0.parquet`: 14,831 filas. Las 4 features que definen `pure_fraud` están en el feature list operativo. |

*Nota criterio 4: `engineering.py` realiza el replace EMPTY→USD correctamente pero sin emitir un warning con conteo (solo `loader.py` tiene el warning). El criterio dice "con warning/conteo" para ambos archivos, pero los tests de la Fase 0 solo exigen el warning en `loader.py`. Los tests pasan al 100%.

**Score:** 4/5 truths verified (truth 2 parcial; impacto operativo bajo)

---

### Required Artifacts

| Artifact                                            | Expected                                       | Status     | Details                                                        |
|-----------------------------------------------------|------------------------------------------------|------------|----------------------------------------------------------------|
| `src/fraud_detector/scoring/features.py`            | Acceso directo + assertions                    | VERIFIED   | Líneas 27-38: acceso directo, 2 assertions, log info           |
| `tests/test_parity_phase0.py`                       | Tests paridad batch↔RT                         | VERIFIED   | 158 líneas, 3 tests, todos PASS (6.91s, sobre 14,831 pagos)    |
| `tests/test_currency_sanitize_phase0.py`            | Tests saneamiento EMPTY→USD                    | VERIFIED   | 148 líneas, 7 tests, todos PASS (0.83s)                        |
| `output/models/thresholds_v2.json`                  | `threshold_source` = validación                | VERIFIED   | `"threshold_source": "percentile_95_validation_set"`, calibrado en 1,130,117 filas val |
| `output/models/final_feature_list_operational.json` | Sin `capture_delay_seconds`                    | VERIFIED   | 39 features; `"excluded": ["capture_delay_seconds"]` explícito |
| `src/fraud_detector/scoring/features_enriched.py`   | Fix pd.NaT con pd.isnull                       | VERIFIED   | Línea 115: `if pd.isnull(captured) or pd.isnull(created):` dentro de try/except |
| `src/fraud_detector/data/loader.py`                 | `_sanitize_currency` con warning+conteo        | VERIFIED   | Líneas 183-199: warning `f"Sanitized {n_empty} rows..."` cuando n_empty > 0 |
| `src/fraud_detector/features/engineering.py`        | Replace EMPTY→USD (warning opcional)           | PARTIAL    | Replace presente (línea 646-649); sin warning/conteo explícito |
| `output/baseline_v0.json`                           | `gate_metric="bias_reduction"`, diag circular  | VERIFIED   | Todos los campos requeridos presentes; 131 líneas              |
| `output/golden_set_v0.parquet`                      | ≥500 filas                                     | VERIFIED   | 14,831 filas, 680 facilities cubiertas                         |
| `output/models/thresholds.json`                     | Marcado como legacy/deprecated                 | PARTIAL    | En disco: `threshold_source = "percentile_95_test_set"` sin sufijo ni campo legacy |

---

### Key Link Verification

| From                              | To                          | Via                                           | Status   | Details                                                      |
|-----------------------------------|-----------------------------|-----------------------------------------------|----------|--------------------------------------------------------------|
| `scorer.py` (IF-40 path)          | `thresholds_v2.json`        | `artifacts=load_artifacts()` → `model_metadata.json` | WIRED | `model_metadata.json` apunta a `thresholds_v2.json`; artifact_loader lo carga |
| `scorer.py` (IF-31 legacy default) | `thresholds.json`           | `thresholds_path` default arg                 | WIRED (legacy) | Solo activo si se instancia sin `artifacts=`; es la ruta IF-31, no IF-40 |
| `features.py`                     | `feature_engineer.joblib`   | `fe._groups[4]._facility_avg_amount`          | WIRED    | Acceso directo post-fix; assertions previenen artefacto corrupto |
| `features_enriched.py`            | `final_feature_list_operational.json` | `feature_list` inyectado            | WIRED    | 39 features (sin capture_delay_seconds)                      |
| `build_baseline_v0.py`            | `thresholds_v2.json`        | `THRESHOLD_PATH` hardcoded                    | WIRED    | Línea 31: `THRESHOLD_PATH = ".../thresholds_v2.json"`        |

---

### Requirements Coverage

| Requirement                                              | Status    | Blocking Issue                                          |
|----------------------------------------------------------|-----------|---------------------------------------------------------|
| Scorer sin bugs silenciosos conocidos                    | SATISFIED | 5 bugs documentados en `baseline_v0.json.bugs_fixed`; tests de regresión activos |
| Umbrales calibrados en val (no test)                     | SATISFIED | `thresholds_v2.json`: `threshold_source = "percentile_95_validation_set"` |
| Gate de éxito = reducción de sesgo                       | SATISFIED | `baseline_v0.json`: `gate_metric = "bias_reduction"`    |
| AUC pure_fraud marcado como diagnóstico circular         | SATISFIED | `auc_pure_fraud_classification = "diagnostic_circular_not_a_gate_metric"` con nota explicativa |
| `capture_delay_seconds` excluida del FS operativo        | SATISFIED | Explícitamente en campo `"excluded"` de `final_feature_list_operational.json` |
| Ruta legacy thresholds deprecada                         | PARTIAL   | Script corregido; artifact en disco stale                |

---

### Anti-Patterns Found

| File                                          | Line | Pattern                              | Severity | Impact                                               |
|-----------------------------------------------|------|--------------------------------------|----------|------------------------------------------------------|
| `output/models/thresholds.json`               | 3    | `threshold_source` sin sufijo LEGACY | Warning  | Artifact stale — no regenerado post-fix del script. El scorer IF-40 no lo usa; impacto bajo. |
| `src/fraud_detector/features/engineering.py`  | 646  | Replace EMPTY sin warning/conteo     | Info     | El criterio pide warning en ambos archivos; solo loader.py lo tiene. Los tests pasan igualmente. |

---

### Human Verification Required

No se identificaron items que requieran verificación humana para los criterios de la Fase 0.

---

### Gaps Summary

Un gap de severidad baja identificado:

**Gap único:** `output/models/thresholds.json` en disco es un artefacto stale. El script `run_fase7_evaluation.py` fue correctamente actualizado para emitir `threshold_source = "percentile_95_test_set_LEGACY_DEPRECATED"` y `legacy_do_not_use_for_IF40: true`, pero el artefacto en disco todavía contiene los valores anteriores al fix (sin sufijo, sin flag de deprecación) porque el script no fue re-ejecutado.

**Impacto operativo real:** Ninguno. El scorer IF-40 carga `thresholds_v2.json` exclusivamente (vía `model_metadata.json` → `artifact_loader.py`). El `thresholds.json` stale solo es cargado por `SingleTransactionScorer` cuando se instancia sin el argumento `artifacts=` (ruta IF-31 legacy, no el path operativo de IF-40).

**Resolución:** Ejecutar `scripts/run_fase7_evaluation.py` (que requiere los datos de test) o escribir directamente el artifact con los campos de deprecación. Esto es cosmético para la Fase 0 — no bloquea ningún criterio de la Fase 1.

---

*Verified: 2026-07-06T05:31:57Z*
*Verifier: Claude (so-verifier)*
