# A3 — Riesgos, Decisiones Cerradas y Checklist por Fase

> Documento transversal consolidado (CODEX + CLAUDE).
> Ultima actualizacion: 2026-03-16.

---

## 1. Decisiones Cerradas

Estas decisiones estan **congeladas**. No se reabren salvo evidencia critica documentada.

| # | Decision |
|---|----------|
| D1 | El universo se extrae con `FINAL` de `pbp_productionDB_optimized.payments`. |
| D2 | El proxy principal es el **unificado** (OR de 5 tipos: A reembolso, B circuito credito, C descuento anomalo, D velocidad extrema, E gratuitas sistematicas); el Tipo A individual y el amplio se usan para analisis de sensibilidad. |
| D3 | El modelo principal es **Isolation Forest**. |
| D4 | La comparacion obligatoria es contra **LOF** y **One-Class SVM**. |
| D5 | El resultado principal debe contrastarse con variante **sin `user_reversal_ratio_30d`** (30 features). |
| D6 | `contamination` se incluye en el grid search (240 combos: 4x4x5x3). |
| D7 | Feature 15 usa `cumulative_nunique_shifted` O(n), no `expanding().apply(nunique)`. |
| D8 | Feature 18 usa `first_txn` del training set, no del split actual. |
| D9 | Los scores siguen la convencion: `-decision_function(X)` (mayor = mas anomalo). |
| D10 | Se aplica correccion **Holm-Bonferroni** a los p-values de HE1-HE4. |
| D11 | Epsilon en ratios es `1e-8` (no `0.01`). |
| D12 | Warm history (Dic 2024) es obligatoria para features con ventana retrospectiva. |

---

## 2. Riesgos Criticos

### R1 — Snapshot incorrecto

| Campo | Detalle |
|-------|---------|
| **Impacto** | Todo el estudio queda invalido. |
| **Mitigacion** | `verify_counts.py`, `dataset_manifest.json`, query canonica congelada. |

### R2 — Data leakage

| Campo | Detalle |
|-------|---------|
| **Impacto** | Metricas artificiales, invalidez de la defensa. |
| **Mitigacion** | Tests de leakage feature por feature, ablacion obligatoria Feature #18, validacion temporal estricta. |

### R3 — Sobrecarga computacional

| Campo | Detalle |
|-------|---------|
| **Impacto** | Retraso en entrenamiento y evaluacion. |
| **Mitigacion** | Cache parquet local, extraccion por split, subsampling OC-SVM (100K), checkpoint grid search. |

### R4 — Sobreingenieria

| Campo | Detalle |
|-------|---------|
| **Impacto** | Perder tiempo en piezas ajenas a la tesis. |
| **Mitigacion** | No API, no realtime, no modelos extra, no dashboards. Solo pipeline offline. |

### R5 — Circularidad Feature #18

| Campo | Detalle |
|-------|---------|
| **Impacto** | Metricas infladas por correlacion mecanica con proxy. |
| **Mitigacion** | Sensitivity analysis obligatoria. Si delta AUC >= 0.02, reportar modelo 30-features como primario. |

### R6 — Escalabilidad OC-SVM

| Campo | Detalle |
|-------|---------|
| **Impacto** | Complejidad O(n^2-n^3) prohibitiva. |
| **Mitigacion** | Subsampling fijo 100K, documentar como baseline comparativo. |

### R7 — SHAP inestable o costoso

| Campo | Detalle |
|-------|---------|
| **Impacto** | Bloqueo del analisis de interpretabilidad. |
| **Mitigacion** | `TreeExplainer` primero, `KernelExplainer` fallback, subsample 5K. Si falla, usar `permutation_importance`. |

### R8 — Drift del origen de datos

| Campo | Detalle |
|-------|---------|
| **Impacto** | Conteos inconsistentes entre corridas. |
| **Mitigacion** | Congelar snapshot parquet, registrar fecha de extraccion, trabajar sobre artefacto local. |

### R9 — Variabilidad entre seeds

| Campo | Detalle |
|-------|---------|
| **Impacto** | Resultados no reproducibles. |
| **Mitigacion** | Multi-seed (42, 52, 62) o justificacion explicita de seed unica. |

### R10 — Normalización multi-moneda

| Campo | Detalle |
|-------|---------|
| **Impacto** | Features monetarias inconsistentes si las tasas de cambio son incorrectas o incompletas. |
| **Mitigacion** | Usar una unica fuente `rate_to_usd` mensual/snapshot; si la fuente original viene como `conversion_rate` con USD base, convertirla internamente a `rate_to_usd`; registrar moneda original y tasa aplicada como columnas de auditoria; spot-check manual de conversiones. |

### R11 — Lenguaje causal en la tesis

| Campo | Detalle |
|-------|---------|
| **Impacto** | Incoherencia con alcance correlacional. |
| **Mitigacion** | Revision sistematica. Eliminar "predice", "causa", "efecto". Usar "asociacion", "capacidad discriminativa", "relacion". |

---

## 3. Checklist por Fase

### Fase 0 — Contrato

- [ ] Contrato tesis-codigo escrito
- [ ] Config limpia sin conceptos supervisados
- [ ] SQL canonico congelado
- [ ] Catalogo de 31 features fijado

### Fase 1 — Datos

- [ ] Snapshot parquet creado (3 splits)
- [ ] Warm history creado (Dic 2024)
- [ ] Conteos validados (+-1% del objetivo)
- [ ] Normalización monetaria aplicada (USD vía `rate_to_usd`)
- [ ] Manifest generado
- [ ] `is_fraud` eliminado del flujo

### Fase 2 — EDA

- [ ] EDA reproducible en notebook
- [ ] Tablas Cap 2 exportadas
- [ ] Figuras Cap 2 exportadas

### Fase 3 — Features

- [ ] 31 features implementadas y documentadas
- [ ] Variante 30 features implementada
- [ ] Tests de leakage pasando
- [ ] Bordes train/val/test validados
- [ ] Cold-start manejado

### Fase 4 — Preprocesamiento

- [ ] Matrices train/val/test congeladas
- [ ] Scaler y metadata guardados
- [ ] `float32` verificado

### Fase 5 — Modelado

- [ ] IF tuneado en validation
- [ ] LOF entrenado (con grid search)
- [ ] OC-SVM entrenado con subsample
- [ ] Scores de test guardados (parquet con `id`, `created_at`)
- [ ] Seeds documentadas

### Fase 6 — Evaluacion

- [ ] HE1 respondida (Mann-Whitney + r_rb)
- [ ] HE2 respondida (AUC + AP)
- [ ] HE3 respondida (EF + top-k)
- [ ] HE4 respondida (comparacion 3 modelos)
- [ ] Bootstrap CI ejecutado
- [ ] Estabilidad temporal evaluada
- [ ] Holm-Bonferroni aplicada

### Fase 7 — Sensibilidad

- [ ] Proxy estricto vs amplio evaluado
- [ ] Feature #18 sensibilidad (delta AUC documentado)
- [ ] SHAP ejecutado (top 10 features)
- [ ] Per-status evaluation completada
- [ ] Sanity baselines verificados

### Fase 8 — Reporting

- [ ] Todas las tablas LaTeX generadas
- [ ] Todas las figuras PDF/PNG generadas
- [ ] Tablas compilan con `pdflatex`
- [ ] Caracteres especiales escapados

### Fase 9 — Orquestador

- [ ] `run_pipeline.py` funcional end-to-end
- [ ] `--step`, `--from-step`, `--fast`, `--dry-run` funcionan
- [ ] Prerequisite validation entre pasos
- [ ] Resumen final impreso

### Fase 10 — Tests, Cleanup e Integracion

- [ ] `pytest tests/ -v` pasa
- [ ] Test de integracion pasa (sin ClickHouse)
- [ ] Archivos obsoletos eliminados
- [ ] `CLAUDE.md` actualizado
- [ ] `.gitignore` cubre outputs
- [ ] Tesis-LaTeX compila con resultados reales
- [ ] 52 marcadores `[POR COMPLETAR]` llenados
- [ ] Conclusiones reflejan evidencia real
- [ ] `requirements-lock.txt` generado

---

## 4. Checklist Final de Tesis

- [ ] OG respondido con evidencia
- [ ] OE1 respondido (Cap 1: marco teorico)
- [ ] OE2 respondido (Cap 2: diagnostico)
- [ ] OE3 respondido (Cap 3: HE1/HE2/HE3)
- [ ] OE4 respondido (Cap 3: HE4)
- [ ] Limitaciones redactadas
- [ ] Trabajo futuro delimitado
- [ ] Defensa apoyada por artefactos reproducibles
- [ ] Lenguaje correlacional verificado (sin causal)
- [ ] APA 7 verificado
- [ ] Compilacion limpia: `pdflatex` -> `biber` -> `pdflatex` -> `pdflatex`
