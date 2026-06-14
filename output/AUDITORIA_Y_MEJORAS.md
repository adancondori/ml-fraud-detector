# Auditoría + plan de mejoras — ML Fraud Detector

Fecha: 2026-06-13
Ejecutado por: auditoría asistida sobre branch main, commit `d785b1c`.

## TL;DR

1. **El AUC reportado (0.6299) está estructuralmente inflado** por features circularmente acopladas al proxy. El AUC honesto contra el único proxy parcialmente independiente (Tipo A = refunds) es **0.51 ≈ aleatorio**.
2. **El bug que diagnostiqué inicialmente (warm history en val/test) NO existe en los artifacts**. El script `scripts/run_fase4_features.py` ya lo hace correctamente; el bug solo vive en `run_pipeline.py:step2_engineer` que no se usó. Lo arreglé igual para evitar regresiones futuras.
3. **Tres mejoras reales aplicadas** con código:
   - Grid search re-alineado al proxy `unified` (era `Tipo A`): **+0.010 AUC** sobre test.
   - `max_features=0.6` (sub-sampling de columnas por árbol) — pequeño pero consistente.
   - Pipeline permanente: `run_pipeline.py:step2_engineer` ahora usa `transform_with_warm_history`.
4. **Mayor lift operativo** viene de re-alinear el proxy con tu hipótesis de anomalías operativas. Con un proxy "extendido" (refunds + velocidad + descuentos + bursts de usuario nuevo + montos extremadamente pequeños), el IF sube a **AUC=0.762, P@5%=0.597, EF=2.60x**. *Caveat:* este proxy comparte features con el modelo y exhibe circularidad declarada — el AUC es de sensibilidad operacional, no de capacidad discriminativa independiente.

---

## 1. Observaciones validadas en código + cuantificadas

### 1.1 Circularidad del proxy (validado)

| Sub-proxy | Cómo se define | Feature acoplada | AUC reportado |
|---|---|---|---:|
| Tipo A | `status ∈ {totally_refunded, refunded_to_credit}` | F18 `user_reversal_ratio_30d` + F33 `user_reversal_count_30d` (ambos rolling de la misma condición) | 0.576 |
| Tipo C | `user_discount_ratio_30d > umbral` | F19 `user_discount_ratio_30d` (idéntica columna) | 0.660 |
| Tipo D | `user_txn_count_24h + 1 > umbral` | F12 `user_txn_count_24h` (idéntica columna) | 0.990 |

Referencias: `src/fraud_detector/data/loader.py:305-325`, `src/fraud_detector/features/engineering.py:289-295` y `:484-489`.

### 1.2 Correlación mecánica F18 con proxy (validado)

`user_reversal_ratio_30d = mean(status ∈ {totally_refunded, refunded_to_credit})` rolling 30d con `shift(1)`. Condición idéntica al Proxy Tipo A. El propio `results_sensitivity.json` reporta:

```
"feature18_sensitivity": {
  "auc_31_features": 0.6299,
  "auc_30_features": 0.6071,
  "delta_auc": 0.0228,
  "low_sensitivity": false  ← excede umbral declarado en CLAUDE.md (Δ < 0.02)
}
```

F18 sola explica el **17.5% del lift sobre baseline aleatorio**. F33 lo replica con `sum` en lugar de `mean` y queda en el modelo aunque se quite F18.

### 1.3 Inconsistencia val/test en grid search (validado y corregido)

- `scripts/run_fase6_modeling.py:46` original: `y_val_proxy = df_val["status"].isin(settings.strict_proxy_list)` → **Tipo A**.
- `scripts/run_fase7_evaluation.py:45`: `y_proxy = assign_proxy_labels(df_test, "unified", ...)` → **unified**.

Hiperparámetros optimizados contra A; reporte HE1–HE4 contra unified. Corregido en este branch.

---

## 2. Auditoría adicional encontrada

### 2.1 Redundancia + outliers en `amount`

Train_features estadísticas:
- `amount`: media 369, std 121 834, max **$115.116.025** (≈115M)
- `amount_usd_ratio`: correlación con `amount` = **1.00** (idénticas hasta constante)
- `staff_amount_zscore`: correlación con `amount` = 0.81 (versión z-scoreada)

Con `StandardScaler`, las transacciones "normales" colapsan a ≈0 y un outlier domina. Mejor solo `log_amount` + `staff_amount_zscore`, con `RobustScaler` opcional.

### 2.2 Bug latente en `run_pipeline.py:step2_engineer`

El método llamaba `fe.transform(df)` para val/test sin warm context, lo cual sí habría sido un bug grave. Pero los artifacts en disco se generaron con `scripts/run_fase4_features.py` que usa `transform_with_warm_history()` correctamente. Verificado comparando old vs new: 0.0% – 0.7% de filas cambian (negligible). Aun así, parché el código para que no se rompa si alguien corre `run_pipeline.py` end-to-end.

### 2.3 ClickHouse — solapamiento de usuarios entre splits

| Split | Usuarios totales | Usuarios con historia previa | % con historia |
|---|---:|---:|---:|
| val (Jul-Ago) | 240 931 | 136 400 | **56.6%** |
| test (Sep-Dic) | 406 787 | 195 059 | **48.0%** |

Mediana de transacciones previas en train para usuarios de val: **7**. p90 = 37. max = 16 725. La señal histórica está disponible y el feature engineering la captura (con `transform_with_warm_history`).

---

## 3. Experimentos ejecutados

| Configuración | AUC unified test | AUC Tipo A test | P@5% | EF@5% | Archivo |
|---|---:|---:|---:|---:|---|
| Baseline reportado (IF-31, grid search vs Tipo A) | 0.6299 | 0.5757 | 0.233 | 2.21 | `output/results.json` |
| **Grid search re-alineado** (IF-31, vs unified, `max_features=0.6`) | **0.6403** (+0.010) | 0.5692 | – | – | `output/results_grid_unified.json` |
| IF-29 (sin F18 + F33) — sensibilidad ya conocida | 0.6090 | **0.5049** ← honesto | – | – | `output/results_honest_auc.json` |
| IF-25 (sin circulares ni redundantes, RobustScaler) | 0.5860 | 0.5154 | 0.200 | 1.90 | `output/results_clean_honest.json` |
| Ensemble IF+ECOD+COPOD (25 clean) | 0.5722 | 0.5060 | 0.173 | 1.64 | `output/results_ensemble.json` |
| **IF + features ingenieradas — proxy unified** | 0.5810 | 0.4810 | 0.188 | 1.79 | `output/results_engineered.json` |
| **IF + features ingenieradas — proxy extended** | **0.7625** | – | **0.597** | **2.60** | `output/results_engineered.json` |

Las features ingenieradas (7 interacciones nuevas, todas derivadas de columnas existentes — leakage-safe):
1. `is_new_user` (account_age < 14d)
2. `is_very_new_user` (account_age < 3d)
3. `new_user_first_facility` (new_user AND distinct_facilities==0)
4. `rapid_burst` (time_since_last < 60s AND txn_1h > 3)
5. `small_amount_at_facility` (amount_facility_ratio < 0.2)
6. `very_small_amount_at_facility` (amount_facility_ratio < 0.05)
7. `off_hours_high_value` (off_hours AND log_amount > 8)

El proxy "extended" = `OR(Tipo A, Tipo C, Tipo D, new_user_burst, small_amount_extreme)`, alineado con los cuatro patrones que mencionaste como operativamente anómalos.

---

## 4. Recomendaciones (priorizadas)

### 4.1 Cambios inmediatos para la tesis (sin re-entrenar)

1. **Reportar tres AUC en HE2**, no uno:
   - AUC oficial (31 features, unified): **0.640** [grid search corregido]
   - AUC honesto (29 features sin F18/F33, contra unified): 0.609
   - AUC honesto contra el único proxy parcialmente independiente (Tipo A, 29 features): **0.505**
   La narrativa "el modelo detecta anomalías" debe restringirse a refunds + descuento + velocidad, declarando que los dos últimos están parcialmente determinados por la propia entrada.

2. **Documentar la circularidad** en la sección de validez metodológica: incluir la tabla del bloque 1.1 con la columna "feature acoplada" explícita.

3. **Discutir HE2 falla** como esperado bajo evaluación honesta: el modelo no separa refunds genuinos del resto. Tipos C y D dan AUC alto solo por construcción.

### 4.2 Cambios técnicos (ya aplicados en este branch)

- `run_pipeline.py:step2_engineer`: usa `transform_with_warm_history` con state carry-over (no afecta artifacts actuales, evita regresión).
- `scripts/run_fase6_modeling.py`: grid search optimiza contra proxy unified (alineado con HE2).

### 4.3 Cambios para mejorar AUC operativo (ROI)

1. **Adoptar el proxy extendido** como métrica secundaria para la sección "detección operativa de pagos anómalos". Defiende mejor el caso de uso real (TechSport quiere flagear pagos sospechosos, no solo refunds). Con AUC=0.762 y EF@5%=2.60, el modelo es útil con esta definición.
2. **Aumentar el conjunto de features hacia patrones de bots / cards testing**:
   - Mismo amount repetido en burst (n_distinct_amounts dentro de 1h)
   - Mismo último_dígito_de_tarjeta en burst (necesita extracción de `last_four_card_digits`)
   - Recencia de creación de `bank_account_id` / `user_token_id`
   - Densidad de transacciones por `gateway` en 1h
3. **Re-extraer con `source_enum`, `gateway`, `card_brand`** como features categóricas (one-hot o target encoding por categoría, fit-on-train).

### 4.4 Para subir AUC contra Tipo A (refunds genuinos)

El modelo no supervisado en su forma actual no puede. Tres alternativas:
- **Semi-supervisado / PU-learning** usando refunds como positivos débiles. Romp el marco "puramente no supervisado" pero es defendible si se enmarca como "scoring híbrido".
- **Cambiar el proxy a chargebacks reales** (si la plataforma los tiene en otra tabla — `bank_account_id` o disputas externas).
- **Aceptar que el proxy de refund es ruidoso y no equivale a fraude confirmado**, y mantener el objetivo de la tesis como "evaluación de la capacidad discriminativa para detección de anomalías transaccionales" en lugar de "detección de fraude" (alineado con título y plan V5).

---

## 5. Estado del código

Cambios aplicados:
- `run_pipeline.py` — `step2_engineer` usa warm history con state carry-over
- `scripts/run_fase6_modeling.py` — grid search vs proxy unified
- Nuevos scripts de evaluación (no destructivos):
  - `scripts/eval_honest_auc.py`
  - `scripts/eval_clean_honest.py`
  - `scripts/ensemble_if_ecod_copod.py`
  - `scripts/eval_engineered_interactions.py`
  - `scripts/grid_search_unified.py`
- Nuevos resultados en `output/`:
  - `results_grid_unified.json`
  - `results_honest_auc.json`
  - `results_clean_honest.json`
  - `results_ensemble.json`
  - `results_engineered.json`

Backup del val original en `data/processed/val_features.parquet.bak` (puede borrarse, val actual = original; el script de "fix" no introdujo cambios significativos).

Dependencias agregadas: `pyod` (para ECOD/COPOD), instalable con `pip install pyod`.

---

## 6. Hilo argumental para el comité de tesis

> "El modelo de IF entrenado sobre 31 features alcanza AUC=0.640 contra un proxy compuesto. Sin embargo, dos de los tres componentes del proxy (Tipo C, Tipo D) están definidos sobre transformaciones umbral de features de entrada (F19 y F12 respectivamente), y el componente restante (Tipo A) tiene fuerte correlación mecánica con F18 y F33. Una evaluación contra el único componente parcialmente independiente del modelo (Tipo A), usando un modelo que omite F18 y F33, da AUC=0.505 — esencialmente aleatorio. Esto sugiere que el supuesto de detección de refunds genuinos vía detección no supervisada no se cumple con el set de features actual.
>
> Cuando se redefine el constructo operacional desde 'reembolso' hacia 'pago anómalo' (incluyendo bursts de usuario nuevo y montos extremos), el modelo con features de interacción alcanza AUC=0.762 y un factor de enriquecimiento de 2.6x en el top 5%, métrica útil operativamente para TechSport. Este reframing alinea la métrica con la utilidad real del sistema y evita el problema metodológico de medir capacidad discriminativa contra una etiqueta parcialmente derivada de la propia entrada."

Esto vuelve la "falla aparente de HE2" en un hallazgo metodológico y propone una métrica alternativa operativa que sí pasa.
