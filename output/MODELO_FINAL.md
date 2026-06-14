# Modelo final — IF-40 para detección de pagos anómalos

Documento de entrega del modelo de producción. Versión 1.0 — 2026-06-13.

## Inventario de artifacts

```
output/models/
  isolation_forest_final.joblib    Modelo IF entrenado (200 árboles, max_samples=512, max_features=0.6)
  scaler_final.joblib              RobustScaler (5-95 IQR) ajustado en train
  final_feature_list.json          Lista ordenada de las 40 features esperadas

scripts/
  score_payment.py                 CLI + clase Python para scorear pagos
  validate_final_model.py          Entrena + valida estadísticamente
  eval_with_raw_features.py        Genera los enriched parquets desde raw

data/processed/
  train_features_enriched.parquet  3.1M filas × 40 features (Ene-Jun 2025)
  val_features_enriched.parquet    1.1M filas × 40 features (Jul-Ago 2025)
  test_features_enriched.parquet   2.5M filas × 40 features (Sep-Dic 2025)
```

## Uso operativo

### Scorear una transacción puntual

```bash
source venv/bin/activate
python scripts/score_payment.py --payment-id 12345678
# Output: {"id": 12345678, "score": 0.612}
```

### Scorear un lote y traer los top-K más anómalos

```bash
python scripts/score_payment.py \
  --batch data/processed/test_features_enriched.parquet \
  --top 1000 \
  --out alertas_top1000.csv
```

### Uso programático

```python
from scripts.score_payment import PaymentScorer
import pandas as pd

scorer = PaymentScorer()             # Carga modelo + scaler + features
df = pd.read_parquet("payments_with_features.parquet")
df_scored = scorer.score_frame(df)    # id, score, decile, is_top_1pct, is_top_5pct
alertas = df_scored[df_scored["is_top_1pct"] == 1]
```

## Feature pipeline esperado

El scorer asume que el DataFrame de entrada ya tiene las 40 features computadas con la lógica anti-leakage del proyecto. Para producir esas features desde raw:

```python
# A partir del parquet raw (mismo schema que data/processed/train_raw.parquet):
python scripts/eval_with_raw_features.py
# Produce *_features_enriched.parquet
```

Para scoring online (una transacción aislada), hay que computar:
1. Las 31 features originales del catálogo (con ventana de historia del usuario)
2. Las 7 interacciones (función pura de las 31)
3. Las 8 raw-derived que requieren contexto de pagos previos (last 35d del usuario):
   - `is_third_party_payment`: from `effective_user_id != user_id`
   - `same_amount_count_1h/24h`: pagos previos del usuario con mismo amount en ventana
   - `gateway_change_recent`: compare con último gateway
   - `capture_delay_seconds`: from `captured_at - created_at`
   - `is_main_gateway`: precomputar mode por usuario (cron mensual)
   - `is_first_gateway_for_user`: cumulative seen check
   - `source_change_recent`: compare con último source

## Validación estadística (resumen)

Ver `output/results_validation_final.json` para CIs completos.

**Multi-seed (5 semillas), test, IF-40:**

| Proxy | AUC promedio | std | Estabilidad |
|---|---:|---:|---:|
| unified | 0.588 | 0.002 | Trivial |
| tipo_a | 0.489 | 0.003 | Trivial |
| extended | 0.747 | 0.002 | Trivial |
| pure_fraud | 0.843 | 0.009 | Estable |

**Bootstrap 95% CI (1000 iter, seed=42):** ver JSON.

**HE1 — Mann-Whitney U:** ver JSON. Esperado: p<0.001 en proxies con señal.

**HE4 — IF vs LOF vs OC-SVM:** IF gana en unified, extended y pure_fraud. Empate aleatorio en tipo_a (todos ≈0.5).

## Interpretación del score para negocio

| Score percentil | Volumen test | Esperado en pure_fraud | Acción sugerida |
|---|---:|---:|---|
| Top 1% (≈25k txns/mes) | 1 cada 3 | 11.3× sobre base | Revisión manual obligatoria |
| Top 5% (≈125k txns/mes) | 1 cada 5.4 | 6.1× sobre base | Revisión automatizada / flag |
| Top 10% (≈250k txns/mes) | 1 cada 11 | 3.3× sobre base | Monitoreo |
| Resto | base 3% | 1.0× | Sin acción |

Frecuencia de revisión y umbral exacto se ajustan según la capacidad operativa de TechSport.

## Limitaciones conocidas

1. **El modelo NO detecta reembolsos operativos** (AUC vs Tipo A = 0.49). Si el caso de uso es "reducir refunds", este modelo no es apropiado — refunds aquí son mayoritariamente operativos, no fraude.
2. **`pure_fraud` es una operacionalización**, no un ground truth. Las transacciones flageadas requieren revisión humana para confirmar fraude.
3. El modelo es **estable temporalmente en 2025** (validado Sep-Dic). Reentrenar trimestralmente si la base de usuarios o el mix de gateways cambia significativamente.

## Reentrenamiento

```bash
# 1. Extracción (~30 min): trae nuevos pagos desde ClickHouse
python run_pipeline.py --step 1

# 2-3. Feature engineering + preprocesamiento (~10 min)
python run_pipeline.py --from-step 2

# 4. Re-entrenamiento del modelo final con validación
python scripts/eval_with_raw_features.py
python scripts/validate_final_model.py
```
