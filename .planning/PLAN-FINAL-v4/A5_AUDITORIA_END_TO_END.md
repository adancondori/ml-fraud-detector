# A5 Auditoria End-to-End v4

## Criterio de suficiencia

v4 esta completa si:

1. El pipeline final corre end-to-end.
2. Test final Oct-Dic supera AUC >= 0.70.
3. Variante sin historial de refund supera AUC >= 0.75.
4. No hay leakage conocido en features principales.
5. Se generan tablas y figuras.
6. Scorer single reproduce batch.

## Auditoria de artefactos

Revisar existencia:

```text
data/processed/v4/
output/v4/features/feature_catalog.csv
output/v4/models/risk_ranker_hgb.joblib
output/v4/evaluation/metrics_final.json
output/v4/sensitivity/ablation_metrics.csv
output/v4/tables/
output/v4/figures/
```

## Auditoria de metricas

Validar:

- AUC.
- AP/base.
- P@1%.
- P@5%.
- Recall@k.
- EF@k.
- CI 95%.
- estabilidad mensual.

## Auditoria de lenguaje

Buscar y corregir frases como:

- "predice fraude".
- "fraude confirmado".
- "causa".

Usar:

- "discrimina reembolsos".
- "ranking de riesgo".
- "asociacion".
- "proxy Tipo A".

## Cierre

Solo marcar v4 como cerrada cuando todos los gates esten en PASS o exista una justificacion documentada.
