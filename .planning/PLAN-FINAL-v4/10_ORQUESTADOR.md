# 10 Orquestador v4

## Objetivo

Agregar una ejecucion reproducible v4 sin romper el pipeline historico.

Estado actual: `run_pipeline_v4.py` todavia no esta implementado. Gate A0 vive en `scripts/run_hgb_benchmark.py` y solo prueba viabilidad.

## CLI propuesta

```bash
python run_pipeline_v4.py benchmark --variant clean-strict
python run_pipeline_v4.py benchmark --baseline simple-rule
python run_pipeline_v4.py extract
python run_pipeline_v4.py eda
python run_pipeline_v4.py features
python run_pipeline_v4.py preprocess
python run_pipeline_v4.py train
python run_pipeline_v4.py evaluate
python run_pipeline_v4.py sensitivity
python run_pipeline_v4.py report
python run_pipeline_v4.py all
```

## Orden

0. `benchmark`: reproduce HGB-CLEAN strict y SIMPLE-RULE antes de tocar pipeline.
1. `extract`: crea snapshot 2025.
2. `eda`: genera lift y leakage candidates.
3. `features`: genera features v4.
4. `preprocess`: fit encoders train-only.
5. `train`: entrena HGB y baselines.
6. `evaluate`: calcula metricas final.
7. `sensitivity`: corre ablations.
8. `report`: genera tablas/figuras.

## Manifiestos

Cada paso debe escribir:

- timestamp.
- git commit si existe.
- input files.
- output files.
- row counts.
- feature list.
- config hash.

## Resume

Permitir:

- `--force` para recomputar.
- `--from-step`.
- `--only-split`.
- `--fast` para benchmark rapido.
- `--final` para corrida completa.
- `--variant clean-strict|clean-no-ru|legacy-31`.
- `--baseline simple-rule|if-v4|lof-v4|ocsvm-v4`.

## Gate

`all --final` debe reproducir metricas dentro de tolerancia:

- AUC +/- 0.005.
- AP +/- 0.005.
- conteos exactos por split.
- USD normalizado aplicado en features monetarias.
- catalogo v4 completo versionado.

Ademas debe bloquear la corrida si:

- `reservations_users` entra a V4-CLEAN sin condicion point-in-time.
- Falta `simple_rule_baseline.json`.
- Falta AUC mensual Oct/Nov/Dic.
- Falta `business_proxy_validation` antes de afirmar fraude real.
- Se intenta publicar resultados de tesis usando solo Gate A0.
