# A2 Protocolo Runbook v4

## Preparacion

```bash
source venv/bin/activate
```

Verificar:

- credenciales ClickHouse.
- espacio en disco.
- branch limpio o cambios documentados.

Estado actual:

- Gate A0 se ejecuta con `scripts/run_hgb_benchmark.py`.
- `run_pipeline_v4.py` todavia no existe; los comandos siguientes son el objetivo de implementacion del orquestador.

## Gate A0 actual

```bash
python scripts/run_hgb_benchmark.py --variant clean-strict --baseline simple-rule
python scripts/run_hgb_benchmark.py --variant clean-no-ru --baseline simple-rule
```

Uso:

- probar viabilidad;
- generar `output/v4/benchmarks/`;
- comparar HGB strict, HGB sin participantes y SIMPLE-RULE.

No usar Gate A0 como resultado final de tesis.

## Corrida rapida

```bash
python run_pipeline_v4.py all --fast
```

Uso:

- validar que no se rompio el pipeline.
- obtener benchmark preliminar.

## Corrida final

```bash
python run_pipeline_v4.py all --final
```

Uso:

- generar resultados oficiales.
- producir artefactos para reporting.

## Orden manual si falla

```bash
python run_pipeline_v4.py extract
python run_pipeline_v4.py features
python run_pipeline_v4.py preprocess
python run_pipeline_v4.py train
python run_pipeline_v4.py evaluate
python run_pipeline_v4.py sensitivity
python run_pipeline_v4.py report
```

## Revision despues de cada corrida

1. Revisar `output/v4/**/manifest.json`.
2. Verificar conteos por split.
3. Revisar `leakage_audit_candidates.csv`.
4. Revisar metricas validation antes de test.
5. Revisar test final Oct-Dic.

## Cierre

Guardar:

- metricas.
- modelos.
- encoders.
- feature catalog.
- tablas.
- figuras.

No sobrescribir resultados finales sin versionar carpeta o manifest.
