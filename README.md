# ML Fraud Detector

Pipeline offline de **detección de pagos anómalos** para la tesis. El sistema es **no supervisado**: entrena modelos de anomalía sobre transacciones históricas y usa los estados de reembolso solo como **proxy de evaluación**, nunca como etiqueta de entrenamiento.

## Qué hace

- Extrae snapshots temporales desde ClickHouse con SQL canónico y `FINAL`
- Normaliza montos multi-moneda a USD antes del feature engineering
- Genera el catálogo oficial de **31 features**
- Escala features con `StandardScaler` fit-on-train
- Entrena modelos no supervisados:
  - `IsolationForest`
  - `LocalOutlierFactor` (`novelty=True`)
  - `OneClassSVM`
- Evalúa con métricas orientadas a scores anómalos:
  - `AUC-ROC`
  - `Average Precision`
  - `Precision@k`
  - `Enrichment Factor`

## Estructura relevante

```text
ml-fraud-detector/
├── config/config.py
├── src/fraud_detector/
│   ├── data/loader.py
│   ├── features/engineering.py
│   ├── features/preprocessor.py
│   ├── models/trainer.py
│   ├── evaluation/metrics.py
│   └── utils/currency.py
├── tests/
└── .planning/PLAN-FINAL/
```

## Uso programático

```python
from config.config import settings
from fraud_detector.data.loader import DataManager
from fraud_detector.features.engineering import FeatureEngineer
from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
from fraud_detector.models.trainer import ModelTrainer

dm = DataManager(settings)
train_df = dm.load_split("train")
val_df = dm.load_split("val")

engineer = FeatureEngineer()
train_features = engineer.fit_transform(train_df)
val_features = engineer.transform_with_warm_history(
    val_df,
    train_df,
    method_state=engineer.get_feature_state(),
)

preprocessor = UnsupervisedPreprocessor(variant="full")
X_train = preprocessor.fit_transform(train_features)
X_val = preprocessor.transform(val_features)

trainer = ModelTrainer(model_type="isolation_forest")
trainer.fit(X_train)
scores_val = trainer.score_samples(X_val)
```

## Tests

```bash
./venv/bin/pytest -q
```

La suite actual valida:

- configuración del pipeline
- extracción y normalización monetaria
- feature engineering de 31 features
- preprocesamiento
- entrenamiento no supervisado
- métricas de evaluación

## Servicio de scoring (FastAPI)

El paquete `scorer/` expone un servicio FastAPI (`scorer.main:app`) con endpoints
`/api/v1/health`, `/api/v1/model/info`, `/api/v1/model/reload`, `/api/v1/score` y
`/api/v1/score/batch`.

### Separación READ (prod) / WRITE (local)

El batch scoring usa **dos clientes ClickHouse independientes** para que ninguna
escritura toque producción:

- **READ** (`CLICKHOUSE_*`): cliente read-only de producción. Resuelve el cursor,
  hace el fetch de `payments` y construye el contexto del batch (6 queries).
- **WRITE** (`ANOMALY_SCORES_CH_*`): cliente local donde se insertan los
  `anomaly_scores`. El destino del INSERT lo define `ANOMALY_SCORES_TABLE`.

Antes de cada INSERT, un **guardrail** (`assert_write_target_is_safe`) aborta si:

1. el fingerprint WRITE (`host, port, database, secure, user`) coincide con el de
   READ — evita escribir sobre el servidor de producción; o
2. el host WRITE no es local (`clickhouse`, `localhost`, `127.0.0.1`) y no se
   activó el bypass explícito `ALLOW_NONLOCAL_ANOMALY_SCORE_WRITES=true`.

`/api/v1/health` reporta `clickhouse_connected = read_ok AND write_ok`.

Variables de entorno: ver `.env.example` (READ, WRITE, tabla y bypass). Para la
corrida local, READ apunta a producción read-only y WRITE al ClickHouse del
`docker-compose.yml`. El flujo end-to-end reproducible se documenta en
`docs/HOWTO-anomaly-local.md`.

## Fuente de verdad

El contrato metodológico y operativo vive en:

- `.planning/PLAN-FINAL/01_CONTRATO_ALCANCE.md`
- `.planning/PLAN-FINAL/04_FEATURE_ENGINEERING.md`
- `.planning/PLAN-FINAL/05_PREPROCESAMIENTO.md`
- `.planning/PLAN-FINAL/06_MODELADO_TUNING.md`
