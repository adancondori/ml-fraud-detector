# HOWTO — Corrida batch de detección de anomalías en LOCAL

End-to-end reproducible: leer `payments` desde ClickHouse de **producción**
(read-only), escribir `anomaly_scores` solo en ClickHouse **local**, y alertas
solo en MySQL **local** de `platform`. La UI `/anomaly-detection` lee ambos
locales.

> **Sin secretos en este documento.** Todas las credenciales van en archivos
> `.env` gitignored y se referencian con placeholders.

## Arquitectura

```
READ  -> ClickHouse PROD read-only   (CLICKHOUSE_*)        payments / users / facilities_users
WRITE -> ClickHouse LOCAL            (ANOMALY_SCORES_CH_*)  anomaly_scores
HTTP  -> platform BatchScoringService -> AlertManager -> MySQL LOCAL (anomaly_detection_*)
UI    -> ClickHouse LOCAL (scores + users) + MySQL LOCAL (alertas + facilities)
```

La separación READ/WRITE vive en el scorer (`scorer/batch/scorer.py`,
`scorer/main.py`). Un guardrail (`assert_write_target_is_safe`) aborta el INSERT
si el destino WRITE coincide con el fingerprint READ o si el host WRITE no es
local sin el bypass explícito `ALLOW_NONLOCAL_ANOMALY_SCORE_WRITES=true`.

## Prerrequisitos

- Docker en ejecución; red `pbc-network` creada:
  ```bash
  docker network create pbc-network || true
  ```
- `ml-fraud-detector/venv` con dependencias instaladas.
- Credenciales **read-only** de ClickHouse de producción (las provee el usuario).

## Paso 1 — Tests del scorer (preflight)

```bash
cd ml-fraud-detector
source venv/bin/activate
pytest -q tests/test_batch_scorer.py tests/test_api.py
# Esperado: 21 passed (incluye separación READ/WRITE + guardrail)
```

## Paso 2 — ClickHouse local

```bash
cd ml-fraud-detector
docker compose up -d clickhouse
```

DDL local en `docker/clickhouse/init/`:
- `01_create_database.sql` — crea `pbp_productionDB_optimized`
- `02_anomaly_scores.sql` — tabla `anomaly_scores` (18 columnas, alineada con `_INSERT_COLUMNS`)
- `04_users.sql` — tabla `users` mínima para cohortes de la UI

> **Nota (volumen preexistente):** si el volumen `clickhouse_data` ya existe, los
> scripts de init **no** se re-ejecutan. Verificar el esquema y, si está
> desactualizado, recrear las tablas manualmente (operación local, sin datos a
> preservar si `count()=0`):
>
> ```bash
> # Verificar columnas de anomaly_scores (deben ser 18)
> docker exec clickhouse clickhouse-client --query \
>   "SELECT count() FROM system.columns WHERE database='pbp_productionDB_optimized' AND table='anomaly_scores'"
>
> # Si difiere y la tabla está vacía, recrear desde el DDL actual:
> docker exec clickhouse clickhouse-client --query "DROP TABLE pbp_productionDB_optimized.anomaly_scores"
> docker exec -i clickhouse clickhouse-client --multiquery < docker/clickhouse/init/02_anomaly_scores.sql
> docker exec -i clickhouse clickhouse-client --multiquery < docker/clickhouse/init/04_users.sql
> ```

Validación:
```bash
docker exec clickhouse clickhouse-client --query "SELECT count() FROM pbp_productionDB_optimized.anomaly_scores"  # 0
docker exec clickhouse clickhouse-client --query "SELECT count() FROM pbp_productionDB_optimized.users"           # 0 hasta copiar
```

### Copiar `users` desde prod (solo SELECT prod + INSERT local)

La UI necesita `users` local para cohortes por IP/email. Copiar **solo** las
columnas mínimas y, para una prueba chica, solo los usuarios referenciados por
los pagos de la ventana (o un subconjunto acotado). Requiere usuario read-only
de prod. Ejemplo de patrón (ajustar el filtro de usuarios):

```bash
# Exportar desde PROD (read-only) e importar a LOCAL — nunca escribir en prod.
# <PROD_*> son placeholders; usar el .env gitignored.
docker exec clickhouse clickhouse-client --query \
  "INSERT INTO pbp_productionDB_optimized.users (id, email, current_sign_in_ip, created_at, _peerdb_is_deleted, _peerdb_version) FORMAT TabSeparated" \
  < usuarios_export.tsv
```

## Paso 3 — Configurar y levantar `ml-scorer`

`ml-fraud-detector/.env` (gitignored; ver `.env.example`):

```env
# READ (producción, read-only)
CLICKHOUSE_HOST=<prod-clickhouse-host>
CLICKHOUSE_PORT=8443
CLICKHOUSE_SECURE=true
CLICKHOUSE_USER=<prod-readonly-user>
CLICKHOUSE_PASSWORD=<prod-readonly-password>
CLICKHOUSE_DATABASE=pbp_productionDB_optimized

# WRITE (local)
ANOMALY_SCORES_CH_HOST=clickhouse
ANOMALY_SCORES_CH_PORT=8123
ANOMALY_SCORES_CH_SECURE=false
ANOMALY_SCORES_CH_USER=default
ANOMALY_SCORES_CH_PASSWORD=
ANOMALY_SCORES_CH_DATABASE=pbp_productionDB_optimized
ANOMALY_SCORES_TABLE=pbp_productionDB_optimized.anomaly_scores

# Modelo / scoring
MODEL_DIR=/app/output/models
SCORING_MODE=shadow
```

```bash
cd ml-fraud-detector
docker compose up -d clickhouse ml-scorer
curl -s http://localhost:8765/api/v1/health | jq
```

Esperado:
```json
{ "model_loaded": true, "clickhouse_connected": true, "model_version": "IF-40-v1", "last_batch_at": null }
```
`clickhouse_connected` es `read_ok AND write_ok`: si falla READ (prod) o WRITE
(local), devuelve `false`.

## Paso 4 — Configurar y levantar `platform`

```bash
cd platform
bundle install                      # si Bundler falla por card_connect, correr antes
docker compose up -d                # o el stack local equivalente (MySQL, etc.)
bundle exec rails db:migrate
```

Tablas MySQL locales esperadas: `anomaly_detection_alerts`,
`anomaly_detection_triage_actions`, `anomaly_detection_settings`.

Envs Rails:
```env
FRAUD_SCORER_URL=http://ml-scorer:8000          # dentro de la red docker
CLICKHOUSE_HOST=clickhouse                       # localhost si Rails corre en host
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=pbp_productionDB_optimized
ANOMALY_DETECTION_CLICKHOUSE_DB=pbp_productionDB_optimized
ANOMALY_DETECTION_CLICKHOUSE_TABLE=pbp_productionDB_optimized.anomaly_scores
```

Desactivar realtime (limitar a batch):
```bash
bundle exec rails runner "AnomalyDetection::Setting.set('realtime_scoring_enabled', false)"
```

Specs enfocadas:
```bash
bundle exec rspec \
  packs/anomaly_detection/spec/queries/anomaly_detection/transaction_query_spec.rb \
  packs/anomaly_detection/spec/queries/anomaly_detection/cohort_query_spec.rb \
  packs/anomaly_detection/spec/services/anomaly_detection/batch_scoring_service_spec.rb
```

## Paso 5 — Batch con ventana chica

```bash
cd platform
bundle exec rails runner "AnomalyDetection::Setting.set('batch_scoring_cursor', 2.hours.ago.iso8601)"
bundle exec rails runner "puts AnomalyDetection::BatchScoringService.new.call.to_json"
```

Validar la respuesta: `processed` coherente con la ventana, `scored <= processed`,
`critical_alerts` es arreglo, `next_cursor` avanza.

## Paso 6 — Validar escrituras locales y CERO escrituras prod

ClickHouse local:
```bash
docker exec clickhouse clickhouse-client --query \
  "SELECT count() FROM pbp_productionDB_optimized.anomaly_scores WHERE scored_at >= now() - INTERVAL 2 HOUR"
docker exec clickhouse clickhouse-client --query \
  "SELECT payment_id, user_id, facility_id, raw_score, percentile, risk_level, model_version FROM pbp_productionDB_optimized.anomaly_scores ORDER BY raw_score DESC LIMIT 20"
```

MySQL local:
```bash
bundle exec rails runner "puts AnomalyDetection::Alert.order(created_at: :desc).limit(10).pluck(:payment_id, :severity, :status).inspect"
bundle exec rails runner "puts AnomalyDetection::Setting.get('batch_scoring_cursor')"
```

Producción (cero escrituras):
- La credencial READ debe carecer de permiso INSERT/ALTER (rechazo por permisos).
- El guardrail del scorer aborta si el destino WRITE = fingerprint READ o si el
  host WRITE no es local.
- Verificación opcional por SELECT: comparar `count()` antes/después en prod para
  la misma ventana/modelo y confirmar que no cambió por esta corrida.

## Paso 7 — Validar UI local

Abrir `/anomaly-detection` con un usuario con permiso
`anomaly_detection.dashboard.view`. Validar KPIs, distribución de scores, tabla
de transacciones (desde `anomaly_scores` local), alertas (MySQL local), cohortes
por IP/email (si `users` local fue copiada) y top facilities (MySQL local).

## Resultado de la corrida (26-jun-2026)

Dos corridas con ventana chica (directa al scorer + vía Rails `BatchScoringService`):

| Métrica | Valor |
|---|---|
| Ventana de cursor | ~2 min cada corrida |
| Directa: `processed` / `scored` / `critical_alerts` | 33 / 33 / 3 |
| Rails: `processed` / `scored` / `critical_alerts` | 25 / 25 / 4 |
| `anomaly_scores` en ClickHouse local | 58 filas (7 `is_anomaly`/critical, 50 facilities) |
| Modelo / features / scoring_mode | IF-40-v1 / enriched-40 / shadow |
| Alertas en MySQL local | 6 (todas `critical`); cursor avanzó a `2026-06-26T18:57:36` |
| `health` | `{"model_loaded":true,"clickhouse_connected":true,"model_version":"IF-40-v1"}` |
| Dashboard (host→CH local) | KPI `anomalous_24h=7, affected_facilities=7, exposure_usd=119.0`; score-dist 3 buckets |
| Cero escrituras prod | credencial READ read-only + guardrail (fingerprint WRITE≠READ, host local) + todos los INSERT al CH local |
| Tests | ml-fraud-detector 181 passed; platform 49 (queries/services) + 11 (scorer client) passed |

### Hallazgos y fixes durante la integración

1. **DDL local desactualizado:** el volumen preexistente tenía `anomaly_scores` con 13 columnas; se recreó con el DDL de 18 (alineado a `_INSERT_COLUMNS`).
2. **`_FETCH_SQL` con columna inexistente:** usaba `payment_id`, pero la tabla `payments` de prod expone `id`. Fix: `SELECT id AS payment_id …` (con test de regresión).
3. **Timeout del cliente Rails:** `FraudScorerClient` tenía timeout fijo de 5 s; un batch real tarda más. Se hizo configurable vía `FRAUD_SCORER_TIMEOUT` (default 5; test añadido). Para el batch local usar `FRAUD_SCORER_TIMEOUT=600`.
4. **ACL de red de ClickHouse local:** la imagen oficial restringe el usuario `default` (sin password) a loopback. Se añadió `docker/clickhouse/users.d/allow-local-networks.xml` permitiendo loopback + redes privadas (RFC1918) para que el scorer (red docker) escriba.
5. **Docker Desktop (Mac) presenta el host con su IP pública** para conexiones host→puerto publicado, lo que rompía el acceso de Rails al CH local. Fix sin debilitar seguridad: publicar el puerto como `127.0.0.1:8123` (la conexión aparece como loopback y entra por el ACL existente).
6. **Rendimiento IF-40 (limitación conocida):** el modelo activo IF-40 usa contexto por-pago (~3.6 s/pago, ~8 queries a prod c/u), por lo que el batch escala linealmente con el tamaño de la ventana. Usar ventanas chicas. Un `BatchContextProvider` enriquecido para 40 features (bulk) sería el siguiente paso de optimización, fuera del alcance acotado de esta integración.

### Pendiente opcional

- **Cohortes por IP/email:** la tabla `users` local está vacía. Para habilitar `CohortQuery`/`CohortDetailQuery`, copiar (SELECT prod read-only → INSERT local) los `users` referenciados por los `anomaly_scores` (~50 usuarios), columnas `id, email, current_sign_in_ip, created_at, _peerdb_is_deleted, _peerdb_version`.

## Estado de implementación (26-jun-2026)

- **Tarea 0 (preflight):** OK. Baseline `pytest` 13 passed; suite completa 181 passed.
- **Tarea 1 (READ/WRITE + guardrail):** OK. Clientes separados en `lifespan`,
  deps `get_read_ch_client`/`get_write_ch_client`,
  `BatchScorer(read_ch_client, write_ch_client, anomaly_scores_table, …)`,
  guardrail `assert_write_target_is_safe`, `health = read_ok AND write_ok`.
- **Tarea 2 (ClickHouse local):** OK (datos opcionales pendientes).
  `anomaly_scores` recreada con 18 columnas; `users` creada (vacía; copia
  opcional). ACL de red + binding loopback resueltos.
- **Tarea 3 (scorer live):** OK. `health` verde (READ prod + WRITE local).
- **Tarea 4 (platform):** OK. Rails arranca en host, sin migraciones pendientes,
  tablas `anomaly_detection_*` presentes; realtime desactivado.
- **Tarea 5 (batch real):** OK. `BatchScoringService` con ventana chica produjo
  scores en CH local y alertas en MySQL local.
- **Tarea 6 (validación escrituras):** OK. 58 filas en CH local; 6 alertas en
  MySQL local; cero escrituras a prod (read-only + guardrail).
- **Tarea 7 (UI):** OK a nivel de datos. Queries del dashboard leen CH local
  (KPIs no-cero) y MySQL local. Render en navegador queda para validación
  manual del usuario; cohortes IP/email requieren copia opcional de `users`.
- **Tarea 8 (HOWTO):** este documento.
