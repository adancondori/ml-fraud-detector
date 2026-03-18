# Anexo A1: ETL, Lineage y Edge Cases

Documento transversal que define las capas de transformación de datos, reglas de
idempotencia, manejo de warm history y los 20 edge cases que deben contemplarse
durante toda la implementación.

---

## Capas ETL

| Capa | Nombre         | Descripción                                            | Artefactos                          |
|------|----------------|--------------------------------------------------------|-------------------------------------|
| 0    | Source          | ClickHouse (producción, réplica de lectura)            | —                                   |
| 1    | Raw extracted   | Parquets deduplicados con `FINAL`, split por período   | `data/raw/*.parquet`                |
| 2    | Canonical       | Tipos sanitizados, proxies calculados                  | `data/processed/*_raw.parquet`      |
| 3    | Feature         | 20 features + 19 variantes                             | `data/processed/*_features.parquet` |
| 4    | Model input     | Matrices escaladas                                     | `output/scores/*.npy`               |
| 5    | Results         | Scores, métricas, tablas, figuras, manifests           | `output/**`                         |

---

## Reglas de idempotencia

1. **No sobreescribir sin `--force`**: si el artefacto de salida ya existe, el paso se salta con warning a menos que se use `--force`
2. **Escritura atómica**: escribir a archivo `.tmp` y luego renombrar (`os.rename`) para evitar artefactos corruptos por interrupciones
3. **Registrar checksums y row counts**: cada artefacto registra su hash MD5 y cantidad de filas en el manifest correspondiente
4. **Mantener manifests** con: versión del snapshot, seed, hash del schema de features

---

## Reanudación ante fallos (resume on failure)

- Reanudar por split o por mes, no reiniciar desde cero
- No recomenzar si existen artefactos válidos de pasos anteriores
- **Verificar integridad antes de reusar**: checksum + row count match contra manifest
- El orquestador (`run_pipeline.py --from-step N`) facilita la reanudación

---

## Warm history (historia previa)

### Features que requieren warm history

| Feature                       | Ventana   | Razón                                           |
|-------------------------------|-----------|--------------------------------------------------|
| `user_txn_count_1h`           | 1 hora    | Conteo de transacciones recientes del usuario    |
| `user_txn_count_24h`          | 24 horas  | Velocidad transaccional diaria                   |
| `user_amount_24h`             | 24 horas  | Monto acumulado diario                           |
| `user_distinct_facilities`    | histórico | Diversidad de establecimientos                   |
| `user_reversal_ratio_30d`     | 30 días   | Tasa de reversiones recientes                    |

### Reglas de warm history

- Las filas de warm history **NUNCA** entran en las métricas finales
- Se usan exclusivamente para calcular features de ventana temporal al inicio de cada split
- **Account age**: usa `first_txn` del training set como referencia
  - **Documentar limitación de left-censoring**: usuarios que existían antes del período de extracción tendrán edad subestimada

---

## Validación por capa

### Capa 1 — Raw

- Row count coincide con query `SELECT count()`
- Rango de fechas correcto por split
- Conteos por status consistentes
- IDs únicos (sin duplicados post-`FINAL`)

### Capa 2 — Canonical

- Proxies asignados correctamente (strict/wide)
- Dominios y tipos de transacción válidos
- Filas descartadas documentadas con razón
- Tipos de datos correctos (downcasting aplicado)

### Capa 3 — Feature

- Tasa de nulls = 0% (todas las features completas)
- Sin inf ni NaN
- Rangos dentro de límites razonables (documentar outliers legítimos)
- Schema freeze: nombres y orden de columnas coinciden con `FEATURE_NAMES`

### Capa 5 — Results

- Todas las tablas referencian el mismo `run_manifest`
- `results.json` contiene todas las claves esperadas
- Figuras generadas con tamaño > 0

---

## Manifests obligatorios

| Manifest                      | Contenido                                                          |
|-------------------------------|--------------------------------------------------------------------|
| `dataset_manifest.json`       | Versión snapshot, fechas, row counts, checksums por split          |
| `feature_manifest.json`       | Lista de features, schema hash, estadísticas de entrenamiento      |
| `run_manifest_*.json`         | Timestamp, git SHA, seed, hiperparámetros, duración, hostname      |

---

## 20 Edge Cases

### Datos de origen (Capas 0-1)

| #  | Edge Case                              | Manejo                                                           |
|----|----------------------------------------|------------------------------------------------------------------|
| 1  | Duplicados después de `FINAL`          | Deduplicar por `id`, documentar cantidad eliminada               |
| 2  | Status vacío                           | Mapear a categoría explícita `'unknown'`                         |
| 3  | Amount = 0                             | Proteger `log(0)` con `log1p()` y divisiones con denominador + ε |
| 4  | Montos negativos o discount > amount   | Registrar en log, decidir: clipear a 0 o excluir                 |
| 5  | Múltiples monedas                      | Verificar que `reservation_paid_out` está en USD; filtrar si no  |
| 6  | `captured_at` nulo                     | Fallback a `created_at`, registrar proporción de fallbacks       |

### Feature engineering (Capa 3)

| #  | Edge Case                              | Manejo                                                           |
|----|----------------------------------------|------------------------------------------------------------------|
| 7  | Usuarios sin historial previo          | Defaults neutrales (0 para conteos, promedio global para montos); contar cold-start |
| 8  | Facilities nuevas (no en train)        | Usar promedio global como fallback                               |
| 9  | Gateways raros (< N transacciones)     | Agrupar como `"other"` para reporting (no afecta features)       |

### Integridad temporal (Capas 1-3)

| #  | Edge Case                              | Manejo                                                           |
|----|----------------------------------------|------------------------------------------------------------------|
| 10 | `_peerdb_version` backfills            | Snapshot congelado, no re-mezclar datos después de extracción    |
| 11 | Frontera Train → Val                   | Verificar continuidad de warm history en la transición           |
| 12 | Frontera Val → Test                    | Misma verificación de continuidad                                |
| 13 | Inicio de año / inicio de datos        | Warm history con defaults documentados para primeras semanas     |

### Modelado (Capa 4)

| #  | Edge Case                              | Manejo                                                           |
|----|----------------------------------------|------------------------------------------------------------------|
| 14 | Todos los scores iguales               | **DETENER pipeline**, investigar causa raíz                      |
| 15 | LOF: problemas de memoria              | Reducir sample o `n_neighbors`                                   |
| 16 | OC-SVM: demasiado lento                | Mantener subsample fijo (documentar tamaño)                      |
| 17 | Alta variabilidad entre seeds          | Ejecutar múltiples seeds, considerar aumentar `n_estimators`     |

### Evaluación y reporting (Capa 5)

| #  | Edge Case                              | Manejo                                                           |
|----|----------------------------------------|------------------------------------------------------------------|
| 18 | Proxy desbalanceado en bootstrap       | Bootstrap estratificado para mantener proporción                 |
| 19 | Sobreescritura accidental de artefactos| Escribir por run con timestamp, promover a `latest/` explícitamente |
| 20 | Edición manual de tablas               | **PROHIBIDO** — siempre corregir el generador                    |

---

## Regla de parada (stop rule)

> Si cualquier edge case cambia el **universo de datos**, la **definición del proxy**,
> los **splits temporales** o el **catálogo de features** → **regresar a Fase 0**
> (Contrato y Alcance) y actualizar el contrato antes de continuar.
