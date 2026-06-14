# 01 Contrato de Alcance v4

## Alcance

Implementar y evaluar un sistema de ranking de riesgo de reembolso transaccional para pagos de la gestion 2025.

## Fuera de alcance

- Confirmar fraude real.
- Entrenar con chargebacks/disputes, porque no se encontraron tablas disponibles con esas etiquetas.
- Usar datos fuera de 2025.
- Usar septiembre 2025 como test final si fue usado para descubrimiento.

## Variable objetivo

`y_tipo_a = status IN ('totally_refunded', 'refunded_to_credit')`

Uso permitido:

- Entrenamiento del ranker supervisado v4.
- Evaluacion de baselines.
- Target encoding calculado solo con train, pero unicamente en sensibilidad/techo operativo.

Uso prohibido:

- Feature directa.
- Calculo de columnas que miren el estado futuro de la misma transaccion.
- Seleccion de thresholds sobre test final.
- Target encoding dentro del resultado principal V4-CLEAN.
- Columnas mutables como feature principal sin snapshot temporal: `updated_at`, `last_change_at`, `deleted_at`, `total_paid_out`, `most_recent_date`, `approval_status`, `status_enum`, `payment_id` de `reservations_users`.

## Nomenclatura

| Termino | Significado |
|---|---|
| Reembolso Tipo A | Proxy operacional de riesgo |
| Anomalia | Transaccion con score alto del modelo |
| Fraude | No se afirma sin etiqueta confirmada |
| Ranker | Modelo que ordena pagos por riesgo |
| Precision top-k | Porcentaje de reembolsos dentro del top de alertas |

## Cambio frente a tesis original

La tesis original se apoyaba en modelos no supervisados. v4 debe documentarse como una extension tecnica necesaria porque el objetivo operativo exige discriminacion sobre un proxy definido. La comparacion academica se conserva con IF/LOF/OC-SVM.

La narrativa academica debe quedar en dos lineas separadas:

- Linea A: IF no supervisado como sistema original evaluado; resultado negativo o insuficiente sobre Tipo A.
- Linea B: ranker supervisado V4-CLEAN como extension operativa para riesgo de reembolso.

No presentar HGB como si fuera la misma hipotesis original de Isolation Forest.

## Criterio de exito

Exito minimo:

- AUC >= 0.70 en octubre-diciembre 2025.

Exito deseado:

- AUC >= 0.80.
- P@1% >= 40%.
- AP/base >= 3.

El benchmark Gate A0 cuenta como evidencia de viabilidad. El exito de tesis exige pipeline final reproducible con USD normalizado y catalogo v4 completo.

## Criterio de fracaso

La v4 no se acepta si:

- No existe benchmark HGB reproducible versionado.
- El AUC final cae por debajo de 0.70.
- V4-CLEAN cae por debajo de 0.75 AUC.
- Hay leakage no corregido en features principales.
- El pipeline no reproduce los resultados desde cero.
- V4-CLEAN queda por debajo de SIMPLE-RULE en AUC y P@1% simultaneamente.
- Sin validacion humana A7, las conclusiones usan "riesgo de reembolso", nunca "fraude".
