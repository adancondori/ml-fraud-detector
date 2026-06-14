# 05 Preprocesamiento v4

## Objetivo

Crear matrices reproducibles para modelos supervisados y baselines sin contaminar validation, discovery ni test.

## Encoding principal limpio

El modelo principal usa frequency encoding train-only para evitar objeciones de circularidad por features derivadas de la etiqueta.

Fit:

- Solo train.

Transform:

- Val, discovery y test usan frecuencias congeladas.
- Categorias nuevas usan 0.

## Target encoding solo sensibilidad

Usar m-estimate smoothing:

```text
encoded = (sum_y_category + m * global_rate) / (count_category + m)
```

Parametros iniciales:

- `m=200` para categorias de alta cardinalidad.
- `m=100` para baja cardinalidad.

El target encoding no forma parte del resultado principal V4-CLEAN. Solo se usa en `V4-CLEAN-BOOSTED` para estimar techo operativo y debe reportarse separado.

## Frequency encoding

Agregar conteo/frecuencia train-only para categorias:

- `facility_id`
- `category`
- `source_enum`
- `payment_method`
- `gateway`
- `card_brand`
- `currency`

`user_role` no se codifica en V4-CLEAN hasta auditar el join a `users` y su mutabilidad. Si se usa, queda en sensibilidad.

## Escalado

HGB no requiere StandardScaler. Mantener scaler solo para:

- Logistic Regression baseline.
- OCSVM.
- LOF.

## Missing values

Reglas:

- Numericos: imputar 0 si representa ausencia historica; mediana train si representa valor desconocido.
- Categoricos: `"UNKNOWN"`.
- IDs sin match de dimension: features de dimension a defaults y flag `missing_*`.

## Artefactos

```text
output/v4/preprocessing/
├── target_encoder.joblib
├── frequency_encoder.joblib
├── imputation_manifest.json
├── feature_list_clean.json
├── feature_list_clean_boosted.json
├── feature_list_full_sensitivity.json
└── X_{train,val,discovery,test}.parquet
```

## Tests

- Encoders no ven val/test.
- Categoria desconocida no rompe transform.
- Mappings no contienen indices de test.
- No existe columna `status` dentro de `X`.
- Target encoder no se usa en `V4-CLEAN`.
