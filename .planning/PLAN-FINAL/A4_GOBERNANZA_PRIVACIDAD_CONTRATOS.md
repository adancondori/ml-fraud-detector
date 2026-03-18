# A4 — Gobernanza, Privacidad y Contratos

## Proposito

Definir reglas de uso de datos productivos, privacidad de outputs, contratos de entrada/salida y límites de interpretación.

## 1. Datos sensibles

Alta sensibilidad:

- `user_id`
- `effective_user_id`
- cualquier nombre de facility si el documento sale del ámbito interno

Sensibilidad media:

- `gateway`
- `currency`
- `paid_by_manager`

Baja sensibilidad:

- métricas agregadas
- percentiles
- curvas y figuras sin identificadores

## 2. Reglas de privacidad

- no incluir ids crudos de usuarios en tablas públicas;
- no incluir nombres personales;
- si la tesis sale del ámbito interno, pseudonimizar actores y facilities;
- los outputs públicos deben ser agregados o pseudonimizados.

## 3. Política de interpretación

Permitido:

- "transacciones con intervención de manager"
- "actores/usuarios asociados"
- "concentración de descuentos en pagos anómalos"
- "señales compatibles con abuso operativo"

Prohibido:

- afirmar fraude individual;
- afirmar culpabilidad;
- atribuir descuentos a una persona si la identidad del actor no está validada;
- usar el modelo como prueba disciplinaria.

## 4. Contrato de datos de entrada

Cada snapshot debe declarar:

- columnas;
- tipos;
- nullability;
- filtros aplicados;
- query exacta;
- fecha de extracción.

Archivo esperado:

- `output/manifests/input_schema.json`

## 5. Contrato de artefactos

### `results.json`

Debe contener:

- versión de snapshot;
- seed;
- schema hash;
- HE1-HE4;
- bootstrap CI;
- sensibilidad;
- comparación de modelos.

### `results_posthoc.json`

Debe contener:

- `actor_identity_validated: true|false`
- nivel de anonimización
- disclaimer metodológico
- agregados post-hoc exportables

### Tablas `.tex`

Deben referenciar en comentario o manifest:

- run id
- fecha
- fuente (`results.json` o `results_posthoc.json`)

## 6. Gate de gobernanza

No cerrar la tesis si:

- no está decidido si el documento es interno o público;
- los outputs post-hoc exponen personas o centros sin política definida;
- no existe `actor_identity_validated` en el post-hoc.
