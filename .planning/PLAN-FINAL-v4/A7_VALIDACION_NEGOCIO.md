# A7 Validacion de Negocio del Proxy Tipo A

## Problema

Tipo A mide reembolso, no fraude confirmado. Aunque HGB-CLEAN discrimine Tipo A con AUC alto, eso no responde por si solo que fraccion de los casos corresponde a fraude, cancelacion legitima o error administrativo.

## Objetivo

Estimar la mezcla operacional del proxy Tipo A mediante revision humana ciega.

## Muestra minima

| Muestra | N | Proposito |
|---|---:|---|
| Tipo A top score | 100 | Ver si el modelo concentra casos sospechosos |
| Tipo A aleatorio | 100 | Estimar mezcla base del proxy |
| No Tipo A top score | 100 | Revisar falsos positivos o anomalias no reembolsadas |

## Revisores

Opcion minima:

- 1 revisor con conocimiento operativo del negocio.
- Tiempo estimado: 2-3 horas por bloque de 100 casos; 6-9 horas total.

Opcion recomendada para defensa:

- 2 revisores independientes.
- Calcular acuerdo interevaluador con Cohen's kappa.
- Resolver desacuerdos por consenso y reportar tanto acuerdo inicial como etiqueta final.

## Etiquetas manuales

- Fraude/sospecha operacional.
- Cancelacion legitima.
- Error administrativo.
- Abuso de politica/reembolso recurrente.
- Indeterminado.

Guia minima:

- Fraude/sospecha operacional: indicios de identidad falsa, tarjeta ajena/robada, patron de abuso deliberado, comportamiento coordinado o evasion de politica.
- Cancelacion legitima: cambio de plan, clima, disponibilidad, error razonable del cliente o cancelacion dentro de reglas normales.
- Error administrativo: correccion de cobro, ajuste de staff, duplicado operativo o reversa por proceso interno.
- Abuso de politica/reembolso recurrente: conducta repetida que no prueba fraude, pero si riesgo operacional.
- Indeterminado: evidencia insuficiente.

## Protocolo

1. Ocultar score y etiqueta del modelo al revisor.
2. Revisar solo informacion disponible operacionalmente.
3. Registrar razon corta y categoria.
4. Calcular distribucion porcentual por muestra.
5. Si hay dos revisores, calcular Cohen's kappa antes de consenso.

## Resultado esperado

El modelo solo puede presentarse como herramienta antifraude si la muestra top score contiene mayor proporcion de fraude/sospecha operacional que la muestra Tipo A aleatoria.

Si no se ejecuta esta muestra, la narrativa queda limitada a:

```text
ranking de riesgo de reembolso Tipo A
```

No usar:

```text
detector de fraude confirmado
```
