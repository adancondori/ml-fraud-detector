# A8 Alineacion con Tesis-Latex

## Riesgo

Si el documento academico mantiene a Isolation Forest como sistema principal, pero la implementacion v4 reporta HGB supervisado como resultado central, la defensa puede objetar un cambio de pregunta de investigacion.

## Regla

Actualizar Tesis-Latex en paralelo a la implementacion tecnica. No cerrar conclusiones con el marco anterior.

## Dos lineas obligatorias

### Linea A: evaluacion academica original

- Objeto: Isolation Forest no supervisado.
- Uso: evaluar capacidad discriminativa frente a Tipo A.
- Resultado: AUC aproximado 0.576 sobre Tipo A; capacidad insuficiente para el objetivo operativo.
- Funcion en tesis: resultado negativo documentado y baseline academico.

### Linea B: extension operativa v4

- Objeto: ranker supervisado HGB-CLEAN.
- Uso: ordenar pagos por riesgo de reembolso Tipo A.
- Resultado Gate A0: AUC Oct-Dic 0.8339 y P@1% 73.4%, sujeto a confirmacion con pipeline final USD + catalogo v4.
- Funcion en tesis: propuesta tecnica de mejora operacional, no sustitucion silenciosa de la hipotesis original.

## Texto recomendado

```text
La evaluacion del sistema no supervisado basado en Isolation Forest mostro capacidad discriminativa insuficiente frente al proxy Tipo A. Por ello, se propone y evalua una extension operativa supervisada, V4-CLEAN, orientada al ranking de riesgo de reembolso transaccional. Esta extension no afirma fraude confirmado; estima riesgo respecto de un proxy observable.
```

## Secciones a actualizar

- Problema general.
- Hipotesis general.
- Objetivo general.
- Objetivos especificos.
- Metodologia.
- Resultados.
- Conclusiones.

## Gate

No entregar version final de tesis si:

- el titulo o problema prometen "Isolation Forest como sistema principal" pero los resultados concluyen con HGB;
- se usa "fraude" sin validacion humana A7;
- se reporta Gate A0 como resultado final en vez de pipeline final.
