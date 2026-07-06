---
status: diagnosed
trigger: "Fase 1 top-5% amount ratio 7.17x en val vs 2.14x en experimento (disjoint30_frames). Localizar por que produccion diverge."
created: 2026-07-06
updated: 2026-07-06
---

## Current Focus

hypothesis: La divergencia NO es features ni fallback de currency; es (1) sensibilidad de la metrica a outliers de cola pesada + (2) muestreo 1/20 del experimento + (3) ventana.
test: reproducir en VAL/TEST completo y mod20 con mismo modelo; analizar top-5%.
expecting: metricas robustas ~1-2x; el 7x dominado por pocas txns de amount>1M.
next_action: entregar diagnostico (modo find_root_cause_only).

## Symptoms

expected: top-5% amount ratio < 4x en retrain frame-v1 (experimento probo 2.14x)
actual: 7.17x en val set (retrain_frame_v1.py)
errors: ninguno; es divergencia de metrica
reproduction: correr retrain_frame_v1.py / medir val_amount[top5].mean()/val_amount.mean()
started: al mover de experimento (TEST mod20) a produccion (VAL completo)

## Eliminated

- hypothesis: H-B (fallback de currency para 1289/1876 facilities reintroduce escala)
  evidence: diag_fmean_divergence.py -> config A (artifact currency-fallback) = 7.046x, config B (stats propias por facility sin fallback) = 7.046x. delta = 0.000x. Cambiar el denominador fmean NO mueve el ratio.
  timestamp: 2026-07-06

- hypothesis: H-C (lista de features distinta / amount_fac_z)
  evidence: frame_version(DISJOINT30) del experimento == FRAME_V1_FEATURE_NAMES exactamente (set diff vacio en ambos sentidos). NINGUNO contiene amount_fac_z. Aritmetica de marco identica.
  timestamp: 2026-07-06

## Evidence

- checked: composicion de features experimento vs prod
  found: identicas (30 features), ambas sin amount_fac_z
  implication: no es H-C

- checked: fmean artifact vs stats propias en VAL completo (mismo modelo)
  found: 7.046x en ambos, delta 0.000x
  implication: no es H-B; el fallback de currency es irrelevante para esta metrica

- checked: mismo modelo/features medido en VAL_full, TEST_full, VAL_mod20, TEST_mod20
  found: VAL_full=7.046x, TEST_full=0.993x, VAL_mod20=0.723x, TEST_mod20=2.038x
  implication: (a) el experimento (TEST mod20=2.04x) reproduce su 2.14x, pero es una muestra 1/20; (b) TEST completo da 0.99x, no 2x -> el 2.14x es artefacto de muestreo; (c) la ventana importa (VAL vs TEST) pero la varianza por muestreo domina

- checked: distribucion de amount
  found: VAL max=100M, top-0.01% (113 txns)=40.75% del monto total; TEST max=10M
  implication: cola extremadamente pesada; media no robusta

- checked: que domina el top-5% de VAL
  found: 2 txns con amount>1M (suma 110M) = 88.4% del monto del top-5%; ratio_medianas=0.62x; ratio winsorizado p99.9=1.56x
  implication: el 7.05x lo causan 2 outliers. Metrica de razon-de-medias sobre cola pesada es inestable.

## Resolution

root_cause: La metrica top5_amount[top5].mean()/amount.mean() es una razon de medias sobre una distribucion de amount de cola extremadamente pesada (VAL max=100M). En VAL, 2 transacciones de amount>1M que el IF coloca en el top-5% aportan el 88.4% del monto del top-5%, inflando el ratio a 7.05x. NO es la ventana (H-A), NO es el fallback de currency (H-B), NI la lista de features (H-C): con features/modelo/ventana identicos, la metrica salta 0.72x (VAL mod20) -> 7.05x (VAL full) por inclusion/exclusion de un puñado de outliers. El 2.14x del experimento fue TEST muestreado 1/20 (TEST completo = 0.99x); es un valor de muestreo, no un objetivo poblacional real.

fix: (a) metrica robusta: usar amount winsorizado a p99.9 antes del ratio (VAL=1.56x) o ratio de medianas; (b) sanear el outlier de 100M en amount (probable error de captura / moneda no normalizada); (c) NO comparar contra 2.14x de muestra 1/20 -> recalibrar el gate contra TEST completo.

verification: pendiente (modo diagnostico, no se aplica fix)

files_changed: []
