# Fase 2. Diagnostico y Capitulo 2

## Proposito

Generar toda la evidencia descriptiva del OE2 y dejar Capitulo 2 soportado por artefactos reproducibles.

## Preguntas del diagnostico

- Como se distribuyen las transacciones por status, canal, gateway y metodo de pago?
- Que diferencia al grupo proxy+ del grupo proxy-?
- Donde estan los montos extremos?
- Existen patrones temporales, de velocidad o de usuario que justifiquen el modelado?
- Que vacios tiene el sistema actual de deteccion?

## Analisis obligatorios

### Distribucion estructural

- status;
- `source_enum`;
- `payment_method`;
- `gateway`;
- mes;
- facility si aporta valor y no infla demasiado el capitulo.

### Montos

- promedio, mediana, P95, P99;
- histogramas o transformacion log;
- comparacion `captured` vs `totally_refunded` vs `refunded_to_credit` vs `partially_refunded`.

### Proxy

- tasa base estricta y amplia;
- comparacion de montos por proxy;
- tablas proxy+ vs proxy-;
- distribuciones por canal y gateway.

### Outliers y patrones

- montos > P99;
- montos extremos absolutos;
- horarios atipicos;
- usuarios con alta velocidad transaccional;
- facilities o gateways con senal desproporcionada.

## Entregables

- notebook `01_cap2_eda.ipynb`;
- `output/tables/cap2_status.tex`;
- `output/tables/cap2_channel.tex`;
- `output/tables/cap2_gateway.tex`;
- `output/tables/cap2_proxy_profiles.tex`;
- `output/figures/cap2_amount_distribution.pdf`;
- `output/figures/cap2_proxy_amounts.pdf`;
- `output/figures/cap2_temporal.pdf`.

## Reglas de calidad

- ninguna tabla se llena manualmente;
- todo sale del snapshot congelado;
- cada figura debe poder regenerarse por script o notebook;
- si una tabla no responde a OE2, no entra.

## Tareas operativas

1. Construir notebook de EDA con celdas ordenadas por tema.
2. Exportar resultados a CSV/JSON y luego a LaTeX.
3. Redactar plantilla de parrafos interpretativos para Capitulo 2.
4. Identificar si algun hallazgo del EDA obliga a ajustar el catalogo de features.

## Gate de salida

No pasar a modelado final sin:

- tablas y figuras base de Capitulo 2 listas;
- confirmacion de que OE2 queda respondido;
- inventario de hallazgos que alimentan el feature engineering.
