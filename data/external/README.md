# External Data Directory

Este directorio contiene **datos externos de terceros** que complementan el análisis.

## Propósito

- Datos de fuentes externas (APIs, datasets públicos, etc.)
- Referencias y metadatos externos
- Datos de enriquecimiento

## Ejemplos de uso

### Datos de referencia
- Listas de códigos de categorías de comercios
- Códigos postales y coordenadas geográficas
- Días festivos y calendario
- Listas de países, ciudades, etc.

### Datos públicos
- Datasets de benchmarking
- Datos demográficos
- Indicadores económicos

### APIs y servicios
- Datos obtenidos de APIs de terceros
- Información de verificación externa

## Estructura ejemplo

```
data/external/
├── merchant_category_codes.csv
├── postal_codes.csv
├── holidays_2024.json
├── public_fraud_datasets/
│   └── kaggle_fraud_data.csv
└── api_responses/
    └── geolocation_cache.json
```

## Gestión

Estos datos **NO se versionan en Git** por defecto.

### Excepciones
Si tienes datos externos pequeños y públicos que quieres versionar:

1. Descomentar en `.gitignore`:
   ```
   !data/external/reference_*.csv
   ```
2. Usar nombres con prefijos específicos

## Fuentes de datos externos

Documenta aquí las fuentes:

### Merchant Category Codes
- **Fuente**: [ISO 18245](https://www.iso.org/standard/33365.html)
- **URL**: https://example.com/mcc-codes
- **Fecha descarga**: 2024-01-15
- **Formato**: CSV

### Ejemplo de carga

```python
from fraud_detector.data.loader import DataLoader

loader = DataLoader()
mcc_codes = loader.load_csv("data/external/merchant_category_codes.csv")
```

## API Keys

⚠️ **NUNCA** guardes API keys en archivos versionados

- Usa variables de entorno en `.env`
- Documenta qué servicios se usan, pero no las keys

```bash
# En .env
GOOGLE_MAPS_API_KEY=your_key_here
FRAUD_CHECK_API_KEY=your_key_here
```

## Actualización

Documenta la frecuencia de actualización:
- Datos estáticos: actualizables manualmente
- Datos dinámicos: considerar cache con TTL
- APIs: considerar rate limits
