# Quick Start Guide - ML Fraud Detector

Guía rápida para empezar a usar el proyecto en 5 minutos.

## Verificación Inicial

El proyecto ya está configurado. Verifica que todo esté bien:

```bash
python3 verify_setup.py
```

Deberías ver todos los checkmarks ✅ en verde.

## Instalación Rápida

### 1. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

O usando el Makefile:

```bash
make install-dev
```

## Ejecutar Random Forest Simple

Una vez instaladas las dependencias, ejecuta el demo:

```bash
python run_simple_rf.py
```

Este script:
- ✅ Genera datos sintéticos
- ✅ Entrena un Random Forest
- ✅ Muestra métricas y predicciones
- ✅ Usa el sistema de logging configurado
- ✅ Demuestra que todo funciona

### Output Esperado

```
============================================================
🚀 Iniciando prueba de Random Forest
============================================================
Entorno: development
Random Seed: 42

📊 Generando datos sintéticos...
✅ Datos generados: 1000 muestras de entrenamiento

🌲 Entrenando Random Forest...
✅ Modelo entrenado exitosamente!

🔮 Realizando predicciones...
✅ Predicciones completadas!
   Accuracy en train: 100%
   Accuracy en test: 95%

✨ ¡Prueba completada exitosamente!
🎉 El proyecto está funcionando correctamente
```

## Otros Comandos Útiles

```bash
# Abrir Jupyter Notebook
make notebook
# o: jupyter notebook notebooks/

# Iniciar MLflow UI
make mlflow
# o: mlflow ui

# Formatear código
make format

# Ejecutar tests
make test

# Ver ayuda del Makefile
make help
```

## Estructura de Archivos Importantes

```
ml-fraud-detector/
├── run_simple_rf.py          ← Ejecuta esto primero
├── verify_setup.py            ← Verifica configuración
├── requirements.txt           ← Dependencias
├── .env                       ← Configuración (ya creado)
├── notebooks/                 ← Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb
└── src/fraud_detector/        ← Tu código
    ├── data/loader.py
    ├── features/preprocessor.py
    ├── models/trainer.py
    └── utils/logger.py
```

## Logs

Después de ejecutar `run_simple_rf.py`, revisa los logs:

```bash
# Ver logs generales
cat logs/fraud_detector.log

# Ver solo errores
cat logs/errors.log
```

## Para tu Tesis

Este proyecto incluye:

1. **Arquitectura modular profesional** → Para capítulo de metodología
2. **Sistema de logging robusto** → Trazabilidad de experimentos
3. **MLflow tracking** → Gestión de experimentos y métricas
4. **Notebooks documentados** → Análisis exploratorio
5. **Tests y calidad de código** → Reproducibilidad

## Troubleshooting

### Si falta algún archivo:
```bash
python3 verify_setup.py
```

### Si hay errores de dependencias:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Si el entorno virtual no funciona:
```bash
deactivate  # Si estás en un venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
make install-dev
```

## Siguiente Paso

Una vez que `run_simple_rf.py` funcione correctamente, abre el notebook:

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

---

**¿Necesitas ayuda?** Revisa el README.md completo para más detalles.
