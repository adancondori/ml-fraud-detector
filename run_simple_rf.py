"""
Script simple para probar Random Forest y verificar que el proyecto funciona.
Solo genera datos sintéticos y entrena un modelo básico.
"""
import sys
from pathlib import Path

# Agregar src al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from config.config import settings
from fraud_detector.utils.logger import logger


def main():
    """Función principal para probar Random Forest."""

    logger.info("=" * 60)
    logger.info("🚀 Iniciando prueba de Random Forest")
    logger.info("=" * 60)

    # Mostrar configuración
    logger.info(f"Entorno: {settings.environment}")
    logger.info(f"Random Seed: {settings.random_seed}")
    logger.info(f"Directorio del proyecto: {settings.project_root}")

    # Generar datos sintéticos simples
    logger.info("\n📊 Generando datos sintéticos...")
    np.random.seed(settings.random_seed)

    n_samples = 1000
    n_features = 5

    # Features aleatorias
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(200, n_features)

    # Target: basado en una regla simple
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    logger.info(f"✅ Datos generados: {n_samples} muestras de entrenamiento")
    logger.info(f"   Features: {n_features}")
    logger.info(f"   Clases en train: {np.bincount(y_train)}")

    # Crear y entrenar Random Forest
    logger.info("\n🌲 Entrenando Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=10,  # Solo 10 árboles para que sea rápido
        max_depth=3,
        random_state=settings.random_seed,
        verbose=0
    )

    rf_model.fit(X_train, y_train)

    logger.info("✅ Modelo entrenado exitosamente!")
    logger.info(f"   Número de árboles: {rf_model.n_estimators}")
    logger.info(f"   Profundidad máxima: {rf_model.max_depth}")

    # Hacer predicciones
    logger.info("\n🔮 Realizando predicciones...")

    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    logger.info(f"✅ Predicciones completadas!")
    logger.info(f"   Accuracy en train: {train_accuracy:.2%}")
    logger.info(f"   Accuracy en test: {test_accuracy:.2%}")

    # Mostrar reporte de clasificación
    logger.info("\n📈 Reporte de clasificación (test):")
    report = classification_report(y_test, y_pred_test, target_names=['Clase 0', 'Clase 1'])
    print(report)

    # Feature importance
    logger.info("\n🎯 Importancia de features:")
    for i, importance in enumerate(rf_model.feature_importances_):
        logger.info(f"   Feature {i}: {importance:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("✨ ¡Prueba completada exitosamente!")
    logger.info("🎉 El proyecto está funcionando correctamente")
    logger.info("=" * 60)

    logger.info("\n💡 Próximos pasos:")
    logger.info("   1. Revisa los logs en: logs/fraud_detector.log")
    logger.info("   2. Abre el notebook: jupyter notebook notebooks/01_exploratory_analysis.ipynb")
    logger.info("   3. Inicia MLflow: mlflow ui")

    return rf_model


if __name__ == "__main__":
    try:
        model = main()
        logger.info("\n✅ Script ejecutado sin errores")
    except Exception as e:
        logger.error(f"\n❌ Error durante la ejecución: {e}")
        logger.exception("Detalles del error:")
        sys.exit(1)
