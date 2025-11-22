"""
Script de verificación simple que comprueba la estructura del proyecto.
No requiere dependencias externas, solo Python estándar.
"""
import sys
from pathlib import Path


def print_header(text):
    """Imprime un encabezado decorado."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_directory_structure():
    """Verifica que existan los directorios principales."""
    print_header("📁 Verificando estructura de directorios")

    project_root = Path(__file__).parent
    required_dirs = [
        "src/fraud_detector",
        "src/fraud_detector/data",
        "src/fraud_detector/features",
        "src/fraud_detector/models",
        "src/fraud_detector/utils",
        "config",
        "data/raw",
        "data/processed",
        "models/saved_models",
        "notebooks",
        "tests",
        "logs",
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}")
        if not exists:
            all_exist = False

    return all_exist


def check_key_files():
    """Verifica que existan los archivos clave."""
    print_header("📄 Verificando archivos clave")

    project_root = Path(__file__).parent
    required_files = [
        "config/config.py",
        "src/fraud_detector/__init__.py",
        "src/fraud_detector/data/loader.py",
        "src/fraud_detector/features/preprocessor.py",
        "src/fraud_detector/models/trainer.py",
        "src/fraud_detector/utils/logger.py",
        "requirements.txt",
        "setup.py",
        ".env",
        "README.md",
        "Makefile",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False

    return all_exist


def simple_random_forest_demo():
    """Demo simple de Random Forest sin dependencias externas."""
    print_header("🌲 Demo Simple de Random Forest (Conceptual)")

    print("\n📚 Concepto de Random Forest:")
    print("   1. Es un ensemble de árboles de decisión")
    print("   2. Cada árbol se entrena con una muestra aleatoria de datos")
    print("   3. Las predicciones se hacen por votación mayoritaria")
    print("   4. Es robusto y previene overfitting")

    print("\n🎯 Uso en Detección de Fraude:")
    print("   • Identifica patrones complejos en transacciones")
    print("   • Maneja datos desbalanceados con sampling")
    print("   • Proporciona importancia de features")
    print("   • Rápido para entrenar y predecir")

    print("\n💡 Ejemplo de código (cuando instales las dependencias):")
    print("""
    from sklearn.ensemble import RandomForestClassifier

    # Crear modelo
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)

    # Entrenar
    rf.fit(X_train, y_train)

    # Predecir
    predictions = rf.predict(X_test)
    """)


def show_next_steps():
    """Muestra los siguientes pasos."""
    print_header("🚀 Próximos pasos para ejecutar el proyecto")

    print("\n1. Crear entorno virtual:")
    print("   python3 -m venv venv")
    print("   source venv/bin/activate")

    print("\n2. Instalar dependencias:")
    print("   pip install --upgrade pip")
    print("   pip install -r requirements.txt")
    print("   pip install -e .")

    print("\n3. Ejecutar el script de Random Forest:")
    print("   python run_simple_rf.py")

    print("\n4. O usar el Makefile:")
    print("   make install-dev")
    print("   make notebook")

    print("\n5. Iniciar MLflow UI:")
    print("   mlflow ui")


def main():
    """Función principal."""
    print("\n" + "🎉" * 30)
    print("  VERIFICACIÓN DEL PROYECTO ML FRAUD DETECTOR")
    print("🎉" * 30)

    # Verificar estructura
    dirs_ok = check_directory_structure()

    # Verificar archivos
    files_ok = check_key_files()

    # Demo conceptual
    simple_random_forest_demo()

    # Siguientes pasos
    show_next_steps()

    # Resumen final
    print_header("📊 Resumen de Verificación")

    if dirs_ok and files_ok:
        print("✅ Estructura del proyecto: OK")
        print("✅ Archivos clave: OK")
        print("\n🎊 ¡El proyecto está correctamente configurado!")
        print("📦 Solo falta instalar las dependencias para ejecutar el código.")
    else:
        print("⚠️  Algunos elementos faltan en la estructura")
        print("   Revisa los elementos marcados con ❌")

    print("\n" + "=" * 60)
    print("  Proyecto listo para desarrollo")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
