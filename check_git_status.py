#!/usr/bin/env python3
"""
Script de validación para verificar que no se están committeando archivos problemáticos.
Ejecuta esto antes de hacer commit para evitar problemas.
"""
import subprocess
import sys
from pathlib import Path


# Configuración de límites
MAX_FILE_SIZE_MB = 10
MAX_TOTAL_SIZE_MB = 50

# Patrones sospechosos (archivos que NO deberían estar en staging)
SUSPICIOUS_PATTERNS = [
    '.env',
    '*.key',
    '*.pem',
    'credentials.json',
    'secrets.yaml',
    '*.log',
    '.DS_Store',
]

# Extensiones grandes comunes en ML
LARGE_FILE_EXTENSIONS = [
    '.csv', '.parquet', '.pkl', '.pickle', '.joblib',
    '.h5', '.hdf5', '.npy', '.npz', '.pth', '.pt',
    '.ckpt', '.weights', '.model'
]


def print_header(text, char='='):
    """Imprime un encabezado."""
    print(f"\n{char * 60}")
    print(f"  {text}")
    print(f"{char * 60}")


def run_command(cmd):
    """Ejecuta un comando shell y retorna el output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout.strip(), result.returncode
    except Exception as e:
        print(f"Error ejecutando comando: {e}")
        return "", 1


def check_git_repo():
    """Verifica que estemos en un repo git."""
    output, returncode = run_command("git rev-parse --git-dir")
    return returncode == 0


def get_staged_files():
    """Obtiene lista de archivos en staging area."""
    output, _ = run_command("git diff --cached --name-only")
    if not output:
        return []
    return output.split('\n')


def get_file_size(filepath):
    """Obtiene el tamaño de un archivo en MB."""
    try:
        path = Path(filepath)
        if path.exists():
            return path.stat().st_size / (1024 * 1024)  # MB
    except:
        pass
    return 0


def check_suspicious_files(staged_files):
    """Verifica archivos sospechosos."""
    print_header("🔍 Verificando archivos sospechosos")

    suspicious_found = []

    for file in staged_files:
        filename = Path(file).name

        # Verificar patrones
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern.startswith('*'):
                # Pattern con wildcard
                if filename.endswith(pattern[1:]):
                    suspicious_found.append((file, pattern))
            else:
                # Match exacto
                if filename == pattern or file.endswith(pattern):
                    suspicious_found.append((file, pattern))

    if suspicious_found:
        print("\n⚠️  ADVERTENCIA: Archivos sospechosos encontrados:\n")
        for file, pattern in suspicious_found:
            print(f"  ❌ {file} (coincide con '{pattern}')")
        print("\n💡 Estos archivos NO deberían estar en Git.")
        print("   Considera agregarlos al .gitignore")
        return False
    else:
        print("✅ No se encontraron archivos sospechosos")
        return True


def check_large_files(staged_files):
    """Verifica archivos grandes."""
    print_header("📦 Verificando tamaño de archivos")

    large_files = []
    total_size = 0

    for file in staged_files:
        size = get_file_size(file)
        total_size += size

        if size > MAX_FILE_SIZE_MB:
            large_files.append((file, size))

    # Verificar archivos individuales grandes
    if large_files:
        print(f"\n⚠️  Archivos mayores a {MAX_FILE_SIZE_MB} MB:\n")
        for file, size in large_files:
            print(f"  ⚠️  {file}: {size:.2f} MB")
        print(f"\n💡 Considera usar Git LFS o no versionar estos archivos")

    # Verificar tamaño total
    if total_size > MAX_TOTAL_SIZE_MB:
        print(f"\n⚠️  Tamaño total en staging: {total_size:.2f} MB")
        print(f"   (límite recomendado: {MAX_TOTAL_SIZE_MB} MB)")
    else:
        print(f"✅ Tamaño total: {total_size:.2f} MB (OK)")

    return len(large_files) == 0 and total_size <= MAX_TOTAL_SIZE_MB


def check_ml_files(staged_files):
    """Verifica archivos típicos de ML que no deberían committearse."""
    print_header("🤖 Verificando archivos ML")

    ml_files = []

    for file in staged_files:
        ext = Path(file).suffix.lower()
        if ext in LARGE_FILE_EXTENSIONS:
            size = get_file_size(file)
            ml_files.append((file, ext, size))

    if ml_files:
        print("\n⚠️  Archivos ML encontrados:\n")
        for file, ext, size in ml_files:
            print(f"  ⚠️  {file} ({ext}): {size:.2f} MB")

        print("\n💡 Estos archivos generalmente NO deberían estar en Git:")
        print("   - Datos: Usa DVC o cloud storage")
        print("   - Modelos: Usa MLflow Model Registry")
        print("   - Si es necesario: Usa Git LFS")
        return False
    else:
        print("✅ No se encontraron archivos ML en staging")
        return True


def check_untracked_important():
    """Verifica archivos importantes no trackeados."""
    print_header("📋 Verificando archivos importantes")

    output, _ = run_command("git ls-files --others --exclude-standard")
    untracked = output.split('\n') if output else []

    important_untracked = []

    # Archivos que deberían estar trackeados
    important_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'pyproject.toml',
        '.gitignore',
    ]

    for important in important_files:
        if important in untracked:
            important_untracked.append(important)

    if important_untracked:
        print("\n⚠️  Archivos importantes no trackeados:\n")
        for file in important_untracked:
            print(f"  ⚠️  {file}")
        print("\n💡 ¿Olvidaste agregarlos con 'git add'?")
        return False
    else:
        print("✅ Todos los archivos importantes están trackeados")
        return True


def main():
    """Función principal."""
    print("\n" + "🔎" * 30)
    print("  VALIDACIÓN DE GIT STATUS")
    print("🔎" * 30)

    # Verificar que estamos en un repo
    if not check_git_repo():
        print("\n❌ No estás en un repositorio Git")
        return 1

    # Obtener archivos en staging
    staged_files = get_staged_files()

    if not staged_files:
        print("\n📭 No hay archivos en staging area")
        print("\n💡 Usa 'git add' para agregar archivos")
        return 0

    print(f"\n📦 Archivos en staging: {len(staged_files)}")
    for file in staged_files[:10]:  # Mostrar primeros 10
        print(f"   - {file}")
    if len(staged_files) > 10:
        print(f"   ... y {len(staged_files) - 10} más")

    # Ejecutar verificaciones
    checks = [
        check_suspicious_files(staged_files),
        check_large_files(staged_files),
        check_ml_files(staged_files),
        check_untracked_important(),
    ]

    # Resultado final
    print_header("📊 RESUMEN")

    if all(checks):
        print("\n✅ Todas las verificaciones pasaron")
        print("🎉 Es seguro hacer commit")
        return 0
    else:
        print("\n⚠️  Algunas verificaciones fallaron")
        print("📝 Revisa las advertencias arriba")
        print("\n❓ ¿Deseas continuar de todas formas?")
        print("   Sí: Haz 'git commit'")
        print("   No: Revisa y ajusta los archivos")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verificación cancelada")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
