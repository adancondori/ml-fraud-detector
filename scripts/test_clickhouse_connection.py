#!/usr/bin/env python3
"""
Script para probar la conexión a ClickHouse y explorar la estructura de datos.
Ejecutar: python test_clickhouse_connection.py
"""
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from config.config import settings
from fraud_detector.data.clickhouse_connector import ClickHouseConnector, FraudDataExtractor
from fraud_detector.utils.logger import logger


def main():
    print("=" * 60)
    print("🔌 Probando conexión a ClickHouse")
    print("=" * 60)
    print()

    # Mostrar configuración (sin password)
    print("📋 Configuración:")
    print(f"   Host: {settings.clickhouse_host}")
    print(f"   Port: {settings.clickhouse_port}")
    print(f"   User: {settings.clickhouse_user}")
    print(f"   Database: {settings.clickhouse_database}")
    print(f"   Table: {settings.clickhouse_table}")
    print(f"   Secure: {settings.clickhouse_secure}")
    print()

    try:
        # Crear conector
        # Auto-detect secure mode based on port (8443 = HTTPS)
        use_secure = settings.clickhouse_secure or settings.clickhouse_port == 8443

        connector = ClickHouseConnector(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            user=settings.clickhouse_user,
            password=settings.clickhouse_password,
            database=settings.clickhouse_database,
            secure=use_secure,
        )

        # Conectar
        print("🔄 Conectando...")
        connector.connect()
        print("✅ Conexión exitosa!")
        print()

        # Listar bases de datos
        print("📂 Bases de datos disponibles:")
        databases = connector.list_databases()
        for db in databases[:10]:  # Mostrar primeras 10
            marker = " 👈" if db == settings.clickhouse_database else ""
            print(f"   - {db}{marker}")
        if len(databases) > 10:
            print(f"   ... y {len(databases) - 10} más")
        print()

        # Listar tablas en la base de datos de interés
        print(f"📑 Tablas en {settings.clickhouse_database}:")
        try:
            tables = connector.list_tables(settings.clickhouse_database)
            for table in tables[:15]:
                marker = " 👈" if table == settings.clickhouse_table else ""
                print(f"   - {table}{marker}")
            if len(tables) > 15:
                print(f"   ... y {len(tables) - 15} más")
        except Exception as e:
            print(f"   ⚠️  No se pudo acceder a la base de datos: {e}")
        print()

        # Información de la tabla de pagos
        print(f"📊 Estructura de {settings.clickhouse_table}:")
        try:
            table_info = connector.get_table_info(
                settings.clickhouse_table,
                settings.clickhouse_database
            )
            print(f"   Total de filas: {table_info['row_count']:,}")
            print(f"   Columnas ({len(table_info['columns'])}):")
            for col in table_info['columns']:
                print(f"      - {col['name']}: {col['type']}")
        except Exception as e:
            print(f"   ⚠️  No se pudo obtener info de la tabla: {e}")
        print()

        # Crear extractor y obtener estadísticas
        print("📈 Estadísticas del dataset de fraude:")
        try:
            extractor = FraudDataExtractor(
                connector=connector,
                table=settings.clickhouse_table,
            )
            stats = extractor.get_statistics()
            print(f"   Total transacciones: {stats['total_transactions']:,}")
            print(f"   Transacciones fraudulentas: {stats['fraud_count']:,}")
            print(f"   Tasa de fraude: {stats['fraud_rate_pct']:.4f}%")
            print(f"   Rango de fechas: {stats['date_range']['min']} - {stats['date_range']['max']}")
            print(f"   Monto promedio: ${stats['avg_amount']:,.2f}")
            print(f"   Monto total: ${stats['total_amount']:,.2f}")
        except Exception as e:
            print(f"   ⚠️  No se pudieron obtener estadísticas: {e}")
        print()

        # Extraer muestra pequeña
        print("🔍 Extrayendo muestra de datos (primeras 5 filas):")
        try:
            sample_query = f"""
            SELECT *
            FROM {settings.clickhouse_database}.{settings.clickhouse_table}
            WHERE status IN ('completed', 'failed', 'refunded')
              AND amount > 0
            LIMIT 5
            """
            sample_df = connector.query_to_dataframe(sample_query)
            print(sample_df.to_string())
        except Exception as e:
            print(f"   ⚠️  No se pudo extraer muestra: {e}")
        print()

        # Desconectar
        connector.disconnect()

        print("=" * 60)
        print("✨ Prueba completada exitosamente!")
        print("=" * 60)
        print()
        print("📝 Próximos pasos:")
        print("   1. Edita .env con tus credenciales reales de ClickHouse")
        print("   2. Ejecuta de nuevo este script para verificar")
        print("   3. Abre el notebook para explorar los datos")

    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        print()
        print("🔧 Verifica:")
        print("   1. Que ClickHouse esté corriendo")
        print("   2. Las credenciales en .env")
        print("   3. Que el host y puerto sean correctos")
        print("   4. Permisos de acceso a la base de datos")
        logger.error(f"Connection error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
