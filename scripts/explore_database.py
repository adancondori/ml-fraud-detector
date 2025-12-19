#!/usr/bin/env python3
"""
Script para explorar la estructura de la base de datos ClickHouse.
"""
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from config.config import settings
from fraud_detector.data.clickhouse_connector import ClickHouseConnector


def main():
    use_secure = settings.clickhouse_secure or settings.clickhouse_port == 8443

    connector = ClickHouseConnector(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_database,
        secure=use_secure,
    )

    with connector:
        print("=" * 80)
        print("EXPLORACIÓN DE LA BASE DE DATOS")
        print("=" * 80)

        # 1. Ver columnas relacionadas con fraude
        print("\n1. BUSCANDO COLUMNAS RELACIONADAS CON FRAUDE:")
        fraud_cols_query = """
        SELECT name, type
        FROM system.columns
        WHERE database = 'pbp_productionDB_optimized'
          AND (name ILIKE '%fraud%' OR name ILIKE '%charg%' OR name ILIKE '%disput%'
               OR name ILIKE '%risk%' OR name ILIKE '%suspicious%' OR name ILIKE '%reversal%'
               OR name ILIKE '%refund%')
        ORDER BY table, name
        """
        fraud_cols = connector.execute_query(fraud_cols_query)
        if fraud_cols:
            for col in fraud_cols:
                print(f"   - {col[0]}: {col[1]}")
        else:
            print("   No se encontraron columnas con nombres relacionados a fraude")

        # 2. Ver valores únicos de status
        print("\n2. VALORES DE STATUS EN PAYMENTS:")
        status_query = """
        SELECT status, count() as cnt
        FROM pbp_productionDB_optimized.payments
        GROUP BY status
        ORDER BY cnt DESC
        """
        statuses = connector.execute_query(status_query)
        for status, count in statuses:
            print(f"   - {status}: {count:,}")

        # 3. Buscar columnas monetarias
        print("\n3. COLUMNAS MONETARIAS EN PAYMENTS:")
        money_cols_query = """
        SELECT name, type
        FROM system.columns
        WHERE database = 'pbp_productionDB_optimized'
          AND table = 'payments'
          AND (type ILIKE '%Decimal%' OR name ILIKE '%amount%' OR name ILIKE '%fee%'
               OR name ILIKE '%paid%' OR name ILIKE '%tax%' OR name ILIKE '%price%')
        ORDER BY name
        """
        money_cols = connector.execute_query(money_cols_query)
        for col in money_cols:
            print(f"   - {col[0]}: {col[1]}")

        # 4. Ver muestra de datos con columnas principales
        print("\n4. MUESTRA DE DATOS (5 registros):")
        sample_query = """
        SELECT
            id,
            user_id,
            facility_id,
            created_at,
            status,
            payment_method,
            card_brand,
            reservation_paid_out,
            technology_fee,
            tax,
            tip,
            debit_refund,
            reversed_id
        FROM pbp_productionDB_optimized.payments
        LIMIT 5
        """
        df = connector.query_to_dataframe(sample_query)
        print(df.to_string())

        # 5. Ver distribución de reversed_id (posible indicador de fraude)
        print("\n5. ANÁLISIS DE REVERSED_ID (posible indicador de fraude):")
        reversed_query = """
        SELECT
            CASE
                WHEN reversed_id > 0 THEN 'Reversed'
                ELSE 'Normal'
            END as type,
            count() as cnt,
            round(count() * 100.0 / sum(count()) OVER (), 4) as pct
        FROM pbp_productionDB_optimized.payments
        GROUP BY type
        """
        reversed_stats = connector.execute_query(reversed_query)
        for row in reversed_stats:
            print(f"   - {row[0]}: {row[1]:,} ({row[2]}%)")

        # 6. Ver distribución de debit_refund
        print("\n6. ANÁLISIS DE DEBIT_REFUND:")
        refund_query = """
        SELECT
            debit_refund,
            count() as cnt,
            round(count() * 100.0 / sum(count()) OVER (), 4) as pct
        FROM pbp_productionDB_optimized.payments
        GROUP BY debit_refund
        """
        refund_stats = connector.execute_query(refund_query)
        for row in refund_stats:
            print(f"   - debit_refund={row[0]}: {row[1]:,} ({row[2]}%)")

        # 7. Buscar tablas relacionadas con disputas o chargebacks
        print("\n7. TABLAS RELACIONADAS CON DISPUTAS/CHARGEBACKS:")
        tables_query = """
        SELECT name
        FROM system.tables
        WHERE database = 'pbp_productionDB_optimized'
          AND (name ILIKE '%disput%' OR name ILIKE '%chargeback%' OR name ILIKE '%fraud%'
               OR name ILIKE '%refund%' OR name ILIKE '%reversal%')
        """
        tables = connector.execute_query(tables_query)
        if tables:
            for t in tables:
                print(f"   - {t[0]}")
        else:
            print("   No se encontraron tablas específicas de fraude/disputas")

        # 8. Rango de fechas
        print("\n8. RANGO DE FECHAS EN PAYMENTS:")
        date_query = """
        SELECT
            min(created_at) as min_date,
            max(created_at) as max_date,
            count() as total
        FROM pbp_productionDB_optimized.payments
        """
        dates = connector.execute_query(date_query)[0]
        print(f"   - Desde: {dates[0]}")
        print(f"   - Hasta: {dates[1]}")
        print(f"   - Total: {dates[2]:,}")

        # 9. Distribución por año
        print("\n9. DISTRIBUCIÓN POR AÑO:")
        year_query = """
        SELECT
            toYear(created_at) as year,
            count() as cnt
        FROM pbp_productionDB_optimized.payments
        GROUP BY year
        ORDER BY year
        """
        years = connector.execute_query(year_query)
        for row in years:
            print(f"   - {row[0]}: {row[1]:,}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
