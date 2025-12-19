#!/usr/bin/env python3
"""
Script para explorar las tablas de reversiones y definir criterio de fraude.
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
        print("ANÁLISIS DE REVERSIONES Y DEFINICIÓN DE FRAUDE")
        print("=" * 80)

        # 1. Explorar tabla de reversiones
        print("\n1. ESTRUCTURA DE genl_orig_payments_reversals_rrr:")
        try:
            cols_query = """
            SELECT name, type
            FROM system.columns
            WHERE database = 'pbp_productionDB_optimized'
              AND table = 'genl_orig_payments_reversals_rrr'
            """
            cols = connector.execute_query(cols_query)
            for col in cols:
                print(f"   - {col[0]}: {col[1]}")
        except Exception as e:
            print(f"   Error: {e}")

        # 2. Muestra de reversiones
        print("\n2. MUESTRA DE REVERSIONES:")
        try:
            sample_query = """
            SELECT *
            FROM pbp_productionDB_optimized.genl_orig_payments_reversals_rrr
            LIMIT 5
            """
            df = connector.query_to_dataframe(sample_query)
            print(df.to_string())
        except Exception as e:
            print(f"   Error: {e}")

        # 3. Análisis de pagos con reversed_id
        print("\n3. ANÁLISIS DE PAGOS CON REVERSED_ID > 0:")
        reversed_query = """
        SELECT
            status,
            count() as cnt,
            round(avg(reservation_paid_out), 2) as avg_amount,
            round(sum(reservation_paid_out), 2) as total_amount
        FROM pbp_productionDB_optimized.payments
        WHERE reversed_id > 0
        GROUP BY status
        ORDER BY cnt DESC
        """
        reversed_stats = connector.execute_query(reversed_query)
        print(f"   {'Status':<25} {'Count':>15} {'Avg Amount':>15} {'Total Amount':>20}")
        print("   " + "-" * 75)
        for row in reversed_stats:
            status = row[0] if row[0] else '(empty)'
            print(f"   {status:<25} {row[1]:>15,} {row[2]:>15,.2f} {row[3]:>20,.2f}")

        # 4. Distribución temporal de reversiones (2024-2025)
        print("\n4. DISTRIBUCIÓN MENSUAL DE REVERSIONES (2024-2025):")
        monthly_query = """
        SELECT
            toStartOfMonth(created_at) as month,
            countIf(reversed_id > 0) as reversed_cnt,
            count() as total_cnt,
            round(countIf(reversed_id > 0) * 100.0 / count(), 2) as reversed_pct
        FROM pbp_productionDB_optimized.payments
        WHERE created_at >= '2024-01-01'
        GROUP BY month
        ORDER BY month
        """
        monthly = connector.execute_query(monthly_query)
        print(f"   {'Month':<12} {'Reversed':>12} {'Total':>12} {'%':>8}")
        print("   " + "-" * 44)
        for row in monthly:
            print(f"   {str(row[0]):<12} {row[1]:>12,} {row[2]:>12,} {row[3]:>8.2f}%")

        # 5. Análisis de payment_method en reversiones
        print("\n5. PAYMENT_METHOD EN TRANSACCIONES REVERSADAS:")
        pm_query = """
        SELECT
            payment_method,
            count() as total,
            countIf(reversed_id > 0) as reversed,
            round(countIf(reversed_id > 0) * 100.0 / count(), 2) as reversed_pct
        FROM pbp_productionDB_optimized.payments
        WHERE created_at >= '2024-01-01'
        GROUP BY payment_method
        ORDER BY total DESC
        """
        pm_stats = connector.execute_query(pm_query)
        print(f"   {'Method':<20} {'Total':>12} {'Reversed':>12} {'%':>8}")
        print("   " + "-" * 52)
        for row in pm_stats:
            method = row[0] if row[0] else '(empty)'
            print(f"   {method:<20} {row[1]:>12,} {row[2]:>12,} {row[3]:>8.2f}%")

        # 6. Análisis de card_brand en reversiones
        print("\n6. CARD_BRAND EN TRANSACCIONES REVERSADAS:")
        cb_query = """
        SELECT
            card_brand,
            count() as total,
            countIf(reversed_id > 0) as reversed,
            round(countIf(reversed_id > 0) * 100.0 / count(), 2) as reversed_pct
        FROM pbp_productionDB_optimized.payments
        WHERE created_at >= '2024-01-01' AND card_brand != ''
        GROUP BY card_brand
        ORDER BY total DESC
        LIMIT 10
        """
        cb_stats = connector.execute_query(cb_query)
        print(f"   {'Card Brand':<20} {'Total':>12} {'Reversed':>12} {'%':>8}")
        print("   " + "-" * 52)
        for row in cb_stats:
            print(f"   {row[0]:<20} {row[1]:>12,} {row[2]:>12,} {row[3]:>8.2f}%")

        # 7. Propuesta de definición de fraude
        print("\n" + "=" * 80)
        print("PROPUESTA DE DEFINICIÓN DE FRAUDE (is_fraud)")
        print("=" * 80)

        # Count different definitions
        definitions_query = """
        SELECT
            'reversed_id > 0' as definition,
            count() as fraud_count,
            (SELECT count() FROM pbp_productionDB_optimized.payments WHERE created_at >= '2024-01-01') as total,
            round(count() * 100.0 / (SELECT count() FROM pbp_productionDB_optimized.payments WHERE created_at >= '2024-01-01'), 4) as pct
        FROM pbp_productionDB_optimized.payments
        WHERE reversed_id > 0 AND created_at >= '2024-01-01'

        UNION ALL

        SELECT
            'status LIKE refund%' as definition,
            count() as fraud_count,
            (SELECT count() FROM pbp_productionDB_optimized.payments WHERE created_at >= '2024-01-01') as total,
            round(count() * 100.0 / (SELECT count() FROM pbp_productionDB_optimized.payments WHERE created_at >= '2024-01-01'), 4) as pct
        FROM pbp_productionDB_optimized.payments
        WHERE status IN ('totally_refunded', 'partially_refunded', 'refunded_to_credit')
          AND created_at >= '2024-01-01'

        UNION ALL

        SELECT
            'debit_refund = true' as definition,
            count() as fraud_count,
            (SELECT count() FROM pbp_productionDB_optimized.payments WHERE created_at >= '2024-01-01') as total,
            round(count() * 100.0 / (SELECT count() FROM pbp_productionDB_optimized.payments WHERE created_at >= '2024-01-01'), 4) as pct
        FROM pbp_productionDB_optimized.payments
        WHERE debit_refund = true AND created_at >= '2024-01-01'
        """
        definitions = connector.execute_query(definitions_query)
        print(f"\n   {'Definición':<30} {'Fraude':>12} {'Total':>12} {'%':>10}")
        print("   " + "-" * 64)
        for row in definitions:
            print(f"   {row[0]:<30} {row[1]:>12,} {row[2]:>12,} {row[3]:>10.4f}%")

        # 8. Datos disponibles para 2025
        print("\n\n8. DATOS DISPONIBLES PARA 2025 (TEMPORAL SPLIT):")
        year2025_query = """
        SELECT
            toStartOfMonth(created_at) as month,
            count() as total,
            countIf(reversed_id > 0) as reversed,
            countIf(status IN ('totally_refunded', 'partially_refunded', 'refunded_to_credit')) as refunded
        FROM pbp_productionDB_optimized.payments
        WHERE toYear(created_at) = 2025
        GROUP BY month
        ORDER BY month
        """
        year2025 = connector.execute_query(year2025_query)
        print(f"   {'Month':<12} {'Total':>12} {'Reversed':>12} {'Refunded':>12}")
        print("   " + "-" * 48)
        total_2025 = 0
        for row in year2025:
            print(f"   {str(row[0]):<12} {row[1]:>12,} {row[2]:>12,} {row[3]:>12,}")
            total_2025 += row[1]
        print(f"   {'TOTAL 2025':<12} {total_2025:>12,}")

        print("\n" + "=" * 80)
        print("CONCLUSIÓN:")
        print("=" * 80)
        print("""
Para la detección de fraude, se recomienda usar:
  is_fraud = 1 CUANDO:
    - reversed_id > 0 (transacción reversada)
    - O status IN ('totally_refunded', 'partially_refunded', 'refunded_to_credit')

Esta definición captura ~10% de las transacciones como "fraudulentas" o problemáticas,
lo cual es un porcentaje razonable para entrenar un modelo de detección.
        """)


if __name__ == "__main__":
    main()
