#!/usr/bin/env python3
"""
Script para extraer muestra de datos de ClickHouse con etiquetas de fraude.
"""
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from config.config import settings
from fraud_detector.data.clickhouse_connector import ClickHouseConnector, FraudDataExtractor


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

    print("=" * 80)
    print("EXTRACCIÓN DE DATOS CON ETIQUETAS DE FRAUDE")
    print("=" * 80)

    with connector:
        extractor = FraudDataExtractor(connector)

        # 1. Estadísticas para 2025
        print("\n1. ESTADÍSTICAS DEL DATASET (2025):")
        stats = extractor.get_statistics(start_date="2025-01-01", end_date="2026-01-01")
        print(f"   Total transacciones: {stats['total_transactions']:,}")
        print(f"   Transacciones fraudulentas: {stats['fraud_count']:,}")
        print(f"   Tasa de fraude: {stats['fraud_rate_pct']:.2f}%")
        print(f"   Rango de fechas: {stats['date_range']['min']} - {stats['date_range']['max']}")
        print(f"   Monto promedio: ${stats['avg_amount']:,.2f}")
        print(f"   Monto total: ${stats['total_amount']:,.2f}")

        # 2. Extraer muestra pequeña
        print("\n2. MUESTRA DE DATOS (1000 registros):")
        sample_df = extractor.get_sample(n=1000, year=2025)

        print(f"   Shape: {sample_df.shape}")
        print(f"   Columnas: {list(sample_df.columns)}")
        print(f"\n   Distribución de is_fraud:")
        fraud_dist = sample_df['is_fraud'].value_counts()
        for val, count in fraud_dist.items():
            label = "Fraude" if val == 1 else "Normal"
            print(f"      {label}: {count} ({count/len(sample_df)*100:.1f}%)")

        print(f"\n   Distribución de payment_method:")
        pm_dist = sample_df['payment_method'].value_counts()
        for pm, count in pm_dist.head(5).items():
            print(f"      {pm}: {count}")

        print(f"\n   Distribución de card_brand:")
        cb_dist = sample_df['card_brand'].value_counts()
        for cb, count in cb_dist.head(5).items():
            cb_name = cb if cb else "(vacío)"
            print(f"      {cb_name}: {count}")

        # 3. Mostrar algunas filas de ejemplo
        print("\n3. PRIMERAS 5 FILAS DE LA MUESTRA:")
        display_cols = ['id', 'user_id', 'facility_id', 'amount', 'payment_method',
                       'card_brand', 'status', 'is_fraud', 'created_at']
        print(sample_df[display_cols].head().to_string())

        # 4. Estadísticas por tipo de fraude
        print("\n4. ANÁLISIS DE REGISTROS FRAUDULENTOS:")
        fraud_records = sample_df[sample_df['is_fraud'] == 1]
        if len(fraud_records) > 0:
            print(f"   Total en muestra: {len(fraud_records)}")
            print(f"   Por status:")
            for status, count in fraud_records['status'].value_counts().items():
                print(f"      {status}: {count}")
            print(f"   Monto promedio fraude: ${fraud_records['amount'].mean():,.2f}")
            print(f"   Monto promedio normal: ${sample_df[sample_df['is_fraud']==0]['amount'].mean():,.2f}")
        else:
            print("   No hay registros fraudulentos en esta muestra")

        # 5. Guardar muestra a archivo
        output_path = project_root / "data/processed/sample_2025.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_parquet(output_path, index=False)
        print(f"\n5. MUESTRA GUARDADA EN: {output_path}")

        # 6. Información del split temporal
        print("\n" + "=" * 80)
        print("INFORMACIÓN PARA SPLIT TEMPORAL (2025)")
        print("=" * 80)
        splits = [
            ("Train (Ene-Jun)", "2025-01-01", "2025-07-01"),
            ("Validation (Jul-Ago)", "2025-07-01", "2025-09-01"),
            ("Test (Sep-Dic)", "2025-09-01", "2026-01-01"),
        ]

        for name, start, end in splits:
            stats = extractor.get_statistics(start_date=start, end_date=end)
            print(f"\n{name}:")
            print(f"   Transacciones: {stats['total_transactions']:,}")
            print(f"   Fraudes: {stats['fraud_count']:,} ({stats['fraud_rate_pct']:.2f}%)")

        print("\n" + "=" * 80)
        print("LISTO PARA ENTRENAR MODELO")
        print("=" * 80)
        print("""
Próximos pasos:
1. Ejecutar feature engineering
2. Balancear clases con SMOTE
3. Entrenar Random Forest
4. Evaluar métricas
        """)


if __name__ == "__main__":
    main()
