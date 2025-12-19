"""
ClickHouse database connector for fraud detection data extraction.
Provides secure connection and efficient data retrieval from TechSport database.
Supports both native protocol (clickhouse-driver) and HTTP protocol (clickhouse-connect).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterator, Literal, Union, List

import pandas as pd
import clickhouse_connect

from fraud_detector.utils.logger import logger


class ClickHouseConnector:
    """Handle ClickHouse database connections and data extraction using HTTP interface."""

    def __init__(
        self,
        host: str,
        port: int = 8443,
        user: str = "default",
        password: str = "",
        database: str = "default",
        secure: bool = True,
    ):
        """
        Initialize ClickHouse connector for HTTP interface (ClickHouse Cloud).

        Args:
            host: ClickHouse server host
            port: ClickHouse HTTP(S) port (default 8443 for Cloud)
            user: Database user
            password: Database password
            database: Default database name
            secure: Use HTTPS (default True for Cloud)
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.secure = secure
        self._client = None

    def connect(self) -> "ClickHouseConnector":
        """
        Establish connection to ClickHouse.

        Returns:
            Self for method chaining
        """
        logger.info(f"Connecting to ClickHouse at {self.host}:{self.port}")

        self._client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            database=self.database,
            secure=self.secure,
        )

        # Test connection
        version = self._client.command("SELECT version()")
        logger.info(f"Connected to ClickHouse version: {version}")

        return self

    def disconnect(self) -> None:
        """Close the database connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Disconnected from ClickHouse")

    def __enter__(self) -> "ClickHouseConnector":
        """Context manager entry."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()

    @property
    def client(self):
        """Get the ClickHouse client, connecting if necessary."""
        if self._client is None:
            self.connect()
        return self._client

    def execute_query(self, query: str, params: Optional[dict] = None) -> list:
        """
        Execute a query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query results as list of tuples
        """
        logger.debug(f"Executing query: {query[:100]}...")
        result = self.client.query(query, parameters=params)
        return result.result_rows

    def query_to_dataframe(
        self,
        query: str,
        params: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query results as pandas DataFrame
        """
        logger.info("Executing query to DataFrame...")

        df = self.client.query_df(query, parameters=params)

        logger.info(f"Retrieved {len(df):,} rows with {len(df.columns)} columns")
        return df

    def query_to_dataframe_chunked(
        self,
        query: str,
        chunk_size: int = 100_000,
        params: Optional[dict] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Execute query and yield results in chunks.
        Useful for large datasets to manage memory.

        Args:
            query: SQL query string
            chunk_size: Number of rows per chunk
            params: Query parameters

        Yields:
            DataFrame chunks
        """
        logger.info(f"Executing chunked query (chunk_size={chunk_size:,})...")

        # Use streaming with clickhouse-connect
        with self.client.query_rows_stream(query, parameters=params) as stream:
            columns = stream.source.column_names
            chunk = []
            total_rows = 0

            for row in stream:
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    df = pd.DataFrame(chunk, columns=columns)
                    total_rows += len(df)
                    logger.info(f"Yielding chunk: {len(df):,} rows (total: {total_rows:,})")
                    yield df
                    chunk = []

            # Yield remaining rows
            if chunk:
                df = pd.DataFrame(chunk, columns=columns)
                total_rows += len(df)
                logger.info(f"Yielding final chunk: {len(df):,} rows (total: {total_rows:,})")
                yield df

    def get_table_info(self, table: str, database: Optional[str] = None) -> dict:
        """
        Get information about a table.

        Args:
            table: Table name
            database: Database name (uses default if None)

        Returns:
            Dictionary with table information
        """
        db = database or self.database

        # Get column info
        columns_query = f"""
        SELECT name, type, comment
        FROM system.columns
        WHERE database = '{db}' AND table = '{table}'
        """
        columns = self.execute_query(columns_query)

        # Get row count
        count_query = f"SELECT count() FROM {db}.{table}"
        row_count = self.execute_query(count_query)[0][0]

        info = {
            "database": db,
            "table": table,
            "row_count": row_count,
            "columns": [
                {"name": col[0], "type": col[1], "comment": col[2] if len(col) > 2 else ""}
                for col in columns
            ],
        }

        logger.info(f"Table {db}.{table}: {row_count:,} rows, {len(columns)} columns")
        return info

    def list_databases(self) -> List[str]:
        """List all accessible databases."""
        result = self.execute_query("SHOW DATABASES")
        return [row[0] for row in result]

    def list_tables(self, database: Optional[str] = None) -> List[str]:
        """List all tables in a database."""
        db = database or self.database
        result = self.execute_query(f"SHOW TABLES FROM {db}")
        return [row[0] for row in result]


class FraudDataExtractor:
    """
    Specialized extractor for TechSport/PayByCourt fraud detection data.
    Handles the specific schema and creates fraud labels from reversals.

    Fraud Definition:
    - A transaction is considered "fraudulent" if it was later reversed
    - Reversals are identified by: reversed_id > 0 OR payment_method = 'reversal'
    - OR status IN ('totally_refunded', 'partially_refunded', 'refunded_to_credit')
    """

    # Query for extracting transactions with fraud labels
    # Excludes reversal records (which are the refund transactions themselves)
    # Only includes actual payment transactions that may have been reversed
    BASE_QUERY = """
    SELECT
        id,
        user_id,
        facility_id,
        facility_name,
        created_at,
        updated_at,
        payment_method,
        card_brand,
        status,
        category,
        -- Monetary columns
        reservation_paid_out as amount,
        technology_fee,
        tax,
        tip,
        discount,
        -- Additional features
        payment_source,
        source_enum,
        paid,
        paid_by_manager,
        club_credit_flag,
        -- Fraud indicators (for labeling)
        reversed_id,
        debit_refund,
        -- Derived fraud label
        CASE
            WHEN reversed_id > 0 THEN 1
            WHEN status IN ('totally_refunded', 'partially_refunded', 'refunded_to_credit') THEN 1
            WHEN debit_refund = true THEN 1
            ELSE 0
        END as is_fraud
    FROM {database}.{table}
    WHERE created_at >= '{start_date}'
      AND created_at < '{end_date}'
      AND payment_method != 'reversal'  -- Exclude reversal records
      AND payment_method != 'free'       -- Exclude free transactions (no fraud risk)
      AND reservation_paid_out > 0       -- Only transactions with actual payments
    ORDER BY created_at ASC
    """

    def __init__(
        self,
        connector: ClickHouseConnector,
        database: Optional[str] = None,
        table: Optional[str] = None,
    ):
        """
        Initialize fraud data extractor.

        Args:
            connector: ClickHouse connector instance
            database: Database name (uses connector's database if None)
            table: Table name (uses 'payments' if None)
        """
        self.connector = connector
        self.database = database or connector.database
        self.table = table or "payments"

    def extract_period(
        self,
        start_date: str,
        end_date: str,
        chunked: bool = False,
        chunk_size: int = 100_000,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Extract transactions for a specific period with fraud labels.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunked: Return iterator of chunks instead of full DataFrame
            chunk_size: Rows per chunk if chunked=True

        Returns:
            DataFrame or Iterator of DataFrames
        """
        query = self.BASE_QUERY.format(
            database=self.database,
            table=self.table,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"Extracting data from {start_date} to {end_date}")

        if chunked:
            return self.connector.query_to_dataframe_chunked(query, chunk_size)
        else:
            return self.connector.query_to_dataframe(query)

    def extract_train_data(self, year: int = 2025) -> pd.DataFrame:
        """Extract training data (January - June)."""
        return self.extract_period(f"{year}-01-01", f"{year}-07-01")

    def extract_validation_data(self, year: int = 2025) -> pd.DataFrame:
        """Extract validation data (July - August)."""
        return self.extract_period(f"{year}-07-01", f"{year}-09-01")

    def extract_test_data(self, year: int = 2025) -> pd.DataFrame:
        """Extract test data (September - December)."""
        return self.extract_period(f"{year}-09-01", f"{year + 1}-01-01")

    def extract_full_dataset(
        self,
        year: int = 2025,
        save_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Extract complete dataset for a year.

        Args:
            year: Year to extract
            save_path: Optional path to save as parquet

        Returns:
            Complete DataFrame
        """
        df = self.extract_period(f"{year}-01-01", f"{year + 1}-01-01")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path, index=False, compression="snappy")
            logger.info(f"Saved dataset to {save_path}")

        return df

    def get_statistics(self, start_date: str = "2024-01-01", end_date: str = "2026-01-01") -> dict:
        """Get basic statistics about the fraud dataset."""
        stats_query = f"""
        SELECT
            count() as total_transactions,
            countIf(
                reversed_id > 0
                OR status IN ('totally_refunded', 'partially_refunded', 'refunded_to_credit')
                OR debit_refund = true
            ) as fraud_count,
            round(countIf(
                reversed_id > 0
                OR status IN ('totally_refunded', 'partially_refunded', 'refunded_to_credit')
                OR debit_refund = true
            ) * 100.0 / count(), 4) as fraud_rate_pct,
            min(created_at) as min_date,
            max(created_at) as max_date,
            round(avg(reservation_paid_out), 2) as avg_amount,
            round(sum(reservation_paid_out), 2) as total_amount
        FROM {self.database}.{self.table}
        WHERE created_at >= '{start_date}'
          AND created_at < '{end_date}'
          AND payment_method != 'reversal'
          AND payment_method != 'free'
          AND reservation_paid_out > 0
        """

        result = self.connector.execute_query(stats_query)[0]

        stats = {
            "total_transactions": result[0],
            "fraud_count": result[1],
            "fraud_rate_pct": float(result[2]),
            "date_range": {"min": result[3], "max": result[4]},
            "avg_amount": float(result[5]),
            "total_amount": float(result[6]),
        }

        logger.info(f"Dataset statistics: {stats['total_transactions']:,} transactions, "
                   f"{stats['fraud_rate_pct']:.2f}% fraud rate")

        return stats

    def get_sample(self, n: int = 1000, year: int = 2025) -> pd.DataFrame:
        """Get a random sample of transactions for quick exploration."""
        sample_query = f"""
        SELECT *
        FROM (
            {self.BASE_QUERY.format(
                database=self.database,
                table=self.table,
                start_date=f'{year}-01-01',
                end_date=f'{year+1}-01-01',
            )}
        )
        ORDER BY rand()
        LIMIT {n}
        """
        return self.connector.query_to_dataframe(sample_query)
