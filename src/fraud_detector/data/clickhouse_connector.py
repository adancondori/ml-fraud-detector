"""
ClickHouse database connector for fraud detection data extraction.
Provides secure connection and efficient data retrieval from TechSport database.
Supports both native protocol (clickhouse-driver) and HTTP protocol (clickhouse-connect).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterator, List

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