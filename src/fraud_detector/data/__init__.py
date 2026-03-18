"""Data loading, extraction and proxy labeling."""
from fraud_detector.data.loader import DataManager
from fraud_detector.data.clickhouse_connector import ClickHouseConnector

__all__ = ["DataManager", "ClickHouseConnector"]
