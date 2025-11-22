"""
Data loading utilities for fraud detection.
Supports CSV, Parquet, and other formats with validation.
"""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import settings
from fraud_detector.utils.logger import logger


class DataLoader:
    """Handle data loading and basic validation."""

    @staticmethod
    def load_csv(
        file_path: str | Path,
        parse_dates: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            parse_dates: List of columns to parse as dates
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        logger.info(f"Loading CSV file: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path, parse_dates=parse_dates, **kwargs)
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

        return df

    @staticmethod
    def load_parquet(file_path: str | Path) -> pd.DataFrame:
        """
        Load data from Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        logger.info(f"Loading Parquet file: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

        return df

    @staticmethod
    def save_parquet(df: pd.DataFrame, file_path: str | Path) -> None:
        """
        Save DataFrame to Parquet format.

        Args:
            df: DataFrame to save
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(df):,} rows to: {file_path}")
        df.to_parquet(file_path, index=False, compression="snappy")
        logger.info("Data saved successfully")

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame,
        required_columns: list[str]
    ) -> None:
        """
        Validate that required columns exist in DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info("All required columns present")

    @staticmethod
    def get_data_info(df: pd.DataFrame) -> dict:
        """
        Get summary information about the DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with data information
        """
        info = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        }

        logger.info(f"Dataset info: {info['n_rows']:,} rows, {info['n_columns']} columns")
        logger.info(f"Memory usage: {info['memory_usage_mb']:.2f} MB")

        return info


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
    stratify: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion of test set (uses settings if None)
        val_size: Proportion of validation set (uses settings if None)
        stratify: Whether to stratify split by target
        random_state: Random seed (uses settings if None)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    test_size = test_size or settings.test_size
    val_size = val_size or settings.validation_size
    random_state = random_state or settings.random_seed

    logger.info(f"Splitting data: test={test_size}, val={val_size}")

    # First split: train+val / test
    stratify_col = df[target_col] if stratify else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Second split: train / val
    val_size_adjusted = val_size / (1 - test_size)
    stratify_col_train = train_val_df[target_col] if stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_col_train,
    )

    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Log class distribution
    if target_col in df.columns:
        logger.info("Class distribution:")
        logger.info(f"  Train: {train_df[target_col].value_counts().to_dict()}")
        logger.info(f"  Val: {val_df[target_col].value_counts().to_dict()}")
        logger.info(f"  Test: {test_df[target_col].value_counts().to_dict()}")

    return train_df, val_df, test_df
