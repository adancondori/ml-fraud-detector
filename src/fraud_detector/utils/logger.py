"""
Robust logging configuration using loguru.
Provides structured logging with rotation, retention, and multiple output formats.
"""
import sys
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from config.config import settings


def setup_logger() -> None:
    """
    Configure loguru logger with file rotation, retention, and formatting.

    Features:
    - Console output with colored formatting
    - File output with rotation (10 MB) and retention (30 days)
    - JSON format support for production
    - Different log levels for console and file
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors (for development)
    if settings.is_development:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.log_level,
            colorize=True,
        )
    else:
        # Production: simple format
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.log_level,
            colorize=False,
        )

    # File handler - General logs
    log_path = settings.get_absolute_path(settings.logs_dir) / "fraud_detector.log"

    if settings.log_format == "json":
        logger.add(
            log_path,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            level="INFO",
            serialize=True,  # JSON format
            backtrace=True,
            diagnose=True,
        )
    else:
        logger.add(
            log_path,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True,
        )

    # Separate error log file
    error_log_path = settings.get_absolute_path(settings.logs_dir) / "errors.log"
    logger.add(
        error_log_path,
        rotation="5 MB",
        retention="60 days",
        compression="zip",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logger initialized successfully")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log level: {settings.log_level}")


def log_experiment_params(params: Dict[str, Any]) -> None:
    """Log experiment parameters in a structured format."""
    logger.info("Experiment parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")


def log_model_metrics(metrics: Dict[str, float]) -> None:
    """Log model metrics in a structured format."""
    logger.info("Model metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


# Initialize logger on import
setup_logger()

# Export configured logger
__all__ = ["logger", "setup_logger", "log_experiment_params", "log_model_metrics"]
