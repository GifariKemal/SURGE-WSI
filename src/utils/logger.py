"""Loguru Logger Configuration
==============================

Configures logging with:
- Console output with colors
- File rotation
- JSON format option
- Level filtering

Author: SURIOTA Team
"""
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    json_format: bool = False,
    console: bool = True
):
    """Setup loguru logger

    Args:
        log_level: Minimum log level
        log_file: Path to log file (None = no file logging)
        rotation: When to rotate (size or time)
        retention: How long to keep old logs
        json_format: Use JSON format for file
        console: Enable console output
    """
    # Remove default handler
    logger.remove()

    # Console format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # File format
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # JSON format for structured logging
    json_format_str = (
        '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
        '"level":"{level}",'
        '"name":"{name}",'
        '"function":"{function}",'
        '"line":{line},'
        '"message":"{message}"}}'
    )

    # Add console handler
    if console:
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )

    # Add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format=json_format_str if json_format else file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            enqueue=True,  # Thread-safe
            backtrace=True,
            diagnose=True
        )

    logger.info(f"Logger configured: level={log_level}, file={log_file}")


def get_logger(name: str = None):
    """Get logger instance

    Args:
        name: Logger name (module name)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Component-specific loggers
def get_kalman_logger():
    """Get logger for Kalman filter module"""
    return logger.bind(component="kalman")


def get_regime_logger():
    """Get logger for regime detection module"""
    return logger.bind(component="regime")


def get_poi_logger():
    """Get logger for POI detection module"""
    return logger.bind(component="poi")


def get_entry_logger():
    """Get logger for entry trigger module"""
    return logger.bind(component="entry")


def get_risk_logger():
    """Get logger for risk management module"""
    return logger.bind(component="risk")


def get_exit_logger():
    """Get logger for exit management module"""
    return logger.bind(component="exit")


def get_executor_logger():
    """Get logger for trade executor module"""
    return logger.bind(component="executor")


def get_telegram_logger():
    """Get logger for telegram module"""
    return logger.bind(component="telegram")


# Filter function for production
def production_filter(record):
    """Filter for production logging (no DEBUG)"""
    return record["level"].name != "DEBUG"


# Setup shortcuts
def setup_development_logger():
    """Setup logger for development"""
    setup_logger(
        log_level="DEBUG",
        log_file="logs/surge_wsi_dev.log",
        rotation="10 MB",
        retention="3 days",
        console=True
    )


def setup_production_logger():
    """Setup logger for production"""
    setup_logger(
        log_level="INFO",
        log_file="logs/surge_wsi.log",
        rotation="50 MB",
        retention="30 days",
        json_format=True,
        console=True
    )
    logger.add(
        lambda msg: None,  # Custom handler
        filter=production_filter,
        level="INFO"
    )


def setup_backtest_logger():
    """Setup logger for backtesting"""
    setup_logger(
        log_level="WARNING",
        log_file="logs/backtest.log",
        rotation="100 MB",
        retention="7 days",
        console=False
    )
