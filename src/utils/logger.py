"""
Logger Module
=============

Provides centralized logging configuration for the project.
Supports both console and file logging with customizable formats.

Example Usage:
    >>> from src.utils.logger import setup_logger
    >>> logger = setup_logger(__name__)
    >>> logger.info("Processing started")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default INFO)
        log_file: Optional file path for log output
        log_format: Optional custom format string
        
    Returns:
        Configured Logger instance
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Hello, World!")
        2024-01-15 10:30:00 - __main__ - INFO - Hello, World!
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


class PipelineLogger:
    """
    Context manager for logging pipeline execution.
    
    Automatically logs start, end, and duration of operations.
    
    Example:
        >>> with PipelineLogger("data_ingestion") as pl:
        ...     # Do work
        ...     pl.log("Processed 1000 rows")
    """
    
    def __init__(self, name: str, log_file: Optional[str] = "logs/pipeline.log"):
        """
        Initialize pipeline logger.
        
        Args:
            name: Pipeline step name
            log_file: Optional log file path
        """
        self.name = name
        self.logger = setup_logger(f"pipeline.{name}", log_file=log_file)
        self.start_time = None
        
    def __enter__(self):
        """Start timing and log entry."""
        self.start_time = datetime.now()
        self.logger.info(f"{'='*50}")
        self.logger.info(f"STARTING: {self.name}")
        self.logger.info(f"Time: {self.start_time.isoformat()}")
        self.logger.info(f"{'='*50}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log exit and duration."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        if exc_type:
            self.logger.error(f"FAILED: {self.name}")
            self.logger.error(f"Error: {exc_val}")
        else:
            self.logger.info(f"COMPLETED: {self.name}")
            
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"{'='*50}\n")
        
        # Don't suppress exceptions
        return False
        
    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log a message during pipeline execution."""
        self.logger.log(level, message)
        
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.log(message, logging.DEBUG)
        
    def info(self, message: str) -> None:
        """Log info message."""
        self.log(message, logging.INFO)
        
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.log(message, logging.WARNING)
        
    def error(self, message: str) -> None:
        """Log error message."""
        self.log(message, logging.ERROR)


def get_file_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Get a logger that writes to a timestamped file.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        Logger configured with file output
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{name}_{timestamp}.log"
    return setup_logger(name, log_file=log_file)
