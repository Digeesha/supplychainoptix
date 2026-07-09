import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs logs in structured JSON format for production environments."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that allows passing contextual information to log messages."""
    
    def process(self, msg, kwargs):
        """Add context to the log record."""
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure and return a logger instance with both file and console handlers.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files. If None, logs only to console
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
        json_format: If True, use structured JSON format; otherwise use human-readable format
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create formatters
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Console handler with color support for different log levels
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation if log directory specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate files for different log levels
        log_file = log_dir / f"{name.replace('.', '_')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Separate error log file for ERROR and CRITICAL levels
        error_log_file = log_dir / f"{name.replace('.', '_')}_errors.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str, context: Optional[dict] = None) -> logging.Logger:
    """
    Get or create a logger instance with optional context.
    
    Args:
        name: Logger name
        context: Optional dictionary of contextual information to include in all log messages
    
    Returns:
        Logger instance or ContextLogger if context provided
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextLogger(logger, context)
    
    return logger


class LoggerMixin:
    """Mixin class to provide logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for the class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__ + "." + self.__class__.__name__)
        return self._logger


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use for logging
    
    Usage:
        @log_execution_time(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(
                    f"Function {func.__name__} completed in {execution_time:.4f}s",
                    extra={"execution_time": execution_time, "function": func.__name__}
                )
                return result
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(
                    f"Function {func.__name__} failed after {execution_time:.4f}s: {str(e)}",
                    extra={"execution_time": execution_time, "function": func.__name__},
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame"):
    """
    Log useful information about a pandas DataFrame.
    
    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name to identify the DataFrame in logs
    """
    logger.info(
        f"{name} info: shape={df.shape}, memory={df.memory_usage(deep=True).sum() / 1024**2:.2f}MB, "
        f"null_count={df.isnull().sum().sum()}, columns={list(df.columns)}"
    )


# Initialize default application logger
_default_log_dir = Path(__file__).parent.parent.parent / "logs"
app_logger = setup_logger(
    name="supply_chain_optix",
    log_level="INFO",
    log_dir=_default_log_dir,
    json_format=False
)