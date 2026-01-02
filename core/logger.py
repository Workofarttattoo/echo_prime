"""
ECH0-PRIME Logging System
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Structured logging for all ECH0-PRIME components.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
import os


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for console output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        timestamp = datetime.now().strftime('%H:%M:%S')
        level = f"{color}[{record.levelname:8}]{reset}"
        module = f"{record.module:15}"
        message = record.getMessage()

        formatted = f"{timestamp} {level} {module} | {message}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logger(
    name: str,
    level: str = None,
    log_format: str = None,
    log_file: str = None
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 'json' for structured logging, 'human' for readable (default)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    # Get configuration from environment or use defaults
    level = level or os.getenv('LOG_LEVEL', 'INFO')
    log_format = log_format or os.getenv('LOG_FORMAT', 'human')

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if log_format == 'json':
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(HumanReadableFormatter())
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger


def log_cognitive_cycle(logger: logging.Logger, cycle_data: Dict[str, Any]):
    """Log a complete cognitive cycle with structured data."""
    logger.info(
        "Cognitive Cycle Complete",
        extra={
            "extra": {
                "free_energy": cycle_data.get("free_energy"),
                "surprise": cycle_data.get("surprise"),
                "actions": len(cycle_data.get("actions", [])),
                "mission_complete": cycle_data.get("mission_complete"),
            }
        }
    )


def log_safety_violation(logger: logging.Logger, intent: str, reason: str):
    """Log a safety violation with details."""
    logger.warning(
        "Safety Violation Detected",
        extra={
            "extra": {
                "intent": intent,
                "reason": reason,
                "severity": "HIGH"
            }
        }
    )


def log_llm_interaction(logger: logging.Logger, model: str, prompt_len: int, response_len: int):
    """Log LLM API interactions."""
    logger.debug(
        "LLM Interaction",
        extra={
            "extra": {
                "model": model,
                "prompt_length": prompt_len,
                "response_length": response_len
            }
        }
    )


# Create default logger for backward compatibility
default_logger = setup_logger("echo_prime")
