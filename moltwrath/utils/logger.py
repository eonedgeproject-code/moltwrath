"""Logging with Rich formatting."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger(name: str = "moltwrath", level: str = "INFO") -> logging.Logger:
    """Setup a Rich-formatted logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def get_logger(name: str = "moltwrath") -> logging.Logger:
    return logging.getLogger(name)
