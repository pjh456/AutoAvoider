"""Shared logging configuration."""

from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Create a basic console logger used across modules."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
