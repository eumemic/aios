"""structlog wiring for the connector.

Mirrors the parent aios project's setup (stdlib LoggerFactory, JSON in
production, console renderer in development based on
``AIOS_TELEGRAM_LOG_FORMAT``).
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging() -> None:
    """Configure structlog once. Idempotent."""
    log_format = os.environ.get("AIOS_TELEGRAM_LOG_FORMAT", "console")
    level_name = os.environ.get("AIOS_TELEGRAM_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )

    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer()
        if log_format == "console"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
