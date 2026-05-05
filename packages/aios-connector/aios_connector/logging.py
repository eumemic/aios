"""structlog wiring for connector subprocesses.

Mirrors the parent aios project's setup (stdlib LoggerFactory, JSON in
production, console renderer in development).  Per-connector env vars
``AIOS_<CONNECTOR_UPPER>_LOG_FORMAT`` and ``AIOS_<CONNECTOR_UPPER>_LOG_LEVEL``
let operators pick format and level without affecting siblings.

Called once from :func:`aios_connector.__main__.main` after the
connector is instantiated; connector authors don't need to call it
themselves.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(*, connector_name: str) -> None:
    """Configure structlog for the connector subprocess. Idempotent.

    Reads ``AIOS_<CONNECTOR_UPPER>_LOG_FORMAT`` (``console``/``json``;
    default ``console``) and ``AIOS_<CONNECTOR_UPPER>_LOG_LEVEL`` (default
    ``INFO``).  ``connector_name`` is the connector's :attr:`Connector.name`
    ClassVar — typically ``"signal"``, ``"telegram"``, etc.
    """
    upper = connector_name.upper()
    log_format = os.environ.get(f"AIOS_{upper}_LOG_FORMAT", "console")
    level_name = os.environ.get(f"AIOS_{upper}_LOG_LEVEL", "INFO").upper()
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
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
