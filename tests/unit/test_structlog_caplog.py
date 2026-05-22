"""Tests verifying structlog is configured so caplog reliably captures log records.

The bare-default structlog config (what tests inherit when nothing calls
aios.logging.configure_logging at process start) uses PrintLoggerFactory, which
writes to stdout and bypasses Python's stdlib logging — making caplog blind to
structlog output. A separate trap is cache_logger_on_first_use=True (set in
production), which freezes a module's bound-logger config at first call.

The autouse fixture in tests/conftest.py neutralizes both: before every test it
installs stdlib.LoggerFactory and disables caching, so caplog captures
everything and reconfiguration (this fixture or any test-local override)
actually takes effect.
"""

from __future__ import annotations

import logging

import structlog


def test_structlog_routed_through_stdlib_for_caplog() -> None:
    """The conftest fixture must route structlog through stdlib.LoggerFactory."""
    config = structlog.get_config()
    assert isinstance(config["logger_factory"], structlog.stdlib.LoggerFactory)


def test_cache_logger_on_first_use_disabled() -> None:
    """The conftest fixture must disable structlog logger caching."""
    assert structlog.get_config()["cache_logger_on_first_use"] is False


def test_caplog_captures_module_level_structlog_logger(caplog) -> None:
    """caplog must capture log records emitted via a module-level structlog logger.

    Imports a module with module-level log binding (a class of modules that
    would otherwise have their loggers frozen at first use). Verifies the
    record reaches caplog through stdlib logging.
    """
    from aios.api.routers import connectors

    marker = "caplog_capture_marker_644"
    with caplog.at_level(logging.INFO):
        connectors.log.info(marker)

    assert any(marker in record.getMessage() for record in caplog.records), (
        f"caplog did not capture structlog message containing {marker!r}. "
        f"Records: {[r.getMessage() for r in caplog.records]}"
    )
