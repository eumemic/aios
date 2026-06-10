"""Unit coverage for the ``aios.db.listen`` context managers.

Every listener allocates a dedicated asyncpg connection and runs
``conn.add_listener`` (and, for :func:`listen_for_events`,
``acquire_subscriber_lock``) before yielding. If any of those setup
steps raise, the conn must still be closed — otherwise the dedicated
asyncpg connection leaks for the worker / API process lifetime,
slowly chewing through Postgres ``max_connections``.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.db import listen


def test_listener_application_name_uses_settings_instance_id() -> None:
    from aios.config import get_settings
    from aios.db.pool import listener_application_name

    expected = f"aios-listener:{get_settings().instance_id}"[:63]
    assert listener_application_name() == expected


def test_listener_application_name_explicit_and_truncates() -> None:
    from aios.db.pool import listener_application_name

    assert listener_application_name("abc") == "aios-listener:abc"
    long = listener_application_name("x" * 100)
    assert long.startswith("aios-listener:")
    assert len(long.encode()) <= 63


def _mk_listener(name: str) -> AbstractAsyncContextManager[Any]:
    """Build the listener context manager keyed by function name.

    Each listener has a different signature; this hides that variance so
    the test body can parametrize uniformly on the failure path.
    """
    db_url = "postgresql://stub/aios"
    if name == "listen_for_connector_result":
        return listen.listen_for_connector_result(db_url, "call_c1")
    if name == "listen_for_events":
        return listen.listen_for_events(db_url, "sess_X")
    if name == "listen_for_connector_calls_by_type":
        return listen.listen_for_connector_calls_by_type(db_url, "telegram")
    if name == "listen_for_management_calls":
        return listen.listen_for_management_calls(db_url, "telegram")
    if name == "listen_for_connection_discovery":
        return listen.listen_for_connection_discovery(db_url, "telegram")
    raise ValueError(f"unknown listener: {name}")


@pytest.mark.parametrize(
    "name",
    [
        "listen_for_connector_result",
        "listen_for_events",
        "listen_for_connector_calls_by_type",
        "listen_for_management_calls",
        "listen_for_connection_discovery",
    ],
)
async def test_add_listener_failure_closes_conn(name: str) -> None:
    """conn.terminate() must run if add_listener raises during setup."""
    conn = MagicMock()
    conn.add_listener = AsyncMock(side_effect=RuntimeError("simulated network blip"))

    with (
        patch("aios.db.listen.asyncpg.connect", AsyncMock(return_value=conn)),
        pytest.raises(RuntimeError, match="simulated network blip"),
    ):
        async with _mk_listener(name):
            pytest.fail("context manager should not yield when setup raises")

    conn.terminate.assert_called_once()


async def test_listen_for_events_acquire_subscriber_lock_failure_terminates_conn() -> None:
    """conn.terminate() must run if acquire_subscriber_lock raises after add_listener."""
    conn = MagicMock()
    conn.add_listener = AsyncMock()
    conn.remove_listener = AsyncMock()

    with (
        patch("aios.db.listen.asyncpg.connect", AsyncMock(return_value=conn)),
        patch(
            "aios.db.listen.acquire_subscriber_lock",
            AsyncMock(side_effect=RuntimeError("lock acquisition failed")),
        ),
        pytest.raises(RuntimeError, match="lock acquisition failed"),
    ):
        async with listen.listen_for_events("postgresql://stub/aios", "sess_X"):
            pytest.fail("context manager should not yield when setup raises")

    conn.terminate.assert_called_once()
