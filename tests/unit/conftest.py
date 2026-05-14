"""Unit-test fixtures.

Provides dummy environment variables so ``get_settings()`` succeeds without
a ``.env`` file or real credentials.  Session-scoped and autouse so every
unit test gets them automatically.
"""

from __future__ import annotations

import base64
import os
import secrets
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from procrastinate import App
from procrastinate.testing import InMemoryConnector


@pytest.fixture(autouse=True, scope="session")
def _unit_env() -> Iterator[None]:
    """Inject minimal env vars required by ``Settings()``."""
    env = {
        "AIOS_API_KEY": secrets.token_urlsafe(16),
        "AIOS_VAULT_KEY": base64.b64encode(secrets.token_bytes(32)).decode(),
        "AIOS_DB_URL": "postgresql://test:test@localhost:5432/test",
    }
    with mock.patch.dict(os.environ, env):
        from aios.config import get_settings

        get_settings.cache_clear()
        yield
        get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _unit_runtime_tool_provider() -> Iterator[None]:
    """Register a no-op ``ToolProvider`` on ``runtime`` for unit tests.

    ``compute_step_prelude`` calls ``runtime.require_tool_provider()`` to
    merge connector-declared tools; without a registered impl the helper
    raises (its load-bearing behavior for production). Unit tests don't
    exercise the provider — they patch ``compose_step_context`` or stop
    short of the model call — so a returns-empty stub is enough.
    """
    from aios.harness import runtime

    class _NoopToolProvider:
        async def list_tools_for_session(self, pool: Any, session_id: str) -> list[Any]:
            return []

    runtime.tool_provider = _NoopToolProvider()
    try:
        yield
    finally:
        runtime.tool_provider = None


@pytest.fixture(autouse=True)
def _unit_runtime_mcp_broker() -> Iterator[None]:
    """Register an unstarted ``McpBroker`` on ``runtime`` for unit tests.

    Sandbox provisioning and teardown paths call
    ``runtime.require_mcp_broker()`` to register and clear per-session
    secrets; without an instance the helper raises. Tests that exercise
    those paths don't need a live HTTP server — only the in-memory
    secret map — so we hand them an instance that was never ``.start()``-ed.
    Tests that actually need ``.port`` should set ``broker._port`` directly.
    """
    from aios.harness import runtime
    from aios.sandbox.mcp_proxy import McpBroker

    runtime.mcp_broker = McpBroker()
    try:
        yield
    finally:
        runtime.mcp_broker = None


@pytest.fixture
async def in_memory_app() -> AsyncIterator[App]:
    """Patch the aios procrastinate app to use an in-memory connector.

    Tests can read ``app.connector.jobs`` directly to inspect deferred
    job rows (lock, queueing_lock, schedule, args).
    """
    from aios.harness.procrastinate_app import app

    with app.replace_connector(InMemoryConnector()) as patched:
        yield patched


def fake_pool_yielding_conn(conn: Any) -> Any:
    """Stand-in for ``asyncpg.Pool`` whose ``async with pool.acquire()`` yields *conn*."""
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    return pool
