"""E2E test fixtures.

The ``harness`` fixture provides a :class:`~tests.e2e.harness.Harness`
instance backed by a real testcontainer Postgres with migrations applied.
Two mocks are installed for the fixture's lifetime:

1. ``litellm.acompletion`` → pops scripted responses from the harness
2. ``defer_wake`` → no-op (tests drive steps manually)

Everything else is real: the step function, async tool dispatch, event
log, context builder, and (in the ``docker_harness`` variant) real
Docker containers.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest import mock

import pytest

from tests.e2e.harness import Harness
from tests.helpers.connections import authed_client


async def wait_for_predicate(
    predicate: Callable[[], bool | Awaitable[bool]],
    *,
    max_wait_s: float = 10.0,
    interval_s: float = 0.05,
) -> None:
    """Poll ``predicate()`` until truthy or ``max_wait_s`` elapses.
    Predicate may be sync or async."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_wait_s
    while loop.time() < deadline:
        result: bool | Awaitable[bool] = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        await asyncio.sleep(interval_s)
    raise AssertionError(f"predicate {predicate!r} did not become true within {max_wait_s}s")


async def _noop_defer_wake(
    pool: Any,
    session_id: str,
    *,
    account_id: str,
    cause: str = "message",
    delay_seconds: float | None = None,
    wake_reason: str | None = None,
) -> None:
    pass


@pytest.fixture
async def real_wake_setup(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    """Real pool against the session-migrated DB.

    Used by tests that exercise the real ``defer_wake`` and a real
    procrastinate worker — the ``harness`` fixture's ``defer_wake`` is
    a no-op and so unsuitable here. The procrastinate schema + aios
    lock-release trigger are applied once at session start by
    ``migrated_db_url``; this fixture only needs a fresh pool.
    """
    from aios.config import get_settings
    from aios.db.pool import create_pool

    pool = await create_pool(get_settings().db_url, min_size=1, max_size=8)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
def crypto_box(aios_env: dict[str, str]) -> Any:
    """The :class:`CryptoBox` keyed by the test instance's ``AIOS_VAULT_KEY``.

    Tests that call into ``aios.services.connections.create_connection`` /
    ``set_connection_secrets`` need this — those services accept a
    ``CryptoBox`` so the in-process encryption is testable without a
    running api process.
    """
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox

    return CryptoBox.from_base64(get_settings().vault_key.get_secret_value())


@pytest.fixture
async def harness(aios_env: dict[str, str]) -> AsyncIterator[Harness]:
    """Function-scoped harness: real Postgres, scripted model, no Docker."""
    import aios.tools  # noqa: F401  — register built-in tools
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios.harness.task_registry import TaskRegistry
    from aios.tools.registry import registry
    from aios_connectors.providers import SubsystemToolProvider

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    task_reg = TaskRegistry()

    # Save runtime globals
    prev = (
        runtime.pool,
        runtime.crypto_box,
        runtime.task_registry,
        runtime.sandbox_registry,
        runtime.worker_id,
        runtime.tool_provider,
    )

    # Install
    runtime.pool = pool
    runtime.crypto_box = crypto_box
    runtime.task_registry = task_reg
    runtime.sandbox_registry = None  # no Docker in fast tier
    runtime.worker_id = "worker_test"
    runtime.tool_provider = SubsystemToolProvider()

    # Snapshot tool registry
    tool_snapshot = dict(registry._tools)

    h = Harness(pool, task_reg)

    # Install mocks at fixture scope so they cover fire-and-forget tool tasks
    async def _fake_acompletion(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return h._pop_streaming_response(**kwargs)
        return h._pop_response(**kwargs)

    def _fake_chunk_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return h._build_chunk_response(*args, **kwargs)

    with (
        mock.patch("aios.harness.completion.litellm.acompletion", _fake_acompletion),
        mock.patch("aios.harness.completion.litellm.stream_chunk_builder", _fake_chunk_builder),
        mock.patch("aios.services.wake.defer_wake", _noop_defer_wake),
    ):
        yield h

    # Restore
    registry._tools = tool_snapshot
    (
        runtime.pool,
        runtime.crypto_box,
        runtime.task_registry,
        runtime.sandbox_registry,
        runtime.worker_id,
        runtime.tool_provider,
    ) = prev
    await task_reg.shutdown()
    await pool.close()


@pytest.fixture
async def docker_harness(aios_env: dict[str, str]) -> AsyncIterator[Harness]:
    """Like ``harness`` but with a real SandboxRegistry for Docker tests."""
    import aios.tools  # noqa: F401
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios.harness.task_registry import TaskRegistry
    from aios.sandbox.backends import make_backend
    from aios.sandbox.mcp_proxy import McpBroker
    from aios.sandbox.network import ensure_sandbox_network
    from aios.sandbox.registry import SandboxRegistry
    from aios.tools.registry import registry
    from aios_connectors.providers import SubsystemToolProvider

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    task_reg = TaskRegistry()
    sandbox_reg = SandboxRegistry(backend=make_backend(settings.sandbox_backend))
    # Mirror worker startup: create+attach the sandbox network before
    # bringing the broker up. Otherwise the very first sandbox a test
    # provisions hits ``docker run --network aios-sandbox`` against a
    # network that doesn't exist on this host yet.
    if settings.sandbox_backend == "docker":
        await ensure_sandbox_network()
    mcp_broker = McpBroker()
    await mcp_broker.start()

    prev = (
        runtime.pool,
        runtime.crypto_box,
        runtime.task_registry,
        runtime.sandbox_registry,
        runtime.mcp_broker,
        runtime.worker_id,
        runtime.tool_provider,
    )
    runtime.pool = pool
    runtime.crypto_box = crypto_box
    runtime.task_registry = task_reg
    runtime.sandbox_registry = sandbox_reg
    runtime.mcp_broker = mcp_broker
    runtime.worker_id = "worker_test"
    runtime.tool_provider = SubsystemToolProvider()

    tool_snapshot = dict(registry._tools)
    h = Harness(pool, task_reg)

    async def _fake_acompletion(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return h._pop_streaming_response(**kwargs)
        return h._pop_response(**kwargs)

    def _fake_chunk_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return h._build_chunk_response(*args, **kwargs)

    with (
        mock.patch("aios.harness.completion.litellm.acompletion", _fake_acompletion),
        mock.patch("aios.harness.completion.litellm.stream_chunk_builder", _fake_chunk_builder),
        mock.patch("aios.services.wake.defer_wake", _noop_defer_wake),
    ):
        yield h

    registry._tools = tool_snapshot
    (
        runtime.pool,
        runtime.crypto_box,
        runtime.task_registry,
        runtime.sandbox_registry,
        runtime.mcp_broker,
        runtime.worker_id,
        runtime.tool_provider,
    ) = prev
    await task_reg.shutdown()
    await sandbox_reg.release_all()
    await mcp_broker.stop()
    await pool.close()


@pytest.fixture
async def http_client(aios_env: dict[str, str]) -> AsyncIterator[Any]:
    """In-process API client wired against the testcontainer DB.

    Wakes are mocked out (no worker) so endpoints that ``defer_wake``
    (POST /messages, POST /tool-results, POST /connectors/inbound) don't
    trip on the absent procrastinate connector.
    """
    import httpx

    from aios.api.app import create_app
    from aios.config import get_settings
    from aios.crypto.vault import CryptoBox
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios_connectors.providers import SubsystemToolProvider

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    app = create_app()
    app.state.pool = pool
    app.state.crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    app.state.db_url = settings.db_url
    app.state.procrastinate = mock.MagicMock()
    # The fixture skips the API lifespan (httpx ASGITransport doesn't
    # fire startup/shutdown), so mirror the lifespan's runtime
    # registrations explicitly — the ``/context`` endpoint reaches
    # ``runtime.require_tool_provider()`` via ``compute_step_prelude``.
    prev_tool_provider = runtime.tool_provider
    runtime.tool_provider = SubsystemToolProvider()
    transport = httpx.ASGITransport(app=app)
    # Mock at every call site that imports ``defer_wake`` directly —
    # patching the source (``aios.services.wake``) is too late since the
    # importing modules already captured the unmocked reference.
    try:
        with (
            mock.patch("aios.api.routers.sessions.defer_wake", new_callable=mock.AsyncMock),
            mock.patch("aios.services.inbound.defer_wake", new_callable=mock.AsyncMock),
        ):
            async with authed_client(
                "http://testserver",
                aios_env["AIOS_API_KEY"],
                transport=transport,
            ) as client:
                yield client
    finally:
        runtime.tool_provider = prev_tool_provider
        await pool.close()
