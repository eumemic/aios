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
        "AIOS_EGRESS_CA_KEY": base64.b64encode(secrets.token_bytes(32)).decode(),
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

        async def list_capabilities_for_session(self, pool: Any, session_id: str) -> dict[str, Any]:
            return {}

    runtime.tool_provider = _NoopToolProvider()
    try:
        yield
    finally:
        runtime.tool_provider = None


@pytest.fixture(autouse=True)
def _unit_runtime_tool_broker() -> Iterator[None]:
    """Register an unstarted ``ToolBroker`` on ``runtime`` for unit tests.

    Sandbox provisioning and teardown paths call
    ``runtime.require_tool_broker()`` to register and clear per-session
    secrets; without an instance the helper raises. Tests that exercise
    those paths don't need a live HTTP server — only the in-memory
    secret map — so we hand them an instance that was never ``.start()``-ed.
    Tests that actually need ``.port`` should set ``broker._port`` directly.
    """
    from aios.harness import runtime
    from aios.sandbox.tool_broker import ToolBroker

    runtime.tool_broker = ToolBroker()
    try:
        yield
    finally:
        runtime.tool_broker = None


@pytest.fixture
async def in_memory_app() -> AsyncIterator[App]:
    """Patch the aios procrastinate app to use an in-memory connector.

    Tests can read ``app.connector.jobs`` directly to inspect deferred
    job rows (lock, queueing_lock, schedule, args).

    The ``app`` is a module-level singleton, so task registration performed by
    one test (importing ``aios.harness.tasks``) leaks into every later test in
    the same process. Under ``pytest -n`` that ordering is non-deterministic and
    silently defeats the ``_assert_no_registered_tasks`` guard in the wake
    routing tests (issue #1699): a test that imported the harness graph earlier
    in the same worker leaves ``harness.*`` registered, so the guard trips even
    though the code under test is correct.

    Snapshot ``app.tasks`` and reset it to the pristine, task-free state the api
    process actually has on entry; restore the snapshot on exit. Each test that
    needs the registration (``TestRoutingMatchesRegistration``) re-imports
    ``aios.harness.tasks`` inside its body, which re-registers on demand.
    """
    from aios.jobs.app import app

    saved_tasks = dict(app.tasks)
    app.tasks.clear()
    try:
        with app.replace_connector(InMemoryConnector()) as patched:
            yield patched
    finally:
        app.tasks.clear()
        app.tasks.update(saved_tasks)


def fake_pool_yielding_conn(conn: Any, **kwargs: Any) -> Any:
    """Stand-in for ``asyncpg.Pool`` whose ``async with pool.acquire()`` yields *conn*."""
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm
    # ``asyncpg.Pool.execute`` is awaitable — used directly (no acquire) by
    # post-commit ``pg_notify`` call sites; keep the stand-in faithful.
    pool.execute = AsyncMock()
    return pool


@pytest.fixture(autouse=True)
def _unit_load_session_account_id_stub() -> Iterator[None]:
    """Auto-mock the worker-side account_id bootstrap.

    Most unit tests mock ``pool`` and ``conn`` to MagicMocks; the real
    ``load_session_account_id`` calls ``conn.fetchrow`` which a MagicMock
    can't await. Patching the function globally with an AsyncMock that
    returns the test stub means tests don't have to know about the
    bootstrap helper at all.
    """
    with mock.patch(
        "aios.services.sessions.load_session_account_id",
        new_callable=AsyncMock,
        return_value="acc_test_stub",
    ):
        yield


@pytest.fixture(autouse=True)
def _unit_no_session_cancel_harvest() -> Iterator[None]:
    """Auto-mock the session step's pre-inference cancel leaf.

    ``_run_session_step_body`` now calls the DB-backed ``harvest_session_cancel_markers``
    before inference. Unit tests drive ``run_session_step`` over a MagicMock pool that can't
    await ``conn.transaction()``, so default the leaf to a no-op (no markers) — exactly as the
    sibling ``load_session_account_id`` stub above. The leaf's real behavior is covered by the
    integration tier (``test_session_cancel_leaf``)."""
    with mock.patch(
        "aios.services.sessions.harvest_session_cancel_markers",
        new_callable=AsyncMock,
        return_value=False,
    ):
        yield


@pytest.fixture(autouse=True)
def _unit_no_scan_floor_advance() -> Iterator[None]:
    """Auto-mock the step prelude's perf-only scan-floor ratchet leaf (#1747).

    ``compute_step_prelude`` calls
    ``_advance_open_request_scan_floor_best_effort`` (which opens its own
    ``conn.transaction()`` savepoint) right after ``get_open_obligations`` on
    every step. Unit tests drive it over a MagicMock pool/conn that can't
    await ``conn.transaction()`` — exactly the same shape as the sibling
    ``harvest_session_cancel_markers`` stub above — so default the leaf to a
    no-op here rather than let every such test hit the mock's ``TypeError``
    (the leaf swallows it internally by design, but it still logs a warning
    and leaves an unawaited-coroutine warning behind from the MagicMock
    transaction context manager). The leaf's real behavior is covered by the
    integration tier (``tests/integration/test_open_request_scan_floor.py``).
    """
    with mock.patch(
        "aios.harness.step_context._advance_open_request_scan_floor_best_effort",
        new_callable=AsyncMock,
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _unit_spend_state_ungated() -> Iterator[None]:
    """Auto-mock the pre-inference spend-admission collaborators.

    ``_run_session_step_body`` reads the rolled-up subtree spend envelope before
    context-build. Unit tests drive ``run_session_step`` over a MagicMock pool
    whose ``acquire().__aenter__().fetchval()`` returns a MagicMock, which the
    real ``get_account_subtree_spend_state`` would compare against the limit
    (``>=``) and raise ``TypeError``. Default both spend-state reads to the
    ungated ``(0, None)`` — exactly as the sibling ``load_session_account_id``
    stub above. The gate's real behavior is covered by ``test_loop_spend_gate``,
    which patches these at the ``aios.harness.loop.accounts_service`` call site.
    """
    with (
        mock.patch(
            "aios.services.accounts.get_account_subtree_spend_state",
            new_callable=AsyncMock,
            return_value=(0, None),
        ),
        mock.patch(
            "aios.services.accounts.get_account_spend_state",
            new_callable=AsyncMock,
            return_value=(0, None),
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _unit_provider_auth_ungated() -> Iterator[None]:
    """Auto-mock the pre-inference provider-auth-conflict guard's collaborator.

    The inline-model-call arm resolves the account's ``model_providers`` config
    and checks it against ``litellm_extra`` before every raw model call.
    ``runtime.require_crypto_box()`` raises outside a real worker context, and
    the real resolver would hit the (fake) pool. Default to a clean pass — no
    resolved row, no conflict — exactly as the sibling spend-state stub above.
    The guard's real behavior is covered by ``test_loop_provider_guard``, which
    patches this at the ``aios.harness.loop.model_providers_service`` call site.
    """
    with (
        mock.patch("aios.harness.runtime.require_crypto_box", return_value=MagicMock()),
        mock.patch(
            "aios.services.model_providers.resolve_provider_auth_or_conflict",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
    ):
        yield
