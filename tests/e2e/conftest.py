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

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import pytest

from tests.e2e.harness import Harness


@pytest.fixture
async def harness(aios_env: dict[str, str]) -> AsyncIterator[Harness]:
    """Function-scoped harness: real Postgres, scripted model, no Docker."""
    import aios.tools  # noqa: F401  — register built-in tools
    from aios.config import get_settings
    from aios.crypto.vault import Vault
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios.harness.task_registry import TaskRegistry
    from aios.tools.registry import registry

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    vault = Vault.from_base64(settings.vault_key.get_secret_value())
    task_reg = TaskRegistry()

    # Save runtime globals
    prev = (
        runtime.pool,
        runtime.vault,
        runtime.task_registry,
        runtime.sandbox_registry,
        runtime.worker_id,
    )

    # Install
    runtime.pool = pool
    runtime.vault = vault
    runtime.task_registry = task_reg
    runtime.sandbox_registry = None  # no Docker in fast tier
    runtime.worker_id = "worker_test"

    # Snapshot tool registry
    tool_snapshot = dict(registry._tools)

    h = Harness(pool, task_reg)

    # Install mocks at fixture scope so they cover fire-and-forget tool tasks
    async def _fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        return await h._pop_response(**kwargs)

    async def _noop_defer_wake(session_id: str, *, cause: str = "message") -> None:
        pass

    with (
        mock.patch("aios.harness.completion.litellm.acompletion", _fake_acompletion),
        mock.patch("aios.harness.tool_dispatch.defer_wake", _noop_defer_wake),
    ):
        yield h

    # Restore
    registry._tools = tool_snapshot
    (
        runtime.pool,
        runtime.vault,
        runtime.task_registry,
        runtime.sandbox_registry,
        runtime.worker_id,
    ) = prev
    await task_reg.shutdown()
    await pool.close()


@pytest.fixture
async def docker_harness(aios_env: dict[str, str]) -> AsyncIterator[Harness]:
    """Like ``harness`` but with a real SandboxRegistry for Docker tests."""
    import aios.tools  # noqa: F401
    from aios.config import get_settings
    from aios.crypto.vault import Vault
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios.harness.task_registry import TaskRegistry
    from aios.sandbox.registry import SandboxRegistry
    from aios.tools.registry import registry

    settings = get_settings()
    pool = await create_pool(settings.db_url, min_size=1, max_size=4)
    vault = Vault.from_base64(settings.vault_key.get_secret_value())
    task_reg = TaskRegistry()
    sandbox_reg = SandboxRegistry()

    prev = (
        runtime.pool,
        runtime.vault,
        runtime.task_registry,
        runtime.sandbox_registry,
        runtime.worker_id,
    )
    runtime.pool = pool
    runtime.vault = vault
    runtime.task_registry = task_reg
    runtime.sandbox_registry = sandbox_reg
    runtime.worker_id = "worker_test"

    tool_snapshot = dict(registry._tools)
    h = Harness(pool, task_reg)

    async def _fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        return await h._pop_response(**kwargs)

    async def _noop_defer_wake(session_id: str, *, cause: str = "message") -> None:
        pass

    with (
        mock.patch("aios.harness.completion.litellm.acompletion", _fake_acompletion),
        mock.patch("aios.harness.tool_dispatch.defer_wake", _noop_defer_wake),
    ):
        yield h

    registry._tools = tool_snapshot
    (
        runtime.pool,
        runtime.vault,
        runtime.task_registry,
        runtime.sandbox_registry,
        runtime.worker_id,
    ) = prev
    await task_reg.shutdown()
    await sandbox_reg.release_all()
    await pool.close()
