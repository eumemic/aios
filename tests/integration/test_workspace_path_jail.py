"""Integration test: ``workspace_path`` must be jailed within the
account's workspace subdirectory.

Pre-fix the ``SessionCreate.workspace_path`` validator only checked
``is_absolute()``.  ``ensure_workspace_path`` then ``mkdir(parents=True,
exist_ok=True)``'d any existing host path, and the sandbox spec
bind-mounted it read-write at ``/workspace``.  Any authenticated
client could:

- Set ``workspace_path="/etc"`` (or ``/var/log``, ``/Users/tom/...``)
  and read/write arbitrary host files through bash / write / edit
  tools.
- Set ``workspace_path=str(workspace_root / other_account_id /
  some_session)`` and cross into another tenant's session files,
  defeating the per-account-subdir isolation that ``insert_session``
  enforces by default.

The fix validates that the resolved ``workspace_path`` is within
``workspace_root/{account_id}/``.  ``Path.resolve()`` collapses
``..`` traversal and dereferences symlinks before the
``is_relative_to`` check, so neither escape vector slips through.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.config import get_settings
from aios.db.pool import create_pool
from aios.errors import ForbiddenError
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def two_accounts_with_agent_env(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str, str]]:
    """Yield ``(pool, acc_a, acc_b, agent_a_id, env_a_id)`` —
    two child accounts plus an agent + env owned by acc_a so we can
    build a sibling test that tenant A can't mount tenant B's path."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root',  NULL,      TRUE,  'tenant-root'),
                       ('acc_a',     'acc_root', FALSE, 'tenant-a'),
                       ('acc_b',     'acc_root', FALSE, 'tenant-b')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_a",
            name="ws-jail-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_a", name="ws-jail-env"
        )
        yield pool, "acc_a", "acc_b", agent.id, env.id
    finally:
        await pool.close()


async def test_create_session_rejects_arbitrary_host_path(
    two_accounts_with_agent_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """``workspace_path="/etc"`` (or any host path outside the
    workspace root) must be rejected before the row lands in the DB
    and before the sandbox provisioner gets a chance to bind-mount
    it."""
    pool, acc_a, _acc_b, agent_id, env_id = two_accounts_with_agent_env

    with pytest.raises(ForbiddenError):
        await sessions_service.create_session(
            pool,
            account_id=acc_a,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            workspace_path="/etc",
        )


async def test_create_session_rejects_cross_tenant_workspace_path(
    two_accounts_with_agent_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """A workspace_path that resolves into another account's
    subdirectory must be rejected.  Closes the per-account-subdir
    invariant ``insert_session`` enforces by default."""
    pool, acc_a, acc_b, agent_id, env_id = two_accounts_with_agent_env
    other_account_dir = str(get_settings().workspace_root / acc_b / "any_session")

    with pytest.raises(ForbiddenError):
        await sessions_service.create_session(
            pool,
            account_id=acc_a,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            workspace_path=other_account_dir,
        )


async def test_create_session_rejects_dotdot_traversal_back_to_other_tenant(
    two_accounts_with_agent_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """A ``..``-laced path that resolves into another account's subdir
    must be rejected.  ``Path.resolve()`` collapses ``..`` segments
    before the ``is_relative_to`` check; pinning this case ensures a
    future refactor swapping ``.resolve()`` for ``.absolute()`` (which
    preserves ``..``) is caught by the suite."""
    pool, acc_a, acc_b, agent_id, env_id = two_accounts_with_agent_env
    traversed = str(get_settings().workspace_root / acc_a / ".." / acc_b / "sneaky")

    with pytest.raises(ForbiddenError):
        await sessions_service.create_session(
            pool,
            account_id=acc_a,
            agent_id=agent_id,
            environment_id=env_id,
            title=None,
            metadata={},
            workspace_path=traversed,
        )


async def test_create_session_accepts_within_account_subdir(
    two_accounts_with_agent_env: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """Operators legitimately override ``workspace_path`` to share a
    workspace between sessions of the same account (per the
    ``clone_session`` docstring's "share a read-only volume between
    clones" example).  Paths within ``workspace_root/{account_id}/``
    must still pass."""
    pool, acc_a, _acc_b, agent_id, env_id = two_accounts_with_agent_env
    own_account_dir = str(get_settings().workspace_root / acc_a / "shared")

    session = await sessions_service.create_session(
        pool,
        account_id=acc_a,
        agent_id=agent_id,
        environment_id=env_id,
        title=None,
        metadata={},
        workspace_path=own_account_dir,
    )

    assert session.id.startswith("sess_")
