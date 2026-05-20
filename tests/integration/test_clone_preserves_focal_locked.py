"""Integration test: ``clone_session`` must preserve ``focal_locked``.

A clone that inherits ``focal_channel`` without ``focal_locked``
gives the clone's model a chat binding without the lock that the
parent's ``switch_channel`` gate relies on — the clone can call
``switch_channel`` to escape the per_chat isolation the resolver
established on the parent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration


@pytest.fixture(params=["telegram/bot/chat1", None], ids=["with-channel", "no-channel"])
async def focal_locked_parent(
    request: pytest.FixtureRequest, migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str | None]]:
    """Yield ``(pool, account_id, parent_session_id, parent_focal_channel)``
    for an idle parent created with ``focal_locked=True``.

    Parametrized over channel presence so the test pins the invariant
    that lock propagation is by the boolean alone, decoupled from
    whether ``focal_channel`` is set.
    """
    parent_focal_channel: str | None = request.param
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_clone_focal', NULL, TRUE, 'clone-focal-locked-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_clone_focal",
            name="clone-focal-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_clone_focal", name="clone-focal-env"
        )
        parent = await sessions_service.create_session(
            pool,
            account_id="acc_clone_focal",
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title="per-chat-parent",
            metadata={},
            focal_channel=parent_focal_channel,
            focal_locked=True,
        )
        # Pin the fixture so a green test cannot be due to the parent
        # itself never being locked in the first place.
        assert parent.focal_locked is True
        assert parent.focal_channel == parent_focal_channel
        yield pool, "acc_clone_focal", parent.id, parent_focal_channel
    finally:
        await pool.close()


async def test_clone_preserves_focal_locked(
    focal_locked_parent: tuple[asyncpg.Pool[Any], str, str, str | None],
) -> None:
    """A clone of a focal-locked parent must itself be focal-locked.

    Otherwise the clone inherits the bound channel without the lock,
    and its first ``switch_channel`` call escapes the per_chat
    isolation the operator (or the resolver) set on the parent.
    """
    pool, account_id, parent_id, parent_focal_channel = focal_locked_parent

    clone = await sessions_service.clone_session(pool, parent_id, account_id=account_id)

    assert clone.focal_channel == parent_focal_channel
    assert clone.focal_locked is True
