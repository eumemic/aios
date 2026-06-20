"""Integration test: ``list_connection_capabilities_for_session`` surfaces a
connector type's typed capability descriptor per session through the same
lineage walk as ``list_connection_tools_for_session`` (#1381).

Bind a session to an ``echo`` connection, publish capabilities for ``echo`` as
root, and assert the per-session read returns the published descriptor keyed by
connector type.  A connector whose row carries the empty ``'{}'`` row-default
reads back as the empty-floor model — the caller never special-cases
"no declared capabilities".
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.connectors import (
    ConnectorCapabilities,
    DraftStreaming,
    NativeButtons,
)
from aios.services import agents as agents_service
from aios.services import connectors as connectors_service
from aios.services import environments as environments_service

pytestmark = pytest.mark.integration


async def _seed_session_bound_to_echo(pool: asyncpg.Pool[Any]) -> str:
    """Create an acc_a session with an active single_session binding to an
    acc_a ``echo`` connection (which upserts the ``echo`` connectors row)."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL,       TRUE,  'tenant-root'),
                   ('acc_a',    'acc_root', FALSE, 'tenant-a')
            """
        )

    agent_a = await agents_service.create_agent(
        pool,
        account_id="acc_a",
        name="a-agent",
        model="openrouter/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env_a = await environments_service.create_environment(
        pool,
        account_id="acc_a",
        name="a-env",
    )
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id="acc_a",
            agent_id=agent_a.id,
            environment_id=env_a.id,
            agent_version=agent_a.version,
            title=None,
            metadata={},
        )
        connection = await queries.insert_connection(
            conn,
            account_id="acc_a",
            connector="echo",
            external_account_id="echo-1",
            metadata={},
        )
        await queries.insert_binding(
            conn,
            account_id="acc_a",
            connection_id=connection.id,
            mode="single_session",
            session_id=session.id,
        )
    return session.id


@pytest.fixture
async def pool_session_bound_to_echo(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        session_id = await _seed_session_bound_to_echo(pool)
        yield pool, session_id
    finally:
        await pool.close()


async def test_published_capabilities_surface_per_session(
    pool_session_bound_to_echo: tuple[asyncpg.Pool[Any], str],
) -> None:
    pool, session_id = pool_session_bound_to_echo

    published = ConnectorCapabilities(
        draft_streaming=DraftStreaming(overflow_limit=4000),
        native_buttons=NativeButtons(max_buttons=5),
    )
    await connectors_service.update_capabilities(
        pool,
        connector="echo",
        account_id="acc_root",
        capabilities=published,
    )

    async with pool.acquire() as conn:
        caps = await queries.list_connection_capabilities_for_session(
            conn, session_id, account_id="acc_a"
        )

    assert caps == {"echo": published}
    # The payoff: shared rendering code branches on a declared KIND.
    assert caps["echo"].draft_streaming is not None
    assert caps["echo"].native_buttons is not None


async def test_empty_row_default_reads_as_floor(
    pool_session_bound_to_echo: tuple[asyncpg.Pool[Any], str],
) -> None:
    """With no capabilities published, the ``'{}'`` row-default surfaces as the
    empty-floor model (every sub-descriptor absent)."""
    pool, session_id = pool_session_bound_to_echo

    async with pool.acquire() as conn:
        caps = await queries.list_connection_capabilities_for_session(
            conn, session_id, account_id="acc_a"
        )

    assert caps == {"echo": ConnectorCapabilities()}
    assert caps["echo"].draft_streaming is None
    assert caps["echo"].native_buttons is None
