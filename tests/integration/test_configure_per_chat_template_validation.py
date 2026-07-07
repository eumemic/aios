"""Integration test: ``configure_per_chat`` must validate the caller-supplied
``session_template_id`` account-scoped (and refuse archived templates) *inside
the bind transaction* — mirroring ``attach_connection``'s session check.

Pre-fix (#1708), ``configure_per_chat`` dropped ``session_template_id`` straight
into ``insert_binding`` with no scoped get. The ``bindings.session_template_id``
FK is single-column (migration 0033; deliberately non-tenant-composite per
0110), so it only enforces *global* existence. Consequences:

* a cross-tenant **existence oracle** — binding a template id that lives in
  *another* account returned 200 (id exists globally) while a nowhere id
  returned 404; the 200/404 split leaks the id's existence across tenants;
* a **silent mis-bind** of a foreign or archived template that only surfaces
  later, as a DETACH from a different actor at inbound resolution.

The fix calls ``get_session_template(conn, id, account_id=account_id)`` and
refuses ``archived_at is not None`` before ``insert_binding``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError, NotFoundError
from aios.services import agents as agents_service
from aios.services import connections as connections_service
from aios.services import environments as environments_service
from aios.services import session_templates as session_templates_service

pytestmark = pytest.mark.integration


async def _make_template(
    pool: asyncpg.Pool[Any], *, account_id: str, name: str
) -> Any:
    agent = await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=f"agent-{name}",
        model="openrouter/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    env = await environments_service.create_environment(
        pool, account_id=account_id, name=f"env-{name}"
    )
    return await session_templates_service.create_session_template(
        pool,
        account_id=account_id,
        name=f"tpl-{name}",
        agent_id=agent.id,
        environment_id=env.id,
        agent_version=agent.version,
        vault_ids=[],
        memory_store_ids=[],
        metadata={},
    )


@pytest.fixture
async def pool_two_tenants(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, connection_id)`` where ``connection_id`` is an
    unbound connection owned by ``acc_a``. Two sibling tenants
    (``acc_a`` / ``acc_b``) exist so cross-tenant template ids are
    exercisable.
    """
    pool = await create_pool(migrated_db_url, min_size=2, max_size=8)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root', NULL,       TRUE,  'tenant-root'),
                       ('acc_a',    'acc_root', FALSE, 'tenant-a'),
                       ('acc_b',    'acc_root', FALSE, 'tenant-b')
                """
            )
            connection = await queries.insert_connection(
                conn,
                account_id="acc_a",
                connector="signal",
                external_account_id="+15550001",
                metadata={},
            )
        yield pool, connection.id
    finally:
        await pool.close()


async def test_foreign_template_raises_not_found(
    pool_two_tenants: tuple[asyncpg.Pool[Any], str],
) -> None:
    """A template that exists only in *another* account must be refused
    with ``NotFoundError`` — closing the cross-tenant existence oracle.
    """
    pool, connection_id = pool_two_tenants
    foreign = await _make_template(pool, account_id="acc_b", name="foreign")

    with pytest.raises(NotFoundError):
        await connections_service.configure_per_chat(
            pool,
            connection_id,
            account_id="acc_a",
            session_template_id=foreign.id,
        )

    # No binding row landed.
    async with pool.acquire() as conn:
        active = await conn.fetchval(
            "SELECT COUNT(*) FROM bindings "
            "WHERE connection_id = $1 AND archived_at IS NULL",
            connection_id,
        )
    assert active == 0


async def test_archived_own_template_raises_conflict(
    pool_two_tenants: tuple[asyncpg.Pool[Any], str],
) -> None:
    """An own-account but *archived* template must be refused at bind
    time (``ConflictError``) rather than silently binding.
    """
    pool, connection_id = pool_two_tenants
    template = await _make_template(pool, account_id="acc_a", name="archived")
    await session_templates_service.archive_session_template(
        pool, template.id, account_id="acc_a"
    )

    with pytest.raises(ConflictError):
        await connections_service.configure_per_chat(
            pool,
            connection_id,
            account_id="acc_a",
            session_template_id=template.id,
        )

    async with pool.acquire() as conn:
        active = await conn.fetchval(
            "SELECT COUNT(*) FROM bindings "
            "WHERE connection_id = $1 AND archived_at IS NULL",
            connection_id,
        )
    assert active == 0


async def test_own_live_template_binds(
    pool_two_tenants: tuple[asyncpg.Pool[Any], str],
) -> None:
    """Regression guard: an own-account, un-archived template still binds."""
    pool, connection_id = pool_two_tenants
    template = await _make_template(pool, account_id="acc_a", name="live")

    connection = await connections_service.configure_per_chat(
        pool,
        connection_id,
        account_id="acc_a",
        session_template_id=template.id,
    )
    assert connection.id == connection_id

    async with pool.acquire() as conn:
        bound = await conn.fetchval(
            "SELECT session_template_id FROM bindings "
            "WHERE connection_id = $1 AND mode = 'per_chat' AND archived_at IS NULL",
            connection_id,
        )
    assert bound == template.id
