"""Integration tests: the two operator-write bind paths must refuse to
bind a connection (single_session ``attach_connection``) or a chat
(``bind_chat_to_session``) to a *workflow-child* session.

A workflow-spawned child session (``parent_run_id IS NOT NULL``,
``focal_channel=NULL``) is the run-attenuated, surface-frozen execution
context of a workflow run — not a human-facing session — and the
connector inbound plane must never route a stranger's (or anyone's)
messages into it (issue #1502, part of the inbound-admission epic #1499).

There are exactly two operator-write paths that make a session
inbound-reachable as a resolver routing target:

* ``attach_connection`` (single_session bind) — writes a ``bindings``
  row the resolver returns as the tier-3 target for *every* ``chat_id``
  on the connection.
* ``bind_chat_to_session`` (per-chat override) — writes a
  ``chat_sessions`` ledger row the resolver short-circuits to as the
  tier-1 target.

Both already load the target ``Session`` (carrying ``parent_run_id``),
so a guard rejecting ``parent_run_id IS NOT NULL`` at each site is
zero extra round-trips. Asserting both guards reject a child covers
all three resolver tiers (tier-2 ``routing_rules`` has no operator
create path to guard today).

These tests mirror the sibling archived-session refusals
(``test_bind_chat_to_session_archived.py`` / the ``attach_connection``
archived check from #549): a synchronous 4xx (409 ``ConflictError``) at
the action that caused the misconfiguration, leaving no ledger /
binding row.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError
from aios.services import connections as connections_service
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration


@pytest.fixture
async def child_session_scaffold(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str, str, str]]:
    """Yield ``(pool, account_id, connection_id, child_session_id,
    normal_session_id)``.

    ``child_session_id`` is a workflow child (``parent_run_id`` non-NULL,
    inserted via :func:`queries.insert_child_session`); ``normal_session_id``
    is the ordinary human session seeded by ``seed_agent_env_session``,
    kept for the happy-path regression assertions.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_child_guard', NULL, TRUE, 'child-guard-test')
                """
            )
        agent, env, session = await seed_agent_env_session(
            pool, account_id="acc_child_guard", prefix="child-guard"
        )
        async with pool.acquire() as conn:
            connection = await queries.insert_connection(
                conn,
                connector="echo",
                external_account_id="test-account",
                metadata={},
                account_id="acc_child_guard",
            )
            # Seed a real parent run row: ``sessions.parent_run_id`` carries a
            # ``REFERENCES wf_runs(id)`` FK, so a non-existent id would trip the
            # FK at ``insert_child_session`` (NotFoundError) before the guard is
            # ever exercised. The guard only reads the column, but the row must
            # exist, so mint a genuine workflow + run and use its id.
            workflow = await wf_queries.insert_workflow(
                conn,
                account_id="acc_child_guard",
                name="child-guard-wf",
                script="async def main(input):\n    return input\n",
            )
            parent_run = await wf_queries.insert_wf_run(
                conn,
                account_id="acc_child_guard",
                workflow_id=workflow.id,
                environment_id=env.id,
                script=workflow.script,
                script_sha="deadbeef",
                host_semantics_epoch=HOST_SEMANTICS_EPOCH,
                depth=10,
            )
            child = await queries.insert_child_session(
                conn,
                session_id="sess_workflow_child",
                account_id="acc_child_guard",
                agent_id=agent.id,
                environment_id=env.id,
                agent_version=agent.version,
                model="openrouter/test",
                parent_run_id=parent_run.id,
                tools=[],
                mcp_servers=[],
                http_servers=[],
            )
        assert child is not None
        assert child.parent_run_id == parent_run.id
        yield pool, "acc_child_guard", connection.id, child.id, session.id
    finally:
        await pool.close()


async def test_attach_connection_refuses_workflow_child(
    child_session_scaffold: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """``attach_connection`` (single_session) must raise ``ConflictError``
    for a child session and stamp no ``bindings`` row — foreclosing the
    resolver tier-3 path."""
    pool, account_id, connection_id, child_session_id, _ = child_session_scaffold

    with pytest.raises(ConflictError) as excinfo:
        await connections_service.attach_connection(
            pool,
            connection_id,
            session_id=child_session_id,
            account_id=account_id,
        )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("session_id") == child_session_id, (
        f"ConflictError must carry the child session_id in detail. Got detail={detail!r}."
    )

    # No binding row may exist — the refusal is atomic.
    async with pool.acquire() as conn:
        active = await queries.get_active_binding(conn, connection_id, account_id=account_id)
    assert active is None, (
        f"bindings must not be stamped when the attach is refused; found {active!r}."
    )


async def test_bind_chat_to_session_refuses_workflow_child(
    child_session_scaffold: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """``bind_chat_to_session`` must raise ``ConflictError`` for a child
    session and stamp no ``chat_sessions`` ledger row — foreclosing the
    resolver tier-1 path."""
    pool, account_id, connection_id, child_session_id, _ = child_session_scaffold
    chat_id = "operator-bound-chat"

    with pytest.raises(ConflictError) as excinfo:
        await connections_service.bind_chat_to_session(
            pool,
            connection_id,
            chat_id=chat_id,
            session_id=child_session_id,
            account_id=account_id,
        )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("session_id") == child_session_id, (
        f"ConflictError must carry the child session_id in detail. Got detail={detail!r}."
    )

    async with pool.acquire() as conn:
        ledger_session_id = await queries.lookup_chat_session(
            conn, connection_id, chat_id, account_id=account_id
        )
    assert ledger_session_id is None, (
        f"chat_sessions ledger must not be stamped when the bind is refused; "
        f"found stamped session_id={ledger_session_id!r}."
    )


async def test_attach_connection_allows_normal_session(
    child_session_scaffold: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """Regression: attaching a connection to a normal human session
    (``parent_run_id IS NULL``) in single_session mode still succeeds and
    stamps the binding."""
    pool, account_id, connection_id, _, normal_session_id = child_session_scaffold

    await connections_service.attach_connection(
        pool,
        connection_id,
        session_id=normal_session_id,
        account_id=account_id,
    )

    async with pool.acquire() as conn:
        active = await queries.get_active_binding(conn, connection_id, account_id=account_id)
    assert active is not None
    assert active.session_id == normal_session_id


async def test_bind_chat_to_session_allows_normal_session(
    child_session_scaffold: tuple[asyncpg.Pool[Any], str, str, str, str],
) -> None:
    """Regression: binding a chat to a normal human session still
    succeeds and stamps the ``chat_sessions`` ledger row."""
    pool, account_id, connection_id, _, normal_session_id = child_session_scaffold
    chat_id = "operator-bound-chat-ok"

    await connections_service.bind_chat_to_session(
        pool,
        connection_id,
        chat_id=chat_id,
        session_id=normal_session_id,
        account_id=account_id,
    )

    async with pool.acquire() as conn:
        ledger_session_id = await queries.lookup_chat_session(
            conn, connection_id, chat_id, account_id=account_id
        )
    assert ledger_session_id == normal_session_id
