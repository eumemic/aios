"""Shared fixtures for ``tests/integration/`` (DB-backed, testcontainer-Postgres).

Anything that seeds reusable account / tenant state belongs here so
individual test modules don't re-roll the same scaffolding.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any
from unittest import mock
from unittest.mock import AsyncMock

import asyncpg
import psycopg
import pytest

from aios.db import queries
from aios.db.pool import register_jsonb_codec
from aios.models.agents import Agent, OutputStyle, ToolSpec
from aios.models.environments import Environment
from aios.models.events import Event
from aios.models.sessions import Session
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service

if TYPE_CHECKING:
    from aios.harness.step_context import StepContext


async def seed_agent_env_session(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    prefix: str,
    tools: list[ToolSpec] | None = None,
    output_style: OutputStyle = "default",
) -> tuple[Agent, Environment, Session]:
    """Seed a default ``(agent, env, session)`` trio scoped to ``account_id``.

    Used by integration tests that need a session-shaped scaffold but
    don't care about the specifics of the agent / environment / session
    rows. The agent name is ``{prefix}-agent``; the env name is
    ``{prefix}-env``. Other agent settings (``model="openrouter/test"``,
    ``window_min=50_000``, ``window_max=150_000``, empty system /
    description / metadata) match the long-standing conventions across
    the existing integration tests.
    """
    agent = await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=f"{prefix}-agent",
        model="openrouter/test",
        system="",
        tools=tools or [],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        output_style=output_style,
    )
    env = await environments_service.create_environment(
        pool, account_id=account_id, name=f"{prefix}-env"
    )
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id=account_id,
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title=None,
            metadata={},
        )
    return agent, env, session


@pytest.fixture
def stub_tool_provider() -> Iterator[None]:
    """A ``runtime.tool_provider`` that declares no connector tools, for tests
    that drive ``compute_step_prelude`` on a session with no connections."""
    from aios.harness import runtime

    prev = runtime.tool_provider
    tp = mock.Mock()
    tp.list_tools_for_session = AsyncMock(return_value=[])
    runtime.tool_provider = tp
    try:
        yield
    finally:
        runtime.tool_provider = prev


async def compose_step_for(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
    events: list[Event] | None = None,
    tail_budget: int | None = None,
    persist: bool = True,
) -> tuple[StepContext, list[Event]]:
    """Prelude → windowed read → compose: the step's own sequence, for real.

    ``events`` hands the composer a caller-chosen slate instead of the
    windowed read (so an ask can be 'scrolled out' without a real overflow);
    ``tail_budget`` sizes the real window for the EVENTS alone (the prelude's
    overhead is added back so the windower's budget is exactly this many
    tokens of log); ``None`` keeps the agent's generous window. Returns the
    composed context and the slate it composed over.
    """
    from aios.harness.step_context import (
        compose_step_context,
        compute_step_prelude,
        prelude_overhead_local,
    )

    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)
    prelude = await compute_step_prelude(
        pool,
        session_id,
        account_id=account_id,
        session=session,
        agent=agent,
        channels=[],
        memory_store_echoes=[],
    )
    omission = None
    if events is None:
        overhead = prelude_overhead_local(prelude)
        windowed = await sessions_service.read_windowed_events(
            pool,
            session_id,
            account_id=account_id,
            window_min=1 if tail_budget is not None else agent.window_min,
            window_max=overhead.total + tail_budget
            if tail_budget is not None
            else agent.window_max,
            model=agent.model,
            overhead_local=overhead,
        )
        events, omission = windowed.events, windowed.omission
    ctx = await compose_step_context(
        pool=pool,
        session=session,
        account_id=account_id,
        agent=agent,
        channels=[],
        prelude=prelude,
        events=events,
        omission=omission,
        persist_reminders=persist,
    )
    return ctx, events


@pytest.fixture
async def conn_two_accounts(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    """Asyncpg conn with one root + two child tenants (``acc_a``, ``acc_b``).

    The partial unique index ``accounts_one_active_root`` permits only
    a single non-archived ``parent_account_id IS NULL`` row at a time,
    so the root + two children layout is the minimum that supports
    cross-tenant tests.
    """
    conn = await asyncpg.connect(migrated_db_url)
    # Mirror the production pool: query functions read jsonb as native Python.
    await register_jsonb_codec(conn)
    try:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL,      TRUE,  'tenant-root'),
                   ('acc_a',    'acc_root', FALSE, 'tenant-a'),
                   ('acc_b',    'acc_root', FALSE, 'tenant-b')
            """
        )
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def _live_db_url(request: pytest.FixtureRequest) -> str:
    """Resolve the live-DB target for :func:`live_conn`, in two ways so the
    same test body runs both in CI and on a workstation with no Docker:

    * ``AIOS_TEST_LIVE_DB_URL`` — an externally provided, already-migrated
      Postgres (local dev / sandboxes where testcontainers can't run);
    * otherwise the standard session-scoped ``migrated_db_url`` testcontainer
      fixture, so these tests execute on the normal integration shard.

    Deliberately SYNC: ``migrated_db_url`` calls ``asyncio.run`` internally,
    so materializing it via ``getfixturevalue`` from inside an async fixture
    raises ``asyncio.run() cannot be called from a running event loop`` — and
    pytest then caches the session-scoped fixture as errored for every later
    test on the worker.  A sync hop materializes it outside the loop.
    """
    import os

    return os.environ.get("AIOS_TEST_LIVE_DB_URL") or request.getfixturevalue("migrated_db_url")


@pytest.fixture
async def live_conn(_live_db_url: str) -> AsyncIterator[asyncpg.Connection[Any]]:
    """Asyncpg connection to a migrated DB, TRUNCATEd before the test."""
    conn = await asyncpg.connect(_live_db_url)
    await register_jsonb_codec(conn)
    rows = await conn.fetch(
        "SELECT tablename FROM pg_tables "
        "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
    )
    if rows:
        await conn.execute(
            "TRUNCATE " + ", ".join(r["tablename"] for r in rows) + " RESTART IDENTITY CASCADE"
        )
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def migration_db_factory(postgres_container: Any) -> Iterator[Callable[[], str]]:
    """Mint fresh, empty databases on the shared session Postgres server.

    Migration tests replay alembic chains from zero, so they can't share the
    already-migrated ``migrated_db_url`` database — but they don't need their
    own *server* either: aios migrations touch only database-local state
    (tables, indexes, extensions), never cluster state.  ``CREATE DATABASE``
    on the shared container (~50 ms) replaces the per-test
    ``PostgresContainer`` boot (~1.5-4 s) the migration files used to pay.
    No per-database teardown — the container's session teardown is the
    cleanup.

    A session-scoped factory (rather than only a function-scoped URL fixture)
    so a module of read-only tests sharing one replayed schema can mint a
    single database at module scope.
    """
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    base = f"postgresql://{postgres_container.username}:{postgres_container.password}@{host}:{port}"
    counter = itertools.count()
    with psycopg.connect(f"{base}/{postgres_container.dbname}", autocommit=True) as admin:

        def make() -> str:
            name = f"aios_mig_{next(counter)}"
            admin.execute(f'CREATE DATABASE "{name}"')
            return f"{base}/{name}"

        yield make


@pytest.fixture
def migration_db_url(migration_db_factory: Callable[[], str]) -> str:
    """A fresh empty database for one test — see ``migration_db_factory``."""
    return migration_db_factory()
