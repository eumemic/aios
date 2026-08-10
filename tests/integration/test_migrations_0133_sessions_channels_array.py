"""Integration tests for migration 0133's ``sessions.channels`` backfill.

Migration 0133 adds ``sessions.channels text[] NOT NULL DEFAULT '{}'`` and
backfills existing rows from the event log's DISTINCT channel set
(issue #1742). These tests seed sessions + events directly via SQL (the
pre-migration schema shape), run the real alembic CLI up to head, and
assert the backfilled array matches the DISTINCT-channel set the old
``list_session_channels`` query would have returned — including the
zero-channel case, which must land as ``'{}'`` rather than NULL.

Mirrors the testcontainer-Postgres/real-alembic-CLI pattern of
``test_migrations_workspace_path_backfill.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres — each test mutates ``alembic_version``."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _seed(db_url: str) -> None:
    """Seed account/agent/env/sessions + events pre-migration (raw SQL).

    Three sessions:
    * ``sess_multi`` — messages on two channels plus a non-message event
      and a NULL-channel message; backfill must == {"slack/c1", "tg/c2"}.
    * ``sess_none`` — no channelled messages at all; backfill must == {}.
    * ``sess_dup`` — repeated messages on the same channel; backfill must
      still == {"sig/c3"} (DISTINCT collapses duplicates).
    """
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL, TRUE, 'root')
            """
        )
        await conn.execute(
            """
            INSERT INTO agents (
                id, account_id, name, model, system, tools, description, metadata,
                window_min, window_max, version, created_at, updated_at
            )
            VALUES (
                'agent_test', 'acc_root', 'test', 'openrouter/test', '', '[]'::jsonb,
                NULL, '{}'::jsonb, 50000, 150000, 1, now(), now()
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO environments (id, account_id, name, config, created_at)
            VALUES ('env_test', 'acc_root', 'test', '{}'::jsonb, now())
            """
        )
        await conn.execute(
            """
            INSERT INTO sessions (
                id, account_id, agent_id, environment_id, agent_version,
                title, metadata, workspace_volume_path, env,
                focal_channel, focal_locked, last_event_seq
            )
            VALUES
                ('sess_multi', 'acc_root', 'agent_test', 'env_test', 1,
                 NULL, '{}'::jsonb, '/tmp/ws-multi', '{}'::jsonb,
                 NULL, FALSE, 4),
                ('sess_none', 'acc_root', 'agent_test', 'env_test', 1,
                 NULL, '{}'::jsonb, '/tmp/ws-none', '{}'::jsonb,
                 NULL, FALSE, 1),
                ('sess_dup', 'acc_root', 'agent_test', 'env_test', 1,
                 NULL, '{}'::jsonb, '/tmp/ws-dup', '{}'::jsonb,
                 NULL, FALSE, 3)
            """
        )
        await conn.execute(
            """
            INSERT INTO events (id, session_id, seq, kind, data, role, account_id, channel)
            VALUES
              ('evt_m1', 'sess_multi', 1, 'message',
               '{"role": "user", "content": "hi"}'::jsonb, 'user', 'acc_root', 'slack/c1'),
              ('evt_m2', 'sess_multi', 2, 'message',
               '{"role": "assistant", "content": "hey"}'::jsonb, 'assistant',
               'acc_root', 'tg/c2'),
              ('evt_m3', 'sess_multi', 3, 'span',
               '{"event": "model_request_end"}'::jsonb, NULL, 'acc_root', NULL),
              ('evt_m4', 'sess_multi', 4, 'message',
               '{"role": "assistant", "content": "no channel"}'::jsonb, 'assistant',
               'acc_root', NULL),
              ('evt_none1', 'sess_none', 1, 'span',
               '{"event": "model_request_end"}'::jsonb, NULL, 'acc_root', NULL),
              ('evt_dup1', 'sess_dup', 1, 'message',
               '{"role": "user", "content": "a"}'::jsonb, 'user', 'acc_root', 'sig/c3'),
              ('evt_dup2', 'sess_dup', 2, 'message',
               '{"role": "user", "content": "b"}'::jsonb, 'user', 'acc_root', 'sig/c3'),
              ('evt_dup3', 'sess_dup', 3, 'message',
               '{"role": "assistant", "content": "c"}'::jsonb, 'assistant',
               'acc_root', 'sig/c3')
            """
        )
    finally:
        await conn.close()


async def _channels(db_url: str) -> dict[str, list[str]]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch("SELECT id, channels FROM sessions ORDER BY id")
        return {r["id"]: list(r["channels"]) for r in rows}
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_backfill_matches_distinct_channel_sets(postgres: object) -> None:
    import asyncio

    db_url = _alembic_url(postgres)

    result = _run_alembic(["upgrade", "0131"], db_url)
    assert result.returncode == 0, result.stderr

    asyncio.run(_seed(db_url))

    result = _run_alembic(["upgrade", "0133"], db_url)
    assert result.returncode == 0, result.stderr

    channels = asyncio.run(_channels(db_url))
    assert channels["sess_multi"] == sorted(["slack/c1", "tg/c2"])
    assert channels["sess_none"] == []
    assert channels["sess_dup"] == ["sig/c3"]


@needs_docker
@pytest.mark.integration
def test_downgrade_drops_column(postgres: object) -> None:
    import asyncio

    db_url = _alembic_url(postgres)

    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, result.stderr

    result = _run_alembic(["downgrade", "0131"], db_url)
    assert result.returncode == 0, result.stderr

    async def _has_column() -> bool:
        conn = await asyncpg.connect(db_url)
        try:
            row = await conn.fetchrow(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'sessions' AND column_name = 'channels'"
            )
            return row is not None
        finally:
            await conn.close()

    assert asyncio.run(_has_column()) is False
