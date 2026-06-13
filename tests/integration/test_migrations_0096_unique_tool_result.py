"""Migration 0096 promotes ``events_tool_result_idx`` to a partial UNIQUE
index on tool-result events ``(session_id, data->>'tool_call_id')`` and
backfills out any pre-existing duplicate rows (keeping the lowest-seq row).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# A minimal account/agent/env/session chain plus an assistant tool_call so a
# tool-role event has a parent to attach to. Two tool-role rows for the SAME
# (session_id, tool_call_id) simulate the historical duplicate-row race; the
# lower seq (1) is the winner, seq 2 is the clobbering late append.
_CHAIN_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO environments (id, name, account_id)
VALUES ('env_a', 'env-a', 'acc_root');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_a', 'agent-a', 'test/model', 'acc_root');
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id,
                      last_event_seq)
VALUES ('sess_a', 'agent_a', 'env_a', '/tmp/ws-a', 'acc_root', 2);
"""

_DUP_ROWS_SQL = """
INSERT INTO events (id, session_id, seq, kind, data, role, account_id)
VALUES
  ('evt_lo', 'sess_a', 1, 'message',
   '{"role": "tool", "tool_call_id": "tc_1", "content": "winner"}'::jsonb,
   'tool', 'acc_root'),
  ('evt_hi', 'sess_a', 2, 'message',
   '{"role": "tool", "tool_call_id": "tc_1", "content": "clobber"}'::jsonb,
   'tool', 'acc_root');
"""

_INSERT_DUP_SQL = """
INSERT INTO events (id, session_id, seq, kind, data, role, account_id)
VALUES ('evt_x', 'sess_a', 3, 'message',
        '{"role": "tool", "tool_call_id": "tc_1", "content": "second"}'::jsonb,
        'tool', 'acc_root');
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres; each test mutates alembic_version."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _fetchval(db_url: str, sql: str) -> object:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval(sql)
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_clean_database_upgrades_to_unique_tool_result_idx(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)

    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"
    indexdef = asyncio.run(
        _fetchval(
            db_url,
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'events_tool_result_idx'",
        )
    )
    assert indexdef is not None, "events_tool_result_idx missing after upgrade"
    assert "UNIQUE" in str(indexdef), f"index is not UNIQUE: {indexdef}"


@needs_docker
@pytest.mark.integration
def test_preexisting_duplicate_backfilled_keeping_lowest_seq(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0095"], db_url)
    assert up.returncode == 0, f"upgrade to 0095 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL + _DUP_ROWS_SQL))

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"

    # Exactly one tool-role row survives — the lowest-seq (winning) one.
    rows = asyncio.run(
        _fetchval(
            db_url,
            "SELECT count(*) FROM events WHERE session_id = 'sess_a' "
            "AND kind = 'message' AND role = 'tool' "
            "AND data->>'tool_call_id' = 'tc_1'",
        )
    )
    assert rows == 1, f"expected 1 surviving row, got {rows}"
    survivor = asyncio.run(
        _fetchval(
            db_url,
            "SELECT data->>'content' FROM events WHERE session_id = 'sess_a' "
            "AND kind = 'message' AND role = 'tool' "
            "AND data->>'tool_call_id' = 'tc_1'",
        )
    )
    assert survivor == "winner", f"kept the wrong row: {survivor}"


@needs_docker
@pytest.mark.integration
def test_unique_index_rejects_second_tool_result(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(
        _execute(
            db_url,
            _CHAIN_SQL + "INSERT INTO events (id, session_id, seq, kind, data, role, account_id) "
            "VALUES ('evt_lo', 'sess_a', 1, 'message', "
            '\'{"role": "tool", "tool_call_id": "tc_1", "content": "winner"}\'::jsonb, '
            "'tool', 'acc_root');",
        )
    )

    with pytest.raises(asyncpg.UniqueViolationError):
        asyncio.run(_execute(db_url, _INSERT_DUP_SQL))
