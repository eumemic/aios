"""Migration 0168 backfills only durable creation evidence.

``archive_when_idle`` is a lifetime choice, not ownership provenance. A public
API client can create a self-archiving root and another session can invoke it
later. The resulting first ``request_opened`` event must not transfer the
target's historical or future subtree spend to that caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import cast

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

_SEED_SQL = r"""
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_0168', NULL, TRUE, '0168 migration');
INSERT INTO environments (id, name, account_id)
VALUES ('env_0168', 'env-0168', 'acc_0168');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_0168', 'agent-0168', 'test/model', 'acc_0168');

INSERT INTO sessions (
    id, agent_id, environment_id, workspace_volume_path, account_id,
    archive_when_idle, last_event_seq, created_by_type, created_by_ref
)
VALUES
    ('ses_caller_0168', 'agent_0168', 'env_0168', '/tmp/caller-0168', 'acc_0168',
     FALSE, 0, 'api_actor', 'key_caller_0168'),
    -- Public API creation: lifetime is ephemeral, ownership is still root.
    ('ses_api_target_0168', 'agent_0168', 'env_0168', '/tmp/api-target-0168', 'acc_0168',
     TRUE, 1, 'api_actor', 'key_target_0168'),
    -- Explicit resource provenance is creation-specific and safe to recover.
    ('ses_provenance_target_0168', 'agent_0168', 'env_0168',
     '/tmp/provenance-target-0168', 'acc_0168', TRUE, 1,
     'session_actor', 'ses_caller_0168');

INSERT INTO events (id, session_id, seq, kind, data, account_id)
VALUES
    ('evt_api_later_0168', 'ses_api_target_0168', 1, 'lifecycle',
     '{"event":"request_opened","request_id":"req_api_later_0168",'
     '"caller":{"kind":"session","id":"ses_caller_0168","awaited":true}}'::jsonb,
     'acc_0168'),
    ('evt_provenance_0168', 'ses_provenance_target_0168', 1, 'lifecycle',
     '{"event":"request_opened","request_id":"req_provenance_0168",'
     '"caller":{"kind":"session","id":"ses_caller_0168","awaited":true}}'::jsonb,
     'acc_0168');
"""


@pytest.fixture
def postgres() -> Iterator[object]:
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


async def _creator(db_url: str, session_id: str) -> str | None:
    conn = await asyncpg.connect(db_url)
    try:
        return cast(
            "str | None",
            await conn.fetchval(
                "SELECT creator_session_id FROM sessions WHERE id = $1", session_id
            ),
        )
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_backfill_does_not_infer_creation_from_archive_when_idle(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0166"], db_url)
    assert up.returncode == 0, f"upgrade to 0166 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0168"], db_url)
    assert up.returncode == 0, f"upgrade to 0168 failed:\n{up.stderr}\n{up.stdout}"

    # The invocation-only edge remains unowned even though the target opted
    # into self-archival. Only explicit creation provenance is backfilled.
    assert asyncio.run(_creator(db_url, "ses_api_target_0168")) is None
    assert asyncio.run(_creator(db_url, "ses_provenance_target_0168")) == "ses_caller_0168"
