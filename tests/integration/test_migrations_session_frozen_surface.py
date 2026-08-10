"""Integration test for migration 0079's grandfather backfill (#794).

0079 adds the frozen-surface columns to ``sessions`` (``tools``/``mcp_servers``/
``http_servers``/``surface_frozen``) and backfills in-flight workflow children — rows
with ``parent_run_id`` set that predate the columns — by freezing the pinned
``AgentVersion``'s surface verbatim. That is exactly what ``load_for_session`` returns
for them today, so behavior is unchanged and no in-flight run bricks on the new
fail-closed read path. Non-child sessions stay ``surface_frozen=false`` with NULL surface.

Seed at 0078 (pre-columns), ``upgrade head``, assert the backfill. Mirrors
``test_migrations_session_status_backfill.py`` (seed-at-N then upgrade against a fresh
per-test Postgres).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

pytestmark = pytest.mark.integration


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres — each test mutates ``alembic_version``."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _seed_at_0078(db_url: str) -> None:
    """Seed an in-flight child + a foreground session at revision 0078 (raw SQL)."""
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('acc_root', NULL, TRUE, 'root')"
        )
        await conn.execute(
            "INSERT INTO environments (id, account_id, name, config, created_at) "
            "VALUES ('env_test', 'acc_root', 'test', '{}'::jsonb, now())"
        )
        await conn.execute(
            """
            INSERT INTO agents (
                id, account_id, name, model, system, tools, description, metadata,
                window_min, window_max, version, created_at, updated_at
            )
            VALUES ('agent_test', 'acc_root', 'test', 'openrouter/test', '',
                    '[]'::jsonb, NULL, '{}'::jsonb, 50000, 150000, 1, now(), now())
            """
        )
        # The pinned version carries a non-empty surface the backfill must copy verbatim.
        await conn.execute(
            """
            INSERT INTO agent_versions (agent_id, version, model, account_id, tools, mcp_servers)
            VALUES ('agent_test', 1, 'openrouter/test', 'acc_root',
                    '[{"type": "bash"}]'::jsonb,
                    '[{"type": "url", "name": "s", "url": "https://s"}]'::jsonb)
            """
        )
        await conn.execute(
            "INSERT INTO workflows (id, account_id, name, script) "
            "VALUES ('wf_1', 'acc_root', 'w', 'async def main(i): return i')"
        )
        await conn.execute(
            """
            INSERT INTO wf_runs (id, workflow_id, account_id, environment_id, script, script_sha)
            VALUES ('run_1', 'wf_1', 'acc_root', 'env_test', 'async def main(i): return i', 'sha')
            """
        )
        # An in-flight child (parent_run_id set, agent_version pinned).
        await conn.execute(
            """
            INSERT INTO sessions (
                id, account_id, agent_id, environment_id, agent_version,
                parent_run_id, workspace_volume_path
            )
            VALUES ('sess_child', 'acc_root', 'agent_test', 'env_test', 1,
                    'run_1', '/ws/child')
            """
        )
        # A foreground session (no parent_run_id) — must stay unfrozen.
        await conn.execute(
            """
            INSERT INTO sessions (
                id, account_id, agent_id, environment_id, workspace_volume_path
            )
            VALUES ('sess_fg', 'acc_root', 'agent_test', 'env_test', '/ws/fg')
            """
        )
    finally:
        await conn.close()


@needs_docker
def test_backfill_freezes_in_flight_children(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    result = _run_alembic(["upgrade", "0078"], db_url)
    assert result.returncode == 0, f"upgrade 0078 failed:\n{result.stderr}\n{result.stdout}"

    asyncio.run(_seed_at_0078(db_url))

    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, (
        f"upgrade head (0079 backfill) failed:\n{result.stderr}\n{result.stdout}"
    )

    async def check() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            child = await conn.fetchrow(
                "SELECT surface_frozen, tools, mcp_servers FROM sessions WHERE id = 'sess_child'"
            )
            assert child is not None
            assert child["surface_frozen"] is True  # grandfathered → frozen
            # Verbatim copy of the pinned AgentVersion's surface (what it read pre-0079).
            # Compare parsed JSON — Postgres stores jsonb with canonical key order.
            assert json.loads(child["tools"]) == [{"type": "bash"}]
            assert json.loads(child["mcp_servers"]) == [
                {"type": "url", "name": "s", "url": "https://s"}
            ]

            fg = await conn.fetchrow(
                "SELECT surface_frozen, tools FROM sessions WHERE id = 'sess_fg'"
            )
            assert fg is not None
            assert fg["surface_frozen"] is False  # foreground → untouched
            assert fg["tools"] is None
        finally:
            await conn.close()

    asyncio.run(check())
