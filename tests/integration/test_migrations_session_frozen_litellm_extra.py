"""Integration test for migration 0104's grandfather backfill (#823).

0104 adds the ``litellm_extra`` jsonb column to ``sessions`` (the frozen, clamped
model-identity snapshot — ``api_base`` foremost) and backfills in-flight workflow
children — frozen rows with ``parent_run_id`` set that predate the column — by freezing
the pinned ``AgentVersion``'s ``litellm_extra`` verbatim. That is exactly what
``load_for_session`` returns for them today, so behavior is unchanged and no in-flight
run shifts its inference endpoint on its next step. Agentless generic children and
foreground sessions stay NULL (→ ``{}``, "no redirect, default endpoint").

Seed at 0103 (pre-column), ``upgrade head``, assert the backfill. Mirrors
``test_migrations_session_frozen_surface.py`` (the #794 surface-snapshot dual).
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


async def _seed_at_0103(db_url: str) -> None:
    """Seed a frozen named child + an agentless child + a foreground session at 0103."""
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
                litellm_extra, window_min, window_max, version, created_at, updated_at
            )
            VALUES ('agent_test', 'acc_root', 'test', 'openrouter/test', '',
                    '[]'::jsonb, NULL, '{}'::jsonb,
                    '{"api_base": "https://trusted.example"}'::jsonb,
                    50000, 150000, 1, now(), now())
            """
        )
        # The pinned version carries the litellm_extra the backfill must copy verbatim.
        await conn.execute(
            """
            INSERT INTO agent_versions (agent_id, version, model, account_id, litellm_extra)
            VALUES ('agent_test', 1, 'openrouter/test', 'acc_root',
                    '{"api_base": "https://trusted.example"}'::jsonb)
            """
        )
        await conn.execute(
            "INSERT INTO workflows (id, account_id, name, script) "
            "VALUES ('wf_1', 'acc_root', 'w', 'async def main(i): return i')"
        )
        await conn.execute(
            """
            INSERT INTO wf_runs (
                id, workflow_id, account_id, environment_id, script, script_sha,
                host_semantics_epoch
            )
            VALUES ('run_1', 'wf_1', 'acc_root', 'env_test',
                    'async def main(i): return i', 'sha', 0)
            """
        )
        # A frozen, in-flight named child (parent_run_id set, agent_version pinned).
        await conn.execute(
            """
            INSERT INTO sessions (
                id, account_id, agent_id, environment_id, agent_version,
                parent_run_id, surface_frozen, workspace_volume_path
            )
            VALUES ('sess_named', 'acc_root', 'agent_test', 'env_test', 1,
                    'run_1', TRUE, '/ws/named')
            """
        )
        # A frozen, in-flight agentless generic child (agent_id NULL) — stays NULL.
        await conn.execute(
            """
            INSERT INTO sessions (
                id, account_id, agent_id, environment_id, agent_version,
                parent_run_id, surface_frozen, model, workspace_volume_path
            )
            VALUES ('sess_generic', 'acc_root', NULL, 'env_test', NULL,
                    'run_1', TRUE, 'openrouter/test', '/ws/generic')
            """
        )
        # A foreground session (no parent_run_id) — must stay NULL.
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
def test_backfill_freezes_in_flight_children_model_identity(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    result = _run_alembic(["upgrade", "0103"], db_url)
    assert result.returncode == 0, f"upgrade 0103 failed:\n{result.stderr}\n{result.stdout}"

    asyncio.run(_seed_at_0103(db_url))

    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, (
        f"upgrade head (0104 backfill) failed:\n{result.stderr}\n{result.stdout}"
    )

    async def check() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            named = await conn.fetchrow(
                "SELECT litellm_extra FROM sessions WHERE id = 'sess_named'"
            )
            assert named is not None
            # Verbatim copy of the pinned AgentVersion's litellm_extra (read pre-0104).
            assert json.loads(named["litellm_extra"]) == {"api_base": "https://trusted.example"}

            # Agentless generic child + foreground session stay NULL (→ {} at read time).
            for sid in ("sess_generic", "sess_fg"):
                row = await conn.fetchrow("SELECT litellm_extra FROM sessions WHERE id = $1", sid)
                assert row is not None
                assert row["litellm_extra"] is None
        finally:
            await conn.close()

    asyncio.run(check())
