"""Integration test: updating an archived session_template must fail
fast instead of silently committing edits that have no observable
effect.

Pre-fix the UPDATE WHERE clause in ``update_session_template`` only
filtered ``id = $1 AND account_id = $N`` — archived rows still match,
so the row's columns get rewritten and the RETURNING-built response
reports the new values back to the caller as if the update succeeded.
The connector resolver's ``_spawn_per_chat_session`` already drops
inbounds whose binding points at an ``archived_at IS NOT NULL``
template (``ResolveDrop.ARCHIVED_TEMPLATE``), so the rewritten fields
never get observed by a downstream spawn — but the API has returned
200 OK with the post-update payload, lying to the operator.

Same defect class as PR #523 (archived-session ``append_event``) and
PR #521 (archived-connection inbound): a write succeeds at the DB
layer that the read/dispatch surface filters out, leaving the caller
with no signal that their request was a no-op.  The fix is symmetric
with the ``update_environment`` / ``update_agent`` path that already
raises ``ConflictError`` on archived rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import session_templates as session_templates_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_session_template(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, template_id)`` for a session_template
    that has been archived after creation."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_archived_tpl', NULL, TRUE, 'archived-template-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_archived_tpl",
            name="archived-tpl-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_archived_tpl", name="archived-tpl-test-env"
        )
        template = await session_templates_service.create_session_template(
            pool,
            account_id="acc_archived_tpl",
            name="original-name",
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )
        async with pool.acquire() as conn:
            await queries.archive_session_template(conn, template.id, account_id="acc_archived_tpl")
        yield pool, "acc_archived_tpl", template.id
    finally:
        await pool.close()


class TestUpdateSessionTemplateArchived:
    async def test_update_session_template_refuses_archived(
        self,
        archived_session_template: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """The service-layer ``update_session_template`` must raise
        ``ConflictError`` (→ 409 at the router) when targeting an
        archived row, instead of silently rewriting it and returning
        200 with the post-update payload.

        Mirrors the asymmetry the same-shape ``update_environment`` /
        ``update_agent`` paths already close — see ``queries.py`` for
        the explicit ``current.archived_at is not None`` raise in both.
        """
        pool, account_id, template_id = archived_session_template

        with pytest.raises(ConflictError):
            await session_templates_service.update_session_template(
                pool,
                template_id,
                account_id=account_id,
                name="post-archive-rewrite",
            )

        # And the row stays untouched — defense in depth so a future
        # regression at the service layer still has a guard at the DB
        # round-trip layer.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name, archived_at FROM session_templates WHERE id = $1",
                template_id,
            )
        assert row is not None
        assert row["name"] == "original-name"
        assert row["archived_at"] is not None
