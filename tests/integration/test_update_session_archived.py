"""Integration test: updating an archived session must fail fast
instead of silently committing edits that have no observable effect.

Pre-fix the UPDATE WHERE clause in ``update_session`` filters only
``id = $1 AND account_id = $N`` — archived rows still match, so the
row's ``title`` / ``metadata`` / ``agent_id`` columns get rewritten
and the RETURNING-built response reports the new values back to the
caller as if the update succeeded.  The read paths that operators and
the worker rely on (``list_sessions`` filters ``archived_at IS NULL``;
the resolver and ``append_event`` refuse archived targets) never see
the rewritten fields — but the API has returned 200 OK with the
post-update payload, lying to the operator.

Same defect class as PR #523 (archived-session ``append_event``), PR
#547 (``update_session_template`` archived rewrite), and PR #554
(``update_vault`` archived rewrite — explicitly enumerated
``update_session`` as the still-open sibling).  The fix is symmetric
with the ``update_agent`` / ``update_environment`` / ``update_vault``
path that already raises ``ConflictError`` on archived rows.
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
from aios.services import sessions as sessions_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_session(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, session_id)`` for a session that has
    been archived after creation."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_sess_arch', NULL, TRUE, 'session-archived-test')
                """
            )
        agent = await agents_service.create_agent(
            pool,
            account_id="acc_sess_arch",
            name="sess-arch-test",
            model="openrouter/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await environments_service.create_environment(
            pool, account_id="acc_sess_arch", name="sess-arch-env"
        )
        session = await sessions_service.create_session(
            pool,
            account_id="acc_sess_arch",
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title="pre-archive",
            metadata={},
        )
        async with pool.acquire() as conn:
            archived = await queries.archive_session(conn, session.id, account_id="acc_sess_arch")
        assert archived.archived_at is not None
        yield pool, "acc_sess_arch", session.id
    finally:
        await pool.close()


@pytest.mark.parametrize(
    "update_kwargs",
    [
        pytest.param({"title": "post-archive"}, id="title-only"),
        pytest.param({"metadata": {"k": "v"}}, id="metadata-only"),
        # No settable fields exercises the ``return current`` branch
        # in ``queries.update_session`` (post-archive-check, pre-UPDATE),
        # distinct from the path the other parametrize cases route
        # through.  A refactor that re-introduced ``get_session(...)``
        # there without re-checking ``archived_at`` would silently
        # regress; this case is the canary.
        pytest.param({}, id="no-fields"),
    ],
)
async def test_update_session_refuses_archived(
    archived_session: tuple[asyncpg.Pool[Any], str, str],
    update_kwargs: dict[str, Any],
) -> None:
    """Pre-fix: ``service.update_session`` returns 200 with the
    post-update session payload; a follow-up SELECT shows the row's
    ``title`` was actually rewritten on the archived row.  Post-fix:
    raises ``ConflictError`` carrying the session id, and the row's
    ``title`` remains unchanged."""
    pool, account_id, session_id = archived_session

    with pytest.raises(ConflictError) as excinfo:
        await sessions_service.update_session(
            pool, session_id, account_id=account_id, **update_kwargs
        )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("id") == session_id

    # Defense-in-depth pin: the row's title must not have been
    # rewritten (the bare UPDATE would have matched and committed the
    # new value even on the archived row).
    async with pool.acquire() as conn:
        actual_title = await conn.fetchval("SELECT title FROM sessions WHERE id = $1", session_id)
    assert actual_title == "pre-archive", (
        f"archived session row was rewritten despite the refusal: "
        f"title is {actual_title!r}, expected 'pre-archive'."
    )
