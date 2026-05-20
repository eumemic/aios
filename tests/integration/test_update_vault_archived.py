"""Integration test: updating an archived vault must fail fast instead
of silently committing edits that have no observable effect.

Pre-fix the UPDATE WHERE clause in ``update_vault`` filters only
``id = $1 AND account_id = $N`` — archived rows still match, so the
row's ``display_name`` / ``metadata`` columns get rewritten and the
RETURNING-built response reports the new values back to the caller as
if the update succeeded.  The vault read path (``get_vault``,
``list_vaults``) filters ``archived_at IS NULL``, so the rewritten
fields are invisible to subsequent reads — but the API returned
200 OK with the post-update payload, lying to the operator.

Same defect class as PR #523 (archived-session ``append_event``), PR
#547 (``update_session_template`` archived rewrite), and the broader
\"archived resource accepts work\" sweep.  The fix is symmetric with
the ``update_environment`` / ``update_agent`` path that already
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
from aios.services import vaults as vaults_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_vault(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, vault_id)`` for a vault that has been
    archived after creation."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_vault_arch', NULL, TRUE, 'vault-archived-test')
                """
            )
        vault = await vaults_service.create_vault(
            pool,
            account_id="acc_vault_arch",
            display_name="pre-archive",
            metadata={},
        )
        async with pool.acquire() as conn:
            archived = await queries.archive_vault(conn, vault.id, account_id="acc_vault_arch")
        assert archived.archived_at is not None
        yield pool, "acc_vault_arch", vault.id
    finally:
        await pool.close()


async def test_update_vault_refuses_archived(
    archived_vault: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Pre-fix: ``update_vault`` returns 200 with the post-update vault
    payload (``display_name='post-archive'``); a follow-up SELECT shows
    the row's ``display_name`` was actually rewritten on the archived
    row.  Post-fix: raises ``ConflictError`` carrying the vault id and
    the row's ``display_name`` is unchanged."""
    pool, account_id, vault_id = archived_vault

    with pytest.raises(ConflictError) as excinfo:
        await vaults_service.update_vault(
            pool, vault_id, display_name="post-archive", account_id=account_id
        )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("id") == vault_id

    # Defense-in-depth pin: the row's display_name must not have been
    # rewritten (the bare UPDATE would have matched and committed the
    # new value even on the archived row).
    async with pool.acquire() as conn:
        actual_display_name = await conn.fetchval(
            "SELECT display_name FROM vaults WHERE id = $1", vault_id
        )
    assert actual_display_name == "pre-archive", (
        f"archived vault row was rewritten despite the refusal: "
        f"display_name is {actual_display_name!r}, expected 'pre-archive'."
    )
