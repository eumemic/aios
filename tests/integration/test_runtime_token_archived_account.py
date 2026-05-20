"""Integration test: runtime-token resolution must refuse archived accounts.

Symmetric to ``lookup_account_by_key_hash`` which correctly filters
``accounts.archived_at IS NULL``. ``resolve_runtime_token`` only
filters ``revoked_at IS NULL`` on the token row, leaving an asymmetric
auth gate: account-key bearers are refused for archived accounts,
runtime-token bearers (used by connector runtime containers per
#328 PR 5) are not.

Reachability: an admin archives a child account; the child's connector
runtime container (Telegram bot, Signal bot, HTTP poller, etc.) is
unaware and keeps POSTing to ``/v1/connectors/runtime/...`` endpoints
with its bearer token. ``require_runtime_auth`` resolves successfully;
downstream queries scope to the archived account. ``handle_inbound``'s
``connection.archived_at`` check (PR #521) catches the case where the
*connection* was also archived, but not the case where only the
*parent account* was archived — connection-level archival isn't a
cascade of account-level archival in the current schema.

The runtime-token resolve is the right layer for the check: it's the
single auth bootstrap that downstream scoping inherits from. The fix
mirrors the JOIN in ``lookup_account_by_key_hash`` — one query, one
``AND accounts.archived_at IS NULL`` clause.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.services import runtime_tokens as runtime_tokens_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def archived_account_with_runtime_token(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, account_id, plaintext_token)`` for a runtime token
    issued before the account was archived."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_rt_arch', NULL, TRUE, 'rt-archived-test')
                """
            )
        _token, plaintext = await runtime_tokens_service.issue(
            pool,
            account_id="acc_rt_arch",
            connector="echo",
            label="pre-archive",
        )
        async with pool.acquire() as conn:
            archived = await queries.archive_account(conn, "acc_rt_arch")
        assert archived is not None and archived.archived_at is not None
        yield pool, "acc_rt_arch", plaintext
    finally:
        await pool.close()


async def test_runtime_token_resolve_refuses_archived_account(
    archived_account_with_runtime_token: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """Pre-fix: ``runtime_tokens_service.resolve`` returns a
    ``ResolvedRuntimeToken`` carrying the archived ``account_id``,
    letting the runtime container continue operating on a
    decommissioned tenant. Post-fix: returns ``None`` so the auth
    middleware raises 401 — same surface ``require_bearer_auth`` uses
    for account-key bearers on archived accounts."""
    pool, _account_id, plaintext = archived_account_with_runtime_token

    resolved = await runtime_tokens_service.resolve(pool, plaintext)

    assert resolved is None, (
        f"runtime-token resolve must refuse archived accounts so the auth "
        f"gate is symmetric with account-key auth (which "
        f"``lookup_account_by_key_hash`` already filters via "
        f"``accounts.archived_at IS NULL``). Pre-fix returns "
        f"ResolvedRuntimeToken on the archived account; post-fix returns "
        f"None and the middleware 401s. Got: {resolved!r}."
    )
