"""Integration tests for the runtime-token ``connection_ids`` allowlist scope (#350).

Issuing a runtime token with ``connection_ids=[...]`` produces a token
that can only see / operate on the listed connections. Omitting the
field (or storing ``NULL``) leaves the token unscoped — the
pre-#350 behaviour. The migration must be backwards-safe: a row
written before the migration (or by a caller that didn't set the
column) resolves with ``connection_ids=None``.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.ids import RUNTIME_TOKEN, make_id
from aios.services import runtime_tokens as runtime_tokens_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool_with_account(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    """Yield ``(pool, account_id)`` for a non-archived account."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_rt_scope', NULL, TRUE, 'rt-scope-test')
                """
            )
            # Insert connector catalog row so runtime_tokens FK is satisfied.
            await conn.execute(
                "INSERT INTO connectors (connector) VALUES ('echo') ON CONFLICT DO NOTHING",
            )
        yield pool, "acc_rt_scope"
    finally:
        await pool.close()


async def test_issue_with_connection_ids_returns_scope(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """``service.issue(connection_ids=[...])`` round-trips the allowlist
    onto the ``RuntimeToken`` read view."""
    pool, account_id = pool_with_account
    token, _plaintext = await runtime_tokens_service.issue(
        pool,
        account_id=account_id,
        connector="echo",
        label="scoped",
        connection_ids=["c1", "c2"],
    )
    assert token.connection_ids == ["c1", "c2"]


async def test_resolve_returns_connection_ids_scope(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """``service.resolve()`` surfaces the allowlist on a ``ResolvedRuntimeToken``."""
    pool, account_id = pool_with_account
    _token, plaintext = await runtime_tokens_service.issue(
        pool,
        account_id=account_id,
        connector="echo",
        label="scoped",
        connection_ids=["c1", "c2"],
    )
    resolved = await runtime_tokens_service.resolve(pool, plaintext)
    assert resolved is not None
    assert resolved.connection_ids == ["c1", "c2"]


async def test_resolve_unscoped_token_returns_none_for_connection_ids(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """Issuing without ``connection_ids`` leaves ``connection_ids=None``
    on resolve. ``None`` is the "unscoped" sentinel; ``[]`` would mean
    "zero connections accessible" — distinct semantics."""
    pool, account_id = pool_with_account
    _token, plaintext = await runtime_tokens_service.issue(
        pool,
        account_id=account_id,
        connector="echo",
        label="unscoped",
    )
    resolved = await runtime_tokens_service.resolve(pool, plaintext)
    assert resolved is not None
    assert resolved.connection_ids is None


async def test_existing_db_row_with_null_connection_ids_resolves(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """Insert a token row directly without setting ``connection_ids`` —
    simulates a pre-migration row. ``resolve()`` must succeed and
    return ``connection_ids=None``. This is the backwards-compatibility
    contract: the migration adds the column nullable, and existing
    rows resolve as unscoped."""
    pool, account_id = pool_with_account
    plaintext = "aios_runtime_" + secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
    token_id = make_id(RUNTIME_TOKEN)
    async with pool.acquire() as conn:
        # Explicitly omit ``connection_ids`` from the INSERT so the
        # column stays at its (NULL) default — same as a row that
        # existed before the migration.
        await conn.execute(
            """
            INSERT INTO runtime_tokens (id, connector, label, token_hash, account_id)
            VALUES ($1, $2, $3, $4, $5)
            """,
            token_id,
            "echo",
            "pre-migration",
            token_hash,
            account_id,
        )
    resolved = await runtime_tokens_service.resolve(pool, plaintext)
    assert resolved is not None
    assert resolved.token_id == token_id
    assert resolved.connection_ids is None
