"""Shared fixtures for ``tests/integration/`` (DB-backed, testcontainer-Postgres).

Anything that seeds reusable account / tenant state belongs here so
individual test modules don't re-roll the same scaffolding.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest


@pytest.fixture
async def conn_two_accounts(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    """Asyncpg conn with one root + two child tenants (``acc_a``, ``acc_b``).

    The partial unique index ``accounts_one_active_root`` permits only
    a single non-archived ``parent_account_id IS NULL`` row at a time,
    so the root + two children layout is the minimum that supports
    cross-tenant tests.
    """
    conn = await asyncpg.connect(migrated_db_url)
    try:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL,      TRUE,  'tenant-root'),
                   ('acc_a',    'acc_root', FALSE, 'tenant-a'),
                   ('acc_b',    'acc_root', FALSE, 'tenant-b')
            """
        )
        yield conn
    finally:
        await conn.close()
