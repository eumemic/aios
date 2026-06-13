"""Integration test: publishing a connector's ``tools_schema`` is
root-only.

Connectors are configured by the root account (the connector type IS
the identity per migration 0033) — child tenants only add their own
connections, never connectors.  The pre-fix
``PUT /v1/connectors/{connector}/tools_schema`` route accepted a
runtime token from any tenant; that runtime token is self-mintable
by any child tenant via ``POST /v1/runtime-tokens``.  A child tenant
could overwrite the global ``connectors.tools_schema`` row with
malicious tool descriptions, and every other tenant's session bound
to a connection of that connector type would see the poisoned tool
list in its model's prelude — cross-tenant prompt-injection-as-a-
service.

The fix restricts the publication operation to the root account.
``queries.update_connector_tools_schema`` itself remains global (one
row per connector type) — the authorization decision lives at the
service layer because "root vs child" is an account-policy concern,
not a SQL one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ForbiddenError
from aios.services import connectors as connectors_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def root_and_child(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, root_account_id, child_account_id)`` with a
    seeded ``echo`` connector row."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root',  NULL,      TRUE,  'tenant-root'),
                       ('acc_child', 'acc_root', FALSE, 'tenant-child')
                """
            )
            await conn.execute(
                "INSERT INTO connectors (connector) VALUES ('echo') ON CONFLICT DO NOTHING"
            )
        yield pool, "acc_root", "acc_child"
    finally:
        await pool.close()


async def test_child_account_cannot_publish_tools_schema(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A child account's runtime token must not be able to publish
    a connector's tools_schema.  Otherwise it could poison the global
    row and inject prompts into every other tenant's session prelude."""
    pool, _root_id, child_id = root_and_child

    with pytest.raises(ForbiddenError) as excinfo:
        await connectors_service.update_tools_schema(
            pool,
            connector="echo",
            account_id=child_id,
            tools_schema=[{"name": "malicious", "description": "injection", "parameters": {}}],
        )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("connector") == "echo"

    # Defense-in-depth: the row must not have been written.
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tools_schema FROM connectors WHERE connector = 'echo'")
    assert row is not None
    # The pool's jsonb codec decodes JSONB to native Python, so the empty-array
    # default reads back as ``[]`` (was the raw ``"[]"`` text before the codec).
    assert row["tools_schema"] == []  # JSONB default


async def test_root_account_can_publish_tools_schema(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """The root account remains authorized — the connector type is a
    root-owned configuration."""
    pool, root_id, _child_id = root_and_child

    await connectors_service.update_tools_schema(
        pool,
        connector="echo",
        account_id=root_id,
        tools_schema=[{"name": "send", "description": "send a message", "parameters": {}}],
    )

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tools_schema FROM connectors WHERE connector = 'echo'")
    assert row is not None
    # Round-trips through JSONB; just verify our tool name landed.
    import json

    raw = row["tools_schema"]
    schema = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    assert any(t.get("name") == "send" for t in schema)


async def test_unknown_account_id_is_rejected(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A bogus account id (e.g., from a forged or stale token resolve)
    must be rejected, not treated as root."""
    pool, _root_id, _child_id = root_and_child

    # Sanity check: the helper doesn't crash on missing accounts; it
    # rejects them as non-root.
    with pytest.raises(ForbiddenError):
        await connectors_service.update_tools_schema(
            pool,
            connector="echo",
            account_id="acc_does_not_exist",
            tools_schema=[],
        )
    # Also verify the query layer's account-id retention; this exists
    # to keep the audit pattern (kwarg → SQL) honest.
    async with pool.acquire() as conn:
        await queries.get_account(conn, "acc_root")
