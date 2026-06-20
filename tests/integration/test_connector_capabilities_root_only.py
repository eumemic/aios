"""Integration tests: publishing a connector's ``capabilities`` is root-only,
and the per-session read query surfaces them through the same lineage walk as
``tools_schema`` (#1381).

``capabilities`` is a typed richness descriptor sibling to ``tools_schema`` on
the single-row-per-connector-type catalog.  It rides the same root-only
publication path: a child tenant publishing it would overwrite the global row
and change how every other tenant's session bound to a connection of that type
is rendered — the same cross-tenant rationale that motivates the
``tools_schema`` root-gate.  And it reads per session through the same lineage
walk so the descriptor surfaces without a second concept.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.errors import ForbiddenError
from aios.models.connectors import (
    ConnectorCapabilities,
    DraftStreaming,
    NativeButtons,
)
from aios.services import connectors as connectors_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def root_and_child(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, root_account_id, child_account_id)`` with a seeded
    ``echo`` connector row.  Mirrors the ``tools_schema`` root-only fixture."""
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


async def test_default_row_reads_as_empty_floor(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A freshly-seeded ``connectors`` row carries the ``'{}'`` default — the
    conservative rendering floor (all sub-descriptors absent)."""
    pool, _root_id, _child_id = root_and_child

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT capabilities FROM connectors WHERE connector = 'echo'")
    assert row is not None
    raw = row["capabilities"]
    caps_dict = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    assert caps_dict == {}
    floor = ConnectorCapabilities.model_validate(caps_dict)
    assert floor.draft_streaming is None
    assert floor.native_buttons is None


async def test_child_account_cannot_publish_capabilities(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A child account's runtime token must not publish capabilities — it
    would poison the global row and change every other tenant's rendering."""
    pool, _root_id, child_id = root_and_child

    with pytest.raises(ForbiddenError) as excinfo:
        await connectors_service.update_capabilities(
            pool,
            connector="echo",
            account_id=child_id,
            capabilities=ConnectorCapabilities(native_buttons=NativeButtons(max_buttons=5)),
        )

    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("connector") == "echo"

    # Defense-in-depth: the row must not have been written.
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT capabilities FROM connectors WHERE connector = 'echo'")
    assert row is not None
    raw = row["capabilities"]
    caps_dict = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    assert caps_dict == {}  # untouched JSONB default


async def test_root_account_can_publish_capabilities(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """The root account remains authorized — the connector type is a
    root-owned configuration; the published JSON lands on the row."""
    pool, root_id, _child_id = root_and_child

    published = ConnectorCapabilities(
        draft_streaming=DraftStreaming(overflow_limit=4000),
        native_buttons=NativeButtons(max_buttons=5),
    )
    await connectors_service.update_capabilities(
        pool,
        connector="echo",
        account_id=root_id,
        capabilities=published,
    )

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT capabilities FROM connectors WHERE connector = 'echo'")
    assert row is not None
    raw = row["capabilities"]
    caps_dict = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    assert ConnectorCapabilities.model_validate(caps_dict) == published


async def test_unknown_account_id_is_rejected(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A bogus account id (forged/stale token resolve) is rejected, not
    treated as root."""
    pool, _root_id, _child_id = root_and_child

    with pytest.raises(ForbiddenError):
        await connectors_service.update_capabilities(
            pool,
            connector="echo",
            account_id="acc_does_not_exist",
            capabilities=ConnectorCapabilities(),
        )
