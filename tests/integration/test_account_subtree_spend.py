"""Integration tests (testcontainer-Postgres) for subtree spend rollup:
the downward recursive aggregation over ``parent_account_id``
(``get_account_subtree_spent_microusd``) so an ancestor's reported spend
equals the sum of its (non-archived) subtree, mirroring the upward
``resolve_effective_spend_limit_usd`` CTE.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import register_jsonb_codec


@pytest.fixture
async def conn(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    c = await asyncpg.connect(migrated_db_url)
    await register_jsonb_codec(c)
    try:
        yield c
    finally:
        await c.close()


async def _make_tree(conn: asyncpg.Connection[Any]) -> None:
    """A small tree:

    root
    ├── childA
    │   └── grand
    └── childB
    """
    await conn.execute(
        """
        INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
        VALUES ('root',   NULL,     TRUE,  'root'),
               ('childA', 'root',   TRUE,  'childA'),
               ('childB', 'root',   FALSE, 'childB'),
               ('grand',  'childA', FALSE, 'grand')
        """
    )


async def _set_spent(conn: asyncpg.Connection[Any], account_id: str, micro: int) -> None:
    await conn.execute("UPDATE accounts SET spent_microusd = $2 WHERE id = $1", account_id, micro)


class TestSubtreeSpentMicrousd:
    async def test_leaf_equals_own_spend(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_tree(conn)
        await _set_spent(conn, "grand", 500)
        # A leaf's subtree is just itself.
        assert await queries.get_account_subtree_spent_microusd(conn, "grand") == 500

    async def test_ancestor_sums_descendants_and_self(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_tree(conn)
        await _set_spent(conn, "root", 1)
        await _set_spent(conn, "childA", 10)
        await _set_spent(conn, "childB", 100)
        await _set_spent(conn, "grand", 1000)
        # root sees the entire tree.
        assert await queries.get_account_subtree_spent_microusd(conn, "root") == 1111
        # childA sees itself + grand only (not childB or root).
        assert await queries.get_account_subtree_spent_microusd(conn, "childA") == 1010
        # childB is a leaf.
        assert await queries.get_account_subtree_spent_microusd(conn, "childB") == 100

    async def test_zero_when_nothing_spent(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_tree(conn)
        assert await queries.get_account_subtree_spent_microusd(conn, "root") == 0

    async def test_archived_descendant_severs_subtree(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_tree(conn)
        await _set_spent(conn, "root", 1)
        await _set_spent(conn, "childA", 10)
        await _set_spent(conn, "grand", 1000)
        # Archiving childA severs it (and its subtree) from root's live rollup,
        # mirroring how the limit-inheritance walk breaks at archived nodes.
        await conn.execute("UPDATE accounts SET archived_at = now() WHERE id = 'childA'")
        assert await queries.get_account_subtree_spent_microusd(conn, "root") == 1

    async def test_archived_root_is_zero(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_tree(conn)
        await _set_spent(conn, "root", 1)
        await conn.execute("UPDATE accounts SET archived_at = now() WHERE id = 'root'")
        # The anchor row itself is archived → nothing to roll up.
        assert await queries.get_account_subtree_spent_microusd(conn, "root") == 0

    async def test_missing_account_is_zero(self, conn: asyncpg.Connection[Any]) -> None:
        assert await queries.get_account_subtree_spent_microusd(conn, "nope") == 0


class TestSubtreeVsFlatRead:
    async def test_flat_read_unchanged(self, conn: asyncpg.Connection[Any]) -> None:
        # The flat single-row meter is independent of the subtree rollup:
        # the ancestor's own meter stays its own write.
        await _make_tree(conn)
        await _set_spent(conn, "root", 7)
        await _set_spent(conn, "grand", 1000)
        assert await queries.get_account_spent_microusd(conn, "root") == 7
        assert await queries.get_account_subtree_spent_microusd(conn, "root") == 1007
