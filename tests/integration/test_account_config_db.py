"""Integration tests (testcontainer-Postgres) for per-account timezone config:
the upward inheritance walk (``resolve_effective_timezone``) and the
partial-update semantics of ``update_account`` for the ``config`` column.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import register_jsonb_codec
from aios.models.accounts import AccountConfig


@pytest.fixture
async def conn(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    c = await asyncpg.connect(migrated_db_url)
    # Mirror the production pool: query functions read jsonb as native Python.
    await register_jsonb_codec(c)
    try:
        yield c
    finally:
        await c.close()


async def _make_chain(conn: asyncpg.Connection[Any]) -> None:
    """root → child → grand (a 3-deep tenant chain)."""
    await conn.execute(
        """
        INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
        VALUES ('root',  NULL,    TRUE,  'root'),
               ('child', 'root',  TRUE,  'child'),
               ('grand', 'child', FALSE, 'grand')
        """
    )


async def _set_tz(conn: asyncpg.Connection[Any], account_id: str, tz: str | None) -> None:
    # Raw write, deliberately bypassing queries.update_account: the resolver
    # tests stay isolated from writer bugs (the writer path is covered by
    # TestUpdateAccountConfig). tz=None writes an explicit JSON null
    # (config->>'timezone' → NULL → unset).
    await conn.execute(
        "UPDATE accounts SET config = $2::jsonb WHERE id = $1",
        account_id,
        json.dumps({"timezone": tz}),
    )


class TestResolveEffectiveTimezone:
    async def test_own_timezone_wins_over_ancestors(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_chain(conn)
        await _set_tz(conn, "child", "Europe/Paris")
        await _set_tz(conn, "grand", "America/New_York")
        assert await queries.resolve_effective_timezone(conn, "grand") == "America/New_York"

    async def test_inherits_nearest_ancestor(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_chain(conn)
        await _set_tz(conn, "root", "Asia/Tokyo")
        await _set_tz(conn, "child", "Europe/Paris")
        # grand has none → nearest set ancestor is child, not root.
        assert await queries.resolve_effective_timezone(conn, "grand") == "Europe/Paris"

    async def test_inherits_from_root(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_chain(conn)
        await _set_tz(conn, "root", "Asia/Tokyo")
        assert await queries.resolve_effective_timezone(conn, "grand") == "Asia/Tokyo"

    async def test_none_when_unset_up_to_root(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_chain(conn)
        # Nothing set anywhere → None (the service maps this to the UTC fallback).
        assert await queries.resolve_effective_timezone(conn, "grand") is None

    async def test_explicit_null_inherits(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_chain(conn)
        await _set_tz(conn, "root", "Asia/Tokyo")
        await _set_tz(conn, "child", None)  # explicit JSON null → treated as unset
        assert await queries.resolve_effective_timezone(conn, "grand") == "Asia/Tokyo"

    async def test_archived_ancestor_breaks_chain(self, conn: asyncpg.Connection[Any]) -> None:
        await _make_chain(conn)
        await _set_tz(conn, "root", "Asia/Tokyo")
        await conn.execute("UPDATE accounts SET archived_at = now() WHERE id = 'child'")
        # The archived parent breaks the upward walk before reaching root → None.
        assert await queries.resolve_effective_timezone(conn, "grand") is None


class TestUpdateAccountConfig:
    async def test_sets_and_returns_config_timezone(self, conn: asyncpg.Connection[Any]) -> None:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('root', NULL, TRUE, 'root')"
        )
        updated = await queries.update_account(
            conn, "root", config=AccountConfig(timezone="Europe/Paris")
        )
        assert updated is not None
        assert updated.config.timezone == "Europe/Paris"

    async def test_non_config_update_preserves_config(self, conn: asyncpg.Connection[Any]) -> None:
        # "Set one item without disrupting others": a later display_name-only
        # update touches only that column, leaving config.timezone intact.
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('root', NULL, TRUE, 'root')"
        )
        await queries.update_account(conn, "root", config=AccountConfig(timezone="Asia/Tokyo"))
        after = await queries.update_account(conn, "root", display_name="renamed")
        assert after is not None
        assert after.display_name == "renamed"
        assert after.config.timezone == "Asia/Tokyo"

    async def test_clear_timezone_reenables_inheritance(
        self, conn: asyncpg.Connection[Any]
    ) -> None:
        # An explicitly-passed None survives exclude_unset → the merge writes a
        # JSON null over the stored value → the child resolves the parent's tz
        # again. Locks the tri-state (omitted / null / value) merge semantics.
        await _make_chain(conn)
        await _set_tz(conn, "root", "Asia/Tokyo")
        await queries.update_account(conn, "child", config=AccountConfig(timezone="Europe/Paris"))
        assert await queries.resolve_effective_timezone(conn, "grand") == "Europe/Paris"
        await queries.update_account(conn, "child", config=AccountConfig(timezone=None))
        assert await queries.resolve_effective_timezone(conn, "grand") == "Asia/Tokyo"
