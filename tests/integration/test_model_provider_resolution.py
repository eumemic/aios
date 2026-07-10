"""Integration tests for `resolve_model_provider`'s nearest-ancestor CTE.

Uses a 3-level chain (root -> mid -> leaf, distinct from the flat
root+2-siblings `conn_two_accounts` fixture) to pin: depth ordering,
row-atomicity (a winning row's api_base and key always come from the SAME
account, never field-merged), an archived ROW falling through to the next
ancestor, and an archived ACCOUNT severing the chain entirely (even when the
account itself never had a provider row).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.crypto.vault import KEY_BYTES, CryptoBox
from aios.db import queries

pytestmark = pytest.mark.integration


@pytest.fixture
async def chain_conn(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    """A 3-level account chain: acc_root -> acc_mid -> acc_leaf."""
    conn = await asyncpg.connect(migrated_db_url)
    try:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('acc_root', NULL,       TRUE,  'root'),
                   ('acc_mid',  'acc_root', TRUE,  'mid'),
                   ('acc_leaf', 'acc_mid',  FALSE, 'leaf')
            """
        )
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(KEY_BYTES))


async def _insert(
    conn: asyncpg.Connection[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    provider: str = "anthropic",
    api_key: str = "sk-x",
    api_base: str | None = None,
) -> str:
    blob = crypto_box.derive_account_subkey(account_id).encrypt(api_key)
    row = await queries.insert_model_provider(
        conn, account_id=account_id, provider=provider, api_base=api_base, blob=blob
    )
    return row.id


async def test_leaf_row_wins_over_ancestors(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    await _insert(chain_conn, crypto_box, account_id="acc_root", api_key="sk-root")
    await _insert(chain_conn, crypto_box, account_id="acc_mid", api_key="sk-mid")
    await _insert(chain_conn, crypto_box, account_id="acc_leaf", api_key="sk-leaf")

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_leaf", provider="anthropic"
    )
    assert resolved is not None
    assert resolved.owner_account_id == "acc_leaf"
    assert crypto_box.derive_account_subkey("acc_leaf").decrypt(resolved.blob) == "sk-leaf"


async def test_no_leaf_row_resolves_nearest_ancestor(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    await _insert(chain_conn, crypto_box, account_id="acc_root", api_key="sk-root")
    await _insert(chain_conn, crypto_box, account_id="acc_mid", api_key="sk-mid")
    # acc_leaf has no row of its own.

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_leaf", provider="anthropic"
    )
    assert resolved is not None
    assert resolved.owner_account_id == "acc_mid"  # nearest, not root
    assert crypto_box.derive_account_subkey("acc_mid").decrypt(resolved.blob) == "sk-mid"


async def test_row_atomicity_no_field_merge_across_levels(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    """The winning row's api_base and key must both come from the SAME
    account — a child with only api_base set must never pick up an
    ancestor's key (or vice versa)."""
    await _insert(
        chain_conn,
        crypto_box,
        account_id="acc_root",
        api_key="sk-root",
        api_base="https://root-proxy.example",
    )
    await _insert(chain_conn, crypto_box, account_id="acc_mid", api_key="sk-mid", api_base=None)

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_leaf", provider="anthropic"
    )
    assert resolved is not None
    assert resolved.owner_account_id == "acc_mid"
    # NOT the root's api_base — the winning row's own (unset) value.
    assert resolved.api_base is None
    assert crypto_box.derive_account_subkey("acc_mid").decrypt(resolved.blob) == "sk-mid"


async def test_archived_row_falls_through_to_next_ancestor(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    await _insert(chain_conn, crypto_box, account_id="acc_root", api_key="sk-root")
    mid_row_id = await _insert(chain_conn, crypto_box, account_id="acc_mid", api_key="sk-mid")
    await queries.archive_model_provider(chain_conn, mid_row_id, account_id="acc_mid")

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_leaf", provider="anthropic"
    )
    assert resolved is not None
    assert resolved.owner_account_id == "acc_root"  # mid's archived row is skipped, not selected


async def test_archived_account_severs_the_chain(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    """An archived ACCOUNT (not just an archived row) must sever the walk —
    a grandparent becomes invisible even though it still has an active row,
    and even though the archived account itself never had one."""
    await _insert(chain_conn, crypto_box, account_id="acc_root", api_key="sk-root")
    # acc_mid never gets a provider row — only its ACCOUNT is archived.
    await chain_conn.execute("UPDATE accounts SET archived_at = now() WHERE id = 'acc_mid'")

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_leaf", provider="anthropic"
    )
    assert resolved is None  # root's row is invisible — the archived mid account cut it off


async def test_no_row_anywhere_returns_none(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_leaf", provider="anthropic"
    )
    assert resolved is None


async def test_different_provider_does_not_match(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    await _insert(chain_conn, crypto_box, account_id="acc_root", provider="openai")
    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_leaf", provider="anthropic"
    )
    assert resolved is None
