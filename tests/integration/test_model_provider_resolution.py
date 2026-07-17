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
    """Real topology: platform_root -> eumemic -> customer."""
    conn = await asyncpg.connect(migrated_db_url)
    try:
        await conn.execute(
            """
            INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
            VALUES ('platform_root', NULL,            TRUE,  'platform-root'),
                   ('acc_eumemic',  'platform_root', TRUE,  'Eumemic'),
                   ('acc_customer', 'acc_eumemic',   FALSE, 'customer')
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
    await _insert(chain_conn, crypto_box, account_id="acc_eumemic", api_key="sk-eumemic")
    await _insert(chain_conn, crypto_box, account_id="acc_customer", api_key="sk-customer")

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_customer", provider="anthropic"
    )
    assert resolved is not None
    assert resolved.owner_account_id == "acc_customer"
    assert crypto_box.derive_account_subkey("acc_customer").decrypt(resolved.blob) == "sk-customer"


async def test_no_leaf_row_resolves_nearest_ancestor(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    await _insert(chain_conn, crypto_box, account_id="acc_eumemic", api_key="sk-eumemic")
    # acc_customer has no row of its own.

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_customer", provider="anthropic"
    )
    assert resolved is not None
    assert resolved.owner_account_id == "acc_eumemic"  # nearest, not root
    assert crypto_box.derive_account_subkey("acc_eumemic").decrypt(resolved.blob) == "sk-eumemic"


async def test_row_atomicity_no_field_merge_across_levels(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    """The winning row's api_base and key must both come from the SAME
    account — a child with only api_base set must never pick up an
    ancestor's key (or vice versa)."""
    await _insert(
        chain_conn,
        crypto_box,
        account_id="acc_eumemic",
        api_key="sk-eumemic",
        api_base="https://eumemic-proxy.example",
    )
    await _insert(
        chain_conn, crypto_box, account_id="acc_customer", api_key="sk-customer", api_base=None
    )

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_customer", provider="anthropic"
    )
    assert resolved is not None
    assert resolved.owner_account_id == "acc_customer"
    # NOT the Eumemic's api_base — the winning row's own (unset) value.
    assert resolved.api_base is None
    assert crypto_box.derive_account_subkey("acc_customer").decrypt(resolved.blob) == "sk-customer"


async def test_archived_row_falls_through_to_next_ancestor(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    await _insert(chain_conn, crypto_box, account_id="acc_eumemic", api_key="sk-eumemic")
    customer_row_id = await _insert(
        chain_conn, crypto_box, account_id="acc_customer", api_key="sk-customer"
    )
    await queries.archive_model_provider(chain_conn, customer_row_id, account_id="acc_customer")

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_customer", provider="anthropic"
    )
    assert resolved is not None
    assert resolved.owner_account_id == "acc_eumemic"  # customer's archived row falls through


async def test_archived_account_severs_the_chain(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    """An archived ACCOUNT (not just an archived row) must sever the walk —
    a grandparent becomes invisible even though it still has an active row,
    and even though the archived account itself never had one."""
    blob = crypto_box.derive_account_subkey("platform_root").encrypt("sentinel-root")
    await chain_conn.execute(
        "INSERT INTO model_providers "
        "(id, account_id, provider, ciphertext, nonce) VALUES ($1, $2, $3, $4, $5)",
        "mp_legacy_root",
        "platform_root",
        "anthropic",
        blob.ciphertext,
        blob.nonce,
    )
    # Eumemic has no provider row; archiving it severs customer from root.
    await chain_conn.execute("UPDATE accounts SET archived_at = now() WHERE id = 'acc_eumemic'")

    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_customer", provider="anthropic"
    )
    assert resolved is None  # root's legacy row is invisible — archived Eumemic cut it off


async def test_no_row_anywhere_returns_none(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_customer", provider="anthropic"
    )
    assert resolved is None


async def test_different_provider_does_not_match(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    await _insert(chain_conn, crypto_box, account_id="acc_eumemic", provider="openai")
    resolved = await queries.resolve_model_provider(
        chain_conn, account_id="acc_customer", provider="anthropic"
    )
    assert resolved is None


async def test_startup_audit_rejects_preexisting_active_root_row(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    blob = crypto_box.derive_account_subkey("platform_root").encrypt("sentinel-root")
    await chain_conn.execute(
        "INSERT INTO model_providers (id, account_id, provider, ciphertext, nonce) VALUES ($1,$2,$3,$4,$5)",
        "mp_root_audit",
        "platform_root",
        "anthropic",
        blob.ciphertext,
        blob.nonce,
    )
    with pytest.raises(RuntimeError, match="credentialless-root invariant violated"):
        await queries.audit_credentialless_root(chain_conn)


async def test_root_row_update_is_rejected(
    chain_conn: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    from aios.errors import ConflictError

    blob = crypto_box.derive_account_subkey("platform_root").encrypt("sentinel-root")
    await chain_conn.execute(
        "INSERT INTO model_providers (id, account_id, provider, ciphertext, nonce) VALUES ($1,$2,$3,$4,$5)",
        "mp_root_update",
        "platform_root",
        "anthropic",
        blob.ciphertext,
        blob.nonce,
    )
    replacement = crypto_box.derive_account_subkey("platform_root").encrypt("replacement")
    with pytest.raises(ConflictError, match="must remain credentialless"):
        await queries.update_model_provider(
            chain_conn, "mp_root_update", account_id="platform_root", blob=replacement
        )


@pytest.mark.parametrize(
    ("policy", "posture"),
    [("account_only", "internal"), ("account_only", "external_byok")],
)
async def test_root_credentials_never_reach_descendant_harness(
    chain_conn: asyncpg.Connection[Any],
    crypto_box: CryptoBox,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    posture: str,
) -> None:
    """Real topology: a legacy ACTIVE root row cannot fund/call for a leaf."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from aios.services import model_providers
    from aios.workflows.run_llm import invoke_call_llm

    sentinel_env = "SENTINEL_ENV_MUST_NOT_EGRESS"
    sentinel_row = "SENTINEL_ROOT_ROW_MUST_NOT_EGRESS"
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel_env)
    blob = crypto_box.derive_account_subkey("platform_root").encrypt(sentinel_row)
    await chain_conn.execute(
        "INSERT INTO model_providers (id, account_id, provider, ciphertext, nonce) VALUES ($1,$2,$3,$4,$5)",
        f"mp_root_{posture}",
        "platform_root",
        "anthropic",
        blob.ciphertext,
        blob.nonce,
    )

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=chain_conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire.return_value = cm
    settings = SimpleNamespace(inference_credential_policy=policy, tenancy_posture=posture)
    request_capture = AsyncMock()
    run = SimpleNamespace(
        id="wfr_root_regression",
        account_id="acc_customer",
        default_child_model="anthropic/claude-x",
    )
    spec = {
        "model": "anthropic/claude-x",
        "messages": [{"role": "user", "content": "do not egress credentials"}],
        "tools": None,
        "params": None,
        "session_id": None,
    }
    with (
        patch.object(model_providers, "get_settings", return_value=settings),
        patch("aios.workflows.run_llm.runtime.require_pool", return_value=pool),
        patch("aios.workflows.run_llm.runtime.require_crypto_box", return_value=crypto_box),
        patch("aios.workflows.run_llm.call_litellm", request_capture),
    ):
        result, cost = await invoke_call_llm(
            run=run,  # type: ignore[arg-type]
            spec=spec,
        )

    assert result["error_kind"] == "model_provider_not_configured"
    assert cost == 0
    request_capture.assert_not_awaited()
    assert sentinel_env not in repr(request_capture.await_args_list)
    assert sentinel_row not in repr(request_capture.await_args_list)
    assert "FUNDED_BY_ROOT" not in repr(result)
