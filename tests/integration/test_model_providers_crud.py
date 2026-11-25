"""Integration tests for the model_providers resource: CRUD, tenant scoping,
archive-zeroing, and the `aios rekey` wiring.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.cli.commands.ops import _REKEY_COLUMNS, _rekey_column
from aios.crypto.vault import KEY_BYTES, CryptoBox, EncryptedBlob
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError, CryptoDecryptError, NotFoundError
from aios.services import model_providers as service

pytestmark = pytest.mark.integration

_MODEL_PROVIDERS_COL = next(c for c in _REKEY_COLUMNS if c.table == "model_providers")


@pytest.fixture
async def pool_with_account(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('platform_root', NULL, TRUE, 'platform-root'),
                       ('acc_mp_test', 'platform_root', TRUE, 'model-providers-test')
                """
            )
        yield pool, "acc_mp_test"
    finally:
        await pool.close()


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(KEY_BYTES))


async def test_create_get_list(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool,
        crypto_box,
        account_id=account_id,
        provider="anthropic",
        api_key="sk-real",
        api_base="https://proxy.example",
    )
    assert created.api_key_set is True
    assert created.api_base == "https://proxy.example"

    fetched = await service.get_model_provider(pool, created.id, account_id=account_id)
    assert fetched == created

    listed = await service.list_model_providers(pool, account_id=account_id)
    assert [p.id for p in listed] == [created.id]


async def test_platform_root_rejects_simultaneous_openai_and_anthropic(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, _ = pool_with_account
    for provider in ("openai", "anthropic"):
        with pytest.raises(ConflictError, match="credentialless"):
            await service.create_model_provider(
                pool,
                crypto_box,
                account_id="platform_root",
                provider=provider,
                api_key=f"sentinel-{provider}",
                api_base=None,
            )


async def test_duplicate_active_provider_conflicts(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    with pytest.raises(ConflictError):
        await service.create_model_provider(
            pool,
            crypto_box,
            account_id=account_id,
            provider="anthropic",
            api_key="k2",
            api_base=None,
        )


async def test_create_after_archive_succeeds(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    first = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    await service.archive_model_provider(pool, first.id, account_id=account_id)
    second = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k2", api_base=None
    )
    assert second.id != first.id
    assert second.archived_at is None


async def test_archive_zeroes_ciphertext(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    archived = await service.archive_model_provider(pool, created.id, account_id=account_id)
    assert archived.archived_at is not None
    assert archived.api_key_set is False

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT ciphertext, nonce FROM model_providers WHERE id = $1", created.id
        )
    assert row is not None
    assert bytes(row["ciphertext"]) == b""
    assert bytes(row["nonce"]) == b""


async def test_archive_is_idempotent(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    first = await service.archive_model_provider(pool, created.id, account_id=account_id)
    second = await service.archive_model_provider(
        pool, created.id, account_id=account_id, idempotent=True
    )
    assert second.id == first.id
    assert second.archived_at == first.archived_at

    with pytest.raises(NotFoundError):
        await service.archive_model_provider(
            pool, created.id, account_id=account_id, idempotent=False
        )


async def test_cross_tenant_isolation(
    conn_two_accounts: asyncpg.Connection[Any], crypto_box: CryptoBox
) -> None:
    """A row created for one account is invisible to (and unmodifiable by) another."""
    blob = crypto_box.derive_account_subkey("acc_a").encrypt("sk-a")
    row = await queries.insert_model_provider(
        conn_two_accounts, account_id="acc_a", provider="anthropic", api_base=None, blob=blob
    )

    with pytest.raises(NotFoundError):
        await queries.get_model_provider(conn_two_accounts, row.id, account_id="acc_b")

    listed_b = await queries.list_model_providers(conn_two_accounts, account_id="acc_b")
    assert listed_b == []

    with pytest.raises(NotFoundError):
        await queries.archive_model_provider(conn_two_accounts, row.id, account_id="acc_b")

    # The row is untouched — still active under its real owner.
    still_active = await queries.get_model_provider(conn_two_accounts, row.id, account_id="acc_a")
    assert still_active.archived_at is None


async def test_rekey_round_trip(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
) -> None:
    """`aios rekey`'s generic column-rewriter correctly covers model_providers:
    a row encrypted under an old master key decrypts under the rotated one
    after `_rekey_column` runs, and no longer decrypts under the old key.
    """
    pool, account_id = pool_with_account
    old_master = CryptoBox(os.urandom(KEY_BYTES))
    new_master = CryptoBox(os.urandom(KEY_BYTES))

    old_blob = old_master.derive_account_subkey(account_id).encrypt("sk-pre-rotation")
    async with pool.acquire() as conn:
        row = await queries.insert_model_provider(
            conn, account_id=account_id, provider="anthropic", api_base=None, blob=old_blob
        )

        rekeyed_count = await _rekey_column(conn, _MODEL_PROVIDERS_COL, new_master, old_master)
        assert rekeyed_count == 1

        refreshed = await conn.fetchrow(
            "SELECT ciphertext, nonce FROM model_providers WHERE id = $1", row.id
        )
    assert refreshed is not None
    new_blob = EncryptedBlob(bytes(refreshed["ciphertext"]), bytes(refreshed["nonce"]))
    assert new_master.derive_account_subkey(account_id).decrypt(new_blob) == "sk-pre-rotation"

    with pytest.raises(CryptoDecryptError):
        old_master.derive_account_subkey(account_id).decrypt(new_blob)


# ─── partial credential updates (PR #2609 review P1-a) ────────────────────────


async def _resolve(pool: asyncpg.Pool[Any], crypto_box: CryptoBox, account_id: str) -> Any:
    return await service._resolve_provider_auth(
        pool, crypto_box, account_id=account_id, model="anthropic/claude-x", litellm_extra=None
    )


async def test_partial_credentials_update_keeps_existing_key(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    """A credentials patch that omits api_key must not drop the stored key while
    the endpoint is retained — the row must still resolve, with both the old
    key and the newly added credential kwarg."""
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool,
        crypto_box,
        account_id=account_id,
        provider="anthropic",
        api_key="sk-keep",
        api_base="https://proxy.example",
    )
    await service.update_model_provider(
        pool,
        crypto_box,
        created.id,
        account_id=account_id,
        api_key=None,
        credentials={"extra_headers": {"x": "y"}},
    )
    auth = await _resolve(pool, crypto_box, account_id)
    assert auth is not None
    assert auth.api_key == "sk-keep"
    assert auth.api_base == "https://proxy.example"
    assert auth.credentials["extra_headers"] == {"x": "y"}


async def test_api_key_rotation_keeps_other_credential_kwargs(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool,
        crypto_box,
        account_id=account_id,
        provider="anthropic",
        api_key="sk-old",
        api_base=None,
        credentials={"extra_headers": {"x": "y"}},
    )
    await service.update_model_provider(
        pool, crypto_box, created.id, account_id=account_id, api_key="sk-new"
    )
    auth = await _resolve(pool, crypto_box, account_id)
    assert auth is not None
    assert auth.api_key == "sk-new"
    assert auth.credentials["extra_headers"] == {"x": "y"}


async def test_api_base_edit_is_what_resolution_sees(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    """create() folds api_base into the encrypted blob, and resolution prefers the
    blob's endpoint — so an api_base edit must reach the blob, not only the
    plaintext column, or the edit is silently ignored at call time."""
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool,
        crypto_box,
        account_id=account_id,
        provider="anthropic",
        api_key="sk-keep",
        api_base="https://old.example",
    )
    updated = await service.update_model_provider(
        pool,
        crypto_box,
        created.id,
        account_id=account_id,
        api_key=None,
        api_base="https://new.example",
    )
    assert updated.api_base == "https://new.example"
    auth = await _resolve(pool, crypto_box, account_id)
    assert auth is not None
    assert auth.api_key == "sk-keep"
    assert auth.api_base == "https://new.example"


async def test_credentials_patch_null_removes_a_kwarg(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool,
        crypto_box,
        account_id=account_id,
        provider="anthropic",
        api_key="sk-keep",
        api_base=None,
        credentials={"extra_headers": {"x": "y"}},
    )
    await service.update_model_provider(
        pool,
        crypto_box,
        created.id,
        account_id=account_id,
        api_key=None,
        credentials={"extra_headers": None},
    )
    auth = await _resolve(pool, crypto_box, account_id)
    assert auth is not None
    assert auth.api_key == "sk-keep"
    assert "extra_headers" not in auth.credentials


async def test_update_that_leaves_endpoint_without_key_is_rejected(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    """Removing the key while an endpoint is retained would write a row that
    _resolve_provider_auth refuses at call time; reject it at write time (422)
    and leave the row untouched."""
    from aios.errors import ValidationError

    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool,
        crypto_box,
        account_id=account_id,
        provider="anthropic",
        api_key="sk-keep",
        api_base="https://proxy.example",
    )
    with pytest.raises(ValidationError, match="requires api_key") as exc:
        await service.update_model_provider(
            pool,
            crypto_box,
            created.id,
            account_id=account_id,
            api_key=None,
            credentials={"api_key": None, "extra_headers": {"x": "y"}},
        )
    assert exc.value.status_code == 422
    auth = await _resolve(pool, crypto_box, account_id)
    assert auth is not None and auth.api_key == "sk-keep"
    fetched = await service.get_model_provider(pool, created.id, account_id=account_id)
    assert fetched.version == created.version


# ─── optimistic concurrency (PR #2609 review P1-b; spec: "version (optimistic
# concurrency, agents-style)") ─────────────────────────────────────────────────


async def test_stale_expected_version_conflicts_and_writes_nothing(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    assert created.version == 1
    first = await service.update_model_provider(
        pool,
        crypto_box,
        created.id,
        account_id=account_id,
        api_key=None,
        litellm_defaults={"reasoning_effort": "high"},
        expected_version=1,
    )
    assert first.version == 2
    # A second client still holding version 1 loses, loudly.
    with pytest.raises(ConflictError, match="version mismatch") as exc:
        await service.update_model_provider(
            pool,
            crypto_box,
            created.id,
            account_id=account_id,
            api_key="k-clobber",
            litellm_defaults={"reasoning_effort": "low"},
            expected_version=1,
        )
    assert exc.value.status_code == 409
    assert exc.value.detail == {"expected": 1, "current": 2, "id": created.id}
    after = await service.get_model_provider(pool, created.id, account_id=account_id)
    assert after.version == 2
    assert after.litellm_defaults == {"reasoning_effort": "high"}
    auth = await _resolve(pool, crypto_box, account_id)
    assert auth is not None and auth.api_key == "k1"


async def test_stale_expected_version_conflicts_even_for_a_noop_body(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    await service.update_model_provider(
        pool, crypto_box, created.id, account_id=account_id, api_key="k2"
    )
    with pytest.raises(ConflictError, match="version mismatch"):
        await service.update_model_provider(
            pool, crypto_box, created.id, account_id=account_id, api_key=None, expected_version=1
        )


async def test_matching_expected_version_updates(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    updated = await service.update_model_provider(
        pool, crypto_box, created.id, account_id=account_id, api_key="k2", expected_version=1
    )
    assert updated.version == 2
    auth = await _resolve(pool, crypto_box, account_id)
    assert auth is not None and auth.api_key == "k2"


async def test_put_route_maps_version_mismatch_to_409(
    pool_with_account: tuple[asyncpg.Pool[Any], str], crypto_box: CryptoBox
) -> None:
    """The wire contract: ``version`` on the PUT body is the expected version."""
    from aios.api.routers import model_providers as router
    from aios.models.model_providers import ModelProviderUpdate

    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    ok = await router.update(
        created.id,
        ModelProviderUpdate.model_validate({"api_key": "k2", "version": 1}),
        pool,
        crypto_box,
        account_id,
    )
    assert ok.version == 2
    with pytest.raises(ConflictError) as exc:
        await router.update(
            created.id,
            ModelProviderUpdate.model_validate({"api_key": "k3", "version": 1}),
            pool,
            crypto_box,
            account_id,
        )
    assert exc.value.status_code == 409


async def test_version_predicate_rejects_a_write_that_raced_past_the_precheck(
    pool_with_account: tuple[asyncpg.Pool[Any], str],
    crypto_box: CryptoBox,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-check reads a snapshot; a concurrent writer can bump the row
    between that read and the UPDATE. Simulate exactly that interleaving: the
    pre-check sees the stale v1 snapshot while the row is really at v2. The
    ``AND version = $expected`` predicate is what must refuse the write."""
    from aios.db.queries import model_providers as mp_queries

    pool, account_id = pool_with_account
    created = await service.create_model_provider(
        pool, crypto_box, account_id=account_id, provider="anthropic", api_key="k1", api_base=None
    )
    await service.update_model_provider(
        pool,
        crypto_box,
        created.id,
        account_id=account_id,
        api_key=None,
        litellm_defaults={"reasoning_effort": "high"},
    )
    real_get = mp_queries.get_model_provider
    calls = {"n": 0}

    async def _stale_first_read(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return created if calls["n"] == 1 else await real_get(*args, **kwargs)

    monkeypatch.setattr(mp_queries, "get_model_provider", _stale_first_read)
    async with pool.acquire() as conn:
        with pytest.raises(ConflictError, match="version mismatch"):
            await mp_queries.update_model_provider(
                conn,
                created.id,
                account_id=account_id,
                litellm_defaults={"reasoning_effort": "low"},
                expected_version=1,
            )
    monkeypatch.setattr(mp_queries, "get_model_provider", real_get)
    after = await service.get_model_provider(pool, created.id, account_id=account_id)
    assert after.version == 2
    assert after.litellm_defaults == {"reasoning_effort": "high"}
