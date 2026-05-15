"""Account lifecycle logic."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

import asyncpg

from aios.db import queries
from aios.errors import ConflictError, ForbiddenError, NotFoundError
from aios.models.accounts import (
    Account,
    AccountKeySummary,
    BootstrapResponse,
    MintAccountResponse,
    MintKeyResponse,
)

_KEY_PREFIX = "aios_"
_KEY_BODY_BYTES = 32
_BOOTSTRAP_KEY_LABEL = "bootstrap"
_FIRST_CHILD_KEY_LABEL = "initial"


def _generate_plaintext_key() -> str:
    """Return a fresh ``aios_<base64url>`` API key with 32 bytes of entropy."""
    body = secrets.token_urlsafe(_KEY_BODY_BYTES)
    return f"{_KEY_PREFIX}{body}"


def hash_key(plaintext: str) -> bytes:
    """SHA-256 of the plaintext key as raw bytes for ``bytea`` storage."""
    return hashlib.sha256(plaintext.encode("ascii")).digest()


async def bootstrap_root(
    pool: asyncpg.Pool[Any],
    *,
    display_name: str,
) -> BootstrapResponse:
    """Create the root account and mint its first API key."""
    plaintext = _generate_plaintext_key()
    key_hash = hash_key(plaintext)
    async with pool.acquire() as conn:
        account, key_id = await queries.bootstrap_root_account(
            conn,
            display_name=display_name,
            key_hash=key_hash,
            key_label=_BOOTSTRAP_KEY_LABEL,
        )
    return BootstrapResponse(
        account_id=account.id,
        key_id=key_id,
        plaintext_key=plaintext,
    )


async def get_account_in_scope(
    pool: asyncpg.Pool[Any], target_id: str, *, caller_account_id: str
) -> Account:
    """Read an account that's either the caller or a direct child.

    Raises :class:`NotFoundError` for "not in scope" — the management API
    deliberately doesn't distinguish "account doesn't exist" from "account
    belongs to a different tenant" so cross-tenant probes can't enumerate
    ids.
    """
    async with pool.acquire() as conn:
        row = await queries.get_account(conn, target_id)
    if row is None or (row.id != caller_account_id and row.parent_account_id != caller_account_id):
        raise NotFoundError(f"account {target_id} not found", detail={"id": target_id})
    return row


async def list_children(pool: asyncpg.Pool[Any], *, parent_account_id: str) -> list[Account]:
    """Return non-archived direct children of ``parent_account_id``."""
    async with pool.acquire() as conn:
        return await queries.list_child_accounts(conn, parent_account_id)


async def mint_child(
    pool: asyncpg.Pool[Any],
    *,
    caller_account_id: str,
    caller_can_mint_children: bool,
    display_name: str,
    can_mint_children: bool,
) -> MintAccountResponse:
    """Mint a direct child under ``caller_account_id`` and its first API key.

    Refuses with :class:`ForbiddenError` if the caller lacks
    ``can_mint_children``. The new key's plaintext is returned exactly
    once in the response — there's no way to recover it after.
    """
    if not caller_can_mint_children:
        raise ForbiddenError(
            "caller account is not authorized to mint children",
            detail={"account_id": caller_account_id},
        )
    plaintext = _generate_plaintext_key()
    key_hash = hash_key(plaintext)
    async with pool.acquire() as conn:
        account, key_id = await queries.insert_child_account(
            conn,
            parent_account_id=caller_account_id,
            display_name=display_name,
            can_mint_children=can_mint_children,
            key_hash=key_hash,
            key_label=_FIRST_CHILD_KEY_LABEL,
        )
    return MintAccountResponse(
        account_id=account.id,
        key_id=key_id,
        plaintext_key=plaintext,
    )


async def mint_key(
    pool: asyncpg.Pool[Any],
    *,
    target_account_id: str,
    caller_account_id: str,
    label: str,
) -> MintKeyResponse:
    """Mint an additional API key on an account that's caller-or-direct-child."""
    # Authorization: the target must be the caller's own account or a
    # direct child. ``get_account_in_scope`` enforces the check + 404s
    # cross-tenant probes.
    await get_account_in_scope(pool, target_account_id, caller_account_id=caller_account_id)
    plaintext = _generate_plaintext_key()
    key_hash = hash_key(plaintext)
    async with pool.acquire() as conn:
        key_id = await queries.insert_account_key(
            conn,
            account_id=target_account_id,
            key_hash=key_hash,
            label=label,
        )
    return MintKeyResponse(key_id=key_id, plaintext_key=plaintext)


async def list_keys(
    pool: asyncpg.Pool[Any],
    *,
    target_account_id: str,
    caller_account_id: str,
) -> list[AccountKeySummary]:
    """Return the key summaries (no hash) for a caller-or-child account."""
    await get_account_in_scope(pool, target_account_id, caller_account_id=caller_account_id)
    async with pool.acquire() as conn:
        rows = await queries.list_account_keys(conn, target_account_id)
    return [AccountKeySummary(**r) for r in rows]


async def revoke_key(
    pool: asyncpg.Pool[Any],
    *,
    target_account_id: str,
    key_id: str,
    caller_account_id: str,
) -> None:
    """Revoke an API key on a caller-or-child account.

    Idempotent: revoking an already-revoked key is a no-op (returns
    silently). Raises :class:`NotFoundError` when the key doesn't exist
    on the target at all.
    """
    await get_account_in_scope(pool, target_account_id, caller_account_id=caller_account_id)
    async with pool.acquire() as conn:
        revoked = await queries.revoke_account_key(
            conn, account_id=target_account_id, key_id=key_id
        )
    if revoked:
        return
    # The UPDATE matched zero rows. Distinguish missing from already-revoked.
    async with pool.acquire() as conn:
        keys = await queries.list_account_keys(conn, target_account_id)
    if not any(k["key_id"] == key_id for k in keys):
        raise NotFoundError(
            f"key {key_id} not found on account {target_account_id}",
            detail={"key_id": key_id, "account_id": target_account_id},
        )


async def archive_child(
    pool: asyncpg.Pool[Any],
    *,
    target_account_id: str,
    caller_account_id: str,
) -> Account:
    """Archive a direct child of the caller. Refuses if the child has children of its own.

    The caller can't archive itself — top-level archival is operator-side
    DB work, not a management API responsibility.
    """
    if target_account_id == caller_account_id:
        raise ConflictError(
            "an account cannot archive itself via the management API",
            detail={"account_id": caller_account_id},
        )
    target = await get_account_in_scope(
        pool, target_account_id, caller_account_id=caller_account_id
    )
    async with pool.acquire() as conn:
        active_kids = await queries.count_active_child_accounts(conn, target_account_id)
    if active_kids > 0:
        raise ConflictError(
            f"account {target_account_id} has {active_kids} non-archived children; "
            "archive them first",
            detail={"account_id": target_account_id, "active_children": active_kids},
        )
    async with pool.acquire() as conn:
        archived = await queries.archive_account(conn, target_account_id)
    assert archived is not None, "scope check just confirmed the account exists"
    # When the row was already archived, ``archive_account`` returns it
    # unchanged. Callers see idempotent behavior.
    _ = target
    return archived
