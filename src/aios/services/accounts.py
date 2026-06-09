"""Account lifecycle logic."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg

from aios.db import queries
from aios.errors import ConflictError, ForbiddenError, NotFoundError
from aios.logging import get_logger
from aios.models.accounts import (
    Account,
    AccountConfig,
    AccountKeySummary,
    AccountUsage,
    BootstrapResponse,
    MintAccountResponse,
    MintKeyResponse,
)

log = get_logger("aios.services.accounts")

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


async def resolve_by_path(pool: asyncpg.Pool[Any], *, caller_account_id: str, path: str) -> Account:
    """Look up ``path`` (slash-separated display names) under the caller.

    The caller's own account is the implicit root. Empty path (or just
    "/") returns the caller. Each segment is matched against
    ``display_name`` at the next depth. Raises :class:`NotFoundError`
    if any segment doesn't resolve — including for paths that walk
    outside the caller's subtree.
    """
    segments = [s for s in path.strip("/").split("/") if s]
    async with pool.acquire() as conn:
        resolved = await queries.resolve_account_by_path(
            conn,
            root_account_id=caller_account_id,
            segments=segments,
        )
    if resolved is None:
        raise NotFoundError(
            f"account path {path!r} not found",
            detail={"path": path, "caller_account_id": caller_account_id},
        )
    return resolved


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


async def update_account(
    pool: asyncpg.Pool[Any],
    *,
    target_account_id: str,
    caller_account_id: str,
    display_name: str | None,
    can_mint_children: bool | None,
    config: AccountConfig | None = None,
) -> Account:
    """Apply a partial update to a caller-or-direct-child account.

    All fields are optional; omitted ones are preserved. ``config`` is merged
    (set keys only). Raises :class:`NotFoundError` if the target is missing,
    archived, or out of scope — the API doesn't distinguish out-of-scope from
    missing.
    """
    await get_account_in_scope(pool, target_account_id, caller_account_id=caller_account_id)
    async with pool.acquire() as conn:
        updated = await queries.update_account(
            conn,
            target_account_id,
            display_name=display_name,
            can_mint_children=can_mint_children,
            config=config,
        )
    if updated is None:
        raise NotFoundError(
            f"account {target_account_id} not found",
            detail={"id": target_account_id},
        )
    return updated


async def resolve_effective_timezone_on(conn: asyncpg.Connection[Any], account_id: str) -> str:
    """The account's effective IANA timezone, inherited up the parent chain,
    or ``"UTC"`` when none is set or the stored value can't be resolved.

    The single guard point for the render path: a stored name that fails to
    construct a ``ZoneInfo`` (e.g. tzdata drift after a once-valid write)
    degrades to UTC here, so the per-message renderer — which runs on every
    wake — never raises (the #446 failure class).
    """
    name = await queries.resolve_effective_timezone(conn, account_id)
    if name is None:
        return "UTC"
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("account.timezone_unresolvable", account_id=account_id, timezone=name)
        return "UTC"
    return name


async def resolve_effective_timezone(pool: asyncpg.Pool[Any], account_id: str) -> str:
    """Pool-level wrapper for :func:`resolve_effective_timezone_on`."""
    async with pool.acquire() as conn:
        return await resolve_effective_timezone_on(conn, account_id)


async def purge_account(
    pool: asyncpg.Pool[Any], *, target_account_id: str, caller_account_id: str
) -> None:
    """Hard-delete a direct child of the caller. Refuses if not archived.

    The compliance hard-delete path. Strict preconditions: the child
    MUST already be soft-archived (call ``archive_child`` first), and
    MUST have zero non-archived children of its own. Re-purging an
    already-deleted account is a no-op success (idempotent — the
    contract is "the row is gone after this call").

    The caller can't purge itself: top-level purges are operator-side
    DB work, not a management API responsibility.
    """
    if target_account_id == caller_account_id:
        raise ConflictError(
            "an account cannot purge itself via the management API",
            detail={"account_id": caller_account_id},
        )
    target = await get_account_in_scope(
        pool, target_account_id, caller_account_id=caller_account_id
    )
    if target.archived_at is None:
        raise ConflictError(
            f"account {target_account_id} is not archived; "
            "soft-archive (DELETE /v1/accounts/{id}) before purging",
            detail={"account_id": target_account_id},
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
        deleted = await queries.hard_delete_account(conn, target_account_id)
    if not deleted:
        # Row already gone (idempotent) OR a resource FK still references it.
        # Re-read to distinguish; if the row is missing the call succeeded
        # under a prior purge.
        async with pool.acquire() as conn:
            still_there = await queries.get_account(conn, target_account_id)
        if still_there is not None:
            raise ConflictError(
                f"account {target_account_id} cannot be hard-deleted while "
                "resources still reference it; archive its resources first",
                detail={"account_id": target_account_id},
            )


async def get_usage(
    pool: asyncpg.Pool[Any], *, target_account_id: str, caller_account_id: str
) -> AccountUsage:
    """Return per-resource counts for a caller-or-direct-child account."""
    await get_account_in_scope(pool, target_account_id, caller_account_id=caller_account_id)
    async with pool.acquire() as conn:
        counts = await queries.count_account_resources(conn, target_account_id)
    return AccountUsage(
        agents=counts.get("agents", 0),
        environments=counts.get("environments", 0),
        sessions=counts.get("sessions", 0),
        vaults=counts.get("vaults", 0),
        memory_stores=counts.get("memory_stores", 0),
        skills=counts.get("skills", 0),
        session_templates=counts.get("session_templates", 0),
        connections=counts.get("connections", 0),
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
