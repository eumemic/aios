"""Model-provider config queries: encrypted per-account API keys + proxy bases.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.

``resolve_model_provider`` is a fourth nearest-ancestor recursive-CTE walk
alongside the three in ``accounts.py`` (``resolve_effective_timezone``,
``resolve_effective_spend_limit_usd``, ``resolve_effective_sandbox_snapshot_bytes``
— the latter's docstring literally says "Mirrors resolve_effective_spend_limit_usd").
It is not unified into a shared helper: those three resolve a *scalar* off
``accounts.config``, while this one resolves a *whole row* (both ``api_base``
and the encrypted key must come from the SAME account — see
``aios.models.model_providers`` for why) from a foreign table via a JOIN. A
helper generic enough to cover both a scalar-off-``config`` shape and a
row-from-a-joined-table shape would mean templating the JOIN/column list in
raw SQL — a mini query-builder, at odds with "raw SQL + asyncpg, no ORM."
"""

from __future__ import annotations

from types import EllipsisType
from typing import Any, NamedTuple

import asyncpg

from aios.crypto.vault import EncryptedBlob
from aios.db.queries import _get_scoped, _list_scoped
from aios.errors import ConflictError, NotFoundError
from aios.ids import MODEL_PROVIDER, make_id
from aios.models.model_providers import ModelProvider


def _row_to_model_provider(row: asyncpg.Record) -> ModelProvider:
    return ModelProvider(
        id=row["id"],
        provider=row["provider"],
        api_base=row["api_base"],
        api_key_set=len(row["ciphertext"]) > 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_model_provider(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    provider: str,
    api_base: str | None,
    blob: EncryptedBlob,
) -> ModelProvider:
    new_id = make_id(MODEL_PROVIDER)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO model_providers (id, account_id, provider, api_base, ciphertext, nonce)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
            """,
            new_id,
            account_id,
            provider,
            api_base,
            blob.ciphertext,
            blob.nonce,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            f"an active model provider config for {provider!r} already exists on this account",
            detail={"provider": provider},
        ) from exc
    assert row is not None
    return _row_to_model_provider(row)


async def get_model_provider(
    conn: asyncpg.Connection[Any], model_provider_id: str, *, account_id: str
) -> ModelProvider:
    return await _get_scoped(
        conn,
        table="model_providers",
        id_=model_provider_id,
        account_id=account_id,
        row=_row_to_model_provider,
        noun="model provider config",
    )


async def list_model_providers(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    provider: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[ModelProvider]:
    return await _list_scoped(
        conn,
        table="model_providers",
        account_id=account_id,
        row=_row_to_model_provider,
        limit=limit,
        after=after,
        filters=[("provider", provider)],
    )


async def update_model_provider(
    conn: asyncpg.Connection[Any],
    model_provider_id: str,
    *,
    account_id: str,
    blob: EncryptedBlob | None = None,
    api_base: str | None | EllipsisType = ...,
) -> ModelProvider:
    """Update a config's key and/or ``api_base``.

    Mirrors ``update_vault`` (pre-check + a ``WHERE ... archived_at IS NULL``
    guard that raises ``ConflictError`` when a concurrent archive won the
    race) rather than ``update_vault_credential`` (which lacks that guard —
    a real, pre-existing bug: a racing update can resurrect an
    archive-zeroed secret into a row that a subsequent archive can no longer
    re-scrub, since its own ``archived_at IS NULL`` guard then no-ops). A
    model-provider row holds a live provider credential, so this path is
    guarded from the start.
    """
    current = await get_model_provider(conn, model_provider_id, account_id=account_id)
    if current.archived_at is not None:
        raise ConflictError(
            f"model provider config {model_provider_id} is archived",
            detail={"id": model_provider_id},
        )

    args: list[Any] = [model_provider_id]
    sets: list[str] = []
    if blob is not None:
        args.append(blob.ciphertext)
        sets.append(f"ciphertext = ${len(args)}")
        args.append(blob.nonce)
        sets.append(f"nonce = ${len(args)}")
    if api_base is not ...:
        args.append(api_base)
        sets.append(f"api_base = ${len(args)}")
    if not sets:
        return current
    sets.append("updated_at = now()")
    args.append(account_id)
    sql = (
        f"UPDATE model_providers SET {', '.join(sets)} "
        f"WHERE id = $1 AND account_id = ${len(args)} AND archived_at IS NULL RETURNING *"
    )
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise ConflictError(
            f"model provider config {model_provider_id} is archived",
            detail={"id": model_provider_id},
        )
    return _row_to_model_provider(row)


async def archive_model_provider(
    conn: asyncpg.Connection[Any],
    model_provider_id: str,
    *,
    account_id: str,
    idempotent: bool = False,
) -> ModelProvider:
    """Archive a config and zero out its encrypted key.

    Mirrors ``archive_vault_credential``: the ciphertext/nonce bytes are
    scrubbed at archive time so a future DB dump can't leak the key, even
    though the read path's ``archived_at IS NULL`` filter already prevents
    resolution. ``idempotent`` backs the bare ``DELETE`` verb — archiving an
    already-archived row returns it unchanged instead of 404ing.
    """
    row = await conn.fetchrow(
        "UPDATE model_providers "
        "SET ciphertext = ''::bytea, nonce = ''::bytea, archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND account_id = $2 AND archived_at IS NULL RETURNING *",
        model_provider_id,
        account_id,
    )
    if row is None:
        if idempotent:
            existing = await conn.fetchrow(
                "SELECT * FROM model_providers WHERE id = $1 AND account_id = $2",
                model_provider_id,
                account_id,
            )
            if existing is not None:
                return _row_to_model_provider(existing)
        raise NotFoundError(
            f"model provider config {model_provider_id} not found or already archived",
            detail={"id": model_provider_id},
        )
    return _row_to_model_provider(row)


class ResolvedModelProvider(NamedTuple):
    owner_account_id: str
    api_base: str | None
    blob: EncryptedBlob


async def resolve_model_provider(
    conn: asyncpg.Connection[Any], *, account_id: str, provider: str
) -> ResolvedModelProvider | None:
    """Nearest-ancestor-wins resolution of ``provider``'s config for ``account_id``.

    Walks ``account_id`` and its ancestors (self first, depth 0), returning
    the whole row — ``api_base`` and the encrypted key together — from the
    nearest account that has an active config for ``provider``. Never
    field-merges across levels: the winning ``api_base`` and key always come
    from the SAME row, so a child with only an ``api_base`` override never
    picks up an ancestor's key by accident (row-atomicity).

    Two INDEPENDENT ``archived_at`` filters, both applied as ``WHERE``
    predicates before ``ORDER BY depth ASC LIMIT 1`` (never post-selection):
    the recursive step's ``a.archived_at IS NULL`` on the joined ``accounts``
    alias severs the walk (an archived ancestor's own parent never enters
    the recursion — a grandparent becomes invisible, not just "this level
    filtered"), and the join's ``mp.archived_at IS NULL`` independently
    excludes an archived (ciphertext-zeroed) config row from ever being
    selected — so decrypting a zeroed row is never reached for the ordinary
    archived case; the caller's ``CryptoDecryptError`` on empty ciphertext
    stays reserved for genuine corruption or key mismatch.
    """
    row = await conn.fetchrow(
        "WITH RECURSIVE chain AS ("
        "  SELECT id, parent_account_id, 0 AS depth "
        "    FROM accounts WHERE id = $1 AND archived_at IS NULL "
        "  UNION ALL "
        "  SELECT a.id, a.parent_account_id, c.depth + 1 "
        "    FROM accounts a JOIN chain c ON a.id = c.parent_account_id "
        "    WHERE a.archived_at IS NULL"
        ") "
        "SELECT mp.account_id, mp.api_base, mp.ciphertext, mp.nonce "
        "FROM chain c "
        "JOIN model_providers mp ON mp.account_id = c.id "
        "WHERE mp.provider = $2 AND mp.archived_at IS NULL "
        "ORDER BY c.depth ASC LIMIT 1",
        account_id,
        provider,
    )
    if row is None:
        return None
    return ResolvedModelProvider(
        owner_account_id=row["account_id"],
        api_base=row["api_base"],
        blob=EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"])),
    )
