"""Vault, vault-credential, OAuth-flow, and credential-resolution queries.

A subsystem module of the ``aios.db.queries`` package — see ``__init__`` for the
shared scoping helpers and the package-level re-export contract. Raw SQL against
asyncpg, same conventions as the rest of the package.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import EllipsisType
from typing import Any, NamedTuple, cast

import asyncpg

from aios.crypto.vault import EncryptedBlob
from aios.db.queries import (
    _build_set_assignments,
    _get_scoped,
    _list_scoped,
    parse_jsonb,
)
from aios.errors import (
    ConflictError,
    NotFoundError,
)
from aios.ids import (
    OAUTH_FLOW,
    VAULT,
    VAULT_CREDENTIAL,
    make_id,
)
from aios.models.vaults import AuthType, Vault, VaultCredential

# ─── vaults ─────────────────────────────────────────────────────────────────


def _row_to_vault(row: asyncpg.Record) -> Vault:
    raw_metadata = row["metadata"]
    metadata = parse_jsonb(raw_metadata)
    return Vault(
        id=row["id"],
        display_name=row["display_name"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_vault(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    display_name: str,
    metadata: dict[str, Any],
) -> Vault:
    new_id = make_id(VAULT)
    metadata_json = json.dumps(metadata)
    row = await conn.fetchrow(
        """
        INSERT INTO vaults (id, display_name, metadata, account_id)
        VALUES ($1, $2, $3::jsonb, $4)
        RETURNING *
        """,
        new_id,
        display_name,
        metadata_json,
        account_id,
    )
    assert row is not None
    return _row_to_vault(row)


async def get_vault(conn: asyncpg.Connection[Any], vault_id: str, *, account_id: str) -> Vault:
    return await _get_scoped(
        conn,
        table="vaults",
        id_=vault_id,
        account_id=account_id,
        row=_row_to_vault,
        noun="vault",
    )


async def list_vaults(
    conn: asyncpg.Connection[Any], *, account_id: str, limit: int = 50, after: str | None = None
) -> list[Vault]:
    return await _list_scoped(
        conn,
        table="vaults",
        account_id=account_id,
        row=_row_to_vault,
        limit=limit,
        after=after,
    )


async def update_vault(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    *,
    account_id: str,
    display_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Vault:
    # Refuse updates to archived vaults: the read path (``get_vault``,
    # ``list_vaults``) filters ``archived_at IS NULL``, so a rewrite of
    # an archived row has no observable effect — but the bare UPDATE
    # below would still commit the new values and the RETURNING-built
    # response would lie back to the caller as if the update took.
    # Mirrors the symmetric raise on archived rows in
    # ``update_agent`` / ``update_environment`` / ``update_session_template``
    # (PR #547).
    current = await get_vault(conn, vault_id, account_id=account_id)
    if current.archived_at is not None:
        raise ConflictError(
            f"vault {vault_id} is archived",
            detail={"id": vault_id},
        )

    args: list[Any] = [vault_id]
    fields: list[tuple[str, Any, str | None]] = []
    if display_name is not None:
        fields.append(("display_name", display_name, None))
    if metadata is not None:
        fields.append(("metadata", metadata, "jsonb"))
    sets = _build_set_assignments(fields, args)
    if not sets:
        return current
    sets.append("updated_at = now()")
    args.append(account_id)
    sql = (
        f"UPDATE vaults SET {', '.join(sets)} "
        f"WHERE id = $1 AND account_id = ${len(args)} AND archived_at IS NULL RETURNING *"
    )
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise ConflictError(f"vault {vault_id} is archived", detail={"id": vault_id})
    return _row_to_vault(row)


async def archive_vault(conn: asyncpg.Connection[Any], vault_id: str, *, account_id: str) -> Vault:
    """Archive a vault and purge the encrypted blobs of its active credentials.

    Archive is an UPDATE, so ``ON DELETE CASCADE`` on the FK does not fire
    here — child credentials must be archived and zeroed explicitly. Both
    operations run in one transaction.
    """
    async with conn.transaction():
        row = await conn.fetchrow(
            "UPDATE vaults SET archived_at = now(), updated_at = now() "
            "WHERE id = $1 AND archived_at IS NULL AND account_id = $2 RETURNING *",
            vault_id,
            account_id,
        )
        if row is None:
            raise NotFoundError(
                f"vault {vault_id} not found or already archived",
                detail={"id": vault_id},
            )
        # Purge every active child credential's secret payload at the same
        # moment we archive the parent vault.
        await conn.execute(
            "UPDATE vault_credentials "
            "SET ciphertext = ''::bytea, nonce = ''::bytea, "
            "    archived_at = now(), updated_at = now() "
            "WHERE vault_id = $1 AND archived_at IS NULL AND account_id = $2",
            vault_id,
            account_id,
        )
    return _row_to_vault(row)


async def delete_vault(conn: asyncpg.Connection[Any], vault_id: str, *, account_id: str) -> None:
    # Child credentials are removed by ``ON DELETE CASCADE`` (migration 0015).
    result = await conn.execute(
        "DELETE FROM vaults WHERE id = $1 AND account_id = $2",
        vault_id,
        account_id,
    )
    if result == "DELETE 0":
        raise NotFoundError(f"vault {vault_id} not found", detail={"id": vault_id})


# ─── vault credentials ──────────────────────────────────────────────────────


def _row_to_vault_credential(row: asyncpg.Record) -> VaultCredential:
    raw_metadata = row["metadata"]
    metadata = parse_jsonb(raw_metadata)
    return VaultCredential(
        id=row["id"],
        vault_id=row["vault_id"],
        display_name=row["display_name"],
        target_url=row["target_url"],
        auth_type=row["auth_type"],
        secret_name=row["secret_name"],
        allowed_hosts=row["allowed_hosts"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


async def insert_vault_credential(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    vault_id: str,
    display_name: str | None,
    target_url: str | None,
    secret_name: str | None,
    allowed_hosts: list[str] | None,
    auth_type: AuthType,
    blob: EncryptedBlob,
    metadata: dict[str, Any],
) -> VaultCredential:
    new_id = make_id(VAULT_CREDENTIAL)
    metadata_json = json.dumps(metadata)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO vault_credentials (
                id, vault_id, display_name, target_url, secret_name,
                allowed_hosts, auth_type, ciphertext, nonce, metadata, account_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11)
            RETURNING *
            """,
            new_id,
            vault_id,
            display_name,
            target_url,
            secret_name,
            allowed_hosts,
            auth_type,
            blob.ciphertext,
            blob.nonce,
            metadata_json,
            account_id,
        )
    except asyncpg.UniqueViolationError as exc:
        # By construction each insert shape can violate exactly one partial
        # unique index: env-var rows have target_url NULL (NULLS DISTINCT, can
        # never trip the url index) and every other kind has secret_name NULL
        # (can never trip the secret_name index). Branch on the kind rather
        # than introspecting the constraint name.
        if auth_type == "environment_variable":
            raise ConflictError(
                f"an active credential named {secret_name!r} already exists in this vault",
                detail={"secret_name": secret_name, "vault_id": vault_id},
            ) from exc
        raise ConflictError(
            f"an active credential for {target_url!r} already exists in this vault",
            detail={"target_url": target_url, "vault_id": vault_id},
        ) from exc
    except asyncpg.ForeignKeyViolationError as exc:
        raise NotFoundError(
            f"vault {vault_id} not found",
            detail={"vault_id": vault_id},
        ) from exc
    assert row is not None
    return _row_to_vault_credential(row)


async def get_active_credential_by_target_url(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    vault_id: str,
    target_url: str,
) -> VaultCredential | None:
    """The active (non-archived) credential for ``(vault_id, target_url)``, if any.

    Used by the interactive OAuth completion to decide between creating a new
    credential and rotating the existing one (the ``(vault_id, target_url)``
    active-row unique index allows only one). Returns metadata only.
    """
    row = await conn.fetchrow(
        "SELECT * FROM vault_credentials "
        "WHERE vault_id = $1 AND target_url = $2 AND account_id = $3 AND archived_at IS NULL",
        vault_id,
        target_url,
        account_id,
    )
    if row is None:
        return None
    return _row_to_vault_credential(row)


async def get_vault_credential(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    credential_id: str,
    *,
    account_id: str,
) -> VaultCredential:
    row = await conn.fetchrow(
        "SELECT * FROM vault_credentials WHERE id = $1 AND vault_id = $2 AND account_id = $3",
        credential_id,
        vault_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found in vault {vault_id}",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    return _row_to_vault_credential(row)


async def lock_oauth_credential_for_refresh(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    target_url: str,
    *,
    account_id: str,
) -> tuple[str, EncryptedBlob] | None:
    """``SELECT FOR UPDATE`` the active credential for ``(vault_id, target_url)``.

    Used by the OAuth refresh path to serialize concurrent refreshes of the
    same credential. Returns ``(credential_id, EncryptedBlob)`` or ``None``
    if no active credential exists. Caller owns the surrounding transaction.
    """
    row = await conn.fetchrow(
        "SELECT id, ciphertext, nonce FROM vault_credentials "
        "WHERE vault_id = $1 AND target_url = $2 AND archived_at IS NULL "
        "AND account_id = $3 FOR UPDATE",
        vault_id,
        target_url,
        account_id,
    )
    if row is None:
        return None
    blob = EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"]))
    return str(row["id"]), blob


# ── interactive OAuth flow state (vault credential "Connect") ────────────────


async def insert_oauth_flow(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    vault_id: str,
    target_url: str,
    state: str,
    redirect_uri: str,
    blob: EncryptedBlob,
    expires_at: datetime,
) -> str:
    """Persist an in-progress OAuth flow; returns the new flow id.

    The encrypted ``blob`` carries the PKCE code_verifier, resolved endpoints,
    and any client_id/secret — see :func:`get_oauth_flow_for_complete`.
    """
    new_id = make_id(OAUTH_FLOW)
    await conn.execute(
        """
        INSERT INTO oauth_flows (
            id, account_id, vault_id, target_url, state, redirect_uri,
            ciphertext, nonce, expires_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        new_id,
        account_id,
        vault_id,
        target_url,
        state,
        redirect_uri,
        blob.ciphertext,
        blob.nonce,
        expires_at,
    )
    return new_id


async def get_oauth_flow_for_complete(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    vault_id: str,
    state: str,
) -> tuple[str, str, str, EncryptedBlob] | None:
    """``SELECT FOR UPDATE`` the live (not-expired) flow for this state.

    Returns ``(flow_id, target_url, redirect_uri, EncryptedBlob)`` or ``None``
    if no matching non-expired flow exists. Caller owns the transaction and
    deletes the row via :func:`delete_oauth_flow` after a successful exchange.
    """
    row = await conn.fetchrow(
        "SELECT id, target_url, redirect_uri, ciphertext, nonce FROM oauth_flows "
        "WHERE account_id = $1 AND vault_id = $2 AND state = $3 AND expires_at > now() "
        "FOR UPDATE",
        account_id,
        vault_id,
        state,
    )
    if row is None:
        return None
    blob = EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"]))
    return str(row["id"]), str(row["target_url"]), str(row["redirect_uri"]), blob


async def delete_oauth_flow(conn: asyncpg.Connection[Any], flow_id: str) -> None:
    """Delete a (single-use) OAuth flow row after the token exchange."""
    await conn.execute("DELETE FROM oauth_flows WHERE id = $1", flow_id)


async def delete_expired_oauth_flows(conn: asyncpg.Connection[Any]) -> None:
    """Prune expired flow rows. Called opportunistically on ``start``."""
    await conn.execute("DELETE FROM oauth_flows WHERE expires_at < now()")


async def get_vault_credential_with_blob(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    credential_id: str,
    *,
    account_id: str,
    for_update: bool = False,
) -> tuple[VaultCredential, EncryptedBlob]:
    """Fetch the credential metadata and decrypted-blob inputs in one round-trip.

    Excludes archived credentials — the blob is meaningless once archived
    (and gets zeroed out at archive time).

    Pass ``for_update=True`` to take a row-level lock for the duration
    of the surrounding transaction. Callers that follow the
    decrypt-merge-encrypt-update pattern (e.g.
    :func:`aios.services.vaults.update_vault_credential`) need this to
    serialize the cross-call read-modify-write so two concurrent PUTs
    don't both read the same pre-race blob.
    """
    sql = (
        "SELECT * FROM vault_credentials "
        "WHERE id = $1 AND vault_id = $2 AND archived_at IS NULL AND account_id = $3"
    )
    if for_update:
        sql += " FOR UPDATE"
    row = await conn.fetchrow(
        sql,
        credential_id,
        vault_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found or archived",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    cred = _row_to_vault_credential(row)
    blob = EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"]))
    return cred, blob


async def list_vault_credentials(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    *,
    account_id: str,
    limit: int = 50,
    after: str | None = None,
) -> list[VaultCredential]:
    if after is None:
        rows = await conn.fetch(
            "SELECT * FROM vault_credentials "
            "WHERE vault_id = $1 AND archived_at IS NULL AND account_id = $2 "
            "ORDER BY id DESC LIMIT $3",
            vault_id,
            account_id,
            limit,
        )
    else:
        rows = await conn.fetch(
            "SELECT * FROM vault_credentials "
            "WHERE vault_id = $1 AND archived_at IS NULL AND id < $2 "
            "AND account_id = $3 ORDER BY id DESC LIMIT $4",
            vault_id,
            after,
            account_id,
            limit,
        )
    return [_row_to_vault_credential(r) for r in rows]


async def update_vault_credential(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    credential_id: str,
    *,
    account_id: str,
    blob: EncryptedBlob | None = None,
    display_name: str | None | EllipsisType = ...,
    metadata: dict[str, Any] | None | EllipsisType = ...,
) -> VaultCredential:
    sets: list[str] = []
    args: list[Any] = [credential_id, vault_id]
    if display_name is not ...:
        args.append(display_name)
        sets.append(f"display_name = ${len(args)}")
    if blob is not None:
        args.append(blob.ciphertext)
        sets.append(f"ciphertext = ${len(args)}")
        args.append(blob.nonce)
        sets.append(f"nonce = ${len(args)}")
    if metadata is not ...:
        args.append(json.dumps(metadata))
        sets.append(f"metadata = ${len(args)}::jsonb")
    if not sets:
        return await get_vault_credential(conn, vault_id, credential_id, account_id=account_id)
    sets.append("updated_at = now()")
    args.append(account_id)
    sql = (
        f"UPDATE vault_credentials SET {', '.join(sets)} "
        f"WHERE id = $1 AND vault_id = $2 AND account_id = ${len(args)} RETURNING *"
    )
    row = await conn.fetchrow(sql, *args)
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found in vault {vault_id}",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    return _row_to_vault_credential(row)


async def archive_vault_credential(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    credential_id: str,
    *,
    account_id: str,
) -> VaultCredential:
    """Archive a credential and zero out its encrypted secret payload.

    The bytes are scrubbed at archive time so a future DB dump or query
    cannot leak the secret, even though ``WHERE archived_at IS NULL``
    filters in the read path already prevent resolution.
    """
    row = await conn.fetchrow(
        "UPDATE vault_credentials "
        "SET ciphertext = ''::bytea, nonce = ''::bytea, "
        "    archived_at = now(), updated_at = now() "
        "WHERE id = $1 AND vault_id = $2 AND archived_at IS NULL AND account_id = $3 RETURNING *",
        credential_id,
        vault_id,
        account_id,
    )
    if row is None:
        raise NotFoundError(
            f"credential {credential_id} not found or already archived",
            detail={"id": credential_id, "vault_id": vault_id},
        )
    return _row_to_vault_credential(row)


async def delete_vault_credential(
    conn: asyncpg.Connection[Any],
    vault_id: str,
    credential_id: str,
    *,
    account_id: str,
) -> None:
    result = await conn.execute(
        "DELETE FROM vault_credentials WHERE id = $1 AND vault_id = $2 AND account_id = $3",
        credential_id,
        vault_id,
        account_id,
    )
    if result == "DELETE 0":
        raise NotFoundError(
            f"credential {credential_id} not found in vault {vault_id}",
            detail={"id": credential_id, "vault_id": vault_id},
        )


async def count_active_vault_credentials(
    conn: asyncpg.Connection[Any], vault_id: str, *, account_id: str
) -> int:
    row = await conn.fetchrow(
        "SELECT count(*) AS cnt FROM vault_credentials "
        "WHERE vault_id = $1 AND archived_at IS NULL AND account_id = $2",
        vault_id,
        account_id,
    )
    assert row is not None
    result: int = row["cnt"]
    return result


# ─── session-vault binding ──────────────────────────────────────────────────


async def set_session_vaults(
    conn: asyncpg.Connection[Any],
    session_id: str,
    vault_ids: list[str],
    *,
    account_id: str,
) -> None:
    """Replace the session's vault bindings, rank-ordered (first-match
    credential resolution). Each vault must exist AND belong to ``account_id``
    — a foreign or unknown id raises ``NotFoundError`` before any insert, so a
    session can never bind another account's vault. The ``vault_id`` FK alone
    checks existence, not ownership; mirrors ``set_run_vaults``. Duplicate ids
    are de-duplicated. Order is preserved via rank.
    """
    deduped = list(dict.fromkeys(vault_ids))
    async with conn.transaction():
        await conn.execute(
            "DELETE FROM session_vaults WHERE session_id = $1 AND account_id = $2",
            session_id,
            account_id,
        )
        if not deduped:
            return
        owned = {
            str(r["id"])
            for r in await conn.fetch(
                "SELECT id FROM vaults WHERE id = ANY($1) AND account_id = $2",
                deduped,
                account_id,
            )
        }
        missing = [v for v in deduped if v not in owned]
        if missing:
            raise NotFoundError(
                f"vault(s) not found in this account: {missing}",
                detail={"vault_ids": missing},
            )
        for rank, vault_id in enumerate(deduped):
            await conn.execute(
                "INSERT INTO session_vaults (session_id, vault_id, rank, account_id) "
                "VALUES ($1, $2, $3, $4)",
                session_id,
                vault_id,
                rank,
                account_id,
            )


async def get_session_vault_ids(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[str]:
    rows = await conn.fetch(
        "SELECT vault_id FROM session_vaults WHERE session_id = $1 AND account_id = $2 ORDER BY rank",
        session_id,
        account_id,
    )
    return [str(r["vault_id"]) for r in rows]


async def batch_get_session_vault_ids(
    conn: asyncpg.Connection[Any],
    session_ids: list[str],
    *,
    account_id: str,
) -> dict[str, list[str]]:
    """Batch-fetch vault_ids for multiple sessions. Returns a dict keyed by session_id."""
    if not session_ids:
        return {}
    rows = await conn.fetch(
        "SELECT session_id, vault_id FROM session_vaults "
        "WHERE session_id = ANY($1) AND account_id = $2 ORDER BY session_id, rank",
        session_ids,
        account_id,
    )
    result: dict[str, list[str]] = {sid: [] for sid in session_ids}
    for r in rows:
        result[str(r["session_id"])].append(str(r["vault_id"]))
    return result


# ─── credential resolution ───────────────────────────────────────────────────


async def resolve_vault_credential(
    conn: asyncpg.Connection[Any],
    *,
    account_id: str,
    vault_id: str,
    target_url: str,
) -> tuple[EncryptedBlob, AuthType] | None:
    """Look up a credential in a specific vault by ``target_url`` — no
    ``session_vaults`` join.  The DB's CHECK constraint guarantees
    ``auth_type`` is one of the ``AuthType`` literals, so the cast on
    the way out is exhaustively safe.
    """
    row = await conn.fetchrow(
        """
        SELECT ciphertext, nonce, auth_type
          FROM vault_credentials
         WHERE vault_id = $1
           AND target_url = $2
           AND archived_at IS NULL
           AND account_id = $3
         LIMIT 1
        """,
        vault_id,
        target_url,
        account_id,
    )
    if row is None:
        return None
    return (
        EncryptedBlob(ciphertext=row["ciphertext"], nonce=row["nonce"]),
        cast(AuthType, str(row["auth_type"])),
    )


async def resolve_session_credential(
    conn: asyncpg.Connection[Any],
    session_id: str,
    target_url: str,
    *,
    account_id: str,
) -> tuple[EncryptedBlob, AuthType, str] | None:
    """Find the first matching credential across a session's bound vaults.

    Joins ``session_vaults`` (rank-ordered) with ``vault_credentials``
    filtered by ``target_url``. Returns
    ``(EncryptedBlob, auth_type, vault_id)`` for the first match, or
    ``None`` if no credential exists. The ``vault_id`` is needed by the
    OAuth refresh path to scope ``SELECT … FOR UPDATE`` to a specific row.
    The DB's CHECK constraint guarantees ``auth_type`` is one of the
    ``AuthType`` literals, so the cast on the way out is exhaustively safe.
    """
    row = await conn.fetchrow(
        """
        SELECT vc.ciphertext, vc.nonce, vc.auth_type, vc.vault_id
          FROM session_vaults sv
          JOIN vault_credentials vc ON vc.vault_id = sv.vault_id
         WHERE sv.session_id = $1
           AND vc.target_url = $2
           AND vc.archived_at IS NULL
           AND sv.account_id = $3
           AND vc.account_id = $3
         ORDER BY sv.rank
         LIMIT 1
        """,
        session_id,
        target_url,
        account_id,
    )
    if row is None:
        return None
    return (
        EncryptedBlob(ciphertext=row["ciphertext"], nonce=row["nonce"]),
        cast(AuthType, str(row["auth_type"])),
        str(row["vault_id"]),
    )


async def resolve_run_credential(
    conn: asyncpg.Connection[Any],
    run_id: str,
    target_url: str,
    *,
    account_id: str,
) -> tuple[EncryptedBlob, AuthType, str] | None:
    """Find the first matching credential across a *run*'s bound vaults.

    The run analog of :func:`resolve_session_credential` — identical query with
    ``wf_run_vaults``/``run_id`` swapped for ``session_vaults``/``session_id``. The
    decrypt + OAuth-refresh + header-render tail downstream is owner-agnostic (it
    keys off ``account_id`` + ``vault_id``), so only this lookup differs by owner.
    """
    row = await conn.fetchrow(
        """
        SELECT vc.ciphertext, vc.nonce, vc.auth_type, vc.vault_id
          FROM wf_run_vaults rv
          JOIN vault_credentials vc ON vc.vault_id = rv.vault_id
         WHERE rv.run_id = $1
           AND vc.target_url = $2
           AND vc.archived_at IS NULL
           AND rv.account_id = $3
           AND vc.account_id = $3
         ORDER BY rv.rank
         LIMIT 1
        """,
        run_id,
        target_url,
        account_id,
    )
    if row is None:
        return None
    return (
        EncryptedBlob(ciphertext=row["ciphertext"], nonce=row["nonce"]),
        cast(AuthType, str(row["auth_type"])),
        str(row["vault_id"]),
    )


class EnvVarCredentialRow(NamedTuple):
    """One ``environment_variable`` credential resolved for a session.

    ``updated_at`` rides along for the recycle-on-rotation drift key
    (#877); ``allowed_hosts`` entries are the stored canonical
    ``host[/path-prefix]`` strings — consumers parse them with
    :func:`aios.models.vaults.parse_allowed_host_entry`, the single
    grammar authority.
    """

    credential_id: str
    secret_name: str
    allowed_hosts: tuple[str, ...]
    blob: EncryptedBlob
    updated_at: datetime


class EnvVarCredentialEcho(NamedTuple):
    """Metadata-only drift echo for a session's env-var credentials (#877).

    The per-step recycle-on-rotation probe needs ONLY the (id, updated_at)
    set membership — never the ciphertext. INTERNAL type (not a pydantic
    model, never on Session.resources or any API response), so the FastAPI
    surface is unchanged.
    """

    credential_id: str
    updated_at: datetime


# Owner-parameterized FROM/WHERE/ORDER-BY body for the env-var credential
# membership/resolution predicate — the security-critical scope shared by ALL
# three credential queries below (the two session sets *and* the run set). The
# cross-tenant scope (``account_id = $2`` on BOTH the binding row and the
# credential row), archival filter (``archived_at IS NULL``), DISTINCT-ON-rank
# predicate and first-vault-wins ordering live here ONCE, across BOTH owners,
# so the provision-set (:func:`list_session_env_var_credentials`), the per-step
# drift echo-set (:func:`list_session_env_var_credential_echoes`) and the
# workflow-run set (:func:`list_run_env_var_credentials`) can NEVER silently
# diverge: a future account-scoping or security fix to this body applies to all
# three. ``$1`` is the owner id (session/run), ``$2`` the account id — bind
# positions are identical across owners, so no call-site change.
#
# The interpolated names (``table``/``a``/``owner_col``) are static module
# literals — never user input, so no injection risk — matching the already
# blessed f-string interpolation idiom in ``db/queries/__init__.py``
# (``_get_scoped``/``_list_scoped`` interpolate ``{table}``/``{column}``).
_ENV_VAR_CREDENTIALS_FROM_WHERE = """
          FROM {table} {a}
          JOIN vault_credentials vc ON vc.vault_id = {a}.vault_id
         WHERE {a}.{owner_col} = $1
           AND vc.auth_type = 'environment_variable'
           AND vc.archived_at IS NULL
           AND {a}.account_id = $2
           AND vc.account_id = $2
         ORDER BY vc.secret_name, {a}.rank
""".strip()

# Session owner: shared by the two session queries (provision set + drift echo
# set) so they can't diverge. ``$1`` is the session id, ``$2`` the account id.
_SESSION_ENV_VAR_CREDENTIALS_FROM_WHERE = _ENV_VAR_CREDENTIALS_FROM_WHERE.format(
    table="session_vaults", a="sv", owner_col="session_id"
)

# Run owner: the workflow-run twin, derived from the SAME template — its body
# can no longer drift from the session predicate by a dropped hand-edit.
# ``$1`` is the run id, ``$2`` the account id.
_RUN_ENV_VAR_CREDENTIALS_FROM_WHERE = _ENV_VAR_CREDENTIALS_FROM_WHERE.format(
    table="wf_run_vaults", a="rv", owner_col="run_id"
)


async def list_run_env_var_credentials(
    conn: asyncpg.Connection[Any],
    run_id: str,
    *,
    account_id: str,
) -> list[EnvVarCredentialRow]:
    """All active ``environment_variable`` credentials across a workflow run's
    bound vaults.

    Run twin of :func:`list_session_env_var_credentials`: duplicate
    ``secret_name`` resolves first-vault-wins by ``wf_run_vaults.rank``.
    Runs have no per-step drift echo, so this is the only run env-var
    credential membership query.

    Embeds ``_RUN_ENV_VAR_CREDENTIALS_FROM_WHERE``, derived from the SAME
    owner-parameterized ``_ENV_VAR_CREDENTIALS_FROM_WHERE`` template the two
    session queries derive their body from, so the cross-tenant credential
    predicate can't diverge between the session and run execution paths.
    """
    rows = await conn.fetch(
        f"""
        SELECT DISTINCT ON (vc.secret_name)
               vc.id, vc.secret_name, vc.allowed_hosts,
               vc.ciphertext, vc.nonce, vc.updated_at
        {_RUN_ENV_VAR_CREDENTIALS_FROM_WHERE}
        """,
        run_id,
        account_id,
    )
    return [
        EnvVarCredentialRow(
            credential_id=str(row["id"]),
            secret_name=str(row["secret_name"]),
            allowed_hosts=tuple(row["allowed_hosts"]),
            blob=EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"])),
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


async def list_session_env_var_credentials(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[EnvVarCredentialRow]:
    """All active ``environment_variable`` credentials across a session's
    bound vaults — a *list*, unlike the single-``target_url`` lookup above.

    Duplicate ``secret_name`` across two attached vaults resolves
    first-vault-wins: ``DISTINCT ON (secret_name)`` keeps the lowest
    ``session_vaults.rank`` row (within one vault duplicates are
    impossible — the partial unique index on ``(vault_id, secret_name)``).
    Rank uniqueness per session is an ``enumerate()`` artifact of
    ``set_session_vaults``, the same invariant ``resolve_session_credential``
    already leans on.

    Shares its account-scoped FROM/WHERE/ORDER-BY body with
    :func:`list_session_env_var_credential_echoes` via
    ``_SESSION_ENV_VAR_CREDENTIALS_FROM_WHERE`` so the two sets can't diverge.
    """
    rows = await conn.fetch(
        f"""
        SELECT DISTINCT ON (vc.secret_name)
               vc.id, vc.secret_name, vc.allowed_hosts,
               vc.ciphertext, vc.nonce, vc.updated_at
        {_SESSION_ENV_VAR_CREDENTIALS_FROM_WHERE}
        """,
        session_id,
        account_id,
    )
    return [
        EnvVarCredentialRow(
            credential_id=str(row["id"]),
            secret_name=str(row["secret_name"]),
            allowed_hosts=tuple(row["allowed_hosts"]),
            blob=EncryptedBlob(ciphertext=bytes(row["ciphertext"]), nonce=bytes(row["nonce"])),
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


async def list_session_env_var_credential_echoes(
    conn: asyncpg.Connection[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[EnvVarCredentialEcho]:
    """Metadata-only env-var credential echoes for the per-step drift probe (#877).

    Selects ONLY ``(id, updated_at)`` — never the ciphertext — under the SAME
    ``DISTINCT ON (secret_name)`` / ``archived_at IS NULL`` / rank-order filter
    as :func:`list_session_env_var_credentials`, so the resolved set (provision)
    and the echo set (step) have IDENTICAL membership. That parity is enforced
    structurally: both queries embed the SAME
    ``_SESSION_ENV_VAR_CREDENTIALS_FROM_WHERE`` body, so an account-scoping or
    archival change touches both at once. Mirrors
    :func:`list_session_github_repo_echoes`.
    """
    rows = await conn.fetch(
        f"""
        SELECT DISTINCT ON (vc.secret_name) vc.id, vc.updated_at
        {_SESSION_ENV_VAR_CREDENTIALS_FROM_WHERE}
        """,
        session_id,
        account_id,
    )
    return [
        EnvVarCredentialEcho(credential_id=str(row["id"]), updated_at=row["updated_at"])
        for row in rows
    ]
