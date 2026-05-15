"""Business logic for github_repository session resources.

Thin orchestration over :mod:`aios.db.queries` plus the libsodium CryptoBox
for token encryption. Mirrors the structure of
:mod:`aios.services.memory_stores` but only exposes the session-bridge
surface — there is no top-level repository resource (each row is the
resource).

Token handling pattern (lifted from :mod:`aios.services.vaults`):
- Create: encrypt-once at the API boundary, persist ``(ciphertext, nonce)``.
- Rotate: decrypt-merge-encrypt within a single transaction, replacing
  only the token while preserving ``url`` and ``mount_path``.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.errors import ConflictError
from aios.models.github_repositories import (
    GithubRepositoryResource,
    GithubRepositoryResourceEcho,
)
from aios.sandbox.github_clone import remove_session_working_tree


def _encrypt_token(crypto_box: CryptoBox, token: str, *, account_id: str) -> Any:
    """Encrypt a github auth token. Returns an :class:`EncryptedBlob`.

    Keyed to ``account_id`` via :meth:`CryptoBox.derive_account_subkey` so
    the embedded token can't be decrypted under another tenant's key.
    """
    return crypto_box.derive_account_subkey(account_id).encrypt(token)


async def _list_attached_resource_ids(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> list[str]:
    """Return the resource_ids currently attached so we can clean up
    their on-disk working trees after the DB rows go away."""
    echoes = await queries.list_session_github_repo_echoes(conn, session_id, account_id=account_id)
    return [e.id for e in echoes]


def _purge_working_trees(session_id: str, resource_ids: list[str]) -> None:
    """Best-effort filesystem cleanup. Each detached resource's working
    tree contains the embedded-token ``.git/config``; leaving it on disk
    after the DB row goes away would be a slow plaintext-token leak."""
    for rid in resource_ids:
        remove_session_working_tree(session_id, rid)


async def attach_to_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resources: list[GithubRepositoryResource],
    crypto_box: CryptoBox,
    *,
    account_id: str,
) -> None:
    """Encrypt each token and insert the attachment rows.

    Caller controls the transaction so a failed attach rolls back the
    earlier session insert (paired flow with
    :func:`aios.services.memory_stores.attach_to_session`).
    """
    if not resources:
        return
    entries: list[tuple[str, str, Any, str | None, str | None]] = []
    for res in resources:
        blob = _encrypt_token(
            crypto_box, res.authorization_token.get_secret_value(), account_id=account_id
        )
        entries.append((res.url, res.mount_path, blob, res.git_user_name, res.git_user_email))
    try:
        await queries.attach_github_repos_to_session(
            conn, session_id, entries, account_id=account_id
        )
    except asyncpg.UniqueViolationError as exc:
        # The unique index hit means the new payload collides with an
        # already-attached row. The model validator catches duplicates
        # within one request, so reaching this means we're racing another
        # writer or there's a programmatic bug somewhere upstream — a 4xx
        # is the right surface either way.
        raise ConflictError(
            "duplicate mount_path among github_repository attachments",
            detail={"session_id": session_id},
        ) from exc


async def set_session_resources(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resources: list[GithubRepositoryResource],
    crypto_box: CryptoBox,
    *,
    account_id: str,
) -> None:
    """Replace attached repositories atomically.

    A failed attach rolls back the delete (parent transaction is the
    caller's). Matches the memory-store full-list-replace semantics.
    Working trees of detached attachments are removed from the host
    after the DB swap commits — the per-session ``.git/config`` carries
    the embedded auth token, so leaving them on disk after detach would
    be a slow plaintext-token leak.
    """
    async with conn.transaction():
        old_ids = await _list_attached_resource_ids(conn, session_id, account_id=account_id)
        await queries.delete_session_github_repos(conn, session_id, account_id=account_id)
        await attach_to_session(conn, session_id, resources, crypto_box, account_id=account_id)
    _purge_working_trees(session_id, old_ids)


async def detach_all_from_session(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> None:
    """Detach every github_repository from a session and clean up the
    working trees. Used by the full-list-replace path when the new
    resource list contains no github entries."""
    async with conn.transaction():
        old_ids = await _list_attached_resource_ids(conn, session_id, account_id=account_id)
        await queries.delete_session_github_repos(conn, session_id, account_id=account_id)
    _purge_working_trees(session_id, old_ids)


async def list_session_echoes(
    pool: asyncpg.Pool[Any],
    session_id: str,
    *,
    account_id: str,
) -> list[GithubRepositoryResourceEcho]:
    async with pool.acquire() as conn:
        return await queries.list_session_github_repo_echoes(
            conn, session_id, account_id=account_id
        )


async def get_resource(
    pool: asyncpg.Pool[Any],
    session_id: str,
    resource_id: str,
    *,
    account_id: str,
) -> GithubRepositoryResourceEcho:
    async with pool.acquire() as conn:
        return await queries.get_session_github_repo(
            conn, session_id, resource_id, account_id=account_id
        )


async def rotate_token(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    session_id: str,
    resource_id: str,
    new_token: str,
    identity: tuple[str | None, str | None] | None = None,
) -> GithubRepositoryResourceEcho:
    """Update the encrypted token (and optionally the git identity);
    ``url`` and ``mount_path`` remain.

    ``identity`` is ``None`` to preserve the existing identity (the
    common token-only rotation) or a ``(name, email)`` tuple to replace
    both fields atomically.  Token-only callers leave ``identity``
    unset and the stored ``git_user_name`` / ``git_user_email`` survive
    the rotation.
    """
    blob = _encrypt_token(crypto_box, new_token, account_id=account_id)
    async with pool.acquire() as conn, conn.transaction():
        return await queries.update_session_github_repo_blob(
            conn,
            session_id,
            resource_id,
            blob,
            identity=identity,
            account_id=account_id,
        )


async def get_session_token(
    conn: asyncpg.Connection[Any],
    crypto_box: CryptoBox,
    session_id: str,
    resource_id: str,
    *,
    account_id: str,
) -> str:
    """Decrypt and return the raw auth token for a single attachment.

    Only callsite today: the sandbox provisioner, which builds the
    auth-embedded clone URL. Lives here (not in queries) because it's
    the boundary that owns the CryptoBox.
    """
    _, blob = await queries.get_session_github_repo_with_blob(
        conn, session_id, resource_id, account_id=account_id
    )
    return crypto_box.derive_account_subkey(account_id).decrypt(blob)
