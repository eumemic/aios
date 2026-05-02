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


def _encrypt_token(crypto_box: CryptoBox, token: str) -> Any:
    """Encrypt a github auth token. Returns an :class:`EncryptedBlob`."""
    return crypto_box.encrypt(token)


async def attach_to_session(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resources: list[GithubRepositoryResource],
    crypto_box: CryptoBox,
) -> None:
    """Encrypt each token and insert the attachment rows.

    Caller controls the transaction so a failed attach rolls back the
    earlier session insert (paired flow with
    :func:`aios.services.memory_stores.attach_to_session`).
    """
    if not resources:
        return
    entries: list[tuple[str, str, Any]] = []
    for res in resources:
        blob = _encrypt_token(crypto_box, res.authorization_token.get_secret_value())
        entries.append((res.url, res.mount_path, blob))
    try:
        await queries.attach_github_repos_to_session(conn, session_id, entries)
    except asyncpg.UniqueViolationError as exc:
        # Hits the (session_id, mount_path) partial unique index. Models
        # validate this in-payload, so a hit here means the new payload
        # collides with an EXISTING attachment in the DB. The full-list-
        # replace path on update issues a DELETE first so this is only
        # reachable on initial create with a programmatic bug — surface it
        # as a 4xx rather than a 500.
        raise ConflictError(
            "duplicate mount_path among github_repository attachments",
            detail={"session_id": session_id},
        ) from exc


async def set_session_resources(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resources: list[GithubRepositoryResource],
    crypto_box: CryptoBox,
) -> None:
    """Replace attached repositories atomically.

    A failed attach rolls back the delete (parent transaction is the
    caller's). Matches the memory-store full-list-replace semantics; the
    aios-vs-CMA difference is documented in
    :doc:`docs/github_repository`.
    """
    async with conn.transaction():
        await queries.delete_session_github_repos(conn, session_id)
        await attach_to_session(conn, session_id, resources, crypto_box)


async def list_session_echoes(
    pool: asyncpg.Pool[Any], session_id: str
) -> list[GithubRepositoryResourceEcho]:
    async with pool.acquire() as conn:
        return await queries.list_session_github_repo_echoes(conn, session_id)


async def get_resource(
    pool: asyncpg.Pool[Any], session_id: str, resource_id: str
) -> GithubRepositoryResourceEcho:
    async with pool.acquire() as conn:
        return await queries.get_session_github_repo(conn, session_id, resource_id)


async def rotate_token(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    session_id: str,
    resource_id: str,
    new_token: str,
) -> GithubRepositoryResourceEcho:
    """Update only the encrypted token; ``url`` and ``mount_path`` remain.

    Decrypt-merge-encrypt isn't actually needed here (we store only the
    token, not a payload of multiple fields), so this is a single
    encrypt-then-update. The transaction guards against torn writes.
    """
    blob = _encrypt_token(crypto_box, new_token)
    async with pool.acquire() as conn, conn.transaction():
        return await queries.update_session_github_repo_blob(conn, session_id, resource_id, blob)


async def get_session_token(
    conn: asyncpg.Connection[Any],
    crypto_box: CryptoBox,
    session_id: str,
    resource_id: str,
) -> str:
    """Decrypt and return the raw auth token for a single attachment.

    Only callsite today: the sandbox provisioner, which builds the
    auth-embedded clone URL. Lives here (not in queries) because it's
    the boundary that owns the CryptoBox.
    """
    _, blob = await queries.get_session_github_repo_with_blob(conn, session_id, resource_id)
    return crypto_box.decrypt(blob)
