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
from aios.db.listen import GITHUB_CLONE_BREAKER_CLEAR_CHANNEL
from aios.errors import ConflictError, RateLimitedError
from aios.models.github_repositories import (
    MAX_REPOS_PER_SESSION,
    GithubRepositoryResource,
    GithubRepositoryResourceEcho,
)
from aios.sandbox.github_clone import remove_session_working_tree


async def _notify_clear_clone_breaker(pool: asyncpg.Pool[Any], resource_id: str) -> None:
    """Signal the worker's github-clone circuit breaker to clear ``resource_id``
    after a token rotation (#1720, re-probe path (a)).

    The rotation runs in the API process; the breaker lives in the worker.
    This fires a ``pg_notify`` on ``GITHUB_CLONE_BREAKER_CLEAR_CHANNEL`` which
    the worker's clear-listener drains and dispatches to
    :meth:`aios.sandbox.github_clone_breaker.GithubCloneBreaker.clear` — so a
    fixed credential re-probes on the very next provision instead of serving
    out a cooldown opened under the old secret. Fire-and-forget, like the MCP
    vault-eviction NOTIFY (#1030): a notification lost during a listener
    reconnect falls back to the breaker's own cooldown/half-open re-probe.
    """
    await pool.execute("SELECT pg_notify($1, $2)", GITHUB_CLONE_BREAKER_CLEAR_CHANNEL, resource_id)


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

    Conn-scoped (mid-transaction): does NOT evict the session's sandbox.
    Github attachments feed build_spec_from_session, so eviction IS
    required — but it must fire AFTER the parent transaction commits, so
    :func:`aios.services.sessions.update_session` owns the post-commit
    eviction hook (#713). Layer 2's ``spec_version`` trigger on
    ``session_github_repositories`` is the direct-SQL / API-process
    safety net behind it.
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
) -> bool:
    """Replace attached repositories atomically.

    A failed attach rolls back the delete (parent transaction is the
    caller's). Matches the memory-store full-list-replace semantics.
    Working trees of detached attachments are removed from the host
    after the DB swap commits — the per-session ``.git/config`` carries
    the embedded auth token, so leaving them on disk after detach would
    be a slow plaintext-token leak.

    Always returns True (the changed signal, mirroring the memory-store
    sibling): a github re-PUT is never idempotent — the incoming
    ``authorization_token`` is re-encrypted (fresh ciphertext, new
    ``updated_at``), and ``updated_at`` is deliberately part of the mount
    snapshot so token rotation reaches the sandbox.

    Conn-scoped: sandbox eviction is fired post-commit by
    :func:`aios.services.sessions.update_session`, not here (#713).
    """
    async with conn.transaction():
        old_ids = await _list_attached_resource_ids(conn, session_id, account_id=account_id)
        await queries.delete_session_github_repos(conn, session_id, account_id=account_id)
        await attach_to_session(conn, session_id, resources, crypto_box, account_id=account_id)
    _purge_working_trees(session_id, old_ids)
    return True


async def detach_all_from_session(
    conn: asyncpg.Connection[Any], session_id: str, *, account_id: str
) -> bool:
    """Detach every github_repository from a session and clean up the
    working trees. Used by the full-list-replace path when the new
    resource list contains no github entries.

    Returns whether any attachment was actually removed, so the caller
    can skip the Layer 1 eviction when there was nothing to detach (#713).

    Conn-scoped: sandbox eviction is fired post-commit by
    :func:`aios.services.sessions.update_session`, not here (#713).
    """
    async with conn.transaction():
        old_ids = await _list_attached_resource_ids(conn, session_id, account_id=account_id)
        await queries.delete_session_github_repos(conn, session_id, account_id=account_id)
    _purge_working_trees(session_id, old_ids)
    return bool(old_ids)


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

    In-place rotation does NOT call the sandbox-eviction hook (#713): the
    UPDATE bumps the github echo's ``updated_at``, which is part of the
    ``mount_snapshot`` key (see
    :func:`aios.sandbox.spec.mount_snapshot_from_echoes`), so the
    per-step :meth:`SandboxRegistry.release_if_mounts_changed` already
    recycles the sandbox on the next step.

    Also fires the github-clone-breaker clear NOTIFY (#1720) after commit
    so a rotated token re-probes on the very next provision instead of
    serving out a cooldown opened under the old secret.
    """
    blob = _encrypt_token(crypto_box, new_token, account_id=account_id)
    async with pool.acquire() as conn, conn.transaction():
        result = await queries.update_session_github_repo_blob(
            conn,
            session_id,
            resource_id,
            blob,
            identity=identity,
            account_id=account_id,
        )
    await _notify_clear_clone_breaker(pool, resource_id)
    return result


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


def _lowest_free_rank(used: list[int], *, cap: int) -> int:
    """Return the lowest rank in ``0..cap-1`` not already used.

    github attachments have no rank CHECK, but they get the same
    lowest-free-rank treatment as memory stores for consistency (#270).
    Callers gate the count against the cap first, so a free slot always
    exists here.
    """
    used_set = set(used)
    for rank in range(cap):
        if rank not in used_set:
            return rank
    raise AssertionError("no free rank — caller must enforce the cap first")


async def add_one(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resource: GithubRepositoryResource,
    crypto_box: CryptoBox,
    *,
    account_id: str,
) -> GithubRepositoryResourceEcho:
    """Attach a single github_repository to a session (granular add-one, #270).

    Caller owns the transaction and holds the per-session advisory lock.

    - Enforces ``MAX_REPOS_PER_SESSION`` against ``existing + 1``.
    - Inserts at the lowest free rank.
    - A ``mount_path`` collision surfaces as
      :class:`asyncpg.UniqueViolationError`, mapped to a 4xx
      :class:`ConflictError` (same as the bulk attach path).
    """
    current = await queries.list_session_github_repo_echoes(conn, session_id, account_id=account_id)
    if len(current) + 1 > MAX_REPOS_PER_SESSION:
        raise RateLimitedError(
            f"session at github-repository cap ({len(current)}/{MAX_REPOS_PER_SESSION}); "
            "detach an existing repository to free a slot"
        )
    used_ranks = await queries.list_session_github_repo_ranks(
        conn, session_id, account_id=account_id
    )
    rank = _lowest_free_rank(used_ranks, cap=MAX_REPOS_PER_SESSION)
    blob = _encrypt_token(
        crypto_box, resource.authorization_token.get_secret_value(), account_id=account_id
    )
    try:
        return await queries.insert_session_github_repo(
            conn,
            session_id,
            rank=rank,
            repo_url=resource.url,
            mount_path=resource.mount_path,
            blob=blob,
            git_user_name=resource.git_user_name,
            git_user_email=resource.git_user_email,
            account_id=account_id,
        )
    except asyncpg.UniqueViolationError as exc:
        raise ConflictError(
            "duplicate mount_path among github_repository attachments",
            detail={"session_id": session_id, "mount_path": resource.mount_path},
        ) from exc


async def remove_one(
    conn: asyncpg.Connection[Any],
    session_id: str,
    resource_id: str,
    *,
    account_id: str,
) -> None:
    """Detach a single github_repository attachment by its row id (#270).

    Raises :class:`NotFoundError` if the attachment doesn't exist. After
    the row is deleted, purges its on-disk working tree — the
    ``.git/config`` embeds the auth token, so leaving it on disk would be
    a slow plaintext-token leak (same cleanup as
    :func:`detach_all_from_session`).
    """
    await queries.delete_session_github_repo(conn, session_id, resource_id, account_id=account_id)
    _purge_working_trees(session_id, [resource_id])
