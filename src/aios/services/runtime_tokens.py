"""Service layer for runtime tokens (#328 PR 5).

Per-connector-type bearer tokens.  One runtime container hosts N
connections of a single ``connector`` type and authenticates with one
token; resolution returns ``(token_id, connector)``, never a single
``connection_id``.

Plaintext format: ``aios_runtime_<32-byte-base64url>``.  The DB stores
only ``sha256(plaintext).hexdigest()``; plaintext is returned exactly
once, in the issue response.

Mirrors :mod:`aios.services.connector_tokens` shape — see that module
for the broader design rationale on hashing, soft revocation, and the
prefix pre-filter.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any, NamedTuple

import asyncpg

from aios.db import queries
from aios.models.runtime_tokens import RuntimeToken


class ResolvedRuntimeToken(NamedTuple):
    """The handful of fields auth needs after a successful token lookup."""

    token_id: str
    connector: str


_TOKEN_PREFIX = "aios_runtime_"
_TOKEN_BYTES = 32  # 256 bits of entropy


def _generate_plaintext() -> str:
    return _TOKEN_PREFIX + secrets.token_urlsafe(_TOKEN_BYTES)


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


async def issue(
    pool: asyncpg.Pool[Any],
    *,
    connector: str,
    label: str | None,
) -> tuple[RuntimeToken, str]:
    """Issue a fresh runtime token scoped to ``connector``.

    Returns ``(record, plaintext)``.  Plaintext is the bearer the
    runtime container will use; never persisted.
    """
    plaintext = _generate_plaintext()
    async with pool.acquire() as conn:
        token = await queries.insert_runtime_token(
            conn,
            connector=connector,
            label=label,
            token_hash=_hash(plaintext),
        )
    return token, plaintext


async def list_tokens(pool: asyncpg.Pool[Any], *, connector: str) -> list[RuntimeToken]:
    async with pool.acquire() as conn:
        return await queries.list_runtime_tokens(conn, connector=connector)


async def revoke(pool: asyncpg.Pool[Any], token_id: str) -> RuntimeToken:
    async with pool.acquire() as conn:
        return await queries.revoke_runtime_token(conn, token_id)


async def resolve(pool: asyncpg.Pool[Any], plaintext: str) -> ResolvedRuntimeToken | None:
    """Resolve a plaintext bearer to a ``ResolvedRuntimeToken`` or ``None``.

    Touches ``last_used_at`` as a side effect.  Returns ``None`` for
    misses, revoked tokens, and plaintext lacking the prefix.
    """
    if not plaintext.startswith(_TOKEN_PREFIX):
        return None
    async with pool.acquire() as conn:
        row = await queries.resolve_runtime_token(conn, _hash(plaintext))
    if row is None:
        return None
    token_id, connector = row
    return ResolvedRuntimeToken(token_id=token_id, connector=connector)
