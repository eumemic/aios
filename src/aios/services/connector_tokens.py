"""Service layer for connector tokens (#301).

Plaintext token format: ``aios_conn_<32-byte-base64url>``.  The DB
stores only ``sha256(plaintext).hexdigest()``; plaintext is returned
exactly once, in the issue response.

The single hot-path entry is :func:`resolve` — used by
:class:`aios.api.deps.ConnectorAuthDep` on every request authenticated
with a connector token.  It runs one ``UPDATE … RETURNING`` (lookup +
``last_used_at`` touch) per call.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any, NamedTuple

import asyncpg

from aios.db import queries
from aios.models.connector_tokens import ConnectorToken


class ResolvedToken(NamedTuple):
    """The handful of fields auth needs after a successful token lookup."""

    token_id: str
    connection_id: str


_TOKEN_PREFIX = "aios_conn_"
_TOKEN_BYTES = 32  # 256 bits of entropy


def _generate_plaintext() -> str:
    return _TOKEN_PREFIX + secrets.token_urlsafe(_TOKEN_BYTES)


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


async def issue(
    pool: asyncpg.Pool[Any],
    *,
    connection_id: str,
    label: str | None,
) -> tuple[ConnectorToken, str]:
    """Issue a fresh token bound to ``connection_id``.

    Returns ``(record, plaintext)``.  The plaintext is the bearer token
    the connector container will use; it is NEVER persisted.
    """
    plaintext = _generate_plaintext()
    async with pool.acquire() as conn:
        token = await queries.insert_connector_token(
            conn,
            connection_id=connection_id,
            label=label,
            token_hash=_hash(plaintext),
        )
    return token, plaintext


async def list_for_connection(pool: asyncpg.Pool[Any], connection_id: str) -> list[ConnectorToken]:
    async with pool.acquire() as conn:
        return await queries.list_connector_tokens(conn, connection_id)


async def revoke(pool: asyncpg.Pool[Any], token_id: str) -> ConnectorToken:
    async with pool.acquire() as conn:
        return await queries.revoke_connector_token(conn, token_id)


async def resolve(pool: asyncpg.Pool[Any], plaintext: str) -> ResolvedToken | None:
    """Resolve a plaintext bearer to a ``ResolvedToken`` or ``None``.

    Touches ``last_used_at`` as a side effect.  Returns ``None`` for
    misses, revoked tokens, and plaintext that doesn't carry our prefix
    (the prefix check is a cheap pre-filter — non-prefix tokens can't
    possibly hash to something we issued).
    """
    if not plaintext.startswith(_TOKEN_PREFIX):
        return None
    async with pool.acquire() as conn:
        row = await queries.resolve_connector_token(conn, _hash(plaintext))
    if row is None:
        return None
    token_id, connection_id = row
    return ResolvedToken(token_id=token_id, connection_id=connection_id)
