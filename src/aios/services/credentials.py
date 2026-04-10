"""Business logic for the credential vault.

Pure functions: take an asyncpg pool and a vault, return pydantic models.
No FastAPI imports — these functions are also used by the harness when it
needs to decrypt a credential mid-loop.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.crypto.vault import Vault
from aios.db import queries
from aios.models.credentials import Credential


async def create_credential(
    pool: asyncpg.Pool[Any],
    vault: Vault,
    *,
    name: str,
    provider: str,
    plaintext_value: str,
) -> Credential:
    blob = vault.encrypt(plaintext_value)
    async with pool.acquire() as conn:
        return await queries.insert_credential(conn, name=name, provider=provider, blob=blob)


async def get_credential(pool: asyncpg.Pool[Any], cred_id: str) -> Credential:
    async with pool.acquire() as conn:
        return await queries.get_credential(conn, cred_id)


async def list_credentials(
    pool: asyncpg.Pool[Any], *, limit: int = 50, after: str | None = None
) -> list[Credential]:
    async with pool.acquire() as conn:
        return await queries.list_credentials(conn, limit=limit, after=after)


async def archive_credential(pool: asyncpg.Pool[Any], cred_id: str) -> None:
    async with pool.acquire() as conn:
        await queries.archive_credential(conn, cred_id)


async def decrypt_credential(pool: asyncpg.Pool[Any], vault: Vault, cred_id: str) -> str:
    """Fetch and decrypt a credential's plaintext value.

    Used by the harness immediately before calling LiteLLM. Plaintext should
    not live longer than the calling function's stack frame.
    """
    async with pool.acquire() as conn:
        blob = await queries.get_credential_blob(conn, cred_id)
    return vault.decrypt(blob)
