"""Account lifecycle logic."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

import asyncpg

from aios.db import queries
from aios.models.accounts import BootstrapResponse

_KEY_PREFIX = "aios_"
_KEY_BODY_BYTES = 32
_BOOTSTRAP_KEY_LABEL = "bootstrap"


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
    account_id: str,
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
            account_id=account_id,
        )
    return BootstrapResponse(
        account_id=account.id,
        key_id=key_id,
        plaintext_key=plaintext,
    )
