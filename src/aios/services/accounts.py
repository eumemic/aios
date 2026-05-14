"""Account lifecycle logic.

PR 1 of the multi-tenancy stack (issue #367) ships only the bootstrap
path: the operator's one-shot route to mint the root account and its
first API key on a fresh deployment. Per-account CRUD, child creation,
key rotation, and the rest of the management surface land in PR 6 once
the auth dep is account-aware.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

import asyncpg

from aios.db import queries
from aios.models.accounts import BootstrapResponse

# Generated bearer tokens carry an opaque random body plus a short prefix
# so they're recognizable in logs without leaking the secret. The body is
# 32 bytes of cryptographic randomness, base32-encoded so the result is
# URL/header-safe and easy to copy-paste from a terminal.
_KEY_PREFIX = "aios_"
_KEY_BODY_BYTES = 32
_BOOTSTRAP_KEY_LABEL = "bootstrap"


def _generate_plaintext_key() -> str:
    """Return a fresh ``aios_<base32>`` API key."""
    body = secrets.token_urlsafe(_KEY_BODY_BYTES)
    return f"{_KEY_PREFIX}{body}"


def hash_key(plaintext: str) -> bytes:
    """SHA-256 of the plaintext key, in raw bytes for ``bytea`` storage.

    Exposed at module level so the auth dep (later PRs) can hash an
    inbound bearer token and look it up by ``hash``.
    """
    return hashlib.sha256(plaintext.encode("ascii")).digest()


async def bootstrap_root(
    pool: asyncpg.Pool[Any],
    *,
    display_name: str,
) -> BootstrapResponse:
    """Create the root account and mint its first API key.

    The caller has already authenticated against ``AIOS_BOOTSTRAP_TOKEN``
    and confirmed no active root exists. Returns the new account id and
    the plaintext key — the *only* time the key is returned in plaintext.
    """
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
