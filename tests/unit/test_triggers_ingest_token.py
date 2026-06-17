"""Unit tests for the external-event ingest secret mint (#1281).

The per-trigger secret mirrors the runtime_tokens precedent: an ``aios_evt_``
prefix, high CSPRNG entropy, SHA-256-at-rest, and a hash that matches what the
ingress recomputes from the path token.
"""

from __future__ import annotations

import hashlib

from aios.services.triggers import (
    _INGEST_TOKEN_PREFIX,
    _mint_ingest_token,
    mint_ingest_token_hash,
)


def test_mint_returns_prefixed_plaintext_and_matching_hash() -> None:
    plaintext, token_hash = _mint_ingest_token()
    assert plaintext.startswith(_INGEST_TOKEN_PREFIX)
    # The stored hash is exactly what the ingress recomputes from the path token.
    assert token_hash == hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
    assert len(token_hash) == 64


def test_mint_is_unique_per_call() -> None:
    a, ah = _mint_ingest_token()
    b, bh = _mint_ingest_token()
    assert a != b
    assert ah != bh


def test_mint_hash_only_helper_drops_plaintext() -> None:
    # The session-create attach path keeps only the hash; it is a valid
    # 64-char sha256 hex of *some* prefixed plaintext (not re-derivable).
    h = mint_ingest_token_hash()
    assert len(h) == 64
    int(h, 16)  # valid hex
