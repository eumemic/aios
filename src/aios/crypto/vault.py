"""Symmetric encryption box using libsodium SecretBox (XChaCha20-Poly1305 + Poly1305 MAC).

The aios server holds a single 32-byte master key in the ``AIOS_VAULT_KEY`` env
var (base64-encoded). Every *active* encrypted row stores a randomly-generated
nonce alongside its ciphertext; encryption is authenticated, so any tampering
or key mismatch produces a clean error rather than silent corruption. New
ciphertexts carry a one-byte format prefix; legacy unprefixed blobs remain
readable so operators can re-encrypt them in place with ``aios rekey``.
Archived rows have their ciphertext and nonce zeroed out so that a future DB
dump or query cannot leak the secret — read paths filter ``WHERE archived_at IS
NULL`` to avoid attempting to decrypt the scrubbed bytes.

Multi-tenancy (#367): the master ``CryptoBox`` can derive a per-account
subkey via HKDF-SHA256, returning a new ``CryptoBox`` whose secrets can't
be decrypted with the master key OR any sibling tenant's derived key.
This is the building block for per-account encryption — see
:meth:`CryptoBox.derive_account_subkey`.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl.exceptions import CryptoError
from nacl.secret import SecretBox
from nacl.utils import random as nacl_random

from aios.errors import CryptoDecryptError

# SecretBox uses 24-byte nonces and 32-byte keys.
KEY_BYTES = SecretBox.KEY_SIZE
NONCE_BYTES = SecretBox.NONCE_SIZE
BLOB_VERSION = b"\x01"


@dataclass(frozen=True, slots=True)
class EncryptedBlob:
    """A ciphertext + nonce pair as stored in encrypted row columns."""

    ciphertext: bytes
    nonce: bytes


class CryptoBox:
    """libsodium-backed encrypt/decrypt wrapper around a single master key."""

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != KEY_BYTES:
            raise ValueError(f"master key must be {KEY_BYTES} bytes, got {len(master_key)}")
        self._key = master_key
        self._box = SecretBox(master_key)

    @property
    def key_bytes(self) -> bytes:
        """Raw key bytes for same-package derivation consumers."""
        return self._key

    def derive_subkey_bytes(self, info: str) -> bytes:
        """Derive a 32-byte HKDF-SHA256 subkey for the domain context ``info``."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=KEY_BYTES,
            salt=b"aios-vault-hkdf-v1",
            info=info.encode(),
        )
        return hkdf.derive(self._key)

    def derive_account_subkey(self, account_id: str) -> CryptoBox:
        """Return a new :class:`CryptoBox` keyed to ``account_id`` via HKDF."""
        if not account_id:
            raise ValueError("account_id must be non-empty")
        return CryptoBox(self.derive_subkey_bytes(f"aios-account-{account_id}"))

    @classmethod
    def from_base64(cls, encoded: str, *, env_name: str = "AIOS_VAULT_KEY") -> CryptoBox:
        """Load a CryptoBox from a base64-encoded 32-byte key string."""
        try:
            key_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"{env_name} is not valid base64: {exc}") from exc
        return cls(key_bytes)

    def encrypt(self, plaintext: str) -> EncryptedBlob:
        """Encrypt ``plaintext`` and return the versioned (ciphertext, nonce) pair."""
        nonce = nacl_random(NONCE_BYTES)
        ciphertext = self._box.encrypt(plaintext.encode("utf-8"), nonce).ciphertext
        return EncryptedBlob(ciphertext=BLOB_VERSION + ciphertext, nonce=nonce)

    def decrypt(self, blob: EncryptedBlob) -> str:
        """Decrypt and return the original plaintext string.

        New blobs carry a one-byte ``BLOB_VERSION`` prefix; legacy blobs do
        not. A legacy blob's leading MAC byte can coincidentally equal the
        version byte (~1/256 of rows), so when the prefix-stripped
        interpretation fails its MAC check we retry the blob verbatim as
        legacy before raising (#858 R4: legacy unversioned blobs must
        round-trip). SecretBox's Poly1305 MAC makes the retry sound: a wrong
        interpretation always fails loudly, never yields wrong plaintext.

        Raises :class:`~aios.errors.CryptoDecryptError` if the key doesn't
        match or the ciphertext has been tampered with.
        """
        ciphertext = blob.ciphertext
        if ciphertext[:1] != BLOB_VERSION:
            return self._decrypt_ciphertext(ciphertext, blob.nonce)
        try:
            return self._decrypt_ciphertext(ciphertext[1:], blob.nonce)
        except CryptoDecryptError:
            # Legacy blob whose first MAC byte happens to be the version byte.
            return self._decrypt_ciphertext(ciphertext, blob.nonce)

    def _decrypt_ciphertext(self, ciphertext: bytes, nonce: bytes) -> str:
        try:
            plaintext_bytes = self._box.decrypt(ciphertext, nonce)
        except CryptoError as exc:
            raise CryptoDecryptError(
                "could not decrypt — wrong key or corrupted ciphertext",
                detail={"reason": str(exc)},
            ) from exc
        return plaintext_bytes.decode("utf-8")

    def encrypt_dict(self, payload: dict[str, Any]) -> EncryptedBlob:
        """Encrypt a JSON-serialisable dict.  Convenience over ``encrypt``."""
        return self.encrypt(json.dumps(payload))

    def decrypt_dict(self, blob: EncryptedBlob) -> dict[str, Any]:
        """Decrypt and JSON-decode to a dict."""
        plaintext = self.decrypt(blob)
        decoded = json.loads(plaintext)
        if not isinstance(decoded, dict):
            raise ValueError(
                f"decrypted blob did not decode to a dict; got {type(decoded).__name__}"
            )
        return decoded
