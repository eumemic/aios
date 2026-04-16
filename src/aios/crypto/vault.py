"""Symmetric encryption box using libsodium SecretBox (XChaCha20-Poly1305 + Poly1305 MAC).

The aios server holds a single 32-byte master key in the ``AIOS_VAULT_KEY`` env
var (base64-encoded). Every *active* encrypted row stores a randomly-generated
nonce alongside its ciphertext; encryption is authenticated, so any tampering
or key mismatch produces a clean error rather than silent corruption. Archived
rows have their ciphertext and nonce zeroed out so that a future DB dump or
query cannot leak the secret — read paths filter ``WHERE archived_at IS NULL``
to avoid attempting to decrypt the scrubbed bytes.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from nacl.exceptions import CryptoError
from nacl.secret import SecretBox
from nacl.utils import random as nacl_random

from aios.errors import CryptoDecryptError

# SecretBox uses 24-byte nonces and 32-byte keys.
KEY_BYTES = SecretBox.KEY_SIZE
NONCE_BYTES = SecretBox.NONCE_SIZE


@dataclass(frozen=True, slots=True)
class EncryptedBlob:
    """A ciphertext + nonce pair as stored in the credentials table."""

    ciphertext: bytes
    nonce: bytes


class CryptoBox:
    """libsodium-backed encrypt/decrypt wrapper around a single master key.

    Construct once at process start with the master key bytes; subsequent
    operations are pure and stateless beyond the held key.
    """

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != KEY_BYTES:
            raise ValueError(f"master key must be {KEY_BYTES} bytes, got {len(master_key)}")
        self._box = SecretBox(master_key)

    @classmethod
    def from_base64(cls, encoded: str) -> CryptoBox:
        """Load a CryptoBox from a base64-encoded master key string.

        This is the form stored in ``AIOS_VAULT_KEY`` and ``.env`` files.
        """
        try:
            key_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"AIOS_VAULT_KEY is not valid base64: {exc}") from exc
        return cls(key_bytes)

    def encrypt(self, plaintext: str) -> EncryptedBlob:
        """Encrypt ``plaintext`` and return the (ciphertext, nonce) pair.

        A fresh random nonce is generated for every call. The plaintext is
        UTF-8-encoded before encryption.
        """
        nonce = nacl_random(NONCE_BYTES)
        ciphertext = self._box.encrypt(plaintext.encode("utf-8"), nonce).ciphertext
        return EncryptedBlob(ciphertext=ciphertext, nonce=nonce)

    def decrypt(self, blob: EncryptedBlob) -> str:
        """Decrypt and return the original plaintext string.

        Raises :class:`~aios.errors.CryptoDecryptError` if the master key
        doesn't match (key rotation without re-encryption) or the ciphertext
        has been tampered with.
        """
        try:
            plaintext_bytes = self._box.decrypt(blob.ciphertext, blob.nonce)
        except CryptoError as exc:
            raise CryptoDecryptError(
                "could not decrypt — wrong key or corrupted ciphertext",
                detail={"reason": str(exc)},
            ) from exc
        return plaintext_bytes.decode("utf-8")
