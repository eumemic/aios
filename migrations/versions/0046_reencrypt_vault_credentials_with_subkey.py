"""Re-encrypt every ``vault_credentials`` row with its per-account subkey.

Pairs with the application-side wiring that derives an HKDF-SHA256 subkey
from the master ``AIOS_VAULT_KEY`` and uses it for all encrypt/decrypt
operations against ``vault_credentials``. Pre-existing rows are encrypted
under the master key alone, so this migration walks each active row,
decrypts with the master, and re-encrypts under
``master.derive_account_subkey(row.account_id)`` in place.

Archived rows are skipped — their ciphertext/nonce columns were scrubbed
to zero bytes at archive time and re-encrypting zero bytes would either
fail or produce a meaningless blob.

Operator requirements:

* ``AIOS_VAULT_KEY`` must be set in the environment when ``alembic
  upgrade`` runs against a database that has existing ``vault_credentials``
  rows. The application already requires this at boot, so the migration
  inherits the same constraint. Empty databases (fresh deployments,
  ``alembic upgrade head`` in CI) skip the env-var check.
* Failure mode is fail-loud: a decrypt error on any row aborts the
  migration. The row is most likely encrypted with a different master
  key than the operator has configured, which is exactly the case where
  re-encrypting would silently lose the secret.

Downgrade reverses the operation (decrypt with subkey, re-encrypt with
master) so the migration is symmetric.

Revision ID: 0046
Revises: 0045
Create Date: 2026-05-14
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl.exceptions import CryptoError
from nacl.secret import SecretBox
from nacl.utils import random as nacl_random

KEY_BYTES = SecretBox.KEY_SIZE
NONCE_BYTES = SecretBox.NONCE_SIZE
BLOB_VERSION = b"\x01"


class EncryptedBlob:
    def __init__(self, *, ciphertext: bytes, nonce: bytes) -> None:
        self.ciphertext = ciphertext
        self.nonce = nonce


class CryptoBox:
    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != KEY_BYTES:
            raise ValueError(f"master key must be {KEY_BYTES} bytes, got {len(master_key)}")
        self._key = master_key
        self._box = SecretBox(master_key)

    def derive_subkey_bytes(self, info: str) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=KEY_BYTES,
            salt=b"aios-vault-hkdf-v1",
            info=info.encode(),
        ).derive(self._key)

    def derive_account_subkey(self, account_id: str) -> CryptoBox:
        if not account_id:
            raise ValueError("account_id must be non-empty")
        return CryptoBox(self.derive_subkey_bytes(f"aios-account-{account_id}"))

    def encrypt(self, plaintext: str) -> EncryptedBlob:
        nonce = nacl_random(NONCE_BYTES)
        ciphertext = self._box.encrypt(plaintext.encode("utf-8"), nonce).ciphertext
        return EncryptedBlob(ciphertext=BLOB_VERSION + ciphertext, nonce=nonce)

    def decrypt(self, blob: EncryptedBlob) -> str:
        ciphertext = blob.ciphertext
        if ciphertext[:1] == BLOB_VERSION:
            ciphertext = ciphertext[1:]
        try:
            plaintext_bytes = self._box.decrypt(ciphertext, blob.nonce)
        except CryptoError as exc:
            raise RuntimeError("could not decrypt — wrong key or corrupted ciphertext") from exc
        return plaintext_bytes.decode("utf-8")

    def encrypt_dict(self, payload: dict[str, Any]) -> EncryptedBlob:
        return self.encrypt(json.dumps(payload))

    def decrypt_dict(self, blob: EncryptedBlob) -> dict[str, Any]:
        decoded = json.loads(self.decrypt(blob))
        if not isinstance(decoded, dict):
            raise ValueError("decrypted blob did not decode to a dict")
        return decoded


revision: str = "0046"
down_revision: str = "0045"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _load_master() -> CryptoBox:
    encoded = os.environ.get("AIOS_VAULT_KEY")
    if not encoded:
        raise RuntimeError(
            "AIOS_VAULT_KEY must be set when running migration 0046 — "
            "the migration re-encrypts vault_credentials and needs the master key"
        )
    return CryptoBox(base64.b64decode(encoded, validate=True))


def _rekey(
    build_src: Callable[[CryptoBox], Callable[[str], CryptoBox]],
    build_dst: Callable[[CryptoBox], Callable[[str], CryptoBox]],
) -> None:
    """Walk active vault_credentials, decrypt with ``src_for(account_id)``,
    encrypt with ``dst_for(account_id)``, and write the new (ciphertext,
    nonce) back.

    The master key is only loaded if there are rows to re-encrypt. An
    empty database — the common case for ``alembic upgrade head`` in CI
    and fresh deployments — skips the env-var check entirely. ``build_src``
    and ``build_dst`` take the master and return ``account_id → CryptoBox``
    so upgrade (master→subkey) and downgrade (subkey→master) share the
    walker."""
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, account_id, ciphertext, nonce "
            "FROM vault_credentials "
            "WHERE archived_at IS NULL"
        )
    ).fetchall()
    if not rows:
        return
    master = _load_master()
    src_for = build_src(master)
    dst_for = build_dst(master)
    for row in rows:
        plaintext = src_for(row.account_id).decrypt(
            EncryptedBlob(ciphertext=row.ciphertext, nonce=row.nonce)
        )
        new_blob = dst_for(row.account_id).encrypt(plaintext)
        bind.execute(
            sa.text(
                "UPDATE vault_credentials "
                "SET ciphertext = :ct, nonce = :nn, updated_at = now() "
                "WHERE id = :id"
            ),
            {"ct": new_blob.ciphertext, "nn": new_blob.nonce, "id": row.id},
        )


def upgrade() -> None:
    _rekey(
        build_src=lambda master: lambda _account_id: master,
        build_dst=lambda master: master.derive_account_subkey,
    )


def downgrade() -> None:
    _rekey(
        build_src=lambda master: master.derive_account_subkey,
        build_dst=lambda master: lambda _account_id: master,
    )
