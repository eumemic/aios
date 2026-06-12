"""Re-encrypt ``connections.secrets_*`` and ``session_github_repositories``
ciphertext columns with per-account HKDF subkeys.

Companion to migration 0046, which did the same for ``vault_credentials``.
After this migration, every encrypted blob in the database is keyed to its
owning account, not the master key. An attacker who recovers one tenant's
derived key cannot decrypt any other tenant's secrets.

For each table, the migration walks active rows that actually have
ciphertext set, decrypts under the master key, and re-encrypts under
``master.derive_account_subkey(row.account_id)``. Same fail-loud behavior
as 0046: a decrypt error aborts the migration rather than silently
overwriting the row with the wrong key.

The master key is only loaded if at least one row needs re-encryption,
so fresh deployments (and the integration test's empty-database upgrade)
don't require ``AIOS_VAULT_KEY`` to be set.

Revision ID: 0047
Revises: 0046
Create Date: 2026-05-14
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl.exceptions import CryptoError
from nacl.secret import SecretBox
from nacl.utils import random as nacl_random
import sqlalchemy as sa
from alembic import op


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

    def derive_account_subkey(self, account_id: str) -> "CryptoBox":
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

revision: str = "0047"
down_revision: str = "0046"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# (table_name, ciphertext_col, nonce_col, extra_where_clause).
# `where` filters to rows that need rekey; archived/scrubbed rows are
# excluded so we don't fail trying to decrypt zero bytes.
_TABLES: Sequence[tuple[str, str, str, str]] = (
    # connections: ciphertext / nonce are NULL when the connection has no
    # secrets configured. archive_connection() scrubs both to NULL, so this
    # filter naturally skips archived rows too.
    ("connections", "secrets_ciphertext", "secrets_nonce", "secrets_ciphertext IS NOT NULL"),
    # session_github_repositories: ciphertext / nonce are NOT NULL on
    # active rows. No archived_at column — rows are DELETEd when detached.
    ("session_github_repositories", "ciphertext", "nonce", "TRUE"),
)


def _load_master() -> CryptoBox:
    encoded = os.environ.get("AIOS_VAULT_KEY")
    if not encoded:
        raise RuntimeError(
            "AIOS_VAULT_KEY must be set when running migration 0047 — "
            "the migration re-encrypts secrets and needs the master key"
        )
    return CryptoBox(base64.b64decode(encoded, validate=True))


def _rekey(
    build_src: Callable[[CryptoBox], Callable[[str], CryptoBox]],
    build_dst: Callable[[CryptoBox], Callable[[str], CryptoBox]],
) -> None:
    """For each :data:`_TABLES` entry, walk rows with non-null ciphertext,
    decrypt under ``src_for(account_id)``, re-encrypt under
    ``dst_for(account_id)``, and write the new bytes back.

    Master key is loaded lazily — only if at least one table has rows to
    rekey. Empty DBs skip the env-var requirement entirely. ``build_src``
    and ``build_dst`` take the master and return ``account_id → CryptoBox``
    so upgrade and downgrade share the walker."""
    bind = op.get_bind()
    master: CryptoBox | None = None
    src_for: Callable[[str], CryptoBox] | None = None
    dst_for: Callable[[str], CryptoBox] | None = None
    for name, ct_col, nn_col, where in _TABLES:
        rows = bind.execute(
            sa.text(
                f"SELECT id, account_id, {ct_col} AS ct, {nn_col} AS nn FROM {name} WHERE {where}"
            )
        ).fetchall()
        if not rows:
            continue
        if master is None:
            master = _load_master()
            src_for = build_src(master)
            dst_for = build_dst(master)
        assert src_for is not None and dst_for is not None  # for mypy
        for row in rows:
            plaintext = src_for(row.account_id).decrypt(
                EncryptedBlob(ciphertext=row.ct, nonce=row.nn)
            )
            new_blob = dst_for(row.account_id).encrypt(plaintext)
            bind.execute(
                sa.text(
                    f"UPDATE {name} "
                    f"SET {ct_col} = :ct, {nn_col} = :nn, updated_at = now() "
                    f"WHERE id = :id"
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
