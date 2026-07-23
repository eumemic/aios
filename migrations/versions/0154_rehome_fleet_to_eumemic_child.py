"""Re-home the production fleet from the platform root to Eumemic.

The platform root must remain credentialless.  This migration creates the
Eumemic tenant below it, moves every account-scoped row, and re-encrypts every
secret whose key is derived from its owning account.  Alembic's PostgreSQL
transaction envelope makes the cutover atomic.

Revision ID: 0154
Revises: 0152
"""

from __future__ import annotations

import base64
import os
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl.exceptions import CryptoError
from nacl.secret import SecretBox
from nacl.utils import random as nacl_random

revision: str = "0154"
down_revision: str = "0152"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CHILD_NAME = "Eumemic"
_MARKER_LABEL = "0154 bootstrap (revoked)"
_BLOB_VERSION = b"\x01"
# These constraints make a bulk account-id rewrite impossible.  They are
# recreated (and validated) before the transaction commits.
_COMPOSITE_FKS: Sequence[tuple[str, str, str]] = (
    (
        "session_vaults",
        "session_vaults_session_account_id_fkey",
        "FOREIGN KEY (session_id, account_id) REFERENCES sessions(id, account_id) ON DELETE CASCADE",
    ),
    (
        "session_vaults",
        "session_vaults_vault_account_id_fkey",
        "FOREIGN KEY (vault_id, account_id) REFERENCES vaults(id, account_id)",
    ),
    (
        "wf_run_vaults",
        "wf_run_vaults_run_account_id_fkey",
        "FOREIGN KEY (run_id, account_id) REFERENCES wf_runs(id, account_id) ON DELETE CASCADE",
    ),
    (
        "wf_run_vaults",
        "wf_run_vaults_vault_account_id_fkey",
        "FOREIGN KEY (vault_id, account_id) REFERENCES vaults(id, account_id)",
    ),
    (
        "vault_credentials",
        "vault_credentials_vault_account_id_fkey",
        "FOREIGN KEY (vault_id, account_id) REFERENCES vaults(id, account_id) ON DELETE CASCADE",
    ),
    (
        "oauth_flows",
        "oauth_flows_vault_account_id_fkey",
        "FOREIGN KEY (vault_id, account_id) REFERENCES vaults(id, account_id) ON DELETE CASCADE",
    ),
    (
        "workflow_versions",
        "workflow_versions_workflow_id_account_id_fkey",
        "FOREIGN KEY (workflow_id, account_id) REFERENCES workflows(id, account_id) ON DELETE CASCADE",
    ),
)
_ENCRYPTED: Sequence[tuple[str, str, str, str]] = (
    ("model_providers", "ciphertext", "nonce", "ciphertext IS NOT NULL"),
    ("vault_credentials", "ciphertext", "nonce", "archived_at IS NULL AND length(ciphertext) > 0"),
    ("connections", "secrets_ciphertext", "secrets_nonce", "secrets_ciphertext IS NOT NULL"),
    ("session_github_repositories", "ciphertext", "nonce", "ciphertext IS NOT NULL"),
    ("oauth_flows", "ciphertext", "nonce", "ciphertext IS NOT NULL"),
)


class _Box:
    def __init__(self, key: bytes) -> None:
        if len(key) != SecretBox.KEY_SIZE:
            raise ValueError(f"master key must be {SecretBox.KEY_SIZE} bytes")
        self._key = key
        self._box = SecretBox(key)

    def account(self, account_id: str) -> _Box:
        key = HKDF(
            algorithm=hashes.SHA256(),
            length=SecretBox.KEY_SIZE,
            salt=b"aios-vault-hkdf-v1",
            info=f"aios-account-{account_id}".encode(),
        ).derive(self._key)
        return _Box(key)

    def decrypt(self, ciphertext: bytes, nonce: bytes) -> str:
        try:
            if ciphertext[:1] != _BLOB_VERSION:
                return self._box.decrypt(ciphertext, nonce).decode()
            try:
                return self._box.decrypt(ciphertext[1:], nonce).decode()
            except CryptoError:
                # A legacy blob's first MAC byte can equal the version byte.
                return self._box.decrypt(ciphertext, nonce).decode()
        except CryptoError as exc:
            raise RuntimeError("migration 0154 could not decrypt account-owned ciphertext") from exc

    def encrypt(self, plaintext: str, nonce: bytes | None = None) -> tuple[bytes, bytes]:
        nonce = nonce or nacl_random(SecretBox.NONCE_SIZE)
        return _BLOB_VERSION + self._box.encrypt(plaintext.encode(), nonce).ciphertext, nonce


def _master() -> _Box:
    value = os.environ.get("AIOS_VAULT_KEY")
    if not value:
        raise RuntimeError("AIOS_VAULT_KEY is required to re-home encrypted fleet rows")
    return _Box(base64.b64decode(value, validate=True))


def _root_and_child(*, create: bool) -> tuple[str | None, str | None]:
    bind = op.get_bind()
    root = bind.execute(
        sa.text(
            "SELECT id FROM accounts WHERE parent_account_id IS NULL AND archived_at IS NULL FOR UPDATE"
        )
    ).scalar_one_or_none()
    if root is None:
        # Fresh database (tests, new installs): no root account, nothing to
        # re-home. The migration is a structural no-op.
        return None, None
    child = bind.execute(
        sa.text(
            "SELECT id FROM accounts WHERE parent_account_id=:root AND display_name=:name AND archived_at IS NULL FOR UPDATE"
        ),
        {"root": root, "name": _CHILD_NAME},
    ).scalar_one_or_none()
    created = child is None and create
    if created:
        # SQL equivalent of insert_child_account: account and first key are
        # inserted together in this migration's transaction.  The bootstrap
        # key is deliberately revoked; operational keys are moved below.
        child = (
            "acc_"
            + bind.execute(sa.text("SELECT replace(gen_random_uuid()::text, '-', '')")).scalar_one()
        )
        key_id = (
            "acckey_"
            + bind.execute(sa.text("SELECT replace(gen_random_uuid()::text, '-', '')")).scalar_one()
        )
        bind.execute(
            sa.text(
                "INSERT INTO accounts (id,parent_account_id,can_mint_children,display_name) VALUES (:id,:root,true,:name)"
            ),
            {"id": child, "root": root, "name": _CHILD_NAME},
        )
        bind.execute(
            sa.text(
                "INSERT INTO account_keys (key_id,account_id,hash,label,revoked_at) "
                "VALUES (:id,:account,:hash,:label,now())"
            ),
            {"id": key_id, "account": child, "hash": os.urandom(32), "label": _MARKER_LABEL},
        )
    if child is None:
        if not create:
            # No child on this database (the upgrade no-opped): nothing to undo.
            return root, None
        raise RuntimeError("migration 0154 upgrade could not create the Eumemic child")
    if child is None:
        return root, None
    marker = bind.execute(
        sa.text(
            "SELECT 1 FROM account_keys WHERE account_id=:child AND label=:label "
            "AND revoked_at IS NOT NULL"
        ),
        {"child": child, "label": _MARKER_LABEL},
    ).scalar_one_or_none()
    if marker is None:
        direction = "upgrade" if create else "downgrade"
        raise RuntimeError(
            f"migration 0154 {direction} refuses pre-existing Eumemic child; "
            "rename/archive it or verify the migration-owned marker key"
        )
    return root, child


def _drop_composite_fks() -> None:
    for table, name, _ in _COMPOSITE_FKS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")


def _restore_composite_fks() -> None:
    for table, name, definition in _COMPOSITE_FKS:
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} {definition} NOT VALID")
    for table, name, _ in _COMPOSITE_FKS:
        op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")


def _rekey(source: str, destination: str) -> None:
    bind = op.get_bind()
    master: _Box | None = None
    for table, ciphertext, nonce, live_filter in _ENCRYPTED:
        # Block concurrent writes until this transaction commits, so the live
        # selection and inverse-predicate scrubbed-row move are exhaustive.
        bind.execute(sa.text(f"LOCK TABLE {table} IN SHARE ROW EXCLUSIVE MODE"))
        rows = bind.execute(
            sa.text(
                f"SELECT id,{ciphertext} AS ciphertext,{nonce} AS nonce FROM {table} WHERE account_id=:source AND {live_filter} FOR UPDATE"
            ),
            {"source": source},
        ).fetchall()
        if rows and master is None:
            master = _master()
        for row in rows:
            assert master is not None
            plaintext = master.account(source).decrypt(row.ciphertext, row.nonce)
            # Reuse the nonce: SecretBox permits this because the key changes,
            # and it makes downgrade restore the original bytes exactly.
            new_ciphertext, new_nonce = master.account(destination).encrypt(plaintext, row.nonce)
            bind.execute(
                sa.text(
                    f"UPDATE {table} SET account_id=:destination,{ciphertext}=:ciphertext,{nonce}=:nonce WHERE id=:id"
                ),
                {
                    "destination": destination,
                    "ciphertext": new_ciphertext,
                    "nonce": new_nonce,
                    "id": row.id,
                },
            )
        # Scrubbed/archived encrypted rows still belong to the fleet but need no decrypt.
        bind.execute(
            sa.text(
                f"UPDATE {table} SET account_id=:destination "
                f"WHERE account_id=:source AND NOT ({live_filter})"
            ),
            {"source": source, "destination": destination},
        )


def _plain_move(source: str, destination: str) -> None:
    bind = op.get_bind()
    excluded = {
        "accounts",
        "account_keys",
        "events",
        # workflow_versions moves via its own guarded path below: the 0112
        # immutability trigger forbids UPDATE, but the composite FK to
        # workflows(id, account_id) forces version rows to move with their
        # workflow. The trigger is disabled for exactly that statement.
        "workflow_versions",
        *[item[0] for item in _ENCRYPTED],
    }
    tables = (
        bind.execute(
            sa.text(
                "SELECT table_name FROM information_schema.columns WHERE table_schema=current_schema() AND column_name='account_id' ORDER BY table_name"
            )
        )
        .scalars()
        .all()
    )
    for table in tables:
        if table not in excluded:
            bind.execute(
                sa.text(f'UPDATE "{table}" SET account_id=:destination WHERE account_id=:source'),
                {"source": source, "destination": destination},
            )
    # workflow_versions: suspend the 0112 no-UPDATE trigger for this one
    # statement (same transaction), because the composite FK to workflows
    # requires the denormalized account_id to move in lockstep.
    bind.execute(sa.text("ALTER TABLE workflow_versions DISABLE TRIGGER workflow_versions_no_update"))
    try:
        bind.execute(
            sa.text("UPDATE workflow_versions SET account_id=:destination WHERE account_id=:source"),
            {"source": source, "destination": destination},
        )
    finally:
        bind.execute(
            sa.text("ALTER TABLE workflow_versions ENABLE TRIGGER workflow_versions_no_update")
        )
    # Events are normally the largest relation. Locks persist until commit;
    # ctid batches only bound each statement's work and memory.
    while (
        bind.execute(
            sa.text(
                "WITH batch AS (SELECT ctid FROM events WHERE account_id=:source LIMIT 10000) UPDATE events e SET account_id=:destination FROM batch WHERE e.ctid=batch.ctid RETURNING 1"
            ),
            {"source": source, "destination": destination},
        ).first()
        is not None
    ):
        pass


def _move_keys(source: str, destination: str, *, retain_one: bool) -> None:
    bind = op.get_bind()
    if retain_one:
        keeper = bind.execute(
            sa.text(
                "SELECT key_id FROM account_keys WHERE account_id=:source AND revoked_at IS NULL ORDER BY created_at,key_id LIMIT 1"
            ),
            {"source": source},
        ).scalar_one_or_none()
        if keeper is None:
            # No active root keys at all (seeded/test databases): there is no
            # operational key set to split; nothing to move or retain.
            return
        bind.execute(
            sa.text(
                "UPDATE account_keys SET account_id=:destination WHERE account_id=:source AND key_id<>:keeper"
            ),
            {"source": source, "destination": destination, "keeper": keeper},
        )
    else:
        # Leave the migration-created revoked bootstrap key on the child so it
        # can be deleted with the account; move every pre-existing key back.
        bind.execute(
            sa.text(
                "UPDATE account_keys SET account_id=:destination WHERE account_id=:source AND label<>'0154 bootstrap (revoked)'"
            ),
            {"source": source, "destination": destination},
        )


def upgrade() -> None:
    root, child = _root_and_child(create=True)
    if root is None or child is None:
        return
    _drop_composite_fks()
    _rekey(root, child)
    _plain_move(root, child)
    _move_keys(root, child, retain_one=True)
    op.get_bind().execute(
        sa.text(
            "UPDATE accounts SET spent_microusd=CASE WHEN id=:child THEN spent_microusd+:spent ELSE 0 END WHERE id IN (:root,:child)"
        ),
        {
            "root": root,
            "child": child,
            "spent": op.get_bind()
            .execute(sa.text("SELECT spent_microusd FROM accounts WHERE id=:root"), {"root": root})
            .scalar_one(),
        },
    )
    _restore_composite_fks()


def downgrade() -> None:
    root, child = _root_and_child(create=False)
    if root is None or child is None:
        return
    _drop_composite_fks()
    _rekey(child, root)
    _plain_move(child, root)
    _move_keys(child, root, retain_one=False)
    op.get_bind().execute(
        sa.text(
            "UPDATE accounts SET spent_microusd=(SELECT spent_microusd FROM accounts WHERE id=:child) WHERE id=:root"
        ),
        {"root": root, "child": child},
    )
    _restore_composite_fks()
    op.get_bind().execute(sa.text("DELETE FROM accounts WHERE id=:child"), {"child": child})
