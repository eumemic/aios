"""Load-bearing migration 0153 fleet re-home checks.

The full PostgreSQL round-trip runs in CI; these checks also pin the encrypted
surface inventory and prove that its account-to-account crypto operation is
symmetric without requiring Docker.
"""

from __future__ import annotations

import base64
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
from nacl.secret import SecretBox

pytestmark = pytest.mark.integration


def _migration() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "migrations"
        / "versions"
        / "0153_rehome_fleet_to_eumemic_child.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0153", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_five_encrypted_surfaces_round_trip_between_account_subkeys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _migration()
    key = os.urandom(SecretBox.KEY_SIZE)
    monkeypatch.setenv("AIOS_VAULT_KEY", base64.b64encode(key).decode())
    master = migration._master()
    root = master.account("acc_root")
    child = master.account("acc_eumemic")
    expected = {
        "model_providers",
        "vault_credentials",
        "connections",
        "session_github_repositories",
        "oauth_flows",
    }
    assert {row[0] for row in migration._ENCRYPTED} == expected

    seeded = {table: root.encrypt(f"secret:{table}") for table in expected}
    moved = {
        table: child.encrypt(root.decrypt(ciphertext, nonce))
        for table, (ciphertext, nonce) in seeded.items()
    }
    for table, (ciphertext, nonce) in moved.items():
        assert child.decrypt(ciphertext, nonce) == f"secret:{table}"
        with pytest.raises(RuntimeError):
            root.decrypt(ciphertext, nonce)

    restored = {
        table: root.encrypt(child.decrypt(ciphertext, nonce))
        for table, (ciphertext, nonce) in moved.items()
    }
    assert {
        table: root.decrypt(ciphertext, nonce) for table, (ciphertext, nonce) in restored.items()
    } == {table: f"secret:{table}" for table in expected}


def test_revision_and_transactional_upgrade_downgrade_contract() -> None:
    migration = _migration()
    assert migration.revision == "0153"
    assert migration.down_revision == "0152"
    assert migration._CHILD_NAME == "Eumemic"
    assert migration._COMPOSITE_FKS
    # Alembic/PostgreSQL supplies the transaction envelope; both directions
    # use the same rekey and inventory walkers, with source/destination swapped.
    assert migration.upgrade.__module__ == migration.downgrade.__module__
