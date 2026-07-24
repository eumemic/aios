"""Load-bearing migration 0153 fleet re-home checks.

The full PostgreSQL round-trip runs in CI; these checks also pin the encrypted
surface inventory and prove that its account-to-account crypto operation is
symmetric without requiring Docker.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import asyncpg
import pytest
from nacl.secret import SecretBox

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import PROJECT_ROOT, _alembic_url

pytestmark = pytest.mark.integration


def _migration() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "migrations"
        / "versions"
        / "0154_rehome_fleet_to_eumemic_child.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0154", path)
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


def test_legacy_unversioned_blob_starting_with_version_byte_round_trips() -> None:
    migration = _migration()
    box = migration._Box(os.urandom(SecretBox.KEY_SIZE))
    plaintext = "legacy-leading-version-byte"
    # MAC bytes are effectively random; find a deterministic-in-test vector
    # whose unversioned ciphertext starts with the version marker.
    for counter in range(10_000):
        nonce = counter.to_bytes(SecretBox.NONCE_SIZE, "big")
        ciphertext = box._box.encrypt(plaintext.encode(), nonce).ciphertext
        if ciphertext[:1] == migration._BLOB_VERSION:
            assert box.decrypt(ciphertext, nonce) == plaintext
            break
    else:  # pragma: no cover - probability is vanishingly small
        pytest.fail("could not construct leading-0x01 legacy ciphertext")


def test_revision_and_transactional_upgrade_downgrade_contract() -> None:
    migration = _migration()
    assert migration.revision == "0154"
    assert migration.down_revision == "0152"
    assert migration._CHILD_NAME == "Eumemic"
    assert migration._COMPOSITE_FKS
    # Alembic/PostgreSQL supplies the transaction envelope; both directions
    # use the same rekey and inventory walkers, with source/destination swapped.
    assert migration.upgrade.__module__ == migration.downgrade.__module__


@pytest.fixture
def postgres() -> Iterator[Any]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


def _run_alembic(args: list[str], db_url: str, key: bytes) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv not found on PATH")
    return subprocess.run(
        [uv, "run", "alembic", *args],
        cwd=PROJECT_ROOT,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "HOME": str(Path.home()),
            "AIOS_DB_URL": db_url,
            "AIOS_VAULT_KEY": base64.b64encode(key).decode(),
        },
        capture_output=True,
        text=True,
        check=False,
    )


async def _seed_fleet(db_url: str, master: Any) -> dict[str, tuple[bytes, bytes, str]]:
    conn = await asyncpg.connect(db_url)
    root = master.account("acc_root")
    secrets = {
        table: (*root.encrypt(f"secret:{table}"), f"secret:{table}")
        for table in (
            "model_providers",
            "vault_credentials",
            "connections",
            "session_github_repositories",
            "oauth_flows",
        )
    }
    try:
        await conn.execute(
            "INSERT INTO accounts (id,display_name,can_mint_children,spent_microusd) "
            "VALUES ('acc_root','root',true,4242)"
        )
        await conn.execute(
            "INSERT INTO account_keys (key_id,account_id,hash,label) VALUES "
            "('key_keep','acc_root',$1,'admin'),('key_move','acc_root',$2,'fleet')",
            os.urandom(32),
            os.urandom(32),
        )
        # Several plain account-scoped relations, including the dependency
        # graph needed by the encrypted surfaces.
        await conn.execute(
            "INSERT INTO environments (id,account_id,name,config) "
            "VALUES ('env_0153','acc_root','env','{}')"
        )
        await conn.execute(
            "INSERT INTO agents (id,account_id,name,model,system,window_min,window_max,version) "
            "VALUES ('agent_0153','acc_root','agent','test/model','',1,2,1)"
        )
        await conn.execute(
            "INSERT INTO sessions (id,account_id,agent_id,environment_id,agent_version,workspace_volume_path) "
            "VALUES ('sess_0153','acc_root','agent_0153','env_0153',1,'/tmp/0153')"
        )
        await conn.execute(
            "INSERT INTO events (id,account_id,session_id,seq,kind,data) "
            "VALUES ('evt_0153','acc_root','sess_0153',1,'message','{}')"
        )
        await conn.execute(
            "INSERT INTO vaults (id,account_id,display_name) VALUES ('vlt_0153','acc_root','vault')"
        )
        ciphertext, nonce, _ = secrets["model_providers"]
        await conn.execute(
            "INSERT INTO model_providers (id,account_id,provider,ciphertext,nonce) "
            "VALUES ('mp_0153','acc_root','test',$1,$2)",
            ciphertext,
            nonce,
        )
        ciphertext, nonce, _ = secrets["vault_credentials"]
        await conn.execute(
            "INSERT INTO vault_credentials "
            "(id,account_id,vault_id,display_name,target_url,auth_type,ciphertext,nonce) "
            "VALUES ('vc_0153','acc_root','vlt_0153','cred','https://example.test',"
            "'bearer_header',$1,$2)",
            ciphertext,
            nonce,
        )
        ciphertext, nonce, _ = secrets["connections"]
        await conn.execute(
            "INSERT INTO connections "
            "(id,account_id,connector,external_account_id,secrets_ciphertext,secrets_nonce) "
            "VALUES ('conn_0153','acc_root','test','external',$1,$2)",
            ciphertext,
            nonce,
        )
        ciphertext, nonce, _ = secrets["session_github_repositories"]
        await conn.execute(
            "INSERT INTO session_github_repositories "
            "(id,account_id,session_id,rank,repo_url,mount_path,ciphertext,nonce) "
            "VALUES ('repo_0153','acc_root','sess_0153',0,'https://example.test/r','/repo',$1,$2)",
            ciphertext,
            nonce,
        )
        ciphertext, nonce, _ = secrets["oauth_flows"]
        await conn.execute(
            "INSERT INTO oauth_flows "
            "(id,account_id,vault_id,target_url,state,redirect_uri,ciphertext,nonce,expires_at) "
            "VALUES ('oauth_0153','acc_root','vlt_0153','https://example.test','state',"
            "'https://callback.test',$1,$2,now()+interval '1 hour')",
            ciphertext,
            nonce,
        )
    finally:
        await conn.close()
    return secrets


async def _assert_upgraded(
    db_url: str, master: Any, expected: dict[str, tuple[bytes, bytes, str]]
) -> str:
    conn = await asyncpg.connect(db_url)
    try:
        child = await conn.fetchval(
            "SELECT id FROM accounts WHERE parent_account_id='acc_root' AND display_name='Eumemic'"
        )
        assert child
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM account_keys WHERE account_id='acc_root' AND revoked_at IS NULL"
            )
            == 1
        )
        assert await conn.fetchval("SELECT spent_microusd FROM accounts WHERE id='acc_root'") == 0
        assert await conn.fetchval("SELECT spent_microusd FROM accounts WHERE id=$1", child) == 4242
        # Root owns no fleet row in any account-scoped table except its one key.
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.columns WHERE table_schema='public' "
            "AND column_name='account_id'"
        )
        for row in tables:
            if row["table_name"] != "account_keys":
                assert (
                    await conn.fetchval(
                        f'SELECT count(*) FROM "{row["table_name"]}" WHERE account_id=$1',
                        "acc_root",
                    )
                    == 0
                )
        columns = {
            "model_providers": ("ciphertext", "nonce"),
            "vault_credentials": ("ciphertext", "nonce"),
            "connections": ("secrets_ciphertext", "secrets_nonce"),
            "session_github_repositories": ("ciphertext", "nonce"),
            "oauth_flows": ("ciphertext", "nonce"),
        }
        for table, (ciphertext_column, nonce_column) in columns.items():
            row = await conn.fetchrow(
                f"SELECT {ciphertext_column} ciphertext,{nonce_column} nonce,account_id FROM {table}"
            )
            assert row["account_id"] == child
            assert (
                master.account(child).decrypt(row["ciphertext"], row["nonce"]) == expected[table][2]
            )
        return str(child)
    finally:
        await conn.close()


async def _assert_restored(db_url: str, expected: dict[str, tuple[bytes, bytes, str]]) -> None:
    conn = await asyncpg.connect(db_url)
    columns = {
        "model_providers": ("ciphertext", "nonce"),
        "vault_credentials": ("ciphertext", "nonce"),
        "connections": ("secrets_ciphertext", "secrets_nonce"),
        "session_github_repositories": ("ciphertext", "nonce"),
        "oauth_flows": ("ciphertext", "nonce"),
    }
    try:
        assert (
            await conn.fetchval("SELECT count(*) FROM accounts WHERE display_name='Eumemic'") == 0
        )
        assert (
            await conn.fetchval("SELECT spent_microusd FROM accounts WHERE id='acc_root'") == 4242
        )
        for table, (ciphertext_column, nonce_column) in columns.items():
            row = await conn.fetchrow(
                f"SELECT {ciphertext_column} ciphertext,{nonce_column} nonce,account_id FROM {table}"
            )
            assert row["account_id"] == "acc_root"
            assert bytes(row["ciphertext"]) == expected[table][0]
            assert bytes(row["nonce"]) == expected[table][1]
    finally:
        await conn.close()


@needs_docker
def test_real_postgres_upgrade_and_downgrade_round_trip(postgres: Any) -> None:
    db_url = _alembic_url(postgres)
    key = os.urandom(SecretBox.KEY_SIZE)
    migration = _migration()
    master = migration._Box(key)
    result = _run_alembic(["upgrade", "0152"], db_url, key)
    assert result.returncode == 0, result.stderr
    expected = asyncio.run(_seed_fleet(db_url, master))

    result = _run_alembic(["upgrade", "0154"], db_url, key)
    assert result.returncode == 0, result.stderr
    asyncio.run(_assert_upgraded(db_url, master, expected))

    result = _run_alembic(["downgrade", "0152"], db_url, key)
    assert result.returncode == 0, result.stderr
    asyncio.run(_assert_restored(db_url, expected))
