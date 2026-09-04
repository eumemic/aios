"""Archived vaults refuse new credential inserts and never resolve credentials.

Regression coverage for the archived-vault resurrection bug:
``create_vault_credential``'s parent-vault lock omitted ``archived_at IS NULL``,
so a fresh active credential could be inserted into an already-archived vault.
The resolver (``resolve_session_credential``/``resolve_run_credential``) filtered
only on ``vault_credentials.archived_at IS NULL``, never ``vaults.archived_at``,
so the newly inserted credential was surfaced to any session/run bound to the
vault *before* it was archived — silently resurrecting a retired credential
source.

The fix closes both the write gate and the read path:
- ``create_vault_credential`` now locks with ``AND archived_at IS NULL`` and
  raises ``ConflictError`` (409) for an exists+owned-but-archived vault, matching
  ``update_vault``'s archived-row guard (PR #554).
- The resolvers (URL-keyed, ssh_key, and env-var) join ``vaults`` with
  ``archived_at IS NULL`` as defense-in-depth so a credential that becomes
  active in an archived vault through any path still does not resolve.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from pydantic import SecretStr

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.db.pool import create_pool
from aios.errors import ConflictError, NotFoundError
from aios.ids import VAULT_CREDENTIAL, make_id
from aios.models.vaults import VaultCredentialCreate
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import vaults as vaults_service
from aios.services.vaults import (
    resolve_run_env_var_credentials,
    resolve_session_env_var_credentials,
)
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH

pytestmark = pytest.mark.integration

ACC = "acc_arch_resurrect"


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(32))


@pytest.fixture
async def pool_fixture(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, NULL, TRUE, 'arch-resurrect-root')",
                ACC,
            )
        yield pool
    finally:
        await pool.close()


async def _make_session(
    pool: asyncpg.Pool[Any], vault_ids: list[str], *, account_id: str = ACC
) -> str:
    agent = await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=f"resurrect-agent-{os.urandom(4).hex()}",
        model="test/dummy",
        system="x",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    env = await environments_service.create_environment(
        pool, account_id=account_id, name=f"resurrect-env-{os.urandom(4).hex()}"
    )
    async with pool.acquire() as conn:
        session = await queries.insert_session(
            conn,
            account_id=account_id,
            agent_id=agent.id,
            environment_id=env.id,
            agent_version=agent.version,
            title=None,
            metadata={},
        )
        await queries.set_session_vaults(conn, session.id, vault_ids, account_id=account_id)
    return session.id


async def _make_run(pool: asyncpg.Pool[Any], vault_ids: list[str], *, account_id: str = ACC) -> str:
    env = await environments_service.create_environment(
        pool, account_id=account_id, name=f"resurrect-run-env-{os.urandom(4).hex()}"
    )
    async with pool.acquire() as conn:
        wf = await queries.workflows.insert_workflow(
            conn,
            account_id=account_id,
            name=f"resurrect-wf-{os.urandom(4).hex()}",
            script="async def main(input):\n    return input\n",
        )
        run = await queries.workflows.insert_wf_run(
            conn,
            account_id=account_id,
            workflow_id=wf.id,
            environment_id=env.id,
            script=wf.script,
            host_semantics_epoch=HOST_SEMANTICS_EPOCH,
            script_sha=hashlib.sha256(wf.script.encode("utf-8")).hexdigest(),
            depth=10,
        )
        await queries.workflows.set_run_vaults(conn, run.id, vault_ids, account_id=account_id)
    return run.id


# ─── primary fix: write gate refuses archived vaults ─────────────────────────


async def test_create_vault_credential_refuses_archived_vault(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Inserting a new credential into an archived vault raises ConflictError.

    Pre-fix the lock ``SELECT 1 FROM vaults … FOR UPDATE`` omitted
    ``archived_at IS NULL``, so the insert succeeded and the resolver later
    surfaced the resurrected credential.
    """
    pool = pool_fixture
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="resurrect-vault", metadata={}
    )
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    with pytest.raises(ConflictError) as excinfo:
        await vaults_service.create_vault_credential(
            pool,
            crypto_box,
            account_id=ACC,
            vault_id=vault.id,
            body=VaultCredentialCreate(
                target_url="https://resurrect.example.com/api",
                auth_type="bearer_header",
                token=SecretStr("post-archive-resurrection-token"),
            ),
        )
    detail = excinfo.value.detail
    assert detail is not None
    assert detail.get("vault_id") == vault.id
    assert excinfo.value.status_code == 409


async def test_create_vault_credential_archived_vault_writes_no_row(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Defense-in-depth pin: the refused insert must not leave a credential row."""
    pool = pool_fixture
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="no-row-vault", metadata={}
    )
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    with pytest.raises(ConflictError):
        await vaults_service.create_vault_credential(
            pool,
            crypto_box,
            account_id=ACC,
            vault_id=vault.id,
            body=VaultCredentialCreate(
                target_url="https://no-row.example.com/api",
                auth_type="bearer_header",
                token=SecretStr("should-not-be-stored"),
            ),
        )

    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM vault_credentials WHERE vault_id = $1", vault.id
        )
    assert count == 0


async def test_create_vault_credential_not_found_for_nonexistent_vault(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """A plainly nonexistent vault id raises NotFoundError (404), not ConflictError.

    Pins the disambiguation between "not found / not owned" (404) and
    "exists, owned, but archived" (409) the lock's None branch now performs.
    """
    pool = pool_fixture
    with pytest.raises(NotFoundError):
        await vaults_service.create_vault_credential(
            pool,
            crypto_box,
            account_id=ACC,
            vault_id="vlt_does_not_exist",
            body=VaultCredentialCreate(
                target_url="https://missing.example.com/api",
                auth_type="bearer_header",
                token=SecretStr("x"),
            ),
        )


async def test_create_vault_credential_active_vault_succeeds(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Positive control: the archived-vault guard does not block active vaults.

    Without this the ConflictError assertions above would still pass if the
    service raised unconditionally.
    """
    pool = pool_fixture
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="active-vault", metadata={}
    )
    cred = await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=ACC,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            target_url="https://active.example.com/api",
            auth_type="bearer_header",
            token=SecretStr("live-token"),
        ),
    )
    assert cred.archived_at is None
    assert cred.vault_id == vault.id


async def test_create_vault_credential_foreign_archived_vault_not_found(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """A foreign account's archived vault is NotFoundError (not ConflictError).

    The disambiguation probe checks ``id AND account_id``; a vault owned by
    another account is invisible to this account regardless of archival state,
    so it surfaces as NotFoundError — the tenant-isolation answer, matching the
    pre-fix behavior for foreign vaults.
    """
    pool = pool_fixture
    # A second account in the same root.
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
            "display_name) VALUES ('acc_other', $1, FALSE, 'other')",
            ACC,
        )
    foreign_vault = await vaults_service.create_vault(
        pool, account_id="acc_other", display_name="foreign-archived", metadata={}
    )
    await vaults_service.archive_vault(pool, foreign_vault.id, account_id="acc_other")

    with pytest.raises(NotFoundError):
        await vaults_service.create_vault_credential(
            pool,
            crypto_box,
            account_id=ACC,
            vault_id=foreign_vault.id,
            body=VaultCredentialCreate(
                target_url="https://foreign.example.com/api",
                auth_type="bearer_header",
                token=SecretStr("x"),
            ),
        )


# ─── defense-in-depth: resolvers never surface archived-vault credentials ───


async def test_resolve_session_credential_none_for_archived_vault_via_direct_sql(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Defense-in-depth: even a credential inserted *directly* into an archived
    vault (bypassing the service-layer write gate) must not resolve.

    The resolver joins ``vaults`` with ``archived_at IS NULL`` so a credential
    that becomes active in an archived vault through any path still does not
    surface to a bound session.
    """
    pool = pool_fixture
    target_url = "https://direct-sql.example.com/api"
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="direct-sql-vault", metadata={}
    )
    session_id = await _make_session(pool, [vault.id])
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    # Bypass the service-layer guard: insert an active credential directly.
    subkey = crypto_box.derive_account_subkey(ACC)
    blob = subkey.encrypt_dict({"token": "direct-sql-resurrection-token"})
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO vault_credentials "
            "(id, vault_id, display_name, target_url, auth_type, ciphertext, nonce, "
            "metadata, account_id) "
            "VALUES ($1, $2, NULL, $3, 'bearer_header', $4, $5, '{}'::jsonb, $6)",
            make_id(VAULT_CREDENTIAL),
            vault.id,
            target_url,
            blob.ciphertext,
            blob.nonce,
            ACC,
        )

    async with pool.acquire() as conn:
        result = await queries.resolve_session_credential(
            conn, session_id, target_url, account_id=ACC
        )
    assert result is None


async def test_resolve_run_credential_none_for_archived_vault_via_direct_sql(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Run twin of the session defense-in-depth test above."""
    pool = pool_fixture
    target_url = "https://run-direct-sql.example.com/api"
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="run-direct-sql-vault", metadata={}
    )
    run_id = await _make_run(pool, [vault.id])
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    subkey = crypto_box.derive_account_subkey(ACC)
    blob = subkey.encrypt_dict({"token": "run-resurrection-token"})
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO vault_credentials "
            "(id, vault_id, display_name, target_url, auth_type, ciphertext, nonce, "
            "metadata, account_id) "
            "VALUES ($1, $2, NULL, $3, 'bearer_header', $4, $5, '{}'::jsonb, $6)",
            make_id(VAULT_CREDENTIAL),
            vault.id,
            target_url,
            blob.ciphertext,
            blob.nonce,
            ACC,
        )

    async with pool.acquire() as conn:
        result = await queries.resolve_run_credential(conn, run_id, target_url, account_id=ACC)
    assert result is None


async def test_resolve_session_credential_none_after_archive_scrub(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """No regression: the pre-existing (scrubbed) credential resolves to None.

    This is the cosmetic case the bug report calls out — archive_vault scrubs
    pre-existing credentials, and the resolver returns None for them. The fix
    must not regress this.
    """
    pool = pool_fixture
    target_url = "https://scrub.example.com/api"
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="scrub-vault", metadata={}
    )
    session_id = await _make_session(pool, [vault.id])
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=ACC,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            target_url=target_url,
            auth_type="bearer_header",
            token=SecretStr("pre-archive-token"),
        ),
    )
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    async with pool.acquire() as conn:
        result = await queries.resolve_session_credential(
            conn, session_id, target_url, account_id=ACC
        )
    assert result is None


async def test_resolve_session_credential_active_vault_resolves(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """No regression: an active vault's credential still resolves and decrypts.

    Positive control for the resolver join — the new ``vaults`` join must not
    filter out a legitimately active vault.
    """
    pool = pool_fixture
    target_url = "https://live.example.com/api"
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="live-vault", metadata={}
    )
    session_id = await _make_session(pool, [vault.id])
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=ACC,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            target_url=target_url,
            auth_type="bearer_header",
            token=SecretStr("live-token"),
        ),
    )

    async with pool.acquire() as conn:
        result = await queries.resolve_session_credential(
            conn, session_id, target_url, account_id=ACC
        )
    assert result is not None
    blob, auth_type, resolved_vault_id = result
    assert resolved_vault_id == vault.id
    assert auth_type == "bearer_header"
    payload = json.loads(crypto_box.derive_account_subkey(ACC).decrypt(blob))
    assert payload["token"] == "live-token"


async def test_resolve_run_credential_active_vault_resolves(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Run twin of the active-vault positive control above."""
    pool = pool_fixture
    target_url = "https://run-live.example.com/api"
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="run-live-vault", metadata={}
    )
    run_id = await _make_run(pool, [vault.id])
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=ACC,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            target_url=target_url,
            auth_type="bearer_header",
            token=SecretStr("run-live-token"),
        ),
    )

    async with pool.acquire() as conn:
        result = await queries.resolve_run_credential(conn, run_id, target_url, account_id=ACC)
    assert result is not None
    blob, auth_type, resolved_vault_id = result
    assert resolved_vault_id == vault.id
    assert auth_type == "bearer_header"
    payload = json.loads(crypto_box.derive_account_subkey(ACC).decrypt(blob))
    assert payload["token"] == "run-live-token"


# ─── defense-in-depth: env-var and ssh_key resolvers ─────────────────────────


async def test_env_var_credentials_none_for_archived_vault(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Env-var credentials in an archived vault do not materialize into the
    bound session's sandbox, even when inserted via direct SQL.

    The shared ``_ENV_VAR_CREDENTIALS_FROM_WHERE`` template now joins
    ``vaults`` with ``archived_at IS NULL``, so all three env-var queries
    (provision set, drift echo set, run set) filter archived vaults in lockstep.
    """
    pool = pool_fixture
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="env-archived-vault", metadata={}
    )
    session_id = await _make_session(pool, [vault.id])
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    # Insert an active env-var credential directly, bypassing the write gate.
    # The shape check requires allowed_hosts (non-empty) for environment_variable.
    subkey = crypto_box.derive_account_subkey(ACC)
    blob = subkey.encrypt_dict({"secret_value": "env-resurrection"})
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO vault_credentials "
            "(id, vault_id, display_name, target_url, secret_name, allowed_hosts, "
            "auth_type, ciphertext, nonce, metadata, account_id) "
            "VALUES ($1, $2, NULL, NULL, 'RESURRECTED_KEY', $3::text[], "
            "'environment_variable', $4, $5, '{}'::jsonb, $6)",
            make_id(VAULT_CREDENTIAL),
            vault.id,
            ["api.example.com"],
            blob.ciphertext,
            blob.nonce,
            ACC,
        )

    async with pool.acquire() as conn:
        resolved = await resolve_session_env_var_credentials(
            conn, crypto_box, session_id, account_id=ACC
        )
    assert resolved == []

    # The echo query (drift probe) must agree — the template is shared.
    async with pool.acquire() as conn:
        echoes = await queries.list_session_env_var_credential_echoes(
            conn, session_id, account_id=ACC
        )
    assert echoes == []


async def test_run_env_var_credentials_none_for_archived_vault(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Run twin of the env-var archived-vault test above."""
    pool = pool_fixture
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="run-env-archived-vault", metadata={}
    )
    run_id = await _make_run(pool, [vault.id])
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    subkey = crypto_box.derive_account_subkey(ACC)
    blob = subkey.encrypt_dict({"secret_value": "run-env-resurrection"})
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO vault_credentials "
            "(id, vault_id, display_name, target_url, secret_name, allowed_hosts, "
            "auth_type, ciphertext, nonce, metadata, account_id) "
            "VALUES ($1, $2, NULL, NULL, 'RUN_KEY', $3::text[], "
            "'environment_variable', $4, $5, '{}'::jsonb, $6)",
            make_id(VAULT_CREDENTIAL),
            vault.id,
            ["api.example.com"],
            blob.ciphertext,
            blob.nonce,
            ACC,
        )

    async with pool.acquire() as conn:
        resolved = await resolve_run_env_var_credentials(conn, crypto_box, run_id, account_id=ACC)
    assert resolved == []


async def test_ssh_key_credential_none_for_archived_vault(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """An ssh_key credential in an archived vault does not resolve, even via
    direct SQL.

    ``resolve_session_ssh_key_credential`` joins ``vaults`` with
    ``archived_at IS NULL`` so an archived vault's ssh key cannot be discovered.
    """
    pool = pool_fixture
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="ssh-archived-vault", metadata={}
    )
    session_id = await _make_session(pool, [vault.id])
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    subkey = crypto_box.derive_account_subkey(ACC)
    blob = subkey.encrypt_dict({"private_key": "RESURRECTED_PRIVATE_KEY"})
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO vault_credentials "
            "(id, vault_id, display_name, target_url, secret_name, auth_type, "
            "ciphertext, nonce, metadata, account_id) "
            "VALUES ($1, $2, NULL, NULL, 'DEPLOY_KEY', 'ssh_key', "
            "$3, $4, '{}'::jsonb, $5)",
            make_id(VAULT_CREDENTIAL),
            vault.id,
            blob.ciphertext,
            blob.nonce,
            ACC,
        )

    async with pool.acquire() as conn:
        result = await queries.resolve_session_ssh_key_credential(
            conn, session_id, "DEPLOY_KEY", account_id=ACC
        )
    assert result is None


async def test_env_var_credentials_active_vault_resolves(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """No regression: an active vault's env-var credentials still materialize.

    Positive control for the env-var template's new vault join.
    """
    pool = pool_fixture
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="env-active-vault", metadata={}
    )
    session_id = await _make_session(pool, [vault.id])
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=ACC,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name="LIVE_KEY",
            secret_value=SecretStr("live-env-value"),
            allowed_hosts=["api.example.com"],
        ),
    )

    async with pool.acquire() as conn:
        resolved = await resolve_session_env_var_credentials(
            conn, crypto_box, session_id, account_id=ACC
        )
    assert [(r.secret_name, r.secret_value) for r in resolved] == [("LIVE_KEY", "live-env-value")]


# ─── end-to-end: the full resurrection timeline from the bug report ──────────


async def test_archived_vault_resurrection_closed_e2e(
    pool_fixture: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """The end-to-end timeline from the bug report no longer resurrects.

    This is the bound-before-archive timeline (the natural production shape):
    bind a vault to a session, insert a credential, archive the vault (which
    scrubs the pre-existing credential), then attempt to insert a NEW credential
    into the archived vault. Pre-fix that insert succeeded and the resolver
    returned the resurrected secret. Post-fix the insert raises ConflictError
    and the resolver returns None.
    """
    pool = pool_fixture
    target_url = "https://resurrect.example.com/api"
    vault = await vaults_service.create_vault(
        pool, account_id=ACC, display_name="e2e-resurrect-vault", metadata={}
    )
    session_id = await _make_session(pool, [vault.id])

    # Pre-existing credential, then archive (scrubs it).
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=ACC,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            target_url=target_url,
            auth_type="bearer_header",
            token=SecretStr("pre-archive-token"),
        ),
    )
    await vaults_service.archive_vault(pool, vault.id, account_id=ACC)

    # Scrubbed credential resolves to None (cosmetic case still holds).
    async with pool.acquire() as conn:
        assert (
            await queries.resolve_session_credential(conn, session_id, target_url, account_id=ACC)
            is None
        )

    # THE BUG: inserting a NEW credential into the ARCHIVED vault must now fail.
    with pytest.raises(ConflictError):
        await vaults_service.create_vault_credential(
            pool,
            crypto_box,
            account_id=ACC,
            vault_id=vault.id,
            body=VaultCredentialCreate(
                target_url=target_url,
                auth_type="bearer_header",
                token=SecretStr("post-archive-resurrection-token"),
            ),
        )

    # And the resolver still returns None — no resurrection.
    async with pool.acquire() as conn:
        assert (
            await queries.resolve_session_credential(conn, session_id, target_url, account_id=ACC)
            is None
        )
