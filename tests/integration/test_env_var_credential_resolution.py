"""Session-scoped ``environment_variable`` credential resolution (#873).

Against a real Postgres: the rank-ordered ``DISTINCT ON`` list query +
the decrypt/mint service tail. The properties that matter:

* a session resolves exactly the active env-var creds across its bound
  vaults — header-style creds and archived rows never appear;
* duplicate ``secret_name`` across two vaults → the lower-rank vault
  wins (and flipping the binding order flips the winner);
* placeholders are deterministic per (session, credential) — stable
  across repeated resolution (so a recycle re-derives the same value)
  and distinct across sessions;
* tenant isolation: an identically-named credential in another
  account's vault is invisible;
* decrypt failures propagate (fail-hard) rather than yielding a
  partial credential set.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from pydantic import SecretStr

from aios.crypto.vault import CryptoBox
from aios.db import queries as db_queries
from aios.db.pool import create_pool
from aios.errors import CryptoDecryptError
from aios.harness import runtime
from aios.models.vaults import VaultCredentialCreate
from aios.services import agents as agents_service
from aios.services import vaults as vaults_service
from aios.services.vaults import (
    SECRET_PLACEHOLDER_PREFIX,
    resolve_session_env_var_credentials,
)

pytestmark = pytest.mark.integration

ACC = "acc_envvar"
ACC_OTHER = "acc_envvar_other"
ENV = "env_envvar"


@pytest.fixture
def crypto_box() -> CryptoBox:
    return CryptoBox(os.urandom(32))


@pytest.fixture
async def vault_pool(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool on ``runtime.pool`` + seeded accounts and an environment."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    prev = runtime.pool
    runtime.pool = pool
    try:
        async with pool.acquire() as conn:
            # Only one active root account is allowed; the second tenant
            # is a child of the first (tenant isolation is account_id
            # scoping, not tree position).
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, NULL, TRUE, 'envvar-root')",
                ACC,
            )
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, $2, FALSE, 'envvar-other')",
                ACC_OTHER,
                ACC,
            )
            await conn.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ($1, 'envvar-env', '{}'::jsonb, $2)",
                ENV,
                ACC,
            )
        yield pool
    finally:
        runtime.pool = prev
        await pool.close()


async def _make_session(
    pool: asyncpg.Pool[Any], *, vault_ids: list[str] | None = None, account_id: str = ACC
) -> str:
    agent = await agents_service.create_agent(
        pool,
        account_id=account_id,
        name=f"envvar-agent-{os.urandom(4).hex()}",
        model="test/dummy",
        system="x",
        tools=[],
        description=None,
        metadata={},
        window_min=1000,
        window_max=100000,
    )
    async with pool.acquire() as conn:
        session = await db_queries.insert_session(
            conn,
            account_id=account_id,
            agent_id=agent.id,
            environment_id=ENV,
            agent_version=agent.version,
            title=None,
            metadata={},
        )
        if vault_ids:
            await db_queries.set_session_vaults(conn, session.id, vault_ids, account_id=account_id)
    return session.id


async def _make_vault(pool: asyncpg.Pool[Any], name: str, *, account_id: str = ACC) -> str:
    vault = await vaults_service.create_vault(
        pool, account_id=account_id, display_name=name, metadata={}
    )
    return vault.id


async def _make_env_cred(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    vault_id: str,
    secret_name: str,
    secret_value: str,
    *,
    account_id: str = ACC,
) -> str:
    cred = await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=account_id,
        vault_id=vault_id,
        body=VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name=secret_name,
            secret_value=SecretStr(secret_value),
            allowed_hosts=["api.example.com/v1", "files.example.com"],
        ),
    )
    return cred.id


async def _resolve(
    pool: asyncpg.Pool[Any], crypto_box: CryptoBox, session_id: str, *, account_id: str = ACC
) -> list[vaults_service.ResolvedEnvVarCredential]:
    async with pool.acquire() as conn:
        return await resolve_session_env_var_credentials(
            conn, crypto_box, session_id, account_id=account_id
        )


async def test_resolves_the_set_across_ranked_vaults(
    vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    pool = vault_pool
    v1, v2 = await _make_vault(pool, "v1"), await _make_vault(pool, "v2")
    await _make_env_cred(pool, crypto_box, v1, "GITHUB_TOKEN", "ghp_one")
    await _make_env_cred(pool, crypto_box, v1, "STRIPE_KEY", "sk_two")
    await _make_env_cred(pool, crypto_box, v2, "TAVILY_KEY", "tvly_three")
    # Noise that must NOT resolve: a header credential in the same vault
    # (the auth_type filter is the only thing excluding it) and an
    # archived env-var credential.
    await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=ACC,
        vault_id=v1,
        body=VaultCredentialCreate(
            target_url="https://api.example/mcp",
            auth_type="bearer_header",
            token=SecretStr("tok-x"),
        ),
    )
    archived_id = await _make_env_cred(pool, crypto_box, v1, "DEAD_KEY", "gone")
    await vaults_service.archive_vault_credential(
        pool, vault_id=v1, credential_id=archived_id, account_id=ACC
    )

    session_id = await _make_session(pool, vault_ids=[v1, v2])
    resolved = await _resolve(pool, crypto_box, session_id)

    assert {(r.secret_name, r.secret_value) for r in resolved} == {
        ("GITHUB_TOKEN", "ghp_one"),
        ("STRIPE_KEY", "sk_two"),
        ("TAVILY_KEY", "tvly_three"),
    }
    for r in resolved:
        assert r.placeholder.startswith(SECRET_PLACEHOLDER_PREFIX)
        assert len(r.placeholder) == len(SECRET_PLACEHOLDER_PREFIX) + 32
        assert r.allowed_hosts == ("api.example.com/v1", "files.example.com")
        # The plaintext never leaks through the dataclass repr.
        assert r.secret_value not in repr(r)
    assert len({r.placeholder for r in resolved}) == 3


async def test_duplicate_secret_name_first_vault_wins(
    vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    pool = vault_pool
    v1, v2 = await _make_vault(pool, "v1"), await _make_vault(pool, "v2")
    await _make_env_cred(pool, crypto_box, v1, "API_KEY", "from-v1")
    await _make_env_cred(pool, crypto_box, v2, "API_KEY", "from-v2")

    forward = await _resolve(pool, crypto_box, await _make_session(pool, vault_ids=[v1, v2]))
    reverse = await _resolve(pool, crypto_box, await _make_session(pool, vault_ids=[v2, v1]))

    assert [r.secret_value for r in forward] == ["from-v1"]
    assert [r.secret_value for r in reverse] == ["from-v2"]


async def test_placeholders_deterministic_per_session_distinct_across(
    vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    pool = vault_pool
    vault_id = await _make_vault(pool, "v1")
    await _make_env_cred(pool, crypto_box, vault_id, "API_KEY", "val")
    s1 = await _make_session(pool, vault_ids=[vault_id])
    s2 = await _make_session(pool, vault_ids=[vault_id])

    (first,) = await _resolve(pool, crypto_box, s1)
    (again,) = await _resolve(pool, crypto_box, s1)
    (other,) = await _resolve(pool, crypto_box, s2)

    # Stable across re-resolution (a container recycle re-derives the
    # same placeholder, so anything persisted into /workspace survives)…
    assert first.placeholder == again.placeholder
    # …but unique per session, and never the secret.
    assert first.placeholder != other.placeholder
    assert "val" not in first.placeholder


async def test_other_tenants_credentials_invisible(
    vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """The query's ``vc.account_id`` filter is load-bearing, so construct
    the state where ONLY it excludes the foreign row: bind the other
    tenant's vault directly into this session (``set_session_vaults``
    FK-checks vault existence, not ownership, so the binding goes
    through) and assert the foreign credential still never resolves."""
    pool = vault_pool
    theirs = await _make_vault(pool, "their-vault", account_id=ACC_OTHER)
    await _make_env_cred(pool, crypto_box, theirs, "API_KEY", "their-secret", account_id=ACC_OTHER)
    mine = await _make_vault(pool, "my-vault")
    await _make_env_cred(pool, crypto_box, mine, "OTHER_KEY", "my-secret")

    session_id = await _make_session(pool, vault_ids=[theirs, mine])
    resolved = await _resolve(pool, crypto_box, session_id)

    assert [(r.secret_name, r.secret_value) for r in resolved] == [("OTHER_KEY", "my-secret")]


async def test_vaultless_session_resolves_empty(
    vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    session_id = await _make_session(vault_pool)
    assert await _resolve(vault_pool, crypto_box, session_id) == []


async def test_wrong_key_fails_hard(vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox) -> None:
    """One undecryptable blob aborts resolution loudly — no partial set."""
    pool = vault_pool
    vault_id = await _make_vault(pool, "v1")
    await _make_env_cred(pool, crypto_box, vault_id, "API_KEY", "val")
    session_id = await _make_session(pool, vault_ids=[vault_id])

    with pytest.raises(CryptoDecryptError):
        await _resolve(pool, CryptoBox(os.urandom(32)), session_id)


async def test_materializer_resolves_through_worker_runtime(
    vault_pool: asyncpg.Pool[Any], crypto_box: CryptoBox
) -> None:
    """Execute the real ``_materialize_env_var_credentials`` (everything
    else patches it out): it pulls pool + crypto box from the worker
    runtime and returns the resolved tuple the plan carries."""
    from aios.sandbox.spec import _materialize_env_var_credentials

    pool = vault_pool
    vault_id = await _make_vault(pool, "v1")
    await _make_env_cred(pool, crypto_box, vault_id, "API_KEY", "val")
    session_id = await _make_session(pool, vault_ids=[vault_id])

    prev = runtime.crypto_box
    runtime.crypto_box = crypto_box
    try:
        resolved = await _materialize_env_var_credentials(session_id, account_id=ACC)
    finally:
        runtime.crypto_box = prev

    assert [(r.secret_name, r.secret_value) for r in resolved] == [("API_KEY", "val")]
    assert resolved[0].placeholder.startswith(SECRET_PLACEHOLDER_PREFIX)
