"""E2E: a session's env-var vault credential surfaces in the real
sandbox container as ``SECRET_NAME=<placeholder>`` (#874).

The C1 safe-foundation acceptance: ``printenv`` inside a provisioned
sandbox shows the opaque placeholder; the decrypted secret never does.
This drives the whole chain the unit tests mock out — resolver → mint →
merged_env → ``docker run --env`` → ``docker exec`` — so it proves the
placeholder reaches a real container's environment, not just the spec
dict. Nothing swaps the placeholder yet (that is #876).
"""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from aios.db import queries
from aios.harness import runtime
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.models.vaults import VaultCredentialCreate
from aios.services import vaults as vaults_service
from aios.services.vaults import mint_secret_placeholder
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, bash, first_tool_result

pytestmark = pytest.mark.docker

_ACCOUNT_ID = "acc_test_stub"
_SECRET_NAME = "GITHUB_TOKEN"
_SENTINEL_SECRET = "ghp_SENTINEL_PLAINTEXT_DO_NOT_LEAK"


@needs_docker
async def test_placeholder_visible_in_container_secret_absent(docker_harness: Harness) -> None:
    pool = docker_harness._pool
    crypto_box = runtime.require_crypto_box()

    vault = await vaults_service.create_vault(
        pool, account_id=_ACCOUNT_ID, display_name="envvar-e2e", metadata={}
    )
    cred = await vaults_service.create_vault_credential(
        pool,
        crypto_box,
        account_id=_ACCOUNT_ID,
        vault_id=vault.id,
        body=VaultCredentialCreate(
            auth_type="environment_variable",
            secret_name=_SECRET_NAME,
            secret_value=SecretStr(_SENTINEL_SECRET),
            allowed_hosts=["api.github.com"],
        ),
    )

    # The placeholder is deterministic per (session, credential), so we
    # can compute the exact string the container must show.
    docker_harness.script_model(
        [
            assistant(tool_calls=[bash(f'printf "%s" "${_SECRET_NAME}"')]),
            assistant("Done."),
        ]
    )
    # The provision gate (#879) requires env-var credentials to ride a
    # Limited-networking environment whose allowed_hosts cover the
    # credential's — this fixture predates the gate and used the harness
    # default (non-Limited) environment, which the gate now correctly
    # rejects at provision.
    session = await docker_harness.start(
        "test",
        tools=["bash"],
        environment_config=EnvironmentConfig(
            networking=LimitedNetworking(type="limited", allowed_hosts=["api.github.com"]),
        ),
    )
    # Bind the vault before the first bash call provisions the sandbox.
    async with pool.acquire() as conn:
        await queries.set_session_vaults(conn, session.id, [vault.id], account_id=_ACCOUNT_ID)

    async with pool.acquire() as conn:
        salt = await queries.get_or_create_account_placeholder_salt(conn, crypto_box, _ACCOUNT_ID)
    expected = mint_secret_placeholder(salt, session.id, cred.id)

    await docker_harness.run_until_idle(session.id)

    content = str(first_tool_result(await docker_harness.events(session.id)).get("content", ""))
    # The bash tool result is a JSON envelope; the placeholder is the
    # command's stdout. Assert the EXACT minted placeholder reached the
    # container env, and the decrypted secret appears nowhere.
    assert json.loads(content)["stdout"].strip() == expected
    assert _SENTINEL_SECRET not in content
