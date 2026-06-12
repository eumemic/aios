"""Raw-SQL proofs for composite tenant FK backstops on secret-bearing chains."""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

pytestmark = pytest.mark.integration


async def _seed_two_tenant_secret_chains(conn: asyncpg.Connection[Any]) -> None:
    await conn.execute(
        """
        INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
        VALUES ('acc_root', NULL, TRUE, 'root'),
               ('acc_a', 'acc_root', FALSE, 'tenant-a'),
               ('acc_b', 'acc_root', FALSE, 'tenant-b')
        ON CONFLICT DO NOTHING;
        INSERT INTO environments (id, name, account_id)
        VALUES ('env_a', 'env-a', 'acc_a'),
               ('env_b', 'env-b', 'acc_b');
        INSERT INTO agents (id, name, model, account_id)
        VALUES ('agent_a', 'agent-a', 'test/model', 'acc_a'),
               ('agent_b', 'agent-b', 'test/model', 'acc_b');
        INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id)
        VALUES ('sess_a', 'agent_a', 'env_a', '/tmp/ws-a', 'acc_a'),
               ('sess_b', 'agent_b', 'env_b', '/tmp/ws-b', 'acc_b');
        INSERT INTO workflows (id, account_id, name, script)
        VALUES ('wf_a', 'acc_a', 'wf-a', 'async def main(i):\n    return i\n'),
               ('wf_b', 'acc_b', 'wf-b', 'async def main(i):\n    return i\n');
        INSERT INTO wf_runs (
            id, workflow_id, account_id, environment_id,
            script, script_sha, status, host_semantics_epoch
        )
        VALUES (
            'run_a', 'wf_a', 'acc_a', 'env_a',
            'async def main(i):\n    return i\n', 'sha-a', 'pending', 1
        ),
        (
            'run_b', 'wf_b', 'acc_b', 'env_b',
            'async def main(i):\n    return i\n', 'sha-b', 'pending', 1
        );
        INSERT INTO vaults (id, display_name, account_id)
        VALUES ('vault_a', 'vault-a', 'acc_a'),
               ('vault_b', 'vault-b', 'acc_b');
        """
    )


async def _assert_fk_violation(conn: asyncpg.Connection[Any], sql: str) -> None:
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await conn.execute(sql)


async def test_session_vaults_rejects_foreign_vault_and_session(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    conn = conn_two_accounts
    await _seed_two_tenant_secret_chains(conn)

    await _assert_fk_violation(
        conn,
        """
        INSERT INTO session_vaults (session_id, vault_id, rank, account_id)
        VALUES ('sess_a', 'vault_b', 0, 'acc_a')
        """,
    )
    await _assert_fk_violation(
        conn,
        """
        INSERT INTO session_vaults (session_id, vault_id, rank, account_id)
        VALUES ('sess_b', 'vault_a', 0, 'acc_a')
        """,
    )


async def test_wf_run_vaults_rejects_foreign_vault_and_run(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    conn = conn_two_accounts
    await _seed_two_tenant_secret_chains(conn)

    await _assert_fk_violation(
        conn,
        """
        INSERT INTO wf_run_vaults (run_id, vault_id, rank, account_id)
        VALUES ('run_a', 'vault_b', 0, 'acc_a')
        """,
    )
    await _assert_fk_violation(
        conn,
        """
        INSERT INTO wf_run_vaults (run_id, vault_id, rank, account_id)
        VALUES ('run_b', 'vault_a', 0, 'acc_a')
        """,
    )


async def test_vault_credentials_rejects_foreign_vault(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    conn = conn_two_accounts
    await _seed_two_tenant_secret_chains(conn)

    await _assert_fk_violation(
        conn,
        """
        INSERT INTO vault_credentials (
            id, vault_id, display_name, target_url, auth_type,
            ciphertext, nonce, account_id
        )
        VALUES (
            'cred_cross', 'vault_b', 'cross', 'https://example.invalid/mcp',
            'bearer_header', '\\x01'::bytea, '\\x02'::bytea, 'acc_a'
        )
        """,
    )


async def test_oauth_flows_rejects_foreign_vault(
    conn_two_accounts: asyncpg.Connection[Any],
) -> None:
    conn = conn_two_accounts
    await _seed_two_tenant_secret_chains(conn)

    await _assert_fk_violation(
        conn,
        """
        INSERT INTO oauth_flows (
            id, account_id, vault_id, target_url, state,
            redirect_uri, ciphertext, nonce, expires_at
        )
        VALUES (
            'flow_cross', 'acc_a', 'vault_b', 'https://example.invalid/mcp', 'state-cross',
            'https://callback.invalid/oauth', '\\x01'::bytea, '\\x02'::bytea, now() + interval '5 minutes'
        )
        """,
    )
