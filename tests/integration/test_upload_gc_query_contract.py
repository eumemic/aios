"""DB-backed contract for the destructive upload-directory GC predicate."""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from aios.db import queries

pytestmark = [pytest.mark.integration, pytest.mark.docker]


async def test_live_session_without_files_is_present_in_upload_keep_map(
    migrated_db_url: str, _reset_db_state: None
) -> None:
    """A missing key authorizes wholesale rmtree, so zero-file live rows must emit one."""
    conn: asyncpg.Connection[Any] = await asyncpg.connect(migrated_db_url)
    try:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('acc_upload_contract', NULL, TRUE, 'upload-contract')"
        )
        await conn.execute(
            "INSERT INTO environments (id, account_id, name) "
            "VALUES ('env_upload_contract', 'acc_upload_contract', 'env')"
        )
        await conn.execute(
            "INSERT INTO agents (id, account_id, name, model, system, version) "
            "VALUES ('agent_upload_contract', 'acc_upload_contract', 'agent', 'test/model', '', 1)"
        )
        await conn.execute(
            "INSERT INTO sessions "
            "(id, agent_id, environment_id, agent_version, title, metadata, "
            " workspace_volume_path, env, account_id, last_event_seq) "
            "VALUES ('sess_upload_contract', 'agent_upload_contract', 'env_upload_contract', "
            "1, 'session', '{}'::jsonb, '/tmp/workspace', '{}'::jsonb, "
            "'acc_upload_contract', 0)"
        )

        assert await queries.list_upload_paths_for_sessions(
            conn, ["sess_upload_contract", "sess_deleted"]
        ) == {"sess_upload_contract": set()}
    finally:
        await conn.close()
