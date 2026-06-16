"""Integration repro for #735: an oversized tool result must be capped at
the append (dispatch) boundary so ``cumulative_tokens`` reflects the capped
size and windowing stays correct by construction.

A single tool result larger than the model context budget wedges the
session — windowing can only drop whole events, never shrink one, so a
lone oversized result can't be shed.  The fix caps the result content at
the SERVICE append sink (``sessions_service.append_tool_result``): the
inline body is replaced with a stub pointing the model at the full output
spilled under the session's attachments mount, recoverable via ``read``.

This test drives the real service sink against a testcontainer Postgres,
then asserts:

* oversized → the stored tool event content is a ``[Tool result
  truncated:`` stub naming ``/mnt/attachments/tool_results/`` AND the full
  body is on the host spill file, byte-for-byte.
* within-cap → the content is stored verbatim, no spill file written.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest import mock

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from aios.sandbox.volumes import ensure_session_attachments_dir
from aios.services import sessions as sessions_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_DEFAULT_MAX_CHARS = 200_000


@pytest.fixture
async def spill_session(
    migrated_db_url: str, _reset_db_state: None, tmp_path: Path
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, session_id, tool_call_id)`` for a session whose event
    log contains an assistant message carrying a single ``tool_calls`` entry
    (so ``append_tool_result`` finds a parent), with ``workspace_root``
    pointed at an isolated ``tmp_path`` so the spill file lands somewhere we
    can read on the host.
    """
    with mock.patch.dict(os.environ, {"AIOS_WORKSPACE_ROOT": str(tmp_path / "workspaces")}):
        get_settings.cache_clear()
        pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
        try:
            account_id = "acc_spill"
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO accounts
                        (id, parent_account_id, can_mint_children, display_name)
                    VALUES ($1, NULL, TRUE, 'spill-test-root')
                    """,
                    account_id,
                )
            _agent, _env, session = await seed_agent_env_session(
                pool,
                account_id=account_id,
                prefix="spill",
                tools=[ToolSpec(type="bash")],
            )
            tool_call_id = "tc_spill_1"
            async with pool.acquire() as conn:
                await queries.append_event(
                    conn,
                    account_id=account_id,
                    session_id=session.id,
                    kind="message",
                    data={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{}"},
                            }
                        ],
                    },
                )
            yield pool, session.id, tool_call_id
        finally:
            await pool.close()
            get_settings.cache_clear()


async def _stored_tool_event(
    pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        event = await queries.find_tool_result_event(
            conn, session_id, tool_call_id, account_id="acc_spill"
        )
    assert event is not None
    return event.data


async def _stored_tool_content(pool: asyncpg.Pool[Any], session_id: str, tool_call_id: str) -> str:
    content = (await _stored_tool_event(pool, session_id, tool_call_id))["content"]
    assert isinstance(content, str)
    return content


class TestToolResultSpill:
    async def test_oversized_result_spilled_and_stubbed(
        self,
        spill_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, session_id, tool_call_id = spill_session
        original = "X" * 300_000  # > the 200_000 default cap
        assert len(original) > _DEFAULT_MAX_CHARS

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id="acc_spill",
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=original,
            )

        stored = await _stored_tool_content(pool, session_id, tool_call_id)
        assert stored.startswith("[Tool result truncated:")
        assert "/mnt/attachments/tool_results/" in stored
        assert len(stored) < len(original)

        spill_file = (
            ensure_session_attachments_dir(session_id) / "tool_results" / f"{tool_call_id}.txt"
        )
        assert spill_file.exists()
        assert spill_file.read_text(encoding="utf-8") == original

        # #1093: the spill file must be recorded under the tool event's
        # ``metadata.attachments`` (the single convention staged inbounds use)
        # so the attachment GC's referenced-set protects it. Its
        # ``in_sandbox_path`` must match the path the GC walk reconstructs for
        # the on-disk file.
        data = await _stored_tool_event(pool, session_id, tool_call_id)
        attachments = data["metadata"]["attachments"]
        assert isinstance(attachments, list) and len(attachments) == 1
        assert (
            attachments[0]["in_sandbox_path"] == f"/mnt/attachments/tool_results/{tool_call_id}.txt"
        )

    async def test_spill_path_is_in_gc_referenced_set(
        self,
        spill_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        """End-to-end #1093 guard: after an oversized result spills, the
        attachment GC's referenced-set query (``list_attachment_paths_for_sessions``)
        — built exclusively from ``data->'metadata'->'attachments'`` — must
        return the spill file's sandbox path. Pre-fix the spill reference lived
        only in the result content stub, so this query returned an empty set
        and the orphan sweep deleted the file on the next worker boot.
        """
        pool, session_id, tool_call_id = spill_session
        original = "Z" * 300_000

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id="acc_spill",
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=original,
            )

        async with pool.acquire() as conn:
            referenced = await queries.list_attachment_paths_for_sessions(conn, [session_id])

        assert f"/mnt/attachments/tool_results/{tool_call_id}.txt" in referenced[session_id]

    async def test_within_cap_result_stored_verbatim(
        self,
        spill_session: tuple[asyncpg.Pool[Any], str, str],
    ) -> None:
        pool, session_id, tool_call_id = spill_session
        original = "Y" * 100  # well under the cap

        async with pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id="acc_spill",
                session_id=session_id,
                tool_call_id=tool_call_id,
                content=original,
            )

        stored = await _stored_tool_content(pool, session_id, tool_call_id)
        assert stored == original

        spill_file = (
            ensure_session_attachments_dir(session_id) / "tool_results" / f"{tool_call_id}.txt"
        )
        # ``ensure_session_attachments_dir`` creates the session dir, but the
        # tool_results spill subdir + file must NOT exist for a within-cap result.
        assert not spill_file.exists()
