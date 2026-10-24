"""Non-string ``tool_calls[].id`` values must not fail the awaiting page (#2393).

A model can emit a tool call whose JSON ``id`` is not a string (litellm keeps
it unvalidated).  The #2393 marker lookup binds the unresolved ids into
``$3::text[]``; binding a raw ``int``/``bool``/``float``/``dict`` there raises
asyncpg ``DataError``.  That fails ``list_unresolved_tool_calls_batch`` for the
WHOLE page, so ``GET /v1/sessions`` 500s for every session on it, which is the
symptom #2393 fixes.

Expected output is master's (the pre-#2393 ``session_id = ANY(page)`` lookup):

* every call is still listed, with its decoded id unchanged;
* both flags are ``False`` for a non-string id, even when a marker whose text
  equals ``item->>'id'`` exists (``'5'``, ``'true'``, ``'1.5'``).  Master tested
  the decoded id for membership in a set of TEXT ids, and ``5 != '5'``;
* string-id calls on the same page keep their flags.

The object id is the one divergence: master raised ``TypeError`` (``dict`` is
unhashable in ``x in set``).  Now its flags read ``False``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db import queries
from aios.db.pool import create_pool
from aios.models.agents import ToolSpec
from aios.models.sessions import Session
from aios.services.sessions import compute_awaiting
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_awaiting_nonstr_ids"

# (label, decoded JSON id, PostgreSQL ``item->>'id'`` text of it)
_IDS: list[tuple[str, Any, str]] = [
    ("int", 5, "5"),
    ("bool", True, "true"),
    ("float", 1.5, "1.5"),
    ("object", {"k": 1}, '{"k": 1}'),
]


def _assistant(*ids: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": i, "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            for i in ids
        ],
    }


async def _lifecycle(conn: asyncpg.Connection[Any], sid: str, data: dict[str, Any]) -> None:
    await queries.append_event(
        conn, account_id=_ACCOUNT, session_id=sid, kind="lifecycle", data=data
    )


@pytest.fixture
async def page(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], dict[str, Session]]]:
    """One session per non-string id type plus a string-id control session.

    Each non-string session has one open call with that id, plus
    ``tool_requested`` and ``tool_confirmed``/``allow`` markers whose
    ``tool_call_id`` is the id's ``->>`` text, and a string-id call ``tc_s``
    that carries an allow marker.
    """
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, "
                "display_name) VALUES ($1, NULL, TRUE, $1)",
                _ACCOUNT,
            )
        sessions: dict[str, Session] = {}
        for label in [lbl for lbl, _v, _t in _IDS] + ["control"]:
            _a, _e, s = await seed_agent_env_session(
                pool, account_id=_ACCOUNT, prefix=f"nonstr-{label}", tools=[ToolSpec(type="bash")]
            )
            sessions[label] = s
        async with pool.acquire() as conn:
            for label, value, text in _IDS:
                sid = sessions[label].id
                await queries.append_event(
                    conn,
                    account_id=_ACCOUNT,
                    session_id=sid,
                    kind="message",
                    data=_assistant(value, "tc_s"),
                )
                await _lifecycle(conn, sid, {"event": "tool_requested", "tool_call_id": text})
                await _lifecycle(
                    conn,
                    sid,
                    {"event": "tool_confirmed", "result": "allow", "tool_call_id": text},
                )
                await _lifecycle(
                    conn,
                    sid,
                    {"event": "tool_confirmed", "result": "allow", "tool_call_id": "tc_s"},
                )
            control = sessions["control"].id
            await queries.append_event(
                conn,
                account_id=_ACCOUNT,
                session_id=control,
                kind="message",
                data=_assistant("tc_c"),
            )
            await _lifecycle(conn, control, {"event": "tool_requested", "tool_call_id": "tc_c"})
        yield pool, sessions
    finally:
        await pool.close()


class TestNonStringToolCallIds:
    async def test_page_does_not_raise_and_matches_master(
        self, page: tuple[asyncpg.Pool[Any], dict[str, Session]]
    ) -> None:
        pool, sessions = page
        async with pool.acquire() as conn:
            batch = await queries.list_unresolved_tool_calls_batch(
                conn, [s.id for s in sessions.values()], account_id=_ACCOUNT
            )
        for label, value, _text in _IDS:
            entries = batch[sessions[label].id]
            got = [
                (e["tool_call_id"], e["has_allow_lifecycle"], e["has_requested_lifecycle"])
                for e in entries
            ]
            # Master's output: the non-string id is listed unchanged with both
            # flags False (its ``->>`` twin markers do not count); the
            # string-id sibling on the same session keeps its allow flag.
            assert got == [(value, False, False), ("tc_s", True, False)], label
            assert type(entries[0]["tool_call_id"]) is type(value), label
        assert [
            (e["tool_call_id"], e["has_allow_lifecycle"], e["has_requested_lifecycle"])
            for e in batch[sessions["control"].id]
        ] == [("tc_c", False, True)]

    @pytest.mark.parametrize("label", [lbl for lbl, _v, _t in _IDS])
    async def test_single_non_string_session_does_not_raise(
        self, page: tuple[asyncpg.Pool[Any], dict[str, Session]], label: str
    ) -> None:
        """One id type per page, so each type is checked on its own (a page
        whose ONLY unresolved id is non-string still answers)."""
        pool, sessions = page
        async with pool.acquire() as conn:
            batch = await queries.list_unresolved_tool_calls_batch(
                conn, [sessions[label].id], account_id=_ACCOUNT
            )
        assert len(batch[sessions[label].id]) == 2

    async def test_compute_awaiting_page_does_not_raise(
        self, page: tuple[asyncpg.Pool[Any], dict[str, Session]]
    ) -> None:
        """The service path behind ``GET /v1/sessions``.  ``bash`` is
        ``always_allow``, so none of these calls is awaiting (as on master);
        the point is that the page answers instead of raising DataError."""
        pool, sessions = page
        awaiting = await compute_awaiting(pool, list(sessions.values()), account_id=_ACCOUNT)
        assert awaiting == {}
