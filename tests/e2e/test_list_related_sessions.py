"""E2E tests for the ``list_related_sessions`` agent tool (#803).

The tool gives an agent a flat listing of the chat sessions routed through
its account's connection(s): each ``(chat_id, session_id, created_at)`` row
from ``chat_sessions``, account-scoped. No ACL, no subset semantics — just
"what chats on my account route to which sessions". An optional
``connection_id`` narrows the listing to one connection (which must belong
to the caller's account; cross-account ids raise ``NotFoundError``).
"""

from __future__ import annotations

import pytest

from aios.crypto.vault import CryptoBox
from aios.db import queries as db_queries
from aios.errors import NotFoundError
from aios.tools.list_related_sessions import list_related_sessions_handler
from tests.conftest import needs_docker
from tests.e2e.harness import Harness

pytestmark = pytest.mark.docker


async def _make_caller_session(harness: Harness, account_id: str, *, suffix: str) -> str:
    """Create an agent + environment + session under ``account_id``; return
    the session id used as the tool's ``session_id`` argument."""
    from aios.services import agents as agents_service
    from aios.services import environments as env_svc
    from aios.services import sessions as sess_svc

    agent = await agents_service.create_agent(
        harness._pool,
        name=f"lrs-caller-{suffix}",
        model="fake/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=account_id,
    )
    env = await env_svc.create_environment(
        harness._pool, name=f"lrs-env-{suffix}", account_id=account_id
    )
    session = await sess_svc.create_session(
        harness._pool,
        agent_id=agent.id,
        environment_id=env.id,
        title=f"lrs-{suffix}",
        metadata={},
        account_id=account_id,
    )
    return session.id


async def _make_bare_session(harness: Harness, account_id: str, *, suffix: str) -> str:
    """Create a session under ``account_id`` to act as a chat-session target."""
    from aios.services import agents as agents_service
    from aios.services import environments as env_svc
    from aios.services import sessions as sess_svc

    agent = await agents_service.create_agent(
        harness._pool,
        name=f"lrs-target-{suffix}",
        model="fake/test",
        system="",
        tools=[],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
        account_id=account_id,
    )
    env = await env_svc.create_environment(
        harness._pool, name=f"lrs-tenv-{suffix}", account_id=account_id
    )
    session = await sess_svc.create_session(
        harness._pool,
        agent_id=agent.id,
        environment_id=env.id,
        title=f"lrs-target-{suffix}",
        metadata={},
        account_id=account_id,
    )
    return session.id


@needs_docker
class TestListRelatedSessions:
    async def test_returns_both_chat_sessions_on_one_connection(
        self, harness: Harness, crypto_box: CryptoBox
    ) -> None:
        from aios.services import connections as connections_service

        account_id = "acc_test_stub"
        caller_session_id = await _make_caller_session(harness, account_id, suffix="both")

        connection = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            external_account_id="echo-both",
            metadata={},
            crypto_box=crypto_box,
            account_id=account_id,
        )

        sess_a = await _make_bare_session(harness, account_id, suffix="a")
        sess_b = await _make_bare_session(harness, account_id, suffix="b")
        async with harness._pool.acquire() as db_conn:
            await db_queries.insert_chat_session(
                db_conn,
                connection_id=connection.id,
                chat_id="chat_a",
                session_id=sess_a,
                account_id=account_id,
            )
            await db_queries.insert_chat_session(
                db_conn,
                connection_id=connection.id,
                chat_id="chat_b",
                session_id=sess_b,
                account_id=account_id,
            )

        result = await list_related_sessions_handler(caller_session_id, {})
        rows = result["sessions"]
        assert {(r["chat_id"], r["session_id"]) for r in rows} == {
            ("chat_a", sess_a),
            ("chat_b", sess_b),
        }
        for r in rows:
            assert r["created_at"]
            assert "chat_name" not in r

    async def test_cross_account_session_not_visible(
        self, harness: Harness, crypto_box: CryptoBox
    ) -> None:
        from aios.services import accounts as accounts_service
        from aios.services import connections as connections_service

        account_a = "acc_test_stub"
        caller_session_id = await _make_caller_session(harness, account_a, suffix="xa")

        conn_a = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            external_account_id="echo-xa",
            metadata={},
            crypto_box=crypto_box,
            account_id=account_a,
        )
        sess_a = await _make_bare_session(harness, account_a, suffix="xa")
        async with harness._pool.acquire() as db_conn:
            await db_queries.insert_chat_session(
                db_conn,
                connection_id=conn_a.id,
                chat_id="chat_a_only",
                session_id=sess_a,
                account_id=account_a,
            )

        # Account B: separate tenant with its own connection + chat session.
        minted = await accounts_service.mint_child(
            harness._pool,
            caller_account_id=account_a,
            caller_can_mint_children=True,
            display_name="tenant-b",
            can_mint_children=False,
        )
        account_b = minted.account_id
        conn_b = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            external_account_id="echo-xb",
            metadata={},
            crypto_box=crypto_box,
            account_id=account_b,
        )
        sess_b = await _make_bare_session(harness, account_b, suffix="xb")
        async with harness._pool.acquire() as db_conn:
            await db_queries.insert_chat_session(
                db_conn,
                connection_id=conn_b.id,
                chat_id="chat_b_only",
                session_id=sess_b,
                account_id=account_b,
            )

        result = await list_related_sessions_handler(caller_session_id, {})
        chat_ids = {r["chat_id"] for r in result["sessions"]}
        session_ids = {r["session_id"] for r in result["sessions"]}
        assert "chat_b_only" not in chat_ids
        assert sess_b not in session_ids
        assert chat_ids == {"chat_a_only"}

    async def test_optional_connection_filter_scopes_and_rejects_cross_account(
        self, harness: Harness, crypto_box: CryptoBox
    ) -> None:
        from aios.services import accounts as accounts_service
        from aios.services import connections as connections_service

        account_a = "acc_test_stub"
        caller_session_id = await _make_caller_session(harness, account_a, suffix="flt")

        conn_a = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            external_account_id="echo-flt-a",
            metadata={},
            crypto_box=crypto_box,
            account_id=account_a,
        )
        sess_a = await _make_bare_session(harness, account_a, suffix="flt-a")
        async with harness._pool.acquire() as db_conn:
            await db_queries.insert_chat_session(
                db_conn,
                connection_id=conn_a.id,
                chat_id="chat_flt_a",
                session_id=sess_a,
                account_id=account_a,
            )

        minted = await accounts_service.mint_child(
            harness._pool,
            caller_account_id=account_a,
            caller_can_mint_children=True,
            display_name="tenant-flt-b",
            can_mint_children=False,
        )
        account_b = minted.account_id
        conn_b = await connections_service.create_connection(
            harness._pool,
            connector="echo",
            external_account_id="echo-flt-b",
            metadata={},
            crypto_box=crypto_box,
            account_id=account_b,
        )
        sess_b = await _make_bare_session(harness, account_b, suffix="flt-b")
        async with harness._pool.acquire() as db_conn:
            await db_queries.insert_chat_session(
                db_conn,
                connection_id=conn_b.id,
                chat_id="chat_flt_b",
                session_id=sess_b,
                account_id=account_b,
            )

        # Filter to A's own connection → only A's rows.
        scoped = await list_related_sessions_handler(
            caller_session_id, {"connection_id": conn_a.id}
        )
        assert {(r["chat_id"], r["session_id"]) for r in scoped["sessions"]} == {
            ("chat_flt_a", sess_a)
        }

        # Filter to B's connection → account guard rejects cross-account id.
        with pytest.raises(NotFoundError):
            await list_related_sessions_handler(caller_session_id, {"connection_id": conn_b.id})
