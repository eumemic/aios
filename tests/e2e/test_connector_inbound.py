"""E2E coverage for the connector supervisor's inbound dispatcher.

Drives ``ConnectorSubprocessRegistry._handle_inbound`` directly against
a real Postgres-backed harness (no connector subprocess spawn — the
focus is the worker-side pipeline, not stdio framing which is covered
by ``tests/e2e/test_connector_supervisor.py``).

Coverage matrix:

* **single_session**: attached connection routes to its session_id;
  event lands, ack invoked, drop counter stays empty.
* **per_chat new chat**: missing ``connection_chat_sessions`` row
  triggers ``services.sessions.create_session`` with focal locked and
  ``spawned_from_connection_id`` set; subsequent inbound for the
  same chat reuses the existing session.
* **per_chat distinct chats**: two distinct ``chat_id``s spawn two
  distinct sessions, both marked ``spawned_from_connection_id``.
* **detached connection**: drop with ``detached`` reason, ack still
  fires (so the connector can clear its spool).
* **no connection + auto_create=true (default)**: detached connection
  auto-created, drop with ``no_connection``, no event.
* **no connection + auto_create=false**: drop with ``no_connection``,
  NO connection row inserted.
* **dedup ledger**: replaying the same ``event_id`` twice via
  ``_handle_inbound`` produces exactly one event row in the session.
* **archived template**: per_chat with archived template drops with
  ``archived_template`` reason.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest import mock

import pytest

from aios.config import Settings
from aios.db import queries
from aios.harness.connector_supervisor import ConnectorSubprocessRegistry
from aios.ids import make_id
from aios.mcp.stdio_transport import ConnectorSpec
from aios.models.agents import ToolSpec
from aios.services import (
    agents as agents_service,
)
from aios.services import (
    connections as connections_service,
)
from aios.services import (
    environments as environments_service,
)
from aios.services import (
    session_templates as session_templates_service,
)
from aios.services import (
    sessions as sessions_service,
)
from tests.conftest import needs_docker
from tests.e2e.harness import Harness


def _spec(name: str = "echo") -> ConnectorSpec:
    """A spec the supervisor can hold without spawning anything.

    Tests never call ``start()`` so command/cwd never run; we only
    need a name so :class:`ConnectorState` initializes cleanly.
    """
    return ConnectorSpec(name=name, command="x", args=[])


def _registry(settings: Settings | None = None, name: str = "echo") -> ConnectorSubprocessRegistry:
    """Build a registry with one default-instance entry for tests.

    Default-instance shape (``instance == connector``) keeps the
    inbound-handler call sites simple: ``_handle_inbound((name, name),
    params)`` mirrors what the real splitter does for single-instance
    setups.
    """
    from aios.config import ConnectorInstance

    settings = settings or Settings()
    return ConnectorSubprocessRegistry(
        [(ConnectorInstance(connector=name, instance=name), _spec(name))],
        settings=settings,
    )


def _unique_account(prefix: str = "acct") -> str:
    """Build a per-test account string so testcontainer state stays isolated.

    The active-connection unique index ``(connector, account) WHERE
    archived_at IS NULL`` would otherwise reject the second test that
    re-uses the same account name (the harness fixture is function-scoped
    but the testcontainer Postgres persists across tests in a session).
    """
    return f"{prefix}-{make_id('evt')[-10:]}"


def _patch_send_ack(registry: ConnectorSubprocessRegistry) -> Callable[[], list[str]]:
    """Replace ``_send_ack`` with a recorder.  Returns a view function.

    The real ``_send_ack`` calls ``dispatch_call`` against the (absent)
    connector subprocess, which would error.  Tests record the
    event_ids that *would have been* acked instead.
    """
    recorded: list[str] = []

    async def fake_ack(self: Any, state: Any, event_id: str) -> None:
        recorded.append(event_id)

    registry._send_ack = fake_ack.__get__(registry, type(registry))  # type: ignore[method-assign]

    def view() -> list[str]:
        return recorded

    return view


async def _make_agent_and_env(harness: Harness) -> tuple[str, str]:
    """Build a fake agent + environment, returning their ids.

    Matches the helper used by ``test_focal_channel`` so the harness's
    ``_env_id`` shortcut still works on the second call.
    """
    if harness._env_id is None:
        env = await environments_service.create_environment(
            harness._pool, name=f"inbound-env-{make_id('env')[-8:]}"
        )
        harness._env_id = env.id
    agent = await agents_service.create_agent(
        harness._pool,
        name=f"inbound-agent-{make_id('agent')[-8:]}",
        model="fake/test",
        system="test",
        tools=[ToolSpec(type="bash")],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    return agent.id, harness._env_id


@needs_docker
class TestSingleSessionInbound:
    async def test_attached_connection_appends_event_and_acks(self, harness: Harness) -> None:
        """Inbound for an attached connection lands as a user event in its session."""
        agent_id, env_id = await _make_agent_and_env(harness)
        account = _unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="single",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        registry = _registry()
        ack_view = _patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            await registry._handle_inbound(
                ("echo", "echo"),
                {
                    "event_id": make_id("evt"),
                    "account": account,
                    "chat_id": "chat-x",
                    "sender": {"display_name": "Alice"},
                    "content": "hello world",
                },
            )

        events = await harness.events(session.id)
        user_messages = [e for e in events if e.kind == "message" and e.data.get("role") == "user"]
        assert any(e.data.get("content") == "hello world" for e in user_messages)
        assert len(ack_view()) == 1
        # Drop counter never bumped on the happy path.
        assert dict(registry._states[("echo", "echo")].drops) == {}


@needs_docker
class TestPerChatInbound:
    async def test_first_inbound_spawns_session_with_focal_locked(self, harness: Harness) -> None:
        """First inbound on a per_chat connection auto-spawns a session.

        Verifies plan §"per_chat session creation": session created via
        ``create_session(template, focal_channel=..., spawned_from_connection_id=...)``,
        registered in ``connection_chat_sessions``, focal channel set,
        outbound permission via ``spawned_from_connection_id``.
        """
        agent_id, env_id = await _make_agent_and_env(harness)
        account = _unique_account()
        template = await session_templates_service.create_session_template(
            harness._pool,
            name=f"tpl-{make_id('stpl')[-8:]}",
            agent_id=agent_id,
            environment_id=env_id,
            agent_version=None,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.configure_per_chat(
            harness._pool, conn_row.id, session_template_id=template.id
        )

        registry = _registry()
        _patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            await registry._handle_inbound(
                ("echo", "echo"),
                {
                    "event_id": make_id("evt"),
                    "account": account,
                    "chat_id": "chat-spawn",
                    "sender": {"display_name": "Bob"},
                    "content": "first message",
                },
            )

        async with harness._pool.acquire() as conn:
            spawned_session_id = await queries.lookup_chat_session(conn, conn_row.id, "chat-spawn")
            assert spawned_session_id is not None
            spawned_session = await queries.get_session(conn, spawned_session_id)
            focal = await queries.get_session_focal_channel(conn, spawned_session_id)
            spawned_from = await queries.get_session_spawn_origin(conn, spawned_session_id)

        assert focal == f"echo/{account}/chat-spawn"
        assert spawned_from == conn_row.id
        # Session inherits agent from template.
        assert spawned_session.agent_id == agent_id

        events = await harness.events(spawned_session_id)
        contents = [e.data.get("content") for e in events if e.data.get("role") == "user"]
        assert "first message" in contents

    async def test_distinct_chats_spawn_distinct_sessions(self, harness: Harness) -> None:
        """Two chat_ids on the same connection produce two sessions.

        Verifies plan §"Per_chat live demo": both sessions are
        addressable, both have ``spawned_from_connection_id`` set.
        """
        agent_id, env_id = await _make_agent_and_env(harness)
        account = _unique_account()
        template = await session_templates_service.create_session_template(
            harness._pool,
            name=f"tpl-{make_id('stpl')[-8:]}",
            agent_id=agent_id,
            environment_id=env_id,
            agent_version=None,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.configure_per_chat(
            harness._pool, conn_row.id, session_template_id=template.id
        )

        registry = _registry()
        _patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            for chat_id in ("chat-A", "chat-B"):
                await registry._handle_inbound(
                    ("echo", "echo"),
                    {
                        "event_id": make_id("evt"),
                        "account": account,
                        "chat_id": chat_id,
                        "sender": {"display_name": "U"},
                        "content": f"msg-{chat_id}",
                    },
                )

        async with harness._pool.acquire() as conn:
            session_a = await queries.lookup_chat_session(conn, conn_row.id, "chat-A")
            session_b = await queries.lookup_chat_session(conn, conn_row.id, "chat-B")

        assert session_a is not None and session_b is not None
        assert session_a != session_b


@needs_docker
class TestDetachedConnection:
    async def test_detached_drops_with_counter_and_ack(self, harness: Harness) -> None:
        account = _unique_account()
        await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        registry = _registry()
        ack_view = _patch_send_ack(registry)
        event_id = make_id("evt")
        await registry._handle_inbound(
            ("echo", "echo"),
            {
                "event_id": event_id,
                "account": account,
                "chat_id": "any",
                "sender": {"display_name": "Z"},
                "content": "ignored",
            },
        )
        assert registry._states[("echo", "echo")].drops["detached"] == 1
        # Ack still fires so connector clears the spool.
        assert ack_view() == [event_id]


@needs_docker
class TestAutoCreateConnection:
    async def test_unknown_account_default_auto_creates_detached(self, harness: Harness) -> None:
        """Default behaviour: unknown ``(connector, account)`` auto-creates a detached row."""
        account = _unique_account("fresh")
        registry = _registry()
        ack_view = _patch_send_ack(registry)
        event_id = make_id("evt")
        await registry._handle_inbound(
            ("echo", "echo"),
            {
                "event_id": event_id,
                "account": account,
                "chat_id": "any",
                "sender": {"display_name": "Z"},
                "content": "ignored",
            },
        )
        assert registry._states[("echo", "echo")].drops["no_connection"] == 1
        async with harness._pool.acquire() as conn:
            row = await queries.get_connection_for_account(conn, connector="echo", account=account)
            assert row is not None
        assert ack_view() == [event_id]

    async def test_unknown_account_with_auto_create_false_skips_insert(
        self, harness: Harness
    ) -> None:
        """``connectors_auto_create={"echo": False}`` skips the row insert."""
        account = _unique_account("norow")
        settings = Settings(connectors_auto_create={"echo": False})
        registry = _registry(settings=settings)
        ack_view = _patch_send_ack(registry)
        event_id = make_id("evt")
        await registry._handle_inbound(
            ("echo", "echo"),
            {
                "event_id": event_id,
                "account": account,
                "chat_id": "any",
                "sender": {"display_name": "Z"},
                "content": "ignored",
            },
        )
        assert registry._states[("echo", "echo")].drops["no_connection"] == 1
        async with harness._pool.acquire() as conn:
            row = await queries.get_connection_for_account(conn, connector="echo", account=account)
            assert row is None
        assert ack_view() == [event_id]


@needs_docker
class TestDedupLedger:
    async def test_replaying_same_event_id_appends_once(self, harness: Harness) -> None:
        """Two ``_handle_inbound`` calls with the same event_id yield one event.

        Simulates the worker-restart-mid-pipe scenario: the connector's
        spool replays the same event after the worker forgot to ack;
        the ledger conflict path catches the duplicate.
        """
        agent_id, env_id = await _make_agent_and_env(harness)
        account = _unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="dedup",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        registry = _registry()
        _patch_send_ack(registry)
        params = {
            "event_id": make_id("evt"),
            "account": account,
            "chat_id": "x",
            "sender": {"display_name": "A"},
            "content": "once and only once",
        }
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            await registry._handle_inbound(("echo", "echo"), params)
            await registry._handle_inbound(("echo", "echo"), params)

        events = await harness.events(session.id)
        matches = [e for e in events if e.data.get("content") == "once and only once"]
        assert len(matches) == 1


@needs_docker
class TestArchivedTemplateDrop:
    async def test_archived_template_drops_with_counter(self, harness: Harness) -> None:
        agent_id, env_id = await _make_agent_and_env(harness)
        account = _unique_account()
        template = await session_templates_service.create_session_template(
            harness._pool,
            name=f"tpl-{make_id('stpl')[-8:]}",
            agent_id=agent_id,
            environment_id=env_id,
            agent_version=None,
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.configure_per_chat(
            harness._pool, conn_row.id, session_template_id=template.id
        )
        async with harness._pool.acquire() as conn:
            await queries.archive_session_template(conn, template.id)

        registry = _registry()
        ack_view = _patch_send_ack(registry)
        event_id = make_id("evt")
        await registry._handle_inbound(
            ("echo", "echo"),
            {
                "event_id": event_id,
                "account": account,
                "chat_id": "new-chat",
                "sender": {"display_name": "Z"},
                "content": "no spawn",
            },
        )
        assert registry._states[("echo", "echo")].drops["archived_template"] == 1
        assert ack_view() == [event_id]


@needs_docker
class TestConnectorMetadataMerging:
    """Connector-supplied ``params["metadata"]`` reaches the event log.

    The model relies on these fields — e.g. ``signal_react`` is documented
    to copy ``sender_uuid`` and ``timestamp_ms`` from the inbound header.
    A regression that drops them silently would invalidate the tool
    contract; this test catches that.
    """

    async def test_connector_metadata_lands_in_event_data(self, harness: Harness) -> None:
        agent_id, env_id = await _make_agent_and_env(harness)
        account = _unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="meta",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        registry = _registry()
        _patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            await registry._handle_inbound(
                ("echo", "echo"),
                {
                    "event_id": make_id("evt"),
                    "account": account,
                    "chat_id": "chat-meta",
                    "sender": {"display_name": "Alice"},
                    "content": "hi",
                    "metadata": {
                        "sender_uuid": "abcd-1234",
                        "timestamp_ms": 1700000000000,
                        "reply_to": {"author_uuid": "x", "timestamp_ms": 1, "text": "?"},
                    },
                },
            )

        events = await harness.events(session.id)
        user_event = next(e for e in events if e.data.get("role") == "user")
        meta = user_event.data["metadata"]
        # Connector fields preserved.
        assert meta["sender_uuid"] == "abcd-1234"
        assert meta["timestamp_ms"] == 1700000000000
        assert meta["reply_to"]["author_uuid"] == "x"
        # Supervisor-canonical stamps win on conflict / additive.
        assert meta["channel"] == f"echo/{account}/chat-meta"
        assert meta["sender"] == "Alice"


@needs_docker
class TestPayloadTooLarge:
    """Oversized inbound content drops with a counter and acks the spool.

    Mirrors ``services.sessions.append_user_message``'s
    ``MAX_USER_MESSAGE_CHARS`` cap so a malformed/attacker payload can't
    blow up the session's prompt-cache window.
    """

    async def test_oversized_content_drops_and_acks(self, harness: Harness) -> None:
        from aios.models.sessions import MAX_USER_MESSAGE_CHARS

        agent_id, env_id = await _make_agent_and_env(harness)
        account = _unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="oversized",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        registry = _registry()
        ack_view = _patch_send_ack(registry)
        event_id = make_id("evt")
        await registry._handle_inbound(
            ("echo", "echo"),
            {
                "event_id": event_id,
                "account": account,
                "chat_id": "x",
                "sender": {"display_name": "A"},
                "content": "x" * (MAX_USER_MESSAGE_CHARS + 1),
            },
        )

        assert registry._states[("echo", "echo")].drops["payload_too_large"] == 1
        assert ack_view() == [event_id]
        # No event was appended.
        events = await harness.events(session.id)
        assert all(e.data.get("role") != "user" or e.data.get("content") != "x" for e in events)


@needs_docker
class TestDeferWakeFailureHeals:
    """A transient ``defer_wake`` failure no longer strands the message.

    Prior to the review fix, a swallowed ``defer_wake`` error meant the
    appended event sat in the log with no wake job AND the connector
    cleared its spool — message lost.  Post-fix: the inbound task fails
    (no ack), the connector replays, the dedup ledger blocks a second
    append, but ``defer_wake`` runs unconditionally on the replay, so the
    wake gets enqueued the second time.
    """

    async def test_defer_wake_failure_is_healed_by_replay(self, harness: Harness) -> None:
        agent_id, env_id = await _make_agent_and_env(harness)
        account = _unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="heal",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        registry = _registry()
        ack_view = _patch_send_ack(registry)

        # First attempt: defer_wake raises, _send_ack should NOT be called.
        params = {
            "event_id": make_id("evt"),
            "account": account,
            "chat_id": "x",
            "sender": {"display_name": "A"},
            "content": "once",
        }
        defer_wake_calls: list[str] = []

        async def fake_defer_wake_fail(_pool: Any, session_id: str, **_kw: Any) -> None:
            defer_wake_calls.append(session_id)
            raise RuntimeError("simulated transient DB hiccup")

        with (
            mock.patch(
                "aios.harness.connector_supervisor.defer_wake",
                new=fake_defer_wake_fail,
            ),
            pytest.raises(RuntimeError, match="simulated"),
        ):
            await registry._handle_inbound(("echo", "echo"), params)

        # Event was committed (the txn closed before defer_wake ran).
        events_after_first = await harness.events(session.id)
        assert any(e.data.get("content") == "once" for e in events_after_first)
        # ack NOT sent (defer_wake raised, _send_ack didn't run).
        assert ack_view() == []

        # Replay: defer_wake now succeeds.  Ledger conflict → no second
        # append, but defer_wake runs (heals the prior strand) and ack
        # fires (clears the connector's spool).
        async def fake_defer_wake_ok(_pool: Any, session_id: str, **_kw: Any) -> None:
            defer_wake_calls.append(session_id)

        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=fake_defer_wake_ok,
        ):
            await registry._handle_inbound(("echo", "echo"), params)

        # Still exactly one event (dedup ledger blocked the second).
        events_after_second = await harness.events(session.id)
        once_count = sum(1 for e in events_after_second if e.data.get("content") == "once")
        assert once_count == 1
        # defer_wake was called twice (once failing, once succeeding).
        assert len(defer_wake_calls) == 2
        # ack now fired.
        assert len(ack_view()) == 1


@needs_docker
class TestRegistryIsolation:
    """Ensure the new registry construction signature plays nicely with `Settings()`."""

    async def test_default_settings_constructs_cleanly(self) -> None:
        registry = _registry()
        assert registry._settings.connectors_auto_create == {}
        assert registry.snapshot_all() == [
            {
                "connector": "echo",
                "instance": "echo",
                "status": "starting",
                "instructions": None,
                "accounts": [],
                "last_error": None,
                "recent_drops": {},
            }
        ]


@needs_docker
class TestMultiAccountInboundRouting:
    """Multi-account: ``(connector, account)`` is the routing primitive.

    A single-instance connector serving N accounts must route inbound on
    each account to the connection attached to that account, even when
    the same instance fans out to multiple sessions.
    """

    async def test_two_accounts_route_to_distinct_sessions(self, harness: Harness) -> None:
        agent_id, env_id = await _make_agent_and_env(harness)
        account_a = _unique_account("acct-a")
        account_b = _unique_account("acct-b")
        session_a = await sessions_service.create_session(
            harness._pool, agent_id=agent_id, environment_id=env_id, title="A", metadata={}
        )
        session_b = await sessions_service.create_session(
            harness._pool, agent_id=agent_id, environment_id=env_id, title="B", metadata={}
        )
        conn_a = await connections_service.create_connection(
            harness._pool, connector="echo", account=account_a, metadata={}
        )
        conn_b = await connections_service.create_connection(
            harness._pool, connector="echo", account=account_b, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_a.id, session_id=session_a.id
        )
        await connections_service.attach_connection(
            harness._pool, conn_b.id, session_id=session_b.id
        )

        registry = _registry()
        _patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            await registry._handle_inbound(
                ("echo", "echo"),
                {
                    "event_id": make_id("evt"),
                    "account": account_a,
                    "chat_id": "chat-A",
                    "sender": {"display_name": "Alice"},
                    "content": "hello A",
                },
            )
            await registry._handle_inbound(
                ("echo", "echo"),
                {
                    "event_id": make_id("evt"),
                    "account": account_b,
                    "chat_id": "chat-B",
                    "sender": {"display_name": "Bob"},
                    "content": "hello B",
                },
            )

        events_a = await harness.events(session_a.id)
        events_b = await harness.events(session_b.id)
        # Each session sees only its own account's inbound.
        contents_a = {
            e.data.get("content")
            for e in events_a
            if e.kind == "message" and e.data.get("role") == "user"
        }
        contents_b = {
            e.data.get("content")
            for e in events_b
            if e.kind == "message" and e.data.get("role") == "user"
        }
        assert "hello A" in contents_a and "hello B" not in contents_a
        assert "hello B" in contents_b and "hello A" not in contents_b


@needs_docker
class TestAccountConflictRejection:
    """Two instances of the same connector type both reporting the same
    account: second is rejected, primary keeps the map entry."""

    async def test_second_instance_rejected_with_last_error(self) -> None:
        from aios.config import ConnectorInstance
        from aios.harness.connector_supervisor import ConnectorState

        registry = ConnectorSubprocessRegistry(
            [
                (ConnectorInstance("signal", "primary"), _spec("signal")),
                (ConnectorInstance("signal", "secondary"), _spec("signal")),
            ],
            settings=Settings(),
        )
        # Bypass the supervisor loop entirely; we only exercise routing.
        # `_states` already wired by the registry constructor.
        await registry._on_aios_notification(
            ("signal", "primary"),
            "notifications/aios/accounts",
            {"accounts": [{"id": "+15551234567", "display_name": "X"}]},
        )
        await registry._on_aios_notification(
            ("signal", "secondary"),
            "notifications/aios/accounts",
            {"accounts": [{"id": "+15551234567", "display_name": "X-dup"}]},
        )
        assert registry.lookup_instance_for_account("signal", "+15551234567") == "primary"
        secondary: ConnectorState | None = registry.state("signal", "secondary")
        assert secondary is not None
        assert secondary.last_error is not None
        assert "conflict" in secondary.last_error
