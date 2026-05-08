"""Unit tests for the new connection + session-template models (#200).

The connector redesign collapses ``connections`` + ``channel_bindings`` +
``connection_routing_rules`` into one ``connections`` table with three
valid shapes (detached / single_session / per_chat) enforced by the
``connections_one_mode_ck`` CHECK constraint.  Wire-level model
validation covers the request shapes.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aios.models.agents import ToolSpec
from aios.models.connections import (
    Connection,
    ConnectionAttach,
    ConnectionConfigurePerChat,
    ConnectionCreate,
)
from aios.models.session_templates import (
    SessionTemplate,
    SessionTemplateCreate,
    SessionTemplateUpdate,
)


class TestConnectionCreate:
    def test_minimal(self) -> None:
        body = ConnectionCreate(connector="signal", account="+15550001")
        assert body.connector == "signal"
        assert body.account == "+15550001"
        assert body.metadata == {}

    def test_rejects_slash_in_connector(self) -> None:
        with pytest.raises(ValueError, match="must not contain '/'"):
            ConnectionCreate(connector="signal/bad", account="acct")

    def test_rejects_slash_in_account(self) -> None:
        with pytest.raises(ValueError, match="must not contain '/'"):
            ConnectionCreate(connector="signal", account="acct/with/slash")

    def test_rejects_extra_fields(self) -> None:
        """``extra="forbid"`` keeps wire shape strict — the old model
        accepted ``mcp_url`` and ``vault_id``; those are gone now and
        the API must 422 anything that still sends them.
        """
        with pytest.raises(ValueError):
            ConnectionCreate.model_validate(
                {
                    "connector": "signal",
                    "account": "acct",
                    "mcp_url": "https://m",  # removed in #200
                }
            )

    def test_accepts_tools(self) -> None:
        """Connections can declare model-facing custom tools (#301)."""
        body = ConnectionCreate(
            connector="signal",
            account="+1",
            tools=[
                ToolSpec(
                    type="custom",
                    name="signal_send",
                    description="Send a Signal message",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "chat_id": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["chat_id", "text"],
                    },
                ),
            ],
        )
        assert len(body.tools) == 1
        assert body.tools[0].name == "signal_send"

    def test_tools_default_empty(self) -> None:
        body = ConnectionCreate(connector="signal", account="+1")
        assert body.tools == []

    def test_rejects_non_custom_tool_types(self) -> None:
        """Per #301: connection-declared tools are always client-executed.
        Built-in (e.g. ``bash``) and ``mcp_toolset`` types belong on the
        agent, not the connection.
        """
        with pytest.raises(ValueError, match="custom"):
            ConnectionCreate(
                connector="signal",
                account="+1",
                tools=[ToolSpec(type="bash")],
            )


class TestConnectionAttach:
    def test_round_trip(self) -> None:
        body = ConnectionAttach(session_id="sess_1")
        assert body.session_id == "sess_1"

    def test_rejects_extra(self) -> None:
        with pytest.raises(ValueError):
            ConnectionAttach.model_validate(
                {"session_id": "sess_1", "session_template_id": "stpl_1"}
            )


class TestConnectionConfigurePerChat:
    def test_round_trip(self) -> None:
        body = ConnectionConfigurePerChat(session_template_id="stpl_1")
        assert body.session_template_id == "stpl_1"


class TestConnectionReadView:
    def _now(self) -> datetime:
        return datetime(2026, 5, 1)

    def test_detached_shape(self) -> None:
        c = Connection(
            id="conn_1",
            connector="signal",
            account="+1",
            metadata={},
            created_at=self._now(),
            updated_at=self._now(),
        )
        assert c.session_id is None
        assert c.session_template_id is None

    def test_single_session_shape(self) -> None:
        c = Connection(
            id="conn_1",
            connector="signal",
            account="+1",
            session_id="sess_1",
            metadata={},
            created_at=self._now(),
            attached_at=self._now(),
            updated_at=self._now(),
        )
        assert c.session_id == "sess_1"
        assert c.session_template_id is None

    def test_per_chat_shape(self) -> None:
        c = Connection(
            id="conn_1",
            connector="signal",
            account="+1",
            session_template_id="stpl_1",
            metadata={},
            created_at=self._now(),
            attached_at=self._now(),
            updated_at=self._now(),
        )
        assert c.session_id is None
        assert c.session_template_id == "stpl_1"

    def test_tools_default_empty(self) -> None:
        c = Connection(
            id="conn_1",
            connector="signal",
            account="+1",
            metadata={},
            created_at=self._now(),
            updated_at=self._now(),
        )
        assert c.tools == []

    def test_round_trips_tools(self) -> None:
        c = Connection(
            id="conn_1",
            connector="signal",
            account="+1",
            metadata={},
            tools=[
                ToolSpec(
                    type="custom",
                    name="signal_send",
                    description="Send",
                    input_schema={"type": "object"},
                ),
            ],
            created_at=self._now(),
            updated_at=self._now(),
        )
        assert len(c.tools) == 1
        assert c.tools[0].name == "signal_send"


class TestSessionTemplateCreate:
    def test_minimal(self) -> None:
        body = SessionTemplateCreate(
            name="dev-template",
            agent_id="agent_1",
            environment_id="env_1",
        )
        assert body.agent_version is None
        assert body.vault_ids == []
        assert body.memory_store_ids == []
        assert body.metadata == {}

    def test_full(self) -> None:
        body = SessionTemplateCreate(
            name="prod",
            agent_id="agent_1",
            environment_id="env_1",
            agent_version=3,
            vault_ids=["vlt_1", "vlt_2"],
            memory_store_ids=["mem_1"],
            metadata={"team": "eng"},
        )
        assert body.agent_version == 3
        assert body.vault_ids == ["vlt_1", "vlt_2"]


class TestSessionTemplateUpdate:
    def test_partial_update_ignores_unset_fields(self) -> None:
        body = SessionTemplateUpdate(name="renamed")
        assert "name" in body.model_fields_set
        assert "agent_version" not in body.model_fields_set
        assert "metadata" not in body.model_fields_set

    def test_explicit_null_agent_version_distinguishable(self) -> None:
        """``agent_version=None`` (latest) and "omitted" (keep current)
        must be distinguishable via ``model_fields_set`` so the router
        can route the two cases correctly.
        """
        latest = SessionTemplateUpdate(name="x", agent_version=None)
        keep = SessionTemplateUpdate(name="x")
        assert "agent_version" in latest.model_fields_set
        assert "agent_version" not in keep.model_fields_set


class TestSessionTemplateReadView:
    def test_round_trip(self) -> None:
        now = datetime(2026, 5, 1)
        t = SessionTemplate(
            id="stpl_1",
            name="prod",
            agent_id="agent_1",
            agent_version=None,
            environment_id="env_1",
            vault_ids=[],
            memory_store_ids=[],
            metadata={},
            created_at=now,
            updated_at=now,
        )
        assert t.archived_at is None
