"""Pure-unit tests for routing Pydantic models (no DB)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.channel_bindings import ChannelBindingCreate
from aios.models.connections import ConnectionCreate, ConnectionUpdate, InboundMessage
from aios.models.routing_rules import (
    RoutingRuleCreate,
    RoutingRuleUpdate,
    SessionParams,
)


class TestConnectionCreate:
    def test_valid(self) -> None:
        c = ConnectionCreate(
            connector="signal",
            account="alice",
            mcp_url="https://mcp.example.com",
            vault_id="vlt_abc",
        )
        assert c.connector == "signal"
        assert c.metadata == {}

    def test_with_metadata(self) -> None:
        c = ConnectionCreate(
            connector="signal",
            account="alice",
            mcp_url="https://mcp.example.com",
            vault_id="vlt_abc",
            metadata={"source": "manual"},
        )
        assert c.metadata == {"source": "manual"}

    def test_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            ConnectionCreate(
                connector="signal",
                account="alice",
                mcp_url="https://m",
                vault_id="vlt_abc",
                bogus="x",  # type: ignore[call-arg]
            )

    def test_rejects_empty_connector(self) -> None:
        with pytest.raises(ValidationError):
            ConnectionCreate(
                connector="",
                account="alice",
                mcp_url="https://m",
                vault_id="vlt_abc",
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [("connector", "signal/x"), ("account", "alice/bob")],
    )
    def test_rejects_slash(self, field: str, value: str) -> None:
        kwargs: dict[str, str] = {
            "connector": "signal",
            "account": "alice",
            "mcp_url": "https://m",
            "vault_id": "vlt_abc",
        }
        kwargs[field] = value
        with pytest.raises(ValidationError, match="must not contain '/'"):
            ConnectionCreate(**kwargs)  # type: ignore[arg-type]


class TestConnectionUpdate:
    def test_all_optional(self) -> None:
        u = ConnectionUpdate()
        assert u.mcp_url is None
        assert u.vault_id is None
        assert u.metadata is None

    def test_rejects_connector_field(self) -> None:
        # connector and account are immutable — Update doesn't accept them
        with pytest.raises(ValidationError):
            ConnectionUpdate(connector="signal")  # type: ignore[call-arg]


class TestChannelBindingCreate:
    def test_valid(self) -> None:
        b = ChannelBindingCreate(address="signal/test/chat-1", session_id="sess_abc")
        assert b.address == "signal/test/chat-1"

    def test_rejects_empty_address(self) -> None:
        with pytest.raises(ValidationError):
            ChannelBindingCreate(address="", session_id="sess_abc")


class TestSessionParams:
    def test_defaults(self) -> None:
        p = SessionParams()
        assert p.environment_id is None
        assert p.vault_ids == []
        assert p.title is None
        assert p.metadata == {}

    def test_full(self) -> None:
        p = SessionParams(
            environment_id="env_xyz",
            vault_ids=["vlt_a", "vlt_b"],
            title="Signal: {address}",
            metadata={"source": "rule"},
        )
        assert p.environment_id == "env_xyz"
        assert p.vault_ids == ["vlt_a", "vlt_b"]


class TestRoutingRuleCreate:
    def test_valid(self) -> None:
        r = RoutingRuleCreate(prefix="signal/test", target="agent:agent_x")
        assert r.prefix == "signal/test"
        assert r.target == "agent:agent_x"
        assert r.session_params == SessionParams()

    def test_rejects_empty_prefix(self) -> None:
        with pytest.raises(ValidationError):
            RoutingRuleCreate(prefix="", target="agent:x")

    def test_rejects_empty_target(self) -> None:
        with pytest.raises(ValidationError):
            RoutingRuleCreate(prefix="x", target="")


class TestRoutingRuleUpdate:
    def test_all_optional(self) -> None:
        u = RoutingRuleUpdate()
        assert u.target is None
        assert u.session_params is None


class TestInboundMessage:
    def test_minimal(self) -> None:
        m = InboundMessage(path="chat-1", content="hi")
        assert m.path == "chat-1"
        assert m.metadata == {}

    def test_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            InboundMessage(path="x", content="y", bogus=1)  # type: ignore[call-arg]
