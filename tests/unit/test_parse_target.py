"""Pure-unit tests for routing-rule target parsing."""

from __future__ import annotations

import pytest

from aios.services.channels import (
    AgentTarget,
    SessionTarget,
    parse_target,
)


class TestParseAgentTarget:
    def test_agent_no_version(self) -> None:
        t = parse_target("agent:agent_01HQR2K7VXBZ9MNPL3WYCT8F")
        assert isinstance(t, AgentTarget)
        assert t.agent_id == "agent_01HQR2K7VXBZ9MNPL3WYCT8F"
        assert t.agent_version is None

    def test_agent_with_version(self) -> None:
        t = parse_target("agent:agent_01HQR2K7VXBZ9MNPL3WYCT8F@7")
        assert isinstance(t, AgentTarget)
        assert t.agent_id == "agent_01HQR2K7VXBZ9MNPL3WYCT8F"
        assert t.agent_version == 7

    def test_agent_version_zero(self) -> None:
        t = parse_target("agent:abc@0")
        assert isinstance(t, AgentTarget)
        assert t.agent_version == 0

    def test_agent_id_with_underscores_preserved(self) -> None:
        t = parse_target("agent:agent_abc_def")
        assert isinstance(t, AgentTarget)
        assert t.agent_id == "agent_abc_def"


class TestParseSessionTarget:
    def test_session(self) -> None:
        t = parse_target("session:sess_01HQR2K7VXBZ9MNPL3WYCT8F")
        assert isinstance(t, SessionTarget)
        assert t.session_id == "sess_01HQR2K7VXBZ9MNPL3WYCT8F"


class TestParseInvalidTargets:
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "agent:",
            "session:",
            "agent:abc@",
            "agent:@7",
            "agent:abc@notanint",
            "agent:abc@1.0",
            "foo:abc",
            "abc",
            "AGENT:abc",  # case-sensitive
        ],
    )
    def test_rejects(self, bad: str) -> None:
        with pytest.raises(ValueError, match="invalid target"):
            parse_target(bad)
