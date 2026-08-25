"""Read tolerance for MCP server URLs persisted before current write rules.

Tolerating an already-persisted policy-blocked URL keeps the read path from
500ing (#2062), but "I tolerated this" must never be indistinguishable from
"this is fine": the hydrated object itself has to carry the distinction, not
just a server-side log line. These tests pin both halves — the marker fires on
the tolerated row, and it stays OFF for an ordinary row (the positive control
that stops a build which marks everything from passing).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from aios.db.queries.agents import _row_to_agent
from aios.models import target_urls
from aios.models.agents import McpServerSpec

_BLOCKED_URL = "http://localhost:8091/mcp"
_VALID_URL = "https://mcp.example.com/mcp"


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every DNS name to a public IP, so ``_VALID_URL`` validates hermetically."""
    monkeypatch.setattr(target_urls, "_resolve_host_ips", lambda host: ["93.184.216.34"])


def _agent_row(servers: list[dict[str, Any]]) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": "agt_legacy_mcp",
        "version": 1,
        "name": "legacy",
        "model": "openai/gpt-4o",
        "system": "",
        "tools": [],
        "skills": [],
        # Deliberately bypass current request validation, as an old JSONB row does.
        "mcp_servers": servers,
        "http_servers": [],
        "description": None,
        "metadata": {},
        "litellm_extra": {},
        "window_min": 1,
        "window_max": 10,
        "preempt_policy": "wait",
        "output_style": "default",
        "created_by_type": None,
        "created_by_ref": None,
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
    }


def test_persisted_loopback_mcp_server_hydrates_without_weakening_writes() -> None:
    legacy_server = {"type": "url", "name": "signal", "url": _BLOCKED_URL}

    agent = _row_to_agent(_agent_row([legacy_server]))

    assert agent.mcp_servers[0].url == _BLOCKED_URL
    with pytest.raises(ValidationError, match="private, internal, or runtime-local"):
        McpServerSpec.model_validate(legacy_server)


def test_tolerated_blocked_server_is_marked() -> None:
    """THE DISTINCTION: a tolerated policy-blocked server is not returned as ordinary.

    Without this, a caller (and every downstream consumer of this
    authority-bearing object) cannot tell tolerated-unsafe from currently-valid.
    """
    agent = _row_to_agent(_agent_row([{"type": "url", "name": "signal", "url": _BLOCKED_URL}]))

    assert agent.mcp_servers[0].url_blocked_by_policy is True


def test_valid_server_hydrates_unmarked(public_dns: None) -> None:
    """POSITIVE CONTROL: an ordinary server still hydrates, and is NOT marked.

    Without this assertion, ``test_tolerated_blocked_server_is_marked`` would
    pass on a build that marks every server.
    """
    agent = _row_to_agent(_agent_row([{"type": "url", "name": "ok", "url": _VALID_URL}]))

    assert agent.mcp_servers[0].url == _VALID_URL
    assert agent.mcp_servers[0].url_blocked_by_policy is False


def test_mixed_row_marks_only_the_blocked_member(public_dns: None) -> None:
    """Per-entry, not per-row: one legacy member must not taint its valid siblings."""
    agent = _row_to_agent(
        _agent_row(
            [
                {"type": "url", "name": "ok", "url": _VALID_URL},
                {"type": "url", "name": "signal", "url": _BLOCKED_URL},
            ]
        )
    )

    assert [s.url_blocked_by_policy for s in agent.mcp_servers] == [False, True]


def test_marker_is_not_persisted_and_not_a_wire_field(public_dns: None) -> None:
    """The marker is derived read state — it must never round-trip into storage.

    Every DB write serializes specs with ``model_dump()``; if the marker landed
    in the JSONB it would go stale the moment policy changed, which is the same
    lie in a new place.
    """
    agent = _row_to_agent(_agent_row([{"type": "url", "name": "signal", "url": _BLOCKED_URL}]))

    assert "url_blocked_by_policy" not in agent.mcp_servers[0].model_dump()
    assert "url_blocked_by_policy" not in McpServerSpec.model_json_schema()["properties"]


def test_ingress_validation_still_rejects_blocked_urls(public_dns: None) -> None:
    """Tolerance is READ-only: the write boundary is unchanged."""
    with pytest.raises(ValidationError, match="private, internal, or runtime-local"):
        McpServerSpec.model_validate({"type": "url", "name": "signal", "url": _BLOCKED_URL})

    ok = McpServerSpec.model_validate({"type": "url", "name": "ok", "url": _VALID_URL})
    assert ok.url_blocked_by_policy is False


def test_unrelated_validation_errors_still_fail_loudly() -> None:
    """Tolerance is scoped to the one retroactively-applied rule.

    A structurally corrupt row (missing/wrong-typed field) must still raise —
    swallowing it would hide real corruption behind a 200.
    """
    with pytest.raises(ValidationError):
        McpServerSpec.model_validate_persisted({"type": "url", "name": "signal"})
    with pytest.raises(ValidationError):
        McpServerSpec.model_validate_persisted(
            {"type": "url", "name": "signal", "url": _BLOCKED_URL, "include_instructions": "nope"}
        )
