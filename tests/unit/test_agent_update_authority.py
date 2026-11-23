"""Agent update authorizes the semantic authority delta, not the merged whole (#1961)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from aios.errors import ConflictError, ForbiddenError
from aios.models.agents import (
    Agent,
    AgentBinding,
    HttpRouteSpec,
    HttpServerSpec,
    McpServerSpec,
    StepSurface,
    ToolSpec,
)
from aios.models.skills import AgentSkillRef
from aios.services import agents as agents_service
from aios.services.agents import _enforce_authority_delta, _field_paths


def _agent(**overrides: object) -> Agent:
    values: dict[str, object] = {
        "model": "openai/gpt-5",
        "litellm_extra": {},
        "skills": [],
    }
    values.update(overrides)
    return cast(Agent, SimpleNamespace(**values))


def test_authority_delta_allows_preserving_authority_the_editor_lacks() -> None:
    prior = _agent(
        model="anthropic/claude-opus",
        litellm_extra={"api_base": "https://target.example"},
        skills=[AgentSkillRef(skill_id="01TARGETSKILL")],
    )
    editor = _agent()

    _enforce_authority_delta(
        model=prior.model,
        litellm_extra=prior.litellm_extra,
        skills=prior.skills,
        prior=prior,
        editor=editor,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other/provider-model"),
        ("litellm_extra", {"base_url": "https://untrusted.example"}),
        ("skills", [AgentSkillRef(skill_id="01NEWSKILL")]),
    ],
)
def test_authority_delta_rejects_authority_absent_from_editor_and_target(
    field: str, value: object
) -> None:
    prior = _agent()
    editor = _agent()
    model = cast(str, value) if field == "model" else prior.model
    litellm_extra = (
        cast(dict[str, object], value) if field == "litellm_extra" else prior.litellm_extra
    )
    skills = cast(list[AgentSkillRef], value) if field == "skills" else prior.skills

    with pytest.raises(ForbiddenError) as exc:
        _enforce_authority_delta(
            model=model,
            litellm_extra=litellm_extra,
            skills=skills,
            prior=prior,
            editor=editor,
        )

    assert field in exc.value.detail["exceeds"]


def test_changed_field_paths_report_nested_metadata_without_values() -> None:
    old = {"metadata": {"reasoning_effort": "low"}, "tools": [ToolSpec(type="read")]}
    new = {"metadata": {"reasoning_effort": "max"}, "tools": [ToolSpec(type="read")]}

    assert _field_paths(old, new) == ["metadata.reasoning_effort"]


# ─── service path: the exact CEO → kedalion-ultron shape (#1961) ──────────────

_DT = datetime(2026, 1, 1, tzinfo=UTC)
_SECRET = "sk-live-DO-NOT-LOG"

_KEDALION_MCP = [
    McpServerSpec(name="github-actions", url="https://mcp.example.test/actions"),
    McpServerSpec(name="github-discussions", url="https://mcp.example.test/discussions"),
]
_KEDALION_HTTP = [
    HttpServerSpec(
        name="site",
        base_url="https://site.example.test",
        routes=[HttpRouteSpec(path_pattern="/api/**", methods=["GET", "POST"])],
    )
]
_KEDALION_TOOLS = [
    ToolSpec(type="bash"),
    ToolSpec(type="mcp_toolset", mcp_server_name="github-actions"),
    ToolSpec(type="mcp_toolset", mcp_server_name="github-discussions"),
    ToolSpec(type="http_request"),
]


def _kedalion(**over: Any) -> Agent:
    base: dict[str, Any] = dict(
        id="agt_kedalion",
        version=7,
        name="kedalion-ultron",
        model="anthropic/claude-opus",
        system="you are kedalion",
        tools=_KEDALION_TOOLS,
        skills=[AgentSkillRef(skill_id="01KEDSKILL")],
        mcp_servers=_KEDALION_MCP,
        http_servers=_KEDALION_HTTP,
        description="d",
        metadata={"reasoning_effort": "high"},
        litellm_extra={"api_key": _SECRET},
        window_min=1000,
        window_max=100000,
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Agent(**base)


_CEO = StepSurface(
    tools=[ToolSpec(type="read"), ToolSpec(type="update_agent"), ToolSpec(type="get_agent")],
    mcp_servers=[],
    http_servers=[],
    model="anthropic/claude-sonnet",
    system="ceo",
    window_min=1000,
    window_max=100000,
    preempt_policy="wait",
    binding=AgentBinding(agent_id="agt_ceo", version=3),
)


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the DB: a stored kedalion-ultron, a CEO editor session, and a writer that
    honors the optimistic-concurrency token and applies the patch like the real query."""
    state: dict[str, Any] = {"agent": _kedalion(), "writes": 0}

    async def get_agent(_pool: Any, _agent_id: str, *, account_id: str) -> Agent:
        return cast(Agent, state["agent"])

    async def update_agent(_conn: Any, _agent_id: str, **kw: Any) -> Agent:
        current: Agent = state["agent"]
        if kw.pop("expected_version") != current.version:
            raise ConflictError("version mismatch")
        kw.pop("account_id")
        kw.pop("skills_json")
        patch = {k: v for k, v in kw.items() if v is not None}
        state["agent"] = current.model_copy(update={**patch, "version": current.version + 1})
        state["writes"] += 1
        return cast(Agent, state["agent"])

    class _Pool:
        def acquire(self) -> Any:
            class _Ctx:
                async def __aenter__(self) -> object:
                    return object()

                async def __aexit__(self, *a: object) -> None:
                    return None

            return _Ctx()

    monkeypatch.setattr("aios.services.agents.get_agent", get_agent)
    monkeypatch.setattr("aios.services.agents.queries.update_agent", update_agent)
    monkeypatch.setattr("aios.services.agents.load_for_session", AsyncMock(return_value=_CEO))
    monkeypatch.setattr(
        "aios.services.sessions.get_session_basic", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(
        "aios.services.agents.skills_service.resolve_skill_refs", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        "aios.services.agents.skills_service.serialize_skills_for_snapshot",
        lambda *a, **k: "[]",
    )
    state["pool"] = _Pool()
    return state


async def _update(stored: dict[str, Any], **kw: Any) -> Agent:
    kw.setdefault("expected_version", stored["agent"].version)
    return await agents_service.update_agent(
        stored["pool"],
        "agt_kedalion",
        account_id="acc_x",
        editor_session_id="ses_ceo",
        **kw,
    )


async def test_ceo_metadata_only_update_on_kedalion_succeeds(stored: dict[str, Any]) -> None:
    before = stored["agent"]
    with capture_logs() as logs:
        updated = await _update(stored, metadata={"reasoning_effort": "max"})

    assert updated.metadata == {"reasoning_effort": "max"}
    assert updated.version == before.version + 1
    for field in agents_service.AUTHORITY_FIELDS:
        assert getattr(updated, field) == getattr(before, field), field
        assert updated.model_dump(mode="json")[field] == before.model_dump(mode="json")[field]

    (event,) = [e for e in logs if e["event"] == "agent_updated"]
    assert event["actor_session_id"] == "ses_ceo"
    assert event["old_version"] == before.version
    assert event["new_version"] == before.version + 1
    assert event["changed_paths"] == ["metadata.reasoning_effort"]
    assert event["changed_authority_fields"] == []
    assert set(event["preserved_authority_fields"]) == {
        "tools",
        "mcp_servers",
        "http_servers",
        "ssh_servers",
        "model",
        "litellm_extra",
        "skills",
    }
    # No values (let alone secrets) in the audit record.
    assert _SECRET not in repr(logs)
    assert "max" not in repr(event)


async def test_restating_preserved_surface_in_reordered_form_is_admitted(
    stored: dict[str, Any],
) -> None:
    reordered_http = [
        _KEDALION_HTTP[0].model_copy(
            update={"routes": [HttpRouteSpec(path_pattern="/api/**", methods=["POST", "GET"])]}
        )
    ]
    updated = await _update(
        stored,
        tools=list(reversed(_KEDALION_TOOLS)),
        mcp_servers=list(reversed(_KEDALION_MCP)),
        http_servers=reordered_http,
        description="restated",
    )
    assert updated.description == "restated"


@pytest.mark.parametrize(
    ("patch", "section"),
    [
        (
            {
                "http_servers": [
                    _KEDALION_HTTP[0].model_copy(
                        update={
                            "routes": [
                                *_KEDALION_HTTP[0].routes,
                                HttpRouteSpec(path_pattern="/admin/**"),
                            ]
                        }
                    )
                ]
            },
            "http_servers",
        ),
        (
            {
                "http_servers": [
                    _KEDALION_HTTP[0].model_copy(
                        update={"routes": [HttpRouteSpec(path_pattern="/api/**")]}
                    )
                ]
            },
            "http_servers",
        ),
        (
            {
                "http_servers": [
                    *_KEDALION_HTTP,
                    HttpServerSpec(name="evil", base_url="https://evil.example.test"),
                ]
            },
            "http_servers",
        ),
        (
            {
                "mcp_servers": [
                    *_KEDALION_MCP,
                    McpServerSpec(name="new-mcp", url="https://mcp.example.test/new"),
                ]
            },
            "mcp_servers",
        ),
        (
            {
                "mcp_servers": [
                    McpServerSpec(name="github-actions", url="https://evil.example.test/mcp"),
                    _KEDALION_MCP[1],
                ]
            },
            "mcp_servers",
        ),
        ({"tools": [*_KEDALION_TOOLS, ToolSpec(type="write")]}, "tools"),
        ({"model": "openrouter/other-model"}, "model"),
        ({"litellm_extra": {"api_key": _SECRET, "api_base": "https://evil.test"}}, "litellm_extra"),
        ({"litellm_extra": {"api_key": "sk-attacker"}}, "litellm_extra"),
        (
            {"skills": [AgentSkillRef(skill_id="01KEDSKILL"), AgentSkillRef(skill_id="01NEW")]},
            "skills",
        ),
    ],
)
async def test_ceo_widening_kedalion_authority_is_rejected(
    stored: dict[str, Any], patch: dict[str, Any], section: str
) -> None:
    before = stored["agent"]
    with pytest.raises(ForbiddenError) as exc:
        await _update(stored, metadata={"reasoning_effort": "max"}, **patch)
    assert exc.value.detail is not None
    assert section in exc.value.detail["exceeds"]
    assert _SECRET not in repr(exc.value.detail)
    assert stored["agent"] is before and stored["writes"] == 0


async def test_stale_concurrency_token_is_rejected_before_any_write(
    stored: dict[str, Any],
) -> None:
    stale = stored["agent"].version
    await _update(stored, expected_version=stale, metadata={"reasoning_effort": "max"})
    with pytest.raises(ConflictError):
        await _update(stored, expected_version=stale, metadata={"reasoning_effort": "low"})
    assert stored["agent"].metadata == {"reasoning_effort": "max"}
    assert stored["writes"] == 1
