from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from aios.errors import ForbiddenError
from aios.models.agents import Agent, ToolSpec
from aios.models.skills import AgentSkillRef
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
