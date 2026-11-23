"""Endpoint and credentials in ``litellm_extra`` are authorized atomically (#1961)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from aios.errors import ForbiddenError
from aios.models.agents import Agent
from aios.services.agents import _enforce_authority_delta


def _agent(litellm_extra: dict[str, Any], model: str = "openai/gpt-5") -> Agent:
    return cast(Agent, SimpleNamespace(model=model, litellm_extra=litellm_extra, skills=[]))


def _check(new: dict[str, Any], prior: Agent, editor: Agent, model: str = "openai/gpt-5") -> None:
    _enforce_authority_delta(model=model, litellm_extra=new, skills=[], prior=prior, editor=editor)


def test_target_secret_cannot_be_rerouted_to_editor_endpoint() -> None:
    prior = _agent({"api_base": "https://target.example", "api_key": "TARGET-SECRET"})
    editor = _agent({"api_base": "https://editor.example"})
    with pytest.raises(ForbiddenError):
        _check({"api_base": "https://editor.example", "api_key": "TARGET-SECRET"}, prior, editor)


def test_target_headers_cannot_be_rerouted_to_editor_endpoint() -> None:
    prior = _agent({"base_url": "https://target.example", "extra_headers": {"X-Auth": "T"}})
    editor = _agent({"base_url": "https://editor.example"})
    with pytest.raises(ForbiddenError):
        _check(
            {"base_url": "https://editor.example", "extra_headers": {"X-Auth": "T"}},
            prior,
            editor,
        )


def test_editor_secret_cannot_be_sent_to_target_endpoint() -> None:
    prior = _agent({"api_base": "https://target.example"})
    editor = _agent({"api_base": "https://editor.example", "api_key": "EDITOR-SECRET"})
    with pytest.raises(ForbiddenError):
        _check({"api_base": "https://target.example", "api_key": "EDITOR-SECRET"}, prior, editor)


def test_target_secret_cannot_follow_model_swap_to_editor_provider() -> None:
    prior = _agent({"api_key": "TARGET-SECRET"}, model="anthropic/claude-opus")
    editor = _agent({}, model="openai/gpt-5")
    with pytest.raises(ForbiddenError):
        _check({"api_key": "TARGET-SECRET"}, prior, editor, model="openai/gpt-5")


def test_whole_prior_or_whole_editor_config_is_admitted() -> None:
    prior = _agent({"api_base": "https://target.example", "api_key": "T", "temperature": 0.1})
    editor = _agent({"api_base": "https://editor.example", "api_key": "E"})
    _check(
        {"api_base": "https://target.example", "api_key": "T", "temperature": 0.9}, prior, editor
    )
    _check({"api_base": "https://editor.example", "api_key": "E"}, prior, editor)


def test_credential_free_model_switch_is_admitted() -> None:
    prior = _agent({}, model="anthropic/claude-opus")
    editor = _agent({"api_key": "E"}, model="openai/gpt-5")
    _check({}, prior, editor, model="openai/gpt-5")


def test_target_secret_cannot_reach_editor_endpoint_via_base_url_alias() -> None:
    # LiteLLM's ``completion`` lets ``base_url`` override ``api_base``: keeping the
    # target's ``api_base`` + key while adding the editor's ``base_url`` would ship
    # the target's key to the editor's endpoint.
    model = "openai/gpt-4o"
    prior = _agent(
        {"api_base": "https://target.example/v1", "api_key": "TARGET-SECRET"}, model=model
    )
    editor = _agent({"api_base": "https://editor.example/v1"}, model=model)
    with pytest.raises(ForbiddenError):
        _check(
            {
                "api_base": "https://target.example/v1",
                "api_key": "TARGET-SECRET",
                "base_url": "https://editor.example/v1",
            },
            prior,
            editor,
            model=model,
        )


def test_editor_secret_cannot_reach_target_endpoint_via_base_url_alias() -> None:
    prior = _agent({"base_url": "https://target.example/v1"})
    editor = _agent({"api_base": "https://editor.example/v1", "api_key": "EDITOR-SECRET"})
    with pytest.raises(ForbiddenError):
        _check(
            {
                "api_base": "https://editor.example/v1",
                "api_key": "EDITOR-SECRET",
                "base_url": "https://target.example/v1",
            },
            prior,
            editor,
        )


def test_credential_free_alias_cannot_name_an_unheld_endpoint() -> None:
    prior = _agent({"api_base": "https://target.example/v1"})
    editor = _agent({})
    with pytest.raises(ForbiddenError):
        _check(
            {"api_base": "https://target.example/v1", "base_url": "https://evil.example/v1"},
            prior,
            editor,
        )


def test_same_endpoint_under_either_alias_is_admitted() -> None:
    prior = _agent({"api_base": "https://target.example/v1", "api_key": "T"})
    editor = _agent({})
    _check({"base_url": "https://target.example/v1", "api_key": "T"}, prior, editor)
