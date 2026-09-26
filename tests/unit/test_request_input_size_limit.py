"""#2122: an oversized request input is refused loudly at the caller, never truncated.

A workflow ``agent()`` input (and the other request writers on the stimulate spine)
is delivered to the servicer whole or refused with an error naming the actual size
and the limit — before any child session exists. The limit is
``MAX_USER_MESSAGE_CHARS`` measured in characters of the *serialized* input.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.errors import PayloadTooLargeError
from aios.models.attenuation import Surface
from aios.models.sessions import MAX_USER_MESSAGE_CHARS
from aios.models.workflows import WfRun
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.services.sessions import (
    AskExistingSession,
    AskNewSession,
    request_content,
    request_input_too_large,
)
from aios.workflows import step as workflow_step
from aios.workflows.host_launcher import EmittedCapability
from aios.workflows.wf_script_host import _agent_error_from

_OVERSIZED = "x" * (MAX_USER_MESSAGE_CHARS + 1)


def test_request_content_serializes_non_strings() -> None:
    assert request_content("hello") == "hello"
    assert request_content({"task": "hi"}) == json.dumps({"task": "hi"})


def test_limit_admits_exact_bound_and_refuses_one_over() -> None:
    assert request_input_too_large("x" * MAX_USER_MESSAGE_CHARS) is None
    message = request_input_too_large(_OVERSIZED)
    assert message is not None
    assert f"{MAX_USER_MESSAGE_CHARS + 1:,}" in message  # the actual size
    assert f"{MAX_USER_MESSAGE_CHARS:,}" in message  # the limit
    assert "refused rather than truncated" in message


def test_limit_measures_serialized_json_not_the_raw_string() -> None:
    # JSON object syntax counts: a string that fits on its own can overflow once
    # wrapped as ``{"task": ...}``.
    body = "x" * (MAX_USER_MESSAGE_CHARS - 5)
    assert request_input_too_large(request_content(body)) is None
    assert request_input_too_large(request_content({"task": body})) is not None


def _run() -> WfRun:
    now = datetime.now(UTC)
    return WfRun(
        id="wfr_1",
        workflow_id="wf_1",
        account_id="acc_1",
        environment_id="env_1",
        script="async def main(input): return None",
        script_sha="sha",
        host_semantics_epoch=1,
        principal="operator",
        status="running",
        last_event_seq=0,
        created_at=now,
        updated_at=now,
        default_child_model="test/model",
    )


@pytest.mark.asyncio
async def test_agent_call_with_oversized_input_is_rejected_before_spawn() -> None:
    cap = EmittedCapability(
        capability_id="agent",
        call_key="sha:abc#0",
        spec={"agent_id": None, "input": {"task": _OVERSIZED}, "output_schema": None},
    )
    with (
        mock.patch.object(workflow_step, "_journal_agent_rejection", new=AsyncMock()) as journal,
        mock.patch.object(workflow_step, "create_child_session", new=AsyncMock()) as spawn,
        mock.patch.object(workflow_step, "defer_wake", new=AsyncMock()) as wake,
    ):
        result = await workflow_step._open_agent_capability(
            MagicMock(), MagicMock(), _run(), cap, agent_spawns=0, max_agent_calls=1000
        )
    assert result.rejected is True
    assert result.quota_exceeded is False
    spawn.assert_not_awaited()  # no child is ever created
    wake.assert_not_awaited()
    journal.assert_awaited_once()
    assert journal.await_args is not None
    kwargs = journal.await_args.kwargs
    assert kwargs["call_key"] == "sha:abc#0"
    assert kwargs["kind"] == "input_too_large"
    assert kwargs["message"].startswith("agent() request input is")
    assert f"{MAX_USER_MESSAGE_CHARS:,}-character limit" in kwargs["message"]

    # And it surfaces at the author's await as a catchable, kind-tagged AgentError.
    err = _agent_error_from({"kind": kwargs["kind"], "message": kwargs["message"]})
    assert err.kind == "input_too_large"
    assert "refused rather than truncated" in str(err)


@pytest.mark.asyncio
async def test_create_child_session_refuses_oversized_input_before_any_write() -> None:
    pool = MagicMock()
    stim = AskNewSession(
        session_id="sess_child",
        agent_id=None,
        environment_id="env_1",
        agent_version=None,
        model="test/model",
        parent_run_id="wfr_1",
        surface=Surface(tools=[], mcp_servers=[], http_servers=[], ssh_servers=[]),
        vault_ids=[],
        request_id="sha:abc#0",
        input=_OVERSIZED,
        depth=1,
    )
    with pytest.raises(PayloadTooLargeError) as exc_info:
        await sessions_service.create_child_session(pool, stim, account_id="acc_1")
    assert exc_info.value.detail == {
        "max_chars": MAX_USER_MESSAGE_CHARS,
        "got_chars": MAX_USER_MESSAGE_CHARS + 1,
    }
    pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_existing_session_ask_refuses_oversized_input_before_any_write() -> None:
    pool = MagicMock()
    stim = AskExistingSession(
        session=MagicMock(),
        caller={"kind": "api", "id": "acc_1"},
        request_id="req_1",
        input=_OVERSIZED,
        output_schema=None,
    )
    with mock.patch.object(agents_service, "load_for_session", new=AsyncMock()) as load:
        with pytest.raises(PayloadTooLargeError):
            await sessions_service._stimulate_existing_ask(pool, stim, account_id="acc_1")
        load.assert_not_awaited()
    pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_invoke_agent_refuses_oversized_input_before_creating_servicer() -> None:
    with mock.patch.object(sessions_service, "create_session", new=AsyncMock()) as create:
        with pytest.raises(PayloadTooLargeError):
            await sessions_service.invoke(
                MagicMock(),
                account_id="acc_1",
                target_kind="agent",
                target="agt_1",
                input=_OVERSIZED,
                environment_id="env_1",
            )
        create.assert_not_awaited()
