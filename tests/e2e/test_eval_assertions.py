"""E2E tests for the structural eval assertion helpers (issue #1350).

These exercise the read-only helpers appended to ``tests/e2e/harness.py``
(``requested_tool_calls`` / ``result_for`` / ``paired_calls`` /
``assert_tool_called`` / ``assert_tool_result_contains`` /
``terminal_lifecycle`` / ``assert_ended``) that assert the *shape* of a
turn over the captured event log.

Deterministic: scripted model, no real LLM. The round-trip uses the
built-in ``bash``/``read`` tools, so it runs in the full (Docker) tier
alongside the other scripted-tool round-trips in ``test_step_model.py``.
The pure-function semantics of ``result_for``/``paired_calls`` on an
unresolved (in-flight / absent) call are pinned with synthetic events so
the ``None``-means-unresolved invariant is codified independently of tool
dispatch internals.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from aios.models.events import Event
from tests.conftest import needs_docker
from tests.e2e.harness import (
    Harness,
    assert_ended,
    assert_tool_called,
    assert_tool_result_contains,
    assistant,
    bash,
    paired_calls,
    requested_tool_calls,
    result_for,
    terminal_lifecycle,
    tool_call,
)

pytestmark = pytest.mark.docker


def _msg_event(seq: int, data: dict[str, Any]) -> Event:
    """Build a synthetic message Event for pure-function helper tests."""
    return Event(
        id=f"evt_{seq}",
        session_id="ses_synthetic",
        seq=seq,
        kind="message",
        data=data,
        created_at=datetime.now(UTC),
    )


def test_unresolved_call_yields_none() -> None:
    """``result_for``/``paired_calls`` report an explicit ``None`` for a call
    with no matching tool result — never a silent skip. Pure-function check on
    synthetic events so the invariant is pinned independently of dispatch.
    """
    events = [
        _msg_event(0, {"role": "user", "content": "go"}),
        _msg_event(
            1,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    tool_call("bash", {"command": "echo hi"}, call_id="call_1"),
                    tool_call("read", {"path": "/x"}, call_id="call_2"),
                ],
            },
        ),
        # Only call_1 ever resolved; call_2 is genuinely in-flight / absent.
        _msg_event(2, {"role": "tool", "tool_call_id": "call_1", "content": "hi"}),
    ]

    assert result_for(events, "call_1") is not None
    assert result_for(events, "call_2") is None

    pairs = paired_calls(events)
    assert len(pairs) == 2
    assert pairs[0][1] is not None  # bash resolved
    assert pairs[1][1] is None  # read unresolved -> explicit None, not skipped


@needs_docker  # built-in bash/read run in the testcontainer sandbox
class TestStructuralAssertions:
    async def test_structural_assertions_over_scripted_round_trip(self, harness: Harness) -> None:
        """Every helper, against a scripted two-tool (bash then read) round-trip."""
        harness.script_model(
            [
                assistant(
                    tool_calls=[
                        bash("echo hi", call_id="call_1"),
                        tool_call("read", {"path": "/x"}, call_id="call_2"),
                    ]
                ),
                assistant("done"),
            ]
        )
        session = await harness.start("go", tools=["bash", "read"])
        await harness.run_until_idle(session.id)
        events = await harness.all_events(session.id)

        # 3. requested_tool_calls: two entries, scripted ids/names, in log order.
        reqs = requested_tool_calls(events)
        assert len(reqs) == 2
        assert reqs[0][0] == "call_1"
        assert reqs[0][1] == "bash"
        assert reqs[1][0] == "call_2"
        assert reqs[1][1] == "read"

        # 4. paired_calls: each call joined to its result by tool_call_id; both
        # resolved (no None) on the canonical run.
        pairs = paired_calls(events)
        assert len(pairs) == 2
        assert all(result is not None for _, result in pairs)
        assert result_for(events, "call_1") is not None
        assert result_for(events, "call_2") is not None

        # 5. assert_tool_called: hit returns the call dict; miss raises.
        called = assert_tool_called(events, "bash")
        assert called["function"]["name"] == "bash"
        with pytest.raises(AssertionError):
            assert_tool_called(events, "nonexistent")

        # 6. assert_tool_result_contains: matching needle passes; wrong raises.
        assert_tool_result_contains(events, "bash", "hi")
        with pytest.raises(AssertionError):
            assert_tool_result_contains(events, "bash", "this-is-not-in-the-output")

        # 7. terminal_lifecycle / assert_ended over all_events(); the
        # message-only read raises (guards the events()/all_events() rule).
        term = terminal_lifecycle(events)
        assert term is not None
        assert term.get("event") == "turn_ended"
        assert_ended(events)

        msg_only = await harness.events(session.id)
        with pytest.raises(AssertionError):
            assert_ended(msg_only)
