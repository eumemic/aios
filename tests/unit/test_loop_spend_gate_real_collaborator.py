"""Regression: the spend-admission collaborator must NOT self-disable to ``(0, None)``.

This module deliberately disables the conftest ``_unit_spend_state_ungated``
autouse fixture so both the service collaborator and the harness gate run for
real over a MagicMock pool — exactly the path the old ``cast(Any, ...) +
isinstance`` escape hatch hid.

On master the escape hatch forced ``(0, None)`` whenever a value read from the
pool failed the runtime type-recheck (e.g. a MagicMock ``fetchval`` result),
silently uncapping spend and skipping the gate. These tests pin the corrected
behavior:

* :func:`test_real_collaborator_does_not_self_disable_over_magicmock_pool`
  asserts the service function returns the *real* scalar tuple it read — NOT the
  fail-open ``(0, None)``. This is the master-failing assertion: on master the
  escape hatch returns ``(0, None)``; after the recheck is deleted it returns
  the genuine ``(spent, limit)``.

* :func:`test_real_gate_trips_when_subtree_over_finite_limit` drives the whole
  ``_run_session_step_body`` prologue over a MagicMock pool with the real
  collaborator (no loop-site patch) and asserts the hard ceiling latches the
  session and refuses the model call once the rolled-up subtree spend breaches
  a finite limit.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from aios.harness.loop import _run_session_step_body, _StepResult
from aios.harness.window import WindowedEvents
from aios.services import accounts as accounts_service

from .conftest import fake_pool_yielding_conn


@pytest.fixture(autouse=True)
def _unit_spend_state_ungated() -> None:
    """Disable the conftest autouse spend-state stub for this module.

    Overriding the sibling fixture by name (pytest resolves the nearest
    definition) lets the REAL ``get_account_subtree_spend_state`` run against
    the MagicMock pool, which is the whole point of the regression.
    """
    return None


async def test_real_collaborator_does_not_self_disable_over_magicmock_pool() -> None:
    """The service collaborator returns the real scalar it read, not fail-open ``(0, None)``.

    This exercises the exact input that tripped the old escape hatch: a MagicMock
    pool whose limit ``fetchval`` returns a bare ``MagicMock`` (an untyped mock
    value that the deleted ``isinstance(cast(Any, limit), int | float | None)``
    arm rejected). The subtree spent meter still comes back as a real int
    (``queries.get_account_subtree_spent_microusd`` wraps its read in ``int(...)``).

    On master this FAILS: the recheck saw the non-scalar ``limit`` and forced the
    ungated ``(0, None)`` sentinel — the silent fail-open uncap. After the recheck
    is deleted the collaborator is total over its real type domain and surfaces
    the genuine ``(spent, limit)`` pair it read (no laundering, no sentinel).
    """
    sentinel_limit = MagicMock()
    conn = MagicMock()
    # 1st fetchval → subtree spent query (wrapped in int(...) → real int);
    # 2nd fetchval → effective-limit query (returned raw → a bare MagicMock).
    conn.fetchval = AsyncMock(side_effect=[2_000_000, sentinel_limit])
    pool = fake_pool_yielding_conn(conn)

    spent, limit = await accounts_service.get_account_subtree_spend_state(pool, "acc_x")

    # No fail-open sentinel: the honest function returns what it actually read.
    assert (spent, limit) != (0, None)
    assert spent == 2_000_000
    assert limit is sentinel_limit


async def test_real_gate_trips_when_subtree_over_finite_limit() -> None:
    """The real (unpatched-at-loop-site) gate latches when the subtree rollup
    the real collaborator reads from the pool breaches a finite effective limit.

    ``conn.fetchval`` returns, in call order:
      1. the subtree spent meter (2_000_000 µ$ = $2.00), then
      2. the effective spend limit ($1.00).
    The real ``get_account_subtree_spend_state`` returns ``(2_000_000, 1.0)``;
    ``$2.00 >= $1.00`` so the gate must latch the session (spend_cap_exceeded)
    and refuse the model call BEFORE context build.
    """
    conn = MagicMock()
    conn.fetchval = AsyncMock(side_effect=[2_000_000, 1.0])
    pool = fake_pool_yielding_conn(conn)

    inflight_tool_registry = MagicMock()
    inflight_tool_registry.in_flight_tool_call_ids.return_value = set()
    session = SimpleNamespace(
        id="sess_x",
        agent_id="agt_x",
        agent_version=None,
        focal_channel=None,
        origin="foreground",
        parent_run_id=None,
    )
    agent = SimpleNamespace(
        model="openrouter/x",
        tools=[],
        mcp_servers=[],
        http_servers=[],
        skills=[],
        system="sys",
        litellm_extra={},
        window_min=1000,
        window_max=10000,
        preempt_policy="wait",
    )
    append_event = AsyncMock(return_value=SimpleNamespace(id="ev"))
    with (
        patch(
            "aios.harness.loop.find_sessions_needing_inference", AsyncMock(return_value={"sess_x"})
        ),
        # Skip the fast-path sweep guard so the yielded conn's ``fetchval`` is
        # consumed ONLY by the real spend-state collaborator below.
        patch("aios.harness.loop.session_has_pending_work", AsyncMock(return_value=True)),
        patch(
            "aios.harness.loop.sessions_service.get_session_basic", AsyncMock(return_value=session)
        ),
        patch("aios.harness.loop.agents_service.load_for_session", AsyncMock(return_value=agent)),
        patch("aios.services.channels.list_session_channels", AsyncMock(return_value=[])),
        patch("aios.harness.loop.refresh_session_mount_state", AsyncMock(return_value=[])),
        patch("aios.harness.loop._list_session_github_repo_echoes", AsyncMock(return_value=[])),
        patch("aios.harness.loop.compute_step_prelude", AsyncMock(return_value=SimpleNamespace())),
        patch("aios.harness.loop.prelude_overhead_local", return_value=0),
        patch(
            "aios.harness.loop.sessions_service.read_windowed_events",
            AsyncMock(return_value=WindowedEvents(events=[], omission=None)),
        ),
        patch("aios.harness.loop._dispatch_confirmed_tools", AsyncMock(return_value=[])),
        # NOTE: get_account_subtree_spend_state is intentionally NOT patched —
        # the real collaborator runs against the pool above.
        patch("aios.harness.loop.sessions_service.append_event", append_event),
        patch("aios.harness.loop.fail_all_open_requests", AsyncMock()) as fail_open,
        patch(
            "aios.harness.loop.sessions_service.set_session_stop_reason", AsyncMock()
        ) as set_stop,
        patch("aios.harness.loop.compose_step_context", AsyncMock()) as compose,
    ):
        result = await _run_session_step_body(
            pool, inflight_tool_registry, "sess_x", cause="message", account_id="acc_x"
        )

    assert result == _StepResult()
    # Gate fired BEFORE context build / the model call.
    compose.assert_not_awaited()
    fail_open.assert_awaited_once_with(
        ANY, "sess_x", account_id="acc_x", error={"kind": "spend_cap_exceeded"}
    )
    assert set_stop.await_args is not None
    stop_reason = set_stop.await_args.args[2]
    assert stop_reason["type"] == "error"
    span_payloads = [c.args[3] for c in append_event.call_args_list if c.args[2] == "span"]
    cap_spans = [p for p in span_payloads if p.get("event") == "spend_cap_exceeded"]
    assert cap_spans, "expected a spend_cap_exceeded span"
    assert cap_spans[0]["spent_microusd"] == 2_000_000
