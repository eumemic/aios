"""The recovery sweep must NOT fabricate an error result for a
confirmation-pending route-gated ``always_ask`` ``http_request`` (#1076).

The historical drift: ``sweep._was_dispatched`` applied only
``resolve_permission`` (the tool's BASE permission) and missed the arg-aware
route refinement (``http_request`` refines a specific route to ``always_ask``).
So a route-gated ``http_request`` parked awaiting a USER confirmation was
reported ``dispatched=True`` → routed to the ghost-error branch → the sweep
fabricated a synthetic error tool-result that KILLED the parked human-in-the-loop
confirmation. That is the exact outcome ``_is_client_result_pending``'s docstring
forbids.

Routing all three consumers through the single classifier
(``tool_disposition.classify_tool_call``) closes the gap by construction: the
refinement now exists in exactly one place. This module is the explicit test for
the forced behavior change — the sweep previously had NO coverage of the
route-gated-``always_ask`` case.
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.sweep import (
    _Candidate,
    _SweepAgentSurface,
    _was_dispatched,
    find_and_repair_ghosts,
)
from aios.models.agents import HttpPermissionPolicy, HttpRouteSpec, HttpServerSpec, ToolSpec
from tests.unit.conftest import fake_pool_yielding_conn

_NOW = dt.datetime.now(dt.UTC)


def _http_surface(policy: str) -> _SweepAgentSurface:
    route = HttpRouteSpec(
        path_pattern="/lights/*",
        enabled=True,
        permission_policy=HttpPermissionPolicy(type=policy),
        methods=None,
    )
    server = HttpServerSpec(name="hue", base_url="https://api.example.com/v1", routes=[route])
    return _SweepAgentSurface(tools=[ToolSpec(type="http_request")], http_servers=[server])


_HTTP_ARGS = '{"server_ref": "hue", "path": "/lights/1", "method": "GET"}'


def _candidate(arguments: Any) -> _Candidate:
    return _Candidate(
        session_id="sess_a",
        tool_call_id="tc_a",
        tool_name="http_request",
        created_at=_NOW,
        arguments=arguments,
    )


# ── direct projection: _was_dispatched ───────────────────────────────────────


class TestWasDispatchedRouteRefinement:
    def test_route_gated_always_ask_unconfirmed_not_dispatched(self) -> None:
        # THE FIX: a route-gated always_ask call awaiting the user is NOT
        # dispatched — it must be left alone, not ghost-errored. Previously this
        # returned True (route refinement was missing).
        surface = _http_surface("always_ask")
        assert (
            _was_dispatched(_candidate(_HTTP_ARGS), confirmed_ids=set(), surface=surface) is False
        )

    def test_route_gated_always_ask_confirmed_dispatched(self) -> None:
        # Once the user confirms, the call IS dispatched — a missing result is a
        # genuine ghost.
        surface = _http_surface("always_ask")
        assert (
            _was_dispatched(_candidate(_HTTP_ARGS), confirmed_ids={"tc_a"}, surface=surface) is True
        )

    def test_route_gated_always_allow_dispatched(self) -> None:
        surface = _http_surface("always_allow")
        assert _was_dispatched(_candidate(_HTTP_ARGS), confirmed_ids=set(), surface=surface) is True

    def test_unparseable_args_fall_through_to_base_dispatched(self) -> None:
        # No route refinement possible → base permission (None) → dispatched, so
        # the schema validator emits a typed error the model self-corrects from.
        surface = _http_surface("always_ask")
        assert _was_dispatched(_candidate("not json"), confirmed_ids=set(), surface=surface) is True


# ── end to end: the parked confirmation is NOT repaired ───────────────────────


async def test_route_gated_always_ask_not_ghost_repaired(monkeypatch: Any) -> None:
    """A parked route-gated always_ask http_request is left alone by the sweep.

    With no result, no in-flight task, and no confirmation lifecycle event, the
    OLD sweep would have fabricated an error tool-result (dispatched=True →
    ghost branch). The fixed sweep classifies it NEEDS_CONFIRM → not dispatched
    → ``append_tool_result`` is never called and nothing is repaired.
    """
    ghost_rows = [
        {
            "session_id": "sess_a",
            "created_at": _NOW,
            "data": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc_a",
                        "type": "function",
                        "function": {"name": "http_request", "arguments": _HTTP_ARGS},
                    }
                ],
            },
        },
    ]
    # The route gates /lights/* to always_ask; the call is NOT confirmed.
    route = {
        "path_pattern": "/lights/*",
        "enabled": True,
        "permission_policy": {"type": "always_ask"},
    }
    agent_rows = [
        {
            "session_id": "sess_a",
            "tools": [{"type": "http_request"}],
            "http_servers": [
                {"name": "hue", "base_url": "https://api.example.com/v1", "routes": [route]}
            ],
        }
    ]
    # Six fetches: GHOST_ASST_SQL, ERRORED_SESSIONS_SQL, ALL_RESULT_ROWS_SQL,
    # GHOST_LIFECYCLE_SQL (no confirmation), agent_rows, GHOST_SPAN_START_SQL.
    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=[ghost_rows, [], [], [], agent_rows, []])
    pool = fake_pool_yielding_conn(conn)

    append_mock = AsyncMock()
    load_account_mock = AsyncMock(return_value="acc_test")

    with (
        patch("aios.harness.sweep.sessions_service.append_tool_result", append_mock),
        patch("aios.harness.sweep.sessions_service.load_session_account_id", load_account_mock),
    ):
        repaired = await find_and_repair_ghosts(pool, InflightToolRegistry())

    assert repaired == [], "a parked route-gated always_ask call must NOT be ghost-repaired"
    append_mock.assert_not_called()


async def test_route_gated_always_ask_confirmed_is_ghost_repaired(monkeypatch: Any) -> None:
    """Once confirmed, a route-gated always_ask call with no result IS a ghost.

    The confirmation lifecycle event makes it dispatched → a missing result is a
    genuine ghost and the sweep repairs it. This pins the OTHER side of the
    behavior so the fix can't silently over-correct into never-repairing.
    """
    ghost_rows = [
        {
            "session_id": "sess_a",
            "created_at": _NOW,
            "data": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc_a",
                        "type": "function",
                        "function": {"name": "http_request", "arguments": _HTTP_ARGS},
                    }
                ],
            },
        },
    ]
    lifecycle_rows = [{"session_id": "sess_a", "tool_call_id": "tc_a"}]
    route = {
        "path_pattern": "/lights/*",
        "enabled": True,
        "permission_policy": {"type": "always_ask"},
    }
    agent_rows = [
        {
            "session_id": "sess_a",
            "tools": [{"type": "http_request"}],
            "http_servers": [
                {"name": "hue", "base_url": "https://api.example.com/v1", "routes": [route]}
            ],
        }
    ]
    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=[ghost_rows, [], [], lifecycle_rows, agent_rows, []])
    pool = fake_pool_yielding_conn(conn)

    append_mock = AsyncMock()
    load_account_mock = AsyncMock(return_value="acc_test")

    with (
        patch("aios.harness.sweep.sessions_service.append_tool_result", append_mock),
        patch("aios.harness.sweep.sessions_service.load_session_account_id", load_account_mock),
    ):
        repaired = await find_and_repair_ghosts(pool, InflightToolRegistry())

    assert repaired == [("sess_a", "tc_a")], "a confirmed-then-lost always_ask call is a ghost"
    append_mock.assert_awaited_once()
