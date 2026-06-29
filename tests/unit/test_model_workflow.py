"""Unit tests for the run-side park/harvest plumbing of the ``workflow:`` model
binding (issue #1634).

These pin the pure read-side projection — ``take_pending_harvest`` reads the
latest park + its matching harvest from the event log and projects them into a
:class:`HarvestedInference` carrying the watermark sealed at park. The park/harvest
DB writes themselves are exercised by the harness integration tests; here we stub
the focused queries to pin the pairing + supersession logic without a database.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.harness import model_workflow
from aios.harness.model_workflow import HarvestedInference, ParkState, take_pending_harvest


class _FakeConn:
    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakePool:
    def acquire(self) -> _FakeConn:
        return _FakeConn()


@pytest.fixture
def patched_queries(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the two focused queries; tests set ``state['park']`` / ``state['harvest']``."""
    state: dict[str, Any] = {"park": None, "harvest": None}

    async def _find_park(conn: object, session_id: str, *, account_id: str) -> Any:
        return state["park"]

    async def _find_harvest(conn: object, session_id: str, *, run_id: str, account_id: str) -> Any:
        harvest = state["harvest"]
        if harvest is None:
            return None
        # Mirror the keyed read: only return when the run id matches.
        return harvest if harvest.get("run_id") == run_id else None

    import aios.db.queries as queries

    monkeypatch.setattr(queries, "find_latest_model_workflow_park", _find_park)
    monkeypatch.setattr(queries, "find_model_workflow_harvest", _find_harvest)
    return state


@pytest.mark.asyncio
async def test_no_park_returns_no_park_state(patched_queries: dict[str, Any]) -> None:
    # No open park → the caller launches a fresh awaited run (the park branch).
    assert await take_pending_harvest(_FakePool(), "s1", account_id="a1") is ParkState.NO_PARK


@pytest.mark.asyncio
async def test_park_without_harvest_returns_park_pending(patched_queries: dict[str, Any]) -> None:
    # Parked, run not resolved yet → the step ends owing the message again WITHOUT
    # re-parking (no new run). Distinguishing this from "no park" is the multi-billing
    # fix: collapsing both into the same value re-dispatched a run on every sweep tick.
    patched_queries["park"] = {"run_id": "run_1", "reacting_to": 7}
    assert await take_pending_harvest(_FakePool(), "s1", account_id="a1") is ParkState.PARK_PENDING


@pytest.mark.asyncio
async def test_resolved_harvest_projects_with_sealed_watermark(
    patched_queries: dict[str, Any],
) -> None:
    patched_queries["park"] = {"run_id": "run_1", "reacting_to": 7}
    patched_queries["harvest"] = {
        "run_id": "run_1",
        "outcome": "ok",
        "output": {"content": "answer"},
        "error": None,
    }
    result = await take_pending_harvest(_FakePool(), "s1", account_id="a1")
    assert result == HarvestedInference(
        outcome="ok",
        output={"content": "answer"},
        error=None,
        reacting_to=7,  # sealed at park, NOT recomputed
        run_id="run_1",
    )


@pytest.mark.asyncio
async def test_harvest_for_other_run_does_not_pair(patched_queries: dict[str, Any]) -> None:
    # A harvest exists but for a stale run id (e.g. a superseded park) → no pairing;
    # the open park is still unresolved, so the caller must NOT re-park.
    patched_queries["park"] = {"run_id": "run_2", "reacting_to": 3}
    patched_queries["harvest"] = {
        "run_id": "run_1",
        "outcome": "ok",
        "output": {"content": "x"},
        "error": None,
    }
    assert await take_pending_harvest(_FakePool(), "s1", account_id="a1") is ParkState.PARK_PENDING


@pytest.mark.asyncio
async def test_errored_outcome_is_carried_through(patched_queries: dict[str, Any]) -> None:
    patched_queries["park"] = {"run_id": "run_1", "reacting_to": 0}
    patched_queries["harvest"] = {
        "run_id": "run_1",
        "outcome": "errored",
        "output": None,
        "error": {"kind": "boom", "message": "inner run failed"},
    }
    result = await take_pending_harvest(_FakePool(), "s1", account_id="a1")
    assert isinstance(result, HarvestedInference)
    assert result.outcome == "errored"
    assert result.output is None
    assert result.error == {"kind": "boom", "message": "inner run failed"}


@pytest.mark.asyncio
async def test_park_with_non_string_run_id_returns_no_park(patched_queries: dict[str, Any]) -> None:
    # A malformed park (no usable run id) does not crash the harvest read; it cannot
    # be harvested and must not wedge the turn, so it reads as NO_PARK (caller re-parks).
    patched_queries["park"] = {"run_id": None, "reacting_to": 1}
    assert await take_pending_harvest(_FakePool(), "s1", account_id="a1") is ParkState.NO_PARK


def test_event_kind_constants() -> None:
    # The park/harvest events are span-kind bookkeeping (excluded from replay).
    assert model_workflow.PARK_EVENT == "model_workflow_park"
    assert model_workflow.HARVEST_EVENT == "model_workflow_harvest"
