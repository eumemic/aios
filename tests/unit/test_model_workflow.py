"""Unit tests for the run-side park/harvest plumbing of the ``workflow:`` model
binding (issue #1634).

These pin the pure read-side projection — ``take_pending_harvest`` reads the
latest park + its matching harvest from the event log and projects them into a
:class:`HarvestedInference` carrying the watermark sealed at park. The park/harvest
DB writes themselves are exercised by the harness integration tests; here we stub
the focused queries to pin the pairing + supersession logic without a database.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.harness import model_workflow
from aios.harness.completion import LlmRequest
from aios.harness.model_binding import WorkflowModelRef
from aios.harness.model_workflow import (
    HarvestedInference,
    ParkState,
    UnlaunchedPark,
    take_pending_harvest,
)
from aios.services import sessions as sessions_service
from aios.services import workflows as workflows_service


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
    """Stub the focused queries; tests set ``state['park']`` / ``state['harvest']`` /
    ``state['run_exists']``."""
    state: dict[str, Any] = {"park": None, "harvest": None, "run_exists": True}

    async def _find_park(conn: object, session_id: str, *, account_id: str) -> Any:
        return state["park"]

    async def _find_harvest(conn: object, session_id: str, *, run_id: str, account_id: str) -> Any:
        harvest = state["harvest"]
        if harvest is None:
            return None
        # Mirror the keyed read: only return when the run id matches.
        return harvest if harvest.get("run_id") == run_id else None

    async def _run_exists(conn: object, run_id: str, *, account_id: str) -> bool:
        return bool(state["run_exists"])

    import aios.db.queries as queries
    from aios.db.queries import workflows as wf_queries

    monkeypatch.setattr(queries, "find_latest_model_workflow_park", _find_park)
    monkeypatch.setattr(queries, "find_model_workflow_harvest", _find_harvest)
    monkeypatch.setattr(wf_queries, "run_exists", _run_exists)
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
async def test_park_whose_run_was_never_created_is_unlaunched(
    patched_queries: dict[str, Any],
) -> None:
    # A crash between the park record and the launch (#2469): the caller must launch the
    # recorded run id, not wait on a run that will never exist or mint a second one.
    patched_queries["park"] = {"run_id": "run_1", "reacting_to": 7}
    patched_queries["run_exists"] = False
    result = await take_pending_harvest(_FakePool(), "s1", account_id="a1")
    assert result == UnlaunchedPark(run_id="run_1")


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


@pytest.mark.asyncio
async def test_model_workflow_launch_inherits_session_vaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sibling trusted session→run edge deliberately uses omitted/inherit semantics."""
    launch = AsyncMock(return_value=(SimpleNamespace(id="run_1"), "req_1"))
    monkeypatch.setattr(workflows_service, "launch_awaited_run", launch)
    monkeypatch.setattr(
        sessions_service,
        "get_session_basic",
        AsyncMock(return_value=SimpleNamespace(environment_id="env_1", parent_run_id=None)),
    )
    monkeypatch.setattr(
        sessions_service, "append_event", AsyncMock(return_value=SimpleNamespace(id="evt_park"))
    )
    monkeypatch.setattr(model_workflow, "_launch_harvest_task", lambda *args, **kwargs: None)

    await model_workflow.launch_model_workflow_park(
        _FakePool(),
        "ses_1",
        ref=WorkflowModelRef("wf_1"),
        request=LlmRequest(messages=[]),
        reacting_to=1,
        request_record={},
        account_id="acc_1",
    )

    assert launch.await_args is not None
    assert launch.await_args.kwargs["authority"].session_id == "ses_1"
    assert launch.await_args.kwargs.get("vault_ids") is None


@pytest.mark.asyncio
async def test_a_second_harvest_task_for_an_in_flight_key_is_not_spawned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#2480: the sweep can re-park a run between its creation and the park step's own
    spawn. The step's spawn must then be a no-op: a second task for the key would let
    the first one's done-callback drop the key while the other still polls."""
    import asyncio

    release = asyncio.Event()
    started: list[str] = []

    async def _park(pool: object, session_id: str, *, run_id: str, account_id: str) -> None:
        started.append(run_id)
        await release.wait()

    monkeypatch.setattr(model_workflow, "_park_and_signal", _park)
    model_workflow.reset_inflight_harvests()
    key = ("ses_1", "run_1")
    try:
        # The sweep's re-park spawns the task, then the park step's own spawn races in.
        assert model_workflow.relaunch_model_dispatch_park(
            _FakePool(), "ses_1", run_id="run_1", account_id="acc_1"
        )
        model_workflow._launch_harvest_task(
            _FakePool(), "ses_1", run_id="run_1", account_id="acc_1"
        )
        await asyncio.sleep(0)

        assert started == ["run_1"], "one harvest task per in-flight key"
        tasks = [t for t in model_workflow._PARK_TASKS if t.get_name().endswith(":ses_1:run_1")]
        assert len(tasks) == 1
        assert key in model_workflow._INFLIGHT_HARVESTS

        release.set()
        await asyncio.gather(*tasks)
        assert key not in model_workflow._INFLIGHT_HARVESTS
    finally:
        release.set()
        model_workflow.reset_inflight_harvests()
