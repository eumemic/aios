"""Machine-observer anomaly-scan fixture (aios#1326).

Drives the production ``build_observer_script`` / ``build_sweep_script`` directly against the
real script host (no DB, no live model) with simulated ``get_run`` / ``list_runs`` resolutions,
walking the deterministic unwrap -> read -> scan -> verdict machine. Because the host re-runs
from the start each wake (replaying the whole growing memo), these tests also assert the
load-bearing durable property: **replay is deterministic** (every call_key is stable across
every replay).

It proves the #1326 acceptance:
- the LEAD cost-on-errored signal [LIVE] -> anomaly (criterion 4),
- a within-bands errored run + any non-errored run -> ok (criterion 8),
- fail-loud: a null/absent telemetry field used by a FIRED predicate -> cannot-determine,
  NEVER ok (criterion 9 — the load-bearing line),
- the trigger-envelope unwrap reads ``input["trigger"]["run"]["id"]`` (the TRACED path, NOT
  the falsely-cited ``trigger.event.run_id``) AND a bare ``{run_id}`` (criterion 2),
- the structured record shape (criterion 3),
- no GitHub issue / obliging action-item (structural: no http_request in surface — crit 11),
- the DEFERRED signals' contract is fixture-locked against a FABRICATED return WITHOUT being
  wired live (criteria 5/6/7): a fabricated non-null iteration_count / net_diff_lines does NOT
  flip a real-run verdict, and a null one yields cannot-determine,
- the dead-man twin (``observer_sweep``): N_runs > N_scans -> alarm; equal -> ok (crit 12);
  a truncated list_runs page -> cannot-determine, never a silent under-count (crit 13).

Mirrors the host-subprocess style of ``test_wf_triage_pipeline_fixture.py``; needs no Postgres.
"""

from __future__ import annotations

from typing import Any

import pytest

from aios.workflows.host_launcher import run_script_host
from aios.workflows.telemetry_observer import (
    DEFAULT_COST_WASTE_FLOOR,
    REQUIRED_TOOLS,
    build_observer_fixture_script,
    build_observer_workflow_create,
    build_sweep_fixture_script,
    build_sweep_workflow_create,
)

pytestmark = pytest.mark.integration


# ─── a get_run return shaped exactly like the real WfRun read row (#1324) ─────


def _wf_run(
    run_id: str,
    *,
    workflow_id: str = "wf_dev_pipeline",
    status: str = "completed",
    usage: dict[str, Any] | None = None,
    omit_usage: bool = False,
) -> dict[str, Any]:
    """A get_run tool return, real-shaped: the full WfRun dump with a ``usage`` object.

    ``usage`` defaults to a complete, real-shaped WfRunUsage (cost/tokens/ wall-clock,
    iteration_count explicit-null as the host emits it today). ``omit_usage`` models a
    pre-#1334 worker that returns a bare WfRun with NO usage object (usage: None)."""
    if usage is None and not omit_usage:
        usage = {
            "cost_microusd": 1000,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "iteration_count": None,  # ALWAYS null today (no host counter)
            "wall_clock_ms": 1234,
        }
    return {
        "id": run_id,
        "workflow_id": workflow_id,
        "status": status,
        "output": {"this": "is the run's self-report — the observer must NEVER read it"},
        "usage": None if omit_usage else usage,
    }


class ObserverScenario:
    """A deterministic responder for the observer: one get_run return, keyed by the call.
    Keeping the response a pure function of the call_key keeps replay stable."""

    def __init__(self, *, get_run_return: dict[str, Any]) -> None:
        self.get_run_return = get_run_return
        self.tool_calls: list[tuple[str, Any]] = []

    def outcome(self, cap: Any) -> dict[str, Any]:
        cid, spec = cap.capability_id, cap.spec
        if cid == "tool":
            name = spec["tool_name"]
            self.tool_calls.append((name, spec["input"]))
            if name == "get_run":
                return {"ok": self.get_run_return}
            raise AssertionError(f"observer must only call get_run, got {name}")
        raise AssertionError(f"observer must not emit capability {cid} (no agent/gate); {spec!r}")


async def _drive_observer(
    scenario: ObserverScenario,
    *,
    input: dict[str, Any],
    max_steps: int = 30,
    **build_kwargs: Any,
) -> tuple[Any, list[str]]:
    """Drive the production observer to a terminal outcome; assert replay-determinism."""
    src = build_observer_fixture_script(**build_kwargs)
    memo: dict[str, Any] = {}
    keys: list[str] = []
    for _ in range(max_steps):
        out = await run_script_host(source=src, input=input, memo=memo)
        if out.kind == "returned":
            assert len(keys) == len(set(keys)), "replay produced a duplicate call_key"
            return out.value, keys
        assert out.kind == "suspended", (out.kind, out.error_repr, out.error_traceback)
        assert len(out.emitted) == 1, [(e.capability_id, e.spec) for e in out.emitted]
        cap = out.emitted[0]
        keys.append(cap.call_key)
        memo[cap.call_key] = scenario.outcome(cap)
    raise AssertionError(f"observer did not terminate within {max_steps} steps")


# ─── criterion 4: LEAD cost-on-errored [LIVE] -> anomaly ──────────────────────


async def test_lead_cost_on_errored_above_floor_is_anomaly() -> None:
    usage = {
        "cost_microusd": DEFAULT_COST_WASTE_FLOOR + 1,
        "input_tokens": 999,
        "output_tokens": 10,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "iteration_count": None,
        "wall_clock_ms": 9999,
    }
    run = _wf_run("wfr_err", status="errored", usage=usage)
    scn = ObserverScenario(get_run_return=run)
    value, _keys = await _drive_observer(scn, input={"run_id": "wfr_err"})

    assert value["verdict"] == "anomaly"
    assert value["signals_fired"] == ["cost-on-errored"]
    assert value["run_id"] == "wfr_err"
    assert value["workflow_id"] == "wf_dev_pipeline"
    assert value["telemetry_snapshot"]["cost_microusd"] == DEFAULT_COST_WASTE_FLOOR + 1
    assert "bands_version" in value
    # the observer read the run by id, never the run's output/prose
    assert scn.tool_calls == [("get_run", {"run_id": "wfr_err"})]


# ─── criterion 8: within-bands errored + non-errored -> ok ────────────────────


async def test_errored_below_floor_is_ok() -> None:
    usage = {
        "cost_microusd": DEFAULT_COST_WASTE_FLOOR - 1,  # cheap fast-fail
        "input_tokens": 10,
        "output_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "iteration_count": None,
        "wall_clock_ms": 5,
    }
    run = _wf_run("wfr_cheap", status="errored", usage=usage)
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_cheap"})
    assert value["verdict"] == "ok"
    assert value["signals_fired"] == []


async def test_completed_run_is_ok() -> None:
    run = _wf_run("wfr_done", status="completed")
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_done"})
    assert value["verdict"] == "ok"
    assert value["signals_fired"] == []


async def test_cancelled_run_is_ok() -> None:
    run = _wf_run("wfr_cxl", status="cancelled")
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_cxl"})
    assert value["verdict"] == "ok"
    assert value["signals_fired"] == []


# ─── criterion 9: fail-loud cannot-determine (the load-bearing criteria) ──────


async def test_errored_with_null_cost_is_cannot_determine_not_ok() -> None:
    """The vault_ids:null disease shape on cost_microusd: a FIRED predicate read a null
    field -> cannot-determine, NEVER silently ok, NEVER anomaly."""
    usage = {
        "cost_microusd": None,  # explicit null (the disease shape)
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "iteration_count": None,
        "wall_clock_ms": 10,
    }
    run = _wf_run("wfr_null", status="errored", usage=usage)
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_null"})
    assert value["verdict"] == "cannot-determine"
    assert value["verdict"] != "ok"
    assert value["signals_fired"] == []


async def test_errored_with_absent_usage_object_is_cannot_determine() -> None:
    """A pre-#1334 worker returns a bare WfRun with usage:None. On an errored run the LEAD
    predicate fires and reads an absent cost -> cannot-determine (this is exactly why the
    arm-time proof needs PR#1334 DEPLOYED, not merely merged — criterion 14)."""
    run = _wf_run("wfr_bare", status="errored", omit_usage=True)
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_bare"})
    assert value["verdict"] == "cannot-determine"


async def test_get_run_error_value_is_cannot_determine() -> None:
    run = {"error": "NotFound: no such run"}
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_missing"})
    assert value["verdict"] == "cannot-determine"


# ─── criterion 2: the TRACED trigger-envelope unwrap ──────────────────────────


async def test_unwraps_real_trigger_envelope_run_path() -> None:
    """The real run_completion WorkflowAction envelope: run id at
    ``input["trigger"]["run"]["id"]`` (TRACED to harness/trigger_runner.py — NOT the falsely
    cited ``trigger.event.run_id``; there is NO trigger.event for run_completion fires)."""
    usage = {
        "cost_microusd": DEFAULT_COST_WASTE_FLOOR + 5,
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "iteration_count": None,
        "wall_clock_ms": 1,
    }
    run = _wf_run("wfr_env", workflow_id="wf_triage", status="errored", usage=usage)
    scn = ObserverScenario(get_run_return=run)
    envelope = {
        "trigger": {
            "id": "trg_1",
            "name": "observer-watch",
            "source": "run_completion",
            "fired_at": "2026-06-17T00:00:00+00:00",
            "run": {
                "id": "wfr_env",
                "workflow_id": "wf_triage",
                "status": "errored",
                "output": {"self": "report — forbidden"},
                "error": {"kind": "boom"},
            },
        },
        "input": None,
    }
    value, _ = await _drive_observer(scn, input=envelope)
    assert value["run_id"] == "wfr_env"
    assert value["verdict"] == "anomaly"
    # it asked get_run for the id pulled from trigger.run.id
    assert scn.tool_calls == [("get_run", {"run_id": "wfr_env"})]


async def test_envelope_without_reachable_run_id_is_cannot_determine() -> None:
    """A trigger envelope carrying no reachable run id is a FINDING surfaced loud, not papered
    over (the issue's explicit instruction)."""
    envelope = {"trigger": {"id": "t", "source": "cron"}, "input": {}}
    src = build_observer_fixture_script()
    out = await run_script_host(source=src, input=envelope, memo={})
    # No run id -> it returns immediately without any get_run call.
    assert out.kind == "returned", (out.kind, out.error_repr)
    assert out.value["verdict"] == "cannot-determine"
    assert out.value["run_id"] is None


# ─── criteria 5/6/7: DEFERRED signals fixture-locked WITHOUT being wired live ──


async def test_fabricated_iteration_count_does_not_flip_a_real_verdict() -> None:
    """iteration-at-max is DEFERRED. Even a FABRICATED non-null iteration_count above the
    band must NOT produce an anomaly on a completed run — the signal is not wired live. (And
    on a real run iteration_count is always null, which criterion 9 would make
    cannot-determine if it were ever consumed.) This pins the contract: it ships DEFERRED."""
    usage = {
        "cost_microusd": 100,
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "iteration_count": 9999,  # FABRICATED non-null, way above any band
        "wall_clock_ms": 10,
    }
    run = _wf_run("wfr_iter", status="completed", usage=usage)
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_iter"})
    # NOT anomaly — the DEFERRED iteration signal is not wired live.
    assert value["verdict"] == "ok"
    assert "iteration-at-max" not in value["signals_fired"]


async def test_fabricated_net_diff_lines_does_not_flip_a_real_verdict() -> None:
    """byte-buffer / cost-per-net-line are DEFERRED (net_diff_lines is not on the host read
    path). Even a FABRICATED high input_tokens + a fabricated net_diff_lines field must NOT
    produce an anomaly — the signals are not wired live."""
    usage = {
        "cost_microusd": 100,
        "input_tokens": 10_000_000,  # huge, above any byte-buffer band
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "iteration_count": None,
        "wall_clock_ms": 10,
    }
    run = _wf_run("wfr_bb", status="completed", usage=usage)
    run["usage"]["net_diff_lines"] = 0  # FABRICATED — not a real host field
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_bb"})
    assert value["verdict"] == "ok"
    assert value["signals_fired"] == []


# ─── criterion 3/11: structured record + structural issue-emission gating ─────


def test_observer_surface_has_no_mutation_or_http() -> None:
    """The observer's only output is its structured return. With no http_request/bash/edit in
    surface it CANNOT file a GitHub issue / obliging action-item — issue-emission gating is
    structural (criterion 11)."""
    types = {t.type for t in REQUIRED_TOOLS}
    assert types == {"get_run", "list_runs"}
    assert "http_request" not in types
    assert "bash" not in types
    assert "edit" not in types
    wc = build_observer_workflow_create()
    wc_types = {t.type for t in wc.tools}
    assert wc_types == {"get_run", "list_runs"}
    assert not wc.http_servers


async def test_record_shape() -> None:
    # a structural check of the verdict record keys (criterion 3)
    run = _wf_run("wfr_shape", status="completed")
    scn = ObserverScenario(get_run_return=run)
    value, _ = await _drive_observer(scn, input={"run_id": "wfr_shape"})
    assert set(value.keys()) == {
        "run_id",
        "workflow_id",
        "verdict",
        "signals_fired",
        "telemetry_snapshot",
        "bands_version",
    }
    assert value["verdict"] in ("ok", "anomaly", "cannot-determine")


# ─── criteria 12/13: the dead-man twin (observer_sweep) ───────────────────────


class SweepScenario:
    """Deterministic responder for observer_sweep: a map of (workflow_id, status) -> list of
    run rows, served once per list_runs call. Keying on the call's args keeps replay stable."""

    def __init__(self, *, pages: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.list_calls: list[dict[str, Any]] = []

    def outcome(self, cap: Any) -> dict[str, Any]:
        cid, spec = cap.capability_id, cap.spec
        if cid == "tool" and spec["tool_name"] == "list_runs":
            args = spec["input"]
            self.list_calls.append(args)
            key = (args["workflow_id"], args["status"])
            runs = self.pages.get(key, [])
            return {"ok": {"runs": runs}}
        raise AssertionError(f"sweep must only call list_runs; got {cid} {spec!r}")


async def _drive_sweep(
    scenario: SweepScenario, *, input: dict[str, Any], max_steps: int = 30
) -> Any:
    src = build_sweep_fixture_script()
    memo: dict[str, Any] = {}
    keys: list[str] = []
    for _ in range(max_steps):
        out = await run_script_host(source=src, input=input, memo=memo)
        if out.kind == "returned":
            assert len(keys) == len(set(keys)), "replay produced a duplicate call_key"
            return out.value
        assert out.kind == "suspended", (out.kind, out.error_repr, out.error_traceback)
        assert len(out.emitted) == 1
        cap = out.emitted[0]
        keys.append(cap.call_key)
        memo[cap.call_key] = scenario.outcome(cap)
    raise AssertionError("sweep did not terminate")


def _run_row(i: int, status: str) -> dict[str, Any]:
    return {"id": f"wfr_{status}_{i}", "workflow_id": "wf_watched", "status": status}


async def test_sweep_alarms_on_run_scan_shortfall() -> None:
    """3 terminal runs but only 2 recorded scans -> alarm (criterion 12)."""
    pages = {
        ("wf_watched", "completed"): [_run_row(0, "completed"), _run_row(1, "completed")],
        ("wf_watched", "errored"): [_run_row(0, "errored")],
        ("wf_watched", "cancelled"): [],
        ("telemetry_observer", "completed"): [
            {"id": "scan_0", "workflow_id": "telemetry_observer", "status": "completed"},
            {"id": "scan_1", "workflow_id": "telemetry_observer", "status": "completed"},
        ],
    }
    scn = SweepScenario(pages=pages)
    value = await _drive_sweep(
        scn,
        input={
            "watched_workflow_id": "wf_watched",
            "observer_workflow_id": "telemetry_observer",
        },
    )
    assert value["alarm"] is True
    assert value["verdict"] == "cannot-determine"
    assert value["n_runs"] == 3
    assert value["n_scans"] == 2


async def test_sweep_ok_when_runs_equal_scans() -> None:
    """3 terminal runs and 3 recorded scans -> ok (criterion 12)."""
    pages = {
        ("wf_watched", "completed"): [_run_row(0, "completed"), _run_row(1, "completed")],
        ("wf_watched", "errored"): [_run_row(0, "errored")],
        ("wf_watched", "cancelled"): [],
        ("telemetry_observer", "completed"): [
            {"id": f"scan_{i}", "workflow_id": "telemetry_observer", "status": "completed"}
            for i in range(3)
        ],
    }
    scn = SweepScenario(pages=pages)
    value = await _drive_sweep(
        scn,
        input={
            "watched_workflow_id": "wf_watched",
            "observer_workflow_id": "telemetry_observer",
        },
    )
    assert value["alarm"] is False
    assert value["verdict"] == "ok"
    assert value["n_runs"] == 3
    assert value["n_scans"] == 3


async def test_sweep_truncated_page_is_cannot_determine_not_undercount() -> None:
    """A FULL list_runs page (len == requested limit) means rows may exist beyond it -> the
    sweep cannot prove the terminal count -> cannot-determine, NEVER a silent under-count read
    as 'all scanned' (criterion 13, the #1323 no-silent-degrade invariant)."""
    limit = 2
    pages = {
        # a full page of `limit` completed runs => possibly more unseen
        ("wf_watched", "completed"): [_run_row(0, "completed"), _run_row(1, "completed")],
        ("wf_watched", "errored"): [],
        ("wf_watched", "cancelled"): [],
        ("telemetry_observer", "completed"): [],
    }
    scn = SweepScenario(pages=pages)
    value = await _drive_sweep(
        scn,
        input={
            "watched_workflow_id": "wf_watched",
            "observer_workflow_id": "telemetry_observer",
            "limit": limit,
        },
    )
    assert value["alarm"] is True
    assert value["verdict"] == "cannot-determine"
    assert "truncated" in value["reason"]


async def test_sweep_list_read_error_is_cannot_determine() -> None:
    """A list_runs read that errors -> cannot-determine, never a silent zero count."""

    class ErroringSweep(SweepScenario):
        def outcome(self, cap: Any) -> dict[str, Any]:
            self.list_calls.append(cap.spec["input"])
            return {"ok": {"error": "boom"}}

    scn = ErroringSweep(pages={})
    value = await _drive_sweep(
        scn,
        input={
            "watched_workflow_id": "wf_watched",
            "observer_workflow_id": "telemetry_observer",
        },
    )
    assert value["alarm"] is True
    assert value["verdict"] == "cannot-determine"


def test_sweep_surface_is_list_runs_only() -> None:
    wc = build_sweep_workflow_create()
    assert {t.type for t in wc.tools} == {"get_run", "list_runs"}
    assert not wc.http_servers


# ─── create-time script validation: surface covers the script, main(input) exists ─


def test_observer_and_sweep_scripts_validate_against_their_surface() -> None:
    """The #1285 create-time validator: each script compiles, defines
    ``async def main(input)``, and the declared ``REQUIRED_TOOLS`` is a superset of the
    script's literal ``tool("…")`` names (so the surface can't drift from the script — #1135)."""
    from aios.errors import ValidationError
    from aios.models.agents import ToolSpec
    from aios.workflows.script_validation import validate_workflow_script
    from aios.workflows.telemetry_observer import build_observer_script, build_sweep_script

    validate_workflow_script(script=build_observer_script(), tools=list(REQUIRED_TOOLS))
    validate_workflow_script(script=build_sweep_script(), tools=list(REQUIRED_TOOLS))

    # the sweep's list_runs call is not covered by a get_run-only surface -> rejected
    with pytest.raises(ValidationError):
        validate_workflow_script(script=build_sweep_script(), tools=[ToolSpec(type="get_run")])
