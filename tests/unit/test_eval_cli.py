"""The operator scripts that register the eval workflows and launch and read gates,
against a fake API."""

from __future__ import annotations

import builtins
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from evals import gate, monitor_check, register
from evals.workflows import eval_item, load, paired_eval

_BARS = Path(__file__).parents[2] / "evals" / "bars"
BAR: dict[str, Any] = json.loads((_BARS / "wam_gate.json").read_text())
MONITOR_BAR: dict[str, Any] = json.loads((_BARS / "wam_monitor.json").read_text())


class FakeApi:
    def __init__(self) -> None:
        self.workflows: dict[str, dict[str, Any]] = {}
        self.versions: dict[tuple[str, int], dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.posts: list[tuple[str, Any]] = []
        self.gets: list[tuple[str, dict[str, Any]]] = []
        self.agent_model = "anthropic/claude-sonnet-4-5"
        self.fires: list[dict[str, Any]] = []

    def get(self, path: str, **params: Any) -> Any:
        self.gets.append((path, params))
        if path == "/v1/workflows":
            return {"data": [w for w in self.workflows.values() if w["name"] == params.get("name")]}
        if path.startswith("/v1/agents/"):
            return {"id": path.rsplit("/", 1)[1], "version": 7, "model": self.agent_model}
        if path.startswith("/v1/triggers/") and path.endswith("/runs"):
            return {"data": self.fires}
        if path.startswith("/v1/triggers/"):
            return {
                "name": path.rsplit("/", 1)[1],
                "action": {
                    "kind": "workflow",
                    "input_template": {
                        "agent": {"agent_id": "agent_1", "version": 7},
                        "candidate": {"workflow_id": "wf_cand", "version": 3},
                    },
                },
            }
        if "/versions/" in path:
            parts = path.split("/")
            return self.versions[(parts[3], int(parts[5]))]
        if path.startswith("/v1/workflows/"):
            return self.workflows[path.rsplit("/", 1)[1]]
        if path == "/v1/runs":
            data = [
                r
                for r in self.runs.values()
                if r["workflow_id"] == params.get("workflow_id")
                and (params.get("include_archived") == "true" or r.get("archived_at") is None)
            ]
            return {"data": data, "next_cursor": None}
        if path.startswith("/v1/runs/"):
            return self.runs[path.rsplit("/", 1)[1]]
        raise AssertionError(path)

    def post(self, path: str, body: Any) -> Any:
        self.posts.append((path, body))
        if path == "/v1/workflows":
            wf = dict(body, id=f"wf_{len(self.workflows)}", version=1)
            self.workflows[wf["id"]] = wf
            self.versions[(wf["id"], 1)] = dict(wf)
            return wf
        if path == "/v1/runs":
            return {"id": f"wfr_{len(self.posts)}"}
        if path == "/v1/triggers":
            return {"name": body["name"], "next_fire": "2026-10-12T03:17:00+00:00"}
        raise AssertionError(path)

    def put(self, path: str, body: Any) -> Any:
        wf = self.workflows[path.rsplit("/", 1)[1]]
        assert body["version"] == wf["version"]
        wf.update(script=body["script"], tools=body["tools"], version=wf["version"] + 1)
        self.versions[(wf["id"], wf["version"])] = dict(wf)
        return wf


# ── register ──────────────────────────────────────────────────────────────────


def test_registering_creates_then_leaves_alone_then_updates() -> None:
    api = FakeApi()
    first = register.register(api, {"wam-gate": BAR, "wam-monitor": MONITOR_BAR})
    assert {ref["version"] for ref in first.values()} == {1}
    gate_wf = api.workflows[first["wam-gate"]["id"]]
    assert gate.bar_of(gate_wf["script"]) == BAR
    assert gate_wf["tools"] == paired_eval.TOOLS
    item_wf = api.workflows[first["eval-item"]["id"]]
    assert item_wf["script"] == eval_item.build(r0=first["eval-r0"], judge=first["eval-judge"])

    assert (
        register.register(api, {"wam-gate": BAR, "wam-monitor": MONITOR_BAR}) == first
    )  # nothing changed

    stricter = dict(BAR, delta=0.05)
    again = register.register(api, {"wam-gate": stricter, "wam-monitor": MONITOR_BAR})
    assert again["wam-gate"] == {"id": first["wam-gate"]["id"], "version": 2}
    assert again["eval-item"] == first["eval-item"]
    assert again["wam-monitor"] == first["wam-monitor"]


def test_the_monitor_is_the_gate_template_in_monitor_mode_with_its_own_bar() -> None:
    api = FakeApi()
    registered = register.register(api, {"wam-gate": BAR, "wam-monitor": MONITOR_BAR})
    script = api.workflows[registered["wam-monitor"]["id"]]["script"]
    config = load(script)["CONFIG"]
    assert config["mode"] == "monitor"
    assert config["bar"] == MONITOR_BAR
    assert config["item"] == registered["eval-item"]


# ── launch ────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 10, 6, 14, 37, tzinfo=UTC)


def _api_with_candidate(**surface: Any) -> FakeApi:
    api = FakeApi()
    register.register(api, {"wam-gate": BAR, "wam-monitor": MONITOR_BAR})
    api.versions[("wf_cand", 3)] = {
        "workflow_id": "wf_cand",
        "version": 3,
        "script": "async def main(input):\n    return await call_llm(input)\n",
        "tools": [],
        "mcp_servers": [],
        "http_servers": [],
        "ssh_servers": [],
        "created_at": "2026-09-20T10:00:00+00:00",
        **surface,
    }
    return api


def _launch(api: FakeApi, **changes: Any) -> tuple[str | None, list[str]]:
    lines: list[str] = []
    kwargs: dict[str, Any] = {
        "agent_id": "agent_1",
        "baseline_model": "anthropic/claude-sonnet-4-5",
        "candidate": "wf_cand@3",
        "seed": "1",
        "budget_usd": 500.0,
        "environment_id": "env_1",
        "now": _NOW,
        "dry_run": False,
    }
    kwargs.update(changes)
    run_id = gate.launch(api, out=lines.append, **kwargs)
    return run_id, lines


def test_launch_opens_the_window_at_the_candidate_and_starts_a_budgeted_run() -> None:
    api = _api_with_candidate()
    run_id, lines = _launch(api)
    assert run_id is not None
    path, body = api.posts[-1]
    assert path == "/v1/runs"
    gate_wf = next(w for w in api.workflows.values() if w["name"] == "wam-gate")
    assert body["workflow_id"] == gate_wf["id"] and body["version"] == gate_wf["version"]
    assert body["budget_usd"] == 500.0
    assert body["input"] == {
        "agent": {"agent_id": "agent_1", "version": 7},
        "baseline_model": "anthropic/claude-sonnet-4-5",
        "candidate": {"workflow_id": "wf_cand", "version": 3},
        "seed": "1",
        "window": {"start": "2026-09-20T10:00:00+00:00", "end": "2026-10-06T14:00:00+00:00"},
        "candidate_created_at": "2026-09-20T10:00:00+00:00",
    }
    estimate = json.loads("\n".join(lines[: lines.index(f"gate run {run_id}")]))
    assert estimate["n_required"] == 208
    assert estimate["cost_usd"] == 208.0
    assert estimate["peak_runs"] == 1 + 6 * 6
    # The candidate's budget follows the cost limit: (1 + 1.0) x $0.5 x margin 2.
    assert estimate["candidate_budget_usd"] == 2.0
    assert estimate["item_budget_usd"] == 2 * 0.5 + 2 * 2.0 + 0.5


def test_an_old_candidate_gets_at_most_one_sample_range() -> None:
    api = _api_with_candidate(created_at="2026-01-01T00:00:00+00:00")
    _launch(api)
    window = api.posts[-1][1]["input"]["window"]
    assert window["start"] == "2026-09-05T14:00:00+00:00"


@pytest.mark.parametrize(
    "surface",
    [
        {"tools": [{"type": "web_search"}]},
        {"mcp_servers": [{"name": "m", "url": "https://m"}]},
        {"http_servers": [{"name": "h"}]},
        {"ssh_servers": [{"name": "s"}]},
    ],
)
def test_a_candidate_with_a_surface_is_refused(surface: dict[str, Any]) -> None:
    api = _api_with_candidate(**surface)
    with pytest.raises(gate.Refused, match="declares"):
        _launch(api)
    assert not [p for p in api.posts if p[0] == "/v1/runs"]


def test_a_workflow_the_candidate_invokes_is_checked_too() -> None:
    """The pre-flight follows literal workflow ids down the candidate's script: a nested
    workflow with tools would only be refused inside the eval, scored as losses."""
    api = _api_with_candidate(
        script="async def main(input):\n    return await invoke_workflow('wf_inner', input)\n"
    )
    api.workflows["wf_inner"] = {
        "id": "wf_inner",
        "name": "inner",
        "version": 1,
        "script": 'async def main(input):\n    return await invoke_workflow("wf_leaf", input)\n',
        "tools": [],
    }
    api.workflows["wf_leaf"] = {
        "id": "wf_leaf",
        "name": "leaf",
        "version": 1,
        "script": "async def main(input):\n    return await tool('web_search', input)\n",
        "tools": [{"type": "web_search"}],
    }
    with pytest.raises(gate.Refused, match="wf_leaf declares tools"):
        _launch(api)


def test_a_workflow_chosen_at_run_time_is_a_warning() -> None:
    api = _api_with_candidate(
        script="async def main(input):\n    return await invoke_workflow(input['wf'], input)\n"
    )
    _, lines = _launch(api, dry_run=True)
    assert any("chosen at run time" in line for line in lines)


def test_the_estimate_uses_the_analysis_version_the_gate_pins() -> None:
    api = _api_with_candidate()
    analysis = next(w for w in api.workflows.values() if w["name"] == "eval-analysis")
    # A newer analysis version the gate doesn't pin is not what the run will use.
    api.put(f"/v1/workflows/{analysis['id']}", {"version": 1, "script": "x = (", "tools": []})
    _launch(api, dry_run=True)
    assert (f"/v1/workflows/{analysis['id']}/versions/1", {}) in api.gets


_PAYLOAD = "\nimport builtins\nbuiltins.EVAL_CLI_RAN_SERVER_TEXT = True\n"


def test_a_gate_someone_else_updated_is_refused_without_running_it() -> None:
    api = _api_with_candidate()
    wam_gate = next(w for w in api.workflows.values() if w["name"] == "wam-gate")
    api.put(
        f"/v1/workflows/{wam_gate['id']}",
        {"version": wam_gate["version"], "script": wam_gate["script"] + _PAYLOAD, "tools": []},
    )
    with pytest.raises(gate.Refused, match="differs from this checkout's template"):
        _launch(api, dry_run=True)
    assert not hasattr(builtins, "EVAL_CLI_RAN_SERVER_TEXT")


def test_a_pinned_analysis_that_differs_from_the_template_is_refused_without_running_it() -> None:
    api = _api_with_candidate()
    analysis = next(w for w in api.workflows.values() if w["name"] == "eval-analysis")
    pinned = api.versions[(analysis["id"], 1)]
    pinned["script"] = pinned["script"] + _PAYLOAD
    with pytest.raises(gate.Refused, match="differs from this checkout's template"):
        _launch(api, dry_run=True)
    assert not hasattr(builtins, "EVAL_CLI_RAN_SERVER_TEXT")


def test_the_bar_is_read_from_a_gate_script_without_running_it() -> None:
    api = _api_with_candidate()
    wam_gate = next(w for w in api.workflows.values() if w["name"] == "wam-gate")
    assert gate.bar_of(wam_gate["script"] + _PAYLOAD) == BAR
    assert not hasattr(builtins, "EVAL_CLI_RAN_SERVER_TEXT")


def test_every_bar_file_registers_as_its_own_gate() -> None:
    """A recipe class with a different cost or latency profile is gated by its own
    registered bar, never by loosening the default."""
    bars = sorted((Path(__file__).parents[2] / "evals" / "bars").glob("wam_gate*.json"))
    names = [register.gate_name(p) for p in bars]
    assert names == ["wam-gate", "wam-gate-fusion"]
    fusion = json.loads(bars[1].read_text())
    assert fusion["limits"]["cost"] == 5.0 and fusion["limits"]["latency"] == 2.0
    assert {k: v for k, v in fusion.items() if k not in ("limits", "planning")} == {
        k: v for k, v in BAR.items() if k not in ("limits", "planning")
    }
    api = FakeApi()
    registered = register.register(
        api, {register.gate_name(p): json.loads(p.read_text()) for p in bars}
    )
    assert {"wam-gate", "wam-gate-fusion"} <= set(registered)
    assert registered["wam-gate"]["id"] != registered["wam-gate-fusion"]["id"]


def test_a_candidate_created_this_hour_has_no_holdout() -> None:
    api = _api_with_candidate(created_at="2026-10-06T14:10:00+00:00")
    with pytest.raises(gate.Refused, match="no holdout"):
        _launch(api)


def test_a_dry_run_starts_nothing_and_warns_about_a_small_budget() -> None:
    api = _api_with_candidate(script="async def main(input):\n    return await agent(input)\n")
    run_id, lines = _launch(api, dry_run=True, budget_usd=10.0)
    assert run_id is None
    assert not [p for p in api.posts if p[0] == "/v1/runs"]
    assert any("agent()" in line for line in lines)
    assert any("below the estimated" in line for line in lines)


# ── report and reanalyze ──────────────────────────────────────────────────────


def _gate_run(id: str, *, start: str, archived: bool = False) -> dict[str, Any]:
    return {
        "id": id,
        "workflow_id": "wf_gate",
        "status": "completed",
        "environment_id": "env_1",
        "created_at": "2026-10-06T00:00:00+00:00",
        "archived_at": "2026-10-06T01:00:00+00:00" if archived else None,
        "input": {
            "agent": {"agent_id": "agent_1", "version": 7},
            "candidate": {"workflow_id": "wf_cand", "version": 3},
        },
        "output": {
            "verdict": "PASS",
            "candidate": "workflow:wf_cand@3",
            "reasons": {"invalid": [], "inconclusive": [], "failed": []},
            "window": {"start": start, "end": "2026-10-06T00:00:00+00:00"},
            "candidate_created_at": "2026-09-20T10:00:00+00:00",
            "stats": {"n": 208},
            "records": [{"x": 1}, {"x": 2}],
            "exclusions": {"missing": 3},
            "flags": [],
            "mode": "gate",
            "alpha": 0.05,
            "bar": BAR,
            "seed": "1",
        },
    }


def test_the_report_lists_earlier_runs_even_archived_and_flags_a_broken_holdout() -> None:
    api = FakeApi()
    api.runs["wfr_new"] = _gate_run("wfr_new", start="2026-09-01T00:00:00+00:00")
    api.runs["wfr_old"] = _gate_run("wfr_old", start="2026-09-21T00:00:00+00:00", archived=True)
    other = _gate_run("wfr_other", start="2026-09-21T00:00:00+00:00")
    other["input"]["candidate"] = {"workflow_id": "wf_cand", "version": 4}
    api.runs["wfr_other"] = other
    lines: list[str] = []
    gate.report(api, "wfr_new", lines.append)
    assert "verdict: PASS  candidate: workflow:wf_cand@3" in lines
    assert any(line.startswith("WARNING: the window opens before") for line in lines)
    earlier = [line for line in lines if line.startswith("earlier:")]
    assert earlier == ["earlier: wfr_old 2026-10-06T00:00:00+00:00 PASS"]


def test_reanalyze_hands_the_stored_records_to_the_analysis() -> None:
    api = FakeApi()
    register.register(api, {"wam-gate": BAR, "wam-monitor": MONITOR_BAR})
    api.runs["wfr_g"] = _gate_run("wfr_g", start="2026-09-21T00:00:00+00:00")
    gate.reanalyze(api, "wfr_g", None, lambda _: None)
    path, body = api.posts[-1]
    analysis = next(w for w in api.workflows.values() if w["name"] == "eval-analysis")
    assert path == "/v1/runs" and body["workflow_id"] == analysis["id"]
    assert body["input"]["records"] == [{"x": 1}, {"x": 2}]
    assert body["input"]["considered"] == 5
    assert body["input"]["attributed"] == []
    assert body["input"]["mode"] == "gate" and body["input"]["bar"] == BAR
    assert body["input"]["alpha"] == 0.05


# ── the monitor ───────────────────────────────────────────────────────────────


def _deploy(api: FakeApi) -> dict[str, Any]:
    return gate.deploy_monitor(
        api,
        name="jarvis-wam",
        agent_id="agent_1",
        baseline_model="anthropic/claude-sonnet-4-5",
        candidate="wf_cand@3",
        seed="m",
        budget_usd=250.0,
        environment_id="env_1",
        schedule="17 3 * * 1",
        out=lambda _: None,
    )


def test_deploy_monitor_creates_a_pinned_one_at_a_time_operator_trigger() -> None:
    api = FakeApi()
    registered = register.register(api, {"wam-gate": BAR, "wam-monitor": MONITOR_BAR})
    api.agent_model = "workflow:wf_cand@3"
    _deploy(api)
    path, body = api.posts[-1]
    assert path == "/v1/triggers"
    assert body["source"] == {"kind": "cron", "schedule": "17 3 * * 1"}
    assert body["action"] == {
        "kind": "workflow",
        "workflow_id": registered["wam-monitor"]["id"],
        "version": registered["wam-monitor"]["version"],
        "max_outstanding_runs": 1,
        "budget_usd": 250.0,
        "input_template": {
            "agent": {"agent_id": "agent_1", "version": 7},
            "baseline_model": "anthropic/claude-sonnet-4-5",
            "candidate": {"workflow_id": "wf_cand", "version": 3},
            "seed": "m",
        },
    }


def test_deploy_monitor_refuses_before_the_deploy() -> None:
    api = FakeApi()
    register.register(api, {"wam-gate": BAR, "wam-monitor": MONITOR_BAR})
    with pytest.raises(gate.Refused, match="deploy before"):
        _deploy(api)
    assert not [p for p in api.posts if p[0] == "/v1/triggers"]


_CHECK_NOW = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)


def _monitor_run(id: str, created_at: str, **output: Any) -> dict[str, Any]:
    out = {
        "candidate": "workflow:wf_cand@3",
        "verdict": "PASS",
        "alarm": False,
        "alarms": [],
        "reasons": {"invalid": [], "inconclusive": [], "failed": []},
        "window": {"start": "2026-10-05T00:00:00+00:00", "end": "2026-10-12T00:00:00+00:00"},
    }
    out.update(output)
    return {"id": id, "status": "completed", "created_at": created_at, "output": out}


def _fire(result_id: str | None, status: str = "ok", error: str | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "result_id": result_id,
        "error_summary": error,
        "created_at": "2026-10-12T03:17:00+00:00",
    }


def _monitor_api() -> FakeApi:
    """An API whose agent runs the workflow the monitor tests."""
    api = FakeApi()
    api.agent_model = "workflow:wf_cand@3"
    return api


def _check(api: FakeApi, now: datetime = _CHECK_NOW) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = monitor_check.check(
        api, "jarvis-wam", now=now, max_age=timedelta(days=8), out=lines.append
    )
    return code, lines


def test_a_quiet_week_is_ok_and_a_failure_to_re_prove_is_not_an_alarm() -> None:
    api = _monitor_api()
    api.fires = [_fire("wfr_1")]
    api.runs["wfr_1"] = _monitor_run("wfr_1", "2026-10-12T03:17:00+00:00", verdict="FAIL")
    assert _check(api)[0] == monitor_check.OK


def test_an_alarm_exits_one_and_names_what_is_worse() -> None:
    api = _monitor_api()
    api.fires = [_fire("wfr_1")]
    api.runs["wfr_1"] = _monitor_run(
        "wfr_1", "2026-10-12T03:17:00+00:00", alarm=True, alarms=["win_rate", "cost"]
    )
    code, lines = _check(api)
    assert code == monitor_check.ALARM
    assert "worse on win_rate, cost" in lines[0]


def test_a_running_fire_falls_back_to_last_weeks_completed_run() -> None:
    api = _monitor_api()
    api.fires = [_fire("wfr_2"), _fire("wfr_1")]
    api.runs["wfr_2"] = dict(_monitor_run("wfr_2", "2026-10-12T03:17:00+00:00"), status="running")
    api.runs["wfr_1"] = _monitor_run("wfr_1", "2026-10-05T03:17:00+00:00")
    # Monday morning: this week's run is still going, last week's is 7 days old.
    assert _check(api, datetime(2026, 10, 12, 9, 0, tzinfo=UTC))[0] == monitor_check.OK


@pytest.mark.parametrize(
    ("fires", "runs"),
    [
        ([], {}),
        ([_fire(None, status="error", error="budget_usd refused")], {}),
        (
            [_fire("wfr_1")],
            {"wfr_1": _monitor_run("wfr_1", "2026-10-01T03:17:00+00:00")},  # 12 days old
        ),
        (
            [_fire("wfr_1")],
            {"wfr_1": _monitor_run("wfr_1", "2026-10-12T03:17:00+00:00", verdict="INVALID")},
        ),
        (
            [_fire("wfr_1")],
            {"wfr_1": dict(_monitor_run("wfr_1", "2026-10-12T03:17:00+00:00"), status="errored")},
        ),
        # This week's run failed: last week's completed run doesn't hide it.
        (
            [_fire("wfr_2"), _fire("wfr_1")],
            {
                "wfr_2": dict(_monitor_run("wfr_2", "2026-10-12T03:17:00+00:00"), status="errored"),
                "wfr_1": _monitor_run("wfr_1", "2026-10-07T03:17:00+00:00"),
            },
        ),
        # INCONCLUSIVE for an operational reason is a monitor that isn't working.
        (
            [_fire("wfr_1")],
            {
                "wfr_1": _monitor_run(
                    "wfr_1",
                    "2026-10-12T03:17:00+00:00",
                    verdict="INCONCLUSIVE",
                    reasons={"invalid": [], "inconclusive": ["budget_stop"], "failed": []},
                )
            },
        ),
    ],
)
def test_a_monitor_that_isnt_watching_exits_two(
    fires: list[dict[str, Any]], runs: dict[str, dict[str, Any]]
) -> None:
    api = _monitor_api()
    api.fires = fires
    api.runs.update(runs)
    assert _check(api)[0] == monitor_check.NOT_WATCHING


def _thin(id: str, created_at: str) -> dict[str, Any]:
    return _monitor_run(
        id,
        created_at,
        verdict="INCONCLUSIVE",
        reasons={"invalid": [], "inconclusive": ["too_few_clusters"], "failed": []},
    )


def test_a_thin_week_warns_without_paging() -> None:
    """A low-traffic agent has too few clusters every week: that is no reason to page."""
    api = _monitor_api()
    api.fires = [_fire("wfr_1")]
    api.runs["wfr_1"] = _thin("wfr_1", "2026-10-12T03:17:00+00:00")
    code, lines = _check(api)
    assert code == monitor_check.OK
    assert lines[0].startswith("jarvis-wam: warning:")


def test_two_thin_weeks_warn_louder() -> None:
    api = _monitor_api()
    api.fires = [_fire("wfr_2"), _fire("wfr_1")]
    api.runs["wfr_2"] = _thin("wfr_2", "2026-10-12T03:17:00+00:00")
    api.runs["wfr_1"] = _thin("wfr_1", "2026-10-05T03:17:00+00:00")
    code, lines = _check(api)
    assert code == monitor_check.OK
    assert "two thin weeks in a row" in lines[0]


def test_a_monitor_of_a_workflow_no_longer_deployed_is_stale() -> None:
    api = FakeApi()  # the agent runs a plain model again: rolled back
    api.fires = [_fire("wfr_1")]
    api.runs["wfr_1"] = _monitor_run("wfr_1", "2026-10-12T03:17:00+00:00")
    code, lines = _check(api)
    assert code == monitor_check.NOT_WATCHING
    assert "stale" in lines[0]


def test_an_unreadable_api_is_not_watching_never_an_alarm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Down:
        def get(self, path: str, **params: Any) -> Any:
            raise OSError("connection refused")

    with mock.patch.object(monitor_check, "Client", Down):
        assert monitor_check.main(["jarvis-wam"]) == monitor_check.NOT_WATCHING
    assert "the check failed: OSError" in capsys.readouterr().out


def test_the_report_reads_a_monitor_run_and_prints_its_alarm() -> None:
    api = FakeApi()
    run = _monitor_run(
        "wfr_m", "2026-10-12T03:17:00+00:00", alarm=True, alarms=["win_rate"], mode="monitor"
    )
    run.update(
        workflow_id="wf_monitor",
        input={
            "trigger": {"fired_at": "2026-10-12T03:17:00+00:00"},
            "input": {
                "agent": {"agent_id": "agent_1", "version": 7},
                "candidate": {"workflow_id": "wf_cand", "version": 3},
            },
        },
    )
    api.runs["wfr_m"] = run
    lines: list[str] = []
    gate.report(api, "wfr_m", lines.append)
    assert "week 2026-10-05..2026-10-12: ALARM on win_rate" in lines


def test_reanalyze_reads_a_gate_output_from_before_the_monitor() -> None:
    api = FakeApi()
    register.register(api, {"wam-gate": BAR})
    run = _gate_run("wfr_g", start="2026-09-21T00:00:00+00:00")
    run["output"].pop("mode", None)
    run["output"].pop("alpha", None)
    api.runs["wfr_g"] = run
    gate.reanalyze(api, "wfr_g", None, lambda _: None)
    body = api.posts[-1][1]
    assert body["input"]["mode"] == "gate" and body["input"]["alpha"] == BAR["alpha"]


@pytest.mark.parametrize(
    "bar_path", sorted((Path(__file__).parents[2] / "evals" / "bars").glob("*.json"))
)
def test_every_bar_renders_and_budgets(bar_path: Path) -> None:
    """Each bar file registers as a working gate or monitor: its script loads and its
    per-item budgets follow from its fields."""
    bar = json.loads(bar_path.read_text())
    ref = {"id": "wf_x", "version": 1}
    script = paired_eval.build(mode=register.mode_of(bar), bar=bar, item=ref, analysis=ref)
    budgets = load(script)["budgets"]()
    assert budgets["candidate_usd"] == (1 + bar["limits"]["cost"]) * 0.5 * 2.0
    assert bar["alpha"] > 0
