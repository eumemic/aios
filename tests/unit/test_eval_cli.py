"""The operator scripts that register the eval workflows and launch and read gates,
against a fake API."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from evals import gate, register
from evals.workflows import eval_item, paired_eval

BAR: dict[str, Any] = json.loads(
    (Path(__file__).parents[2] / "evals" / "bars" / "wam_gate.json").read_text()
)


class FakeApi:
    def __init__(self) -> None:
        self.workflows: dict[str, dict[str, Any]] = {}
        self.versions: dict[tuple[str, int], dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.posts: list[tuple[str, Any]] = []
        self.gets: list[tuple[str, dict[str, Any]]] = []

    def get(self, path: str, **params: Any) -> Any:
        self.gets.append((path, params))
        if path == "/v1/workflows":
            return {"data": [w for w in self.workflows.values() if w["name"] == params.get("name")]}
        if path.startswith("/v1/agents/"):
            return {"id": path.rsplit("/", 1)[1], "version": 7}
        if "/versions/" in path:
            parts = path.split("/")
            return self.versions[(parts[3], int(parts[5]))]
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
            return wf
        if path == "/v1/runs":
            return {"id": f"wfr_{len(self.posts)}"}
        raise AssertionError(path)

    def put(self, path: str, body: Any) -> Any:
        wf = self.workflows[path.rsplit("/", 1)[1]]
        assert body["version"] == wf["version"]
        wf.update(script=body["script"], tools=body["tools"], version=wf["version"] + 1)
        return wf


# ── register ──────────────────────────────────────────────────────────────────


def test_registering_creates_then_leaves_alone_then_updates() -> None:
    api = FakeApi()
    first = register.register(api, BAR)
    assert {ref["version"] for ref in first.values()} == {1}
    gate_wf = api.workflows[first["wam-gate"]["id"]]
    assert gate.bar_of(gate_wf["script"]) == BAR
    assert gate_wf["tools"] == paired_eval.TOOLS
    item_wf = api.workflows[first["eval-item"]["id"]]
    assert item_wf["script"] == eval_item.build(r0=first["eval-r0"], judge=first["eval-judge"])

    assert register.register(api, BAR) == first  # nothing changed: no new versions

    stricter = dict(BAR, delta=0.05)
    again = register.register(api, stricter)
    assert again["wam-gate"] == {"id": first["wam-gate"]["id"], "version": 2}
    assert again["eval-item"] == first["eval-item"]


# ── launch ────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 10, 6, 14, 37, tzinfo=UTC)


def _api_with_candidate(**surface: Any) -> FakeApi:
    api = FakeApi()
    register.register(api, BAR)
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
    register.register(api, BAR)
    api.runs["wfr_g"] = _gate_run("wfr_g", start="2026-09-21T00:00:00+00:00")
    gate.reanalyze(api, "wfr_g", None, lambda _: None)
    path, body = api.posts[-1]
    analysis = next(w for w in api.workflows.values() if w["name"] == "eval-analysis")
    assert path == "/v1/runs" and body["workflow_id"] == analysis["id"]
    assert body["input"]["records"] == [{"x": 1}, {"x": 2}]
    assert body["input"]["considered"] == 5
    assert body["input"]["mode"] == "gate" and body["input"]["bar"] == BAR
