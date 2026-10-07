"""Launch the WaM deploy gate, read its verdict, re-run its analysis.

    AIOS_URL=... AIOS_API_KEY=<operator key> uv run python -m evals.gate launch \\
        --agent AGENT_ID --baseline-model MODEL --candidate WF_ID@VERSION \\
        --environment-id ENV_ID --seed 1 --budget-usd 300
    uv run python -m evals.gate report RUN_ID
    uv run python -m evals.gate reanalyze RUN_ID [--analysis-version N]

``launch`` runs the gate named by ``--gate`` (default ``wam-gate``; each bar file in
``bars/`` registers as its own gate). It refuses a candidate whose version declares
tools, MCP, HTTP or SSH servers, and does the same for every workflow the candidate's
script names by id, the whole way down: an arm acts within the gate's surface, which
holds only the replay tools, so such a candidate would be scored on refusals it
doesn't get in production. A workflow the script picks at run time can't be checked,
so that is a warning. It opens the window at the candidate version's creation (so the
sample is a holdout), prints the items the bar needs (from the analysis version the
gate pins) with a cost and run-slot estimate, and starts the run as the operator with
``budget_usd``.

A verdict is advisory. On PASS, the deploy is one ``PUT`` of the agent's model to the
``workflow:<id>@<version>`` the verdict names.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from evals.client import Api, Client
from evals.workflows import eval_analysis, load, paired_eval

MAX_WINDOW = timedelta(days=31)  # the longest range one sample may cover
SURFACE_FIELDS = ("tools", "mcp_servers", "http_servers", "ssh_servers")
# Runs one in-flight item holds: the item, its three arms and the judge.
ITEM_RUNS = 5
_INVOKE = re.compile(r"invoke_workflow\(\s*([^,)]*)")
_WORKFLOW_ID = re.compile(r"^[\"'](wf_[0-9A-Za-z]+)[\"']$")


class Refused(Exception):
    """The launch was refused before anything ran."""


def _workflow(api: Api, name: str) -> dict[str, Any]:
    found = [w for w in api.get("/v1/workflows", name=name)["data"] if w["name"] == name]
    if not found:
        raise Refused(f"no workflow named {name!r}: run evals.register first")
    return dict(found[0])


def config_of(script: str) -> dict[str, Any]:
    """The ``CONFIG`` a registered gate script was rendered with, read from its source
    without running it: the script text comes from the API, so whoever can update
    the workflow controls it."""
    for node in ast.parse(script).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if (
            isinstance(target, ast.Name)
            and target.id == "CONFIG"
            and isinstance(value, ast.Call)
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
        ):
            return dict(json.loads(value.args[0].value))
    raise Refused("the registered gate script has no CONFIG literal")


def bar_of(script: str) -> dict[str, Any]:
    """The bar baked into a registered gate script."""
    return dict(config_of(script)["bar"])


def trusted(registered: str, local: str, name: str) -> dict[str, Any]:
    """The namespace of ``local``, a script this checkout built from its own templates,
    once it matches the registered text byte for byte. Only local text is executed,
    so a workflow someone else updated is refused rather than run on this machine."""
    if registered != local:
        raise Refused(
            f"the registered {name} differs from this checkout's template: register it "
            "again with evals.register, or launch from the checkout that registered it"
        )
    return load(local)


def _parse_candidate(candidate: str) -> tuple[str, int]:
    workflow_id, sep, version = candidate.partition("@")
    if not sep or not version.isdigit():
        raise Refused(f"--candidate is WF_ID@VERSION, got {candidate!r}")
    return workflow_id, int(version)


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def check_surface(api: Api, label: str, version: dict[str, Any], warnings: list[str]) -> None:
    """Refuse when the version, or a workflow its script invokes by literal id (at its
    current version, recursively), declares a surface. A workflow chosen at run time
    can't be checked, which is a warning."""
    seen: set[str] = set()
    todo = [(label, version)]
    while todo:
        name, current = todo.pop()
        declared = [field for field in SURFACE_FIELDS if current.get(field)]
        if declared:
            raise Refused(
                f"{name} declares {', '.join(declared)}; the gate runs only candidates "
                "that use call_llm, invoke_workflow and agent() (an arm acts within the "
                "gate's surface)"
            )
        script = current["script"]
        if "agent(" in script and "agent() children" not in " ".join(warnings):
            warnings.append(
                "the candidate calls agent(): its agent() children run with no tools in the eval"
            )
        for arg in _INVOKE.findall(script):
            literal = _WORKFLOW_ID.match(arg.strip())
            if literal is None:
                warnings.append(
                    f"{name} invokes a workflow chosen at run time ({arg.strip()}): "
                    "its surface can't be checked before the run"
                )
                continue
            nested = literal.group(1)
            if nested not in seen:
                seen.add(nested)
                todo.append((nested, api.get(f"/v1/workflows/{nested}")))


def plan(
    api: Api,
    *,
    agent_id: str,
    baseline_model: str,
    candidate: str,
    seed: str,
    budget_usd: float,
    environment_id: str,
    now: datetime,
    gate_name: str = paired_eval.NAME,
) -> dict[str, Any]:
    """Everything ``launch`` checks and estimates, and the run it would create."""
    gate = _workflow(api, gate_name)
    config = config_of(gate["script"])
    registered = trusted(gate["script"], paired_eval.build(**config), gate_name)
    bar = dict(registered["BAR"])
    agent = api.get(f"/v1/agents/{agent_id}")
    workflow_id, version = _parse_candidate(candidate)
    cand = api.get(f"/v1/workflows/{workflow_id}/versions/{version}")
    warnings: list[str] = []
    check_surface(api, f"candidate {candidate}", cand, warnings)
    created = _iso(cand["created_at"])
    end = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    start = max(created, end - MAX_WINDOW)
    if start >= end:
        raise Refused(f"candidate {candidate} was created after {end.isoformat()}: no holdout yet")
    # The power and the budgets the run itself will use: the analysis version the gate
    # pins, and the gate's own budget rule.
    pinned = registered["CONFIG"]["analysis"]
    analysis = api.get(f"/v1/workflows/{pinned['id']}/versions/{pinned['version']}")
    needed = trusted(analysis["script"], eval_analysis.build(), eval_analysis.NAME)["required_n"](
        bar
    )
    budgets = registered["budgets"]()
    estimate = {
        "n_required": needed["n"],
        "power": needed["power"],
        "cost_usd": None if needed["n"] is None else needed["n"] * bar["planning"]["item_cost_usd"],
        "item_budget_usd": budgets["item_usd"],
        "candidate_budget_usd": budgets["candidate_usd"],
        "peak_runs": 1 + bar["wave_size"] * (ITEM_RUNS + bar["planning"]["candidate_fanout"]),
    }
    if estimate["cost_usd"] is not None and budget_usd < estimate["cost_usd"]:
        warnings.append(
            f"--budget-usd {budget_usd} is below the estimated ${estimate['cost_usd']:.0f}: "
            "the run may stop on its budget (INCONCLUSIVE)"
        )
    run = {
        "workflow_id": gate["id"],
        "version": gate["version"],
        "environment_id": environment_id,
        "budget_usd": budget_usd,
        "input": {
            "agent": {"agent_id": agent_id, "version": agent["version"]},
            "baseline_model": baseline_model,
            "candidate": {"workflow_id": workflow_id, "version": version},
            "seed": seed,
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "candidate_created_at": created.isoformat(),
        },
    }
    return {"run": run, "estimate": estimate, "warnings": warnings}


def launch(api: Api, *, dry_run: bool, out: Callable[[str], None], **kwargs: Any) -> str | None:
    planned = plan(api, **kwargs)
    for warning in planned["warnings"]:
        out(f"warning: {warning}")
    out(json.dumps(planned["estimate"], indent=2))
    if planned["estimate"]["n_required"] is None:
        raise Refused("the bar can't reach its power within its sample size")
    if dry_run:
        out(json.dumps(planned["run"], indent=2))
        return None
    run = api.post("/v1/runs", planned["run"])
    out(f"gate run {run['id']}")
    return str(run["id"])


def earlier_runs(api: Api, run: dict[str, Any]) -> list[dict[str, Any]]:
    """Earlier gate runs for the same agent and candidate, newest first."""
    key = (run["input"]["agent"]["agent_id"], run["input"]["candidate"])
    found: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = api.get(
            "/v1/runs",
            workflow_id=None if cursor else run["workflow_id"],
            include_archived=None if cursor else "true",
            cursor=cursor,
        )
        for other in page["data"]:
            other_input = other.get("input") or {}
            other_key = (other_input.get("agent", {}).get("agent_id"), other_input.get("candidate"))
            if other["id"] != run["id"] and other_key == key:
                found.append(other)
        cursor = page.get("next_cursor")
        if not cursor:
            return found


def report(api: Api, run_id: str, out: Callable[[str], None]) -> dict[str, Any]:
    run = api.get(f"/v1/runs/{run_id}")
    output = run.get("output") or {}
    out(f"run {run_id}: {run['status']}")
    out(f"verdict: {output.get('verdict')}  candidate: {output.get('candidate')}")
    out(f"reasons: {json.dumps(output.get('reasons'))}")
    if output.get("analysis_error"):
        out(f"analysis failed: {output['analysis_error']} (run `reanalyze {run_id}`)")
    stats = output.get("stats") or {}
    for key in ("n", "clusters", "w", "control", "degenerate", "tool_calls", "cost", "latency_p95"):
        if key in stats:
            out(f"  {key}: {json.dumps(stats[key])}")
    created = output.get("candidate_created_at")
    window = output.get("window") or {}
    if created and window.get("start") and _iso(window["start"]) < _iso(created):
        out("WARNING: the window opens before the candidate version: the sample is no holdout")
    for other in earlier_runs(api, run):
        other_output = other.get("output") or {}
        out(f"earlier: {other['id']} {other['created_at']} {other_output.get('verdict')}")
    return dict(output)


def reanalyze(
    api: Api, run_id: str, analysis_version: int | None, out: Callable[[str], None]
) -> str:
    """Run the analysis again on a finished gate run's records: no item is paid for
    again."""
    run = api.get(f"/v1/runs/{run_id}")
    output = run["output"]
    records = output["records"]
    exclusions = output.get("exclusions", {})
    analysis = _workflow(api, eval_analysis.NAME)
    created = api.post(
        "/v1/runs",
        {
            "workflow_id": analysis["id"],
            "version": analysis_version or analysis["version"],
            "environment_id": run["environment_id"],
            "input": {
                "mode": "gate",
                "bar": output["bar"],
                "alpha": output["bar"]["alpha"],
                "records": records,
                "exclusions": exclusions,
                "attributed": output.get("attributed", []),
                "considered": len(records) + sum(exclusions.values()),
                "flags": output.get("flags", []),
                "seed": output["seed"],
            },
        },
    )
    out(f"analysis run {created['id']}")
    return str(created["id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    go = commands.add_parser("launch")
    go.add_argument("--agent", required=True)
    go.add_argument("--baseline-model", required=True)
    go.add_argument("--candidate", required=True, help="WF_ID@VERSION")
    go.add_argument("--environment-id", required=True)
    go.add_argument("--seed", required=True)
    go.add_argument("--budget-usd", type=float, required=True)
    go.add_argument("--dry-run", action="store_true")
    go.add_argument("--gate", default=paired_eval.NAME, help="the gate (bar) to run")
    show = commands.add_parser("report")
    show.add_argument("run_id")
    again = commands.add_parser("reanalyze")
    again.add_argument("run_id")
    again.add_argument("--analysis-version", type=int)
    args = parser.parse_args(argv)
    api = Client()
    try:
        if args.command == "launch":
            launch(
                api,
                dry_run=args.dry_run,
                out=print,
                agent_id=args.agent,
                baseline_model=args.baseline_model,
                candidate=args.candidate,
                seed=args.seed,
                budget_usd=args.budget_usd,
                environment_id=args.environment_id,
                now=datetime.now(UTC),
                gate_name=args.gate,
            )
        elif args.command == "report":
            report(api, args.run_id, print)
        else:
            reanalyze(api, args.run_id, args.analysis_version, print)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
